"""Tests for ``nestor.eval.long_context`` against plain reference implementations."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from nestor.backbones import NestorConfig
from nestor.configs.adapters import TextAdapterConfig
from nestor.eval.long_context import PerPositionLoss, binned_mean, moving_average, sequence_nll, token_nll
from nestor.heads import LanguageModelingHeadConfig
from nestor.models import Model, SequenceModelConfig


def _tiny_model(num_layers: int = 2, seed: int = 0) -> Model:
    """A tiny causal dense Nestor LM (vocab 256, width 16)."""

    torch.manual_seed(seed)
    backbone = NestorConfig(
        input_dim=16,
        num_blocks=1,
        feature_dims=[16],
        causal=True,
        pooling_strides=[1],
        block_num_layers=num_layers,
        num_groups=[1],
        num_groups_per_token=[1],
        block_num_heads=2,
        block_head_dims=8,
        block_mlp_hidden_dims=[32],
        block_mlp_activations="gelu",
        router_num_experts_per_group=[1],
        router_num_experts_per_group_per_token=1,
        router_score_mlp_hidden_dims=[16],
        router_noise_mlp_hidden_dims=[16],
    )
    config = SequenceModelConfig(
        adapter_config=TextAdapterConfig(embed_dim=16, vocab_size=256, max_length=64),
        backbone_config=backbone,
        head_config=LanguageModelingHeadConfig(vocab_size=256),
    )
    return Model.from_config(config).eval()


def reference_moving_average(values: list[float], window: int) -> list[float]:
    """Centred window mean over the finite entries, NaN when none."""

    half = window // 2
    out = []
    for i in range(len(values)):
        vals = [v for v in values[max(0, i - half) : i + half + 1] if math.isfinite(v)]
        out.append(sum(vals) / len(vals) if vals else math.nan)
    return out


class PerPositionLossTest:
    """The accumulator must equal a dict-of-lists recount."""

    def test_matches_reference(self) -> None:
        rng = np.random.default_rng(0)
        acc = PerPositionLoss(10)
        by_position: dict[int, list[float]] = {}
        for length, start in ((5, 1), (9, 1), (3, 4)):
            nll = rng.random(length)
            acc.add(nll, start)
            for i, v in enumerate(nll):
                by_position.setdefault(start + i, []).append(float(v))
        mean = acc.mean_by_position()
        for pos in range(10):
            if pos in by_position:
                assert math.isclose(mean[pos], sum(by_position[pos]) / len(by_position[pos]))
            else:
                assert math.isnan(mean[pos])
        all_values = [v for vs in by_position.values() for v in vs]
        assert acc.tokens == len(all_values)
        assert math.isclose(acc.cross_entropy(), sum(all_values) / len(all_values))
        out = acc.to_dict(ma_window=3, bin_size=4)
        assert out["mean_by_position"][0] is None and out["tokens"] == len(all_values)
        assert len(out["binned"]) == 3 and out["binned"][0]["start"] == 0

    def test_tensor_input_and_overflow(self) -> None:
        acc = PerPositionLoss(4)
        acc.add(torch.tensor([1.0, 2.0]), 2)
        assert acc.tokens == 2 and acc.counts.tolist() == [0, 0, 1, 1]
        try:
            acc.add(torch.tensor([1.0, 2.0, 3.0]), 2)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on overflow")


class MovingAverageAndBinsTest:
    """Smoothing and binning against reference loops."""

    def test_moving_average_matches_reference(self) -> None:
        values = [1.0, math.nan, 3.0, 4.0, math.nan, math.nan, 7.0, 8.0]
        for window in (1, 3, 5):
            got = moving_average(np.asarray(values), window=window)
            ref = reference_moving_average(values, window)
            for g, r in zip(got, ref):
                assert (math.isnan(g) and math.isnan(r)) or math.isclose(g, r)

    def test_even_window_rejected(self) -> None:
        try:
            moving_average(np.zeros(3), window=2)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    def test_binned_mean(self) -> None:
        sums = np.asarray([1.0, 2.0, 3.0, 0.0, 5.0])
        counts = np.asarray([1, 1, 1, 0, 2])
        bins = binned_mean(sums, counts, bin_size=2)
        assert [b["tokens"] for b in bins] == [2, 1, 2]
        assert math.isclose(bins[0]["cross_entropy"], 1.5)
        assert math.isclose(bins[1]["cross_entropy"], 3.0)
        assert math.isclose(bins[2]["cross_entropy"], 2.5)


class TokenNllTest:
    """``token_nll`` on a slice equals the full-logits reference."""

    def test_slice_matches_full_forward(self) -> None:
        model = _tiny_model()
        torch.manual_seed(1)
        ids = torch.randint(0, 256, (1, 12))
        with torch.no_grad():
            logits = model(ids).pred  # (1, 12, vocab)
            reference = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")
            nll, aux = token_nll(model, ids, target_start=3, target_end=9)
        assert nll.shape == (6,)
        assert torch.allclose(nll, reference[2:8], atol=1e-5)
        assert isinstance(aux, dict)
        with torch.no_grad():
            whole, _ = token_nll(model, ids, target_start=1, target_end=12)
        assert torch.allclose(whole, reference, atol=1e-5)

    def test_bad_ranges_rejected(self) -> None:
        model = _tiny_model()
        ids = torch.randint(0, 256, (1, 6))
        for start, end in ((0, 3), (3, 3), (2, 7)):
            try:
                token_nll(model, ids, target_start=start, target_end=end)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {(start, end)}")


class SequenceNllTest:
    """``sequence_nll`` equals the full-logits reference and chunked ``token_nll``."""

    def test_matches_reference_for_every_chunk_size(self) -> None:
        model = _tiny_model()
        torch.manual_seed(2)
        ids = torch.randint(0, 256, (1, 13))
        with torch.no_grad():
            logits = model(ids).pred
            reference = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")
            for head_chunk in (1, 4, 5, 12, 64):
                nll, aux = sequence_nll(model, ids, head_chunk=head_chunk)
                assert nll.shape == (12,), head_chunk
                assert torch.allclose(nll, reference, atol=1e-5), head_chunk
                assert isinstance(aux, dict)
            chunked = torch.cat(
                [token_nll(model, ids, target_start=s, target_end=min(s + 4, 13))[0] for s in range(1, 13, 4)]
            )
        assert torch.allclose(nll, chunked, atol=1e-5)

    def test_bad_inputs_rejected(self) -> None:
        model = _tiny_model()
        for ids, head_chunk in (
            (torch.randint(0, 256, (1, 1)), 4),
            (torch.randint(0, 256, (2, 6)), 4),
            (torch.randint(0, 256, (1, 6)), 0),
        ):
            try:
                sequence_nll(model, ids, head_chunk=head_chunk)
            except ValueError:
                continue
            raise AssertionError("expected ValueError")
