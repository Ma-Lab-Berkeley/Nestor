"""Correctness tests for dispatch-capacity bucketing.

``NESTOR_GATHERED_CAP_BUCKET`` rounds the routed-attention capacity up to a
fixed multiple inside
:func:`nestor.backbones.nestor._bucket_capacity`, which every capacity site
calls (``_build_attention_dispatch_plan``, ``forward_gathered``,
``forward_gathered_full``, ``_gathered_proj_bmm_attention``).

The capacity is derived from live routing counts, so its exact value drifts
with the router and each distinct value is a fresh ``torch.compile``
specialization. Bucketing bounds the number of those shapes; it must buy that
by carrying padding *only*, never by changing results. The padded slots are
already excluded by each caller's ``valid`` mask, so the flag must be a pure
no-op on outputs and gradients.

The flag is read from the environment on every call, so these tests set the
real variable (no monkeypatching) and compare against the unbucketed path,
which serves as the reference implementation throughout.
"""

from __future__ import annotations

import os
from typing import Self

import pytest
import torch

from nestor.backbones.nestor import _bucket_capacity

_BUCKET_FLAG = "NESTOR_GATHERED_CAP_BUCKET"


class _BucketEnv:
    """Context manager setting the real capacity-bucket env var.

    Not a mock: it sets the variable the production helper reads and restores
    the previous value on exit.

    Attributes:
        value (str): Value assigned to ``NESTOR_GATHERED_CAP_BUCKET`` inside
            the block.
        previous (str | None): Value restored on exit (``None`` -> unset).
    """

    def __init__(self, value: int) -> None:
        self.value: str = str(value)
        self.previous: str | None = None

    def __enter__(self) -> Self:
        self.previous = os.environ.get(_BUCKET_FLAG)
        os.environ[_BUCKET_FLAG] = self.value
        return self

    def __exit__(self, *exc: object) -> None:
        if self.previous is None:
            os.environ.pop(_BUCKET_FLAG, None)
        else:
            os.environ[_BUCKET_FLAG] = self.previous


def _reference_bucket(cap: int, seq_len: int, bucket: int) -> int:
    """Slow, obvious reference for the bucketing rule.

    Walks candidate capacities upward instead of computing the round-up in
    closed form, so it is easy to see it returns the smallest multiple of
    ``bucket`` that is >= ``cap`` (clamped to ``seq_len``).

    Args:
        cap (int): Exact required capacity.
        seq_len (int): Sequence length (upper clamp).
        bucket (int): Bucket granularity.

    Returns:
        int: Reference bucketed capacity.
    """

    if bucket <= 1 or cap <= 0:
        return cap
    candidate = bucket
    while candidate < cap:
        candidate += bucket
    return min(candidate, seq_len)


class BucketCapacityHelperTest:
    """Unit tests for :func:`_bucket_capacity`."""

    def test_matches_reference_over_full_range(self) -> None:
        """Closed-form round-up equals the walk-upward reference."""

        seq_len = 512
        for bucket in (2, 8, 64, 128):
            with _BucketEnv(bucket):
                for cap in range(seq_len + 1):
                    assert _bucket_capacity(cap, seq_len) == _reference_bucket(
                        cap, seq_len, bucket
                    ), f"cap={cap} bucket={bucket}"

    def test_never_drops_a_token(self) -> None:
        """Capacity only ever grows, so no routed token loses its slot."""

        seq_len = 4096
        with _BucketEnv(128):
            for cap in range(1, seq_len + 1):
                assert _bucket_capacity(cap, seq_len) >= cap

    def test_never_exceeds_seq_len(self) -> None:
        """Bucketing can never ask for more slots than there are tokens."""

        seq_len = 4096
        with _BucketEnv(128):
            for cap in range(seq_len + 1):
                assert _bucket_capacity(cap, seq_len) <= seq_len

    def test_bounds_distinct_shape_count(self) -> None:
        """The point of the flag: few distinct shapes, hence few recompiles."""

        seq_len = 4096
        with _BucketEnv(128):
            shapes = {_bucket_capacity(c, seq_len) for c in range(seq_len + 1)}
        # ceil(4096/128) buckets, plus the pass-through 0.
        assert len(shapes) == 33
        assert len(shapes) < 64, "must stay under the dynamo recompile limit"

    def test_empty_group_guard_preserved(self) -> None:
        """``cap == 0`` must stay 0 so callers keep their empty-group path."""

        with _BucketEnv(128):
            assert _bucket_capacity(0, 4096) == 0

    def test_disabled_is_exact(self) -> None:
        """Bucket <= 1 restores the exact pre-bucketing capacity."""

        for disabled in (0, 1):
            with _BucketEnv(disabled):
                for cap in (1, 7, 517, 4096):
                    assert _bucket_capacity(cap, 4096) == cap


class BucketedDispatchEquivalenceTest:
    """Bucketing must not change routed-attention outputs or gradients."""

    @staticmethod
    def _build_model() -> torch.nn.Module:
        """Build a tiny grouped token-dispatch LM on CPU.

        Block 0 has 4 attention groups with top-2 routing through the gathered
        path, so the capacity sites under test are actually exercised. Built
        under a fixed seed so two settings share identical weights.

        Returns:
            torch.nn.Module: The model, in eval mode (eval freezes the router
            noise, so routing -- and hence the capacity -- is deterministic).
        """

        from nestor.backbones import NestorConfig
        from nestor.configs.adapters import TextAdapterConfig
        from nestor.heads import LanguageModelingHeadConfig
        from nestor.models import Model, SequenceModel, SequenceModelConfig

        config = SequenceModelConfig(
            adapter_config=TextAdapterConfig(embed_dim=32, vocab_size=64),
            backbone_config=NestorConfig(
                input_dim=32,
                num_blocks=1,
                feature_dims=[32],
                pooling_strides=[1],
                causal=True,
                block_num_layers=2,
                num_groups=[4],
                num_groups_per_token=[2],
                block_num_heads=2,
                block_head_dims=16,
                attention_token_dispatch=True,
                block_mlp_hidden_dims=[[24]],
                block_mlp_activations="gelu",
                router_num_experts_per_group=[1],
                router_num_shared_experts=[1],
                router_num_experts_per_group_per_token=1,
                router_score_mlp_hidden_dims=[[8]],
                router_noise_mlp_hidden_dims=[[8]],
            ),
            head_config=LanguageModelingHeadConfig(vocab_size=64),
        )
        torch.manual_seed(13)
        model = Model.from_config(config)
        assert isinstance(model, SequenceModel)
        return model.eval()

    def _run(self, bucket: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one forward/backward at a given bucket setting.

        Rebuilds the model under a fixed seed so both settings start from
        identical weights and see identical inputs; the bucket flag is then the
        only difference.

        Args:
            bucket (int): Value for ``NESTOR_GATHERED_CAP_BUCKET``.
            seq_len (int): Sequence length.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Prediction tensor and the
            gradient of the first parameter that receives one.
        """

        with _BucketEnv(bucket):
            model = self._build_model()
            torch.manual_seed(1234)
            tokens = torch.randint(0, 64, (2, seq_len))
            out = model(tokens)
            tensor = out.pred
            tensor.sum().backward()
            grad = next(
                p.grad for p in model.parameters() if p.grad is not None
            )
            return tensor.detach().clone(), grad.detach().clone()

    @pytest.mark.parametrize("bucket", [8, 32, 128])
    def test_outputs_match_unbucketed(self, bucket: int) -> None:
        """Bucketed dispatch reproduces the exact-capacity reference."""

        seq_len = 64
        ref_out, ref_grad = self._run(0, seq_len)  # exact capacity = reference
        got_out, got_grad = self._run(bucket, seq_len)
        torch.testing.assert_close(got_out, ref_out, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(got_grad, ref_grad, rtol=1e-5, atol=1e-6)
