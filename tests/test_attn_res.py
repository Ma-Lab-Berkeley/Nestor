"""Tests for Attention Residuals (Kimi arXiv:2603.15031), residual_mode='attn_res'.

The production ``depth_attention`` reads the depth history in place (cached
inverse RMS + gain folded into the pseudo-query) so autograd never retains
stacked ``(H, batch, seq, dim)`` copies of the history. These tests pin its
numerics (forward and gradients) to a naive transcription of the paper's
Eq 2-4, and smoke-test the end-to-end AttnRes model paths.

They also pin the read's *launch-count* optimization: the per-source scale,
select, multiply and add loop was replaced by one stacked scale, one ``unbind``
and an in-place ``addcmul_`` accumulate, and the read compiles again (the
``@torch.compiler.disable`` that used to graph-break it twice per layer is now
opt-in via ``NESTOR_DEPTH_READ_EAGER=1``). The reference implementation below is
the loop it replaced, verbatim.
"""

import os
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from nestor.backbones import ModelOutput, Nestor, NestorConfig
from nestor.backbones.nestor import DepthHistory, depth_attention
from nestor.scaling.blocks import RMSNorm, ScaledLinear


def _reference_depth_attention(
    values: list[torch.Tensor],
    query: ScaledLinear,
    key_norm: RMSNorm,
) -> torch.Tensor:
    """Naive paper Eq 2-4: stack history, RMSNorm keys, softmax over depth."""

    stacked = torch.stack(values, dim=0)  # (H, batch, seq, dim)
    logits = F.linear(key_norm(stacked), query.weight).squeeze(-1)
    alpha = logits.softmax(dim=0)
    return torch.einsum("hbsd,hbs->bsd", stacked, alpha)


def _make_history(
    num_sources: int, batch: int = 2, seq: int = 5, dim: int = 16
) -> tuple[DepthHistory, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    leaves = [
        torch.randn(batch, seq, dim, generator=generator, requires_grad=True)
        for _ in range(num_sources)
    ]
    history = DepthHistory()
    for leaf in leaves:
        # Scale per-source so the key RMSNorm actually matters.
        history.append(leaf * (0.5 + 0.5 * len(history)))
    return history, leaves


class DepthAttentionTest:
    """Numerical equivalence of depth_attention with the paper formulation."""

    def test_matches_paper_reference(self) -> None:
        dim = 16
        history, _ = _make_history(num_sources=4, dim=dim)
        query = ScaledLinear(dim, 1, role="score", bias=False)
        key_norm = RMSNorm(dim, bias=False)
        with torch.no_grad():
            query.weight.normal_(0.0, 1.0)
            key_norm.weight.uniform_(0.5, 1.5)

        out = depth_attention(history, query, key_norm)
        ref = _reference_depth_attention(history.values, query, key_norm)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_gradients_match_paper_reference(self) -> None:
        dim = 16
        history, leaves = _make_history(num_sources=4, dim=dim)
        query = ScaledLinear(dim, 1, role="score", bias=False)
        key_norm = RMSNorm(dim, bias=False)
        with torch.no_grad():
            query.weight.normal_(0.0, 1.0)
            key_norm.weight.uniform_(0.5, 1.5)
        params = [query.weight, key_norm.weight, *leaves]

        out = depth_attention(history, query, key_norm)
        grads = torch.autograd.grad((out * out).sum(), params)

        ref_history, ref_leaves = _make_history(num_sources=4, dim=dim)
        ref = _reference_depth_attention(ref_history.values, query, key_norm)
        ref_grads = torch.autograd.grad(
            (ref * ref).sum(), [query.weight, key_norm.weight, *ref_leaves]
        )
        for grad, ref_grad in zip(grads, ref_grads):
            torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=1e-5)

    def test_uniform_mix_at_zero_init(self) -> None:
        """Zero-init pseudo-query => exact equal-weight average (paper Sec. 5)."""

        dim = 16
        history, _ = _make_history(num_sources=3, dim=dim)
        query = ScaledLinear(dim, 1, role="score", bias=False, zero_init=True)
        key_norm = RMSNorm(dim, bias=False)

        out = depth_attention(history, query, key_norm)
        uniform = torch.stack(history.values, dim=0).mean(dim=0)
        torch.testing.assert_close(out, uniform, rtol=1e-6, atol=1e-6)


def _create_dense_attn_res_config() -> NestorConfig:
    """1-group / 1-expert AttnRes config: every layer is dense-fastpath
    eligible, mirroring the production ViT-S ablation arms."""

    return NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[32, 32],
        router_num_experts_per_group=[1, 1],
        router_num_shared_experts=[1, 1],
        pooling_strides=[1, 1],
        num_groups=[1, 1],
        num_groups_per_token=[1, 1],
        block_num_layers=[2, 1],
        block_num_heads=[2, 2],
        block_head_dims=[16, 16],
        block_mlp_hidden_dims=[[64], [64]],
        block_mlp_activations=["gelu", "gelu"],
        router_num_experts_per_group_per_token=[1, 1],
        router_score_mlp_hidden_dims=[16],
        router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[16],
        router_noise_mlp_activations="gelu",
        residual_mode="attn_res",
    )


class AttnResDenseFastpathTest:
    """The AttnRes dense fast path must match the general path exactly."""

    def _forward_pair(
        self, *, full_attn: bool, noroute: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        config = _create_dense_attn_res_config()
        model = Nestor.from_config(config).eval()
        with torch.no_grad():
            # Randomize the (zero-init) pseudo-queries and key-norm gains so
            # the depth-attention weights are non-uniform in the comparison.
            for name, param in model.named_parameters():
                if "query.weight" in name:
                    param.normal_(0.0, 1.0)
                elif "key_norm.weight" in name:
                    param.uniform_(0.5, 1.5)

        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, config.input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        layers = [
            layer for block in model.blocks for layer in block.layers
        ]
        assert all(layer._dense_fastpath for layer in layers)
        for layer in layers:
            layer._dense_full_attn = full_attn
            layer._dense_noroute = noroute
        with torch.no_grad():
            out_fast = model(x, pos, attn_mask).pred
            for layer in layers:
                layer._dense_fastpath = False
                layer._dense_full_attn = False
                layer._dense_noroute = False
            out_general = model(x, pos, attn_mask).pred
        return out_fast, out_general

    def test_dense_fastpath_matches_general(self) -> None:
        out_fast, out_general = self._forward_pair(
            full_attn=False, noroute=False
        )
        torch.testing.assert_close(out_fast, out_general, rtol=1e-5, atol=1e-5)

    def test_dense_full_attn_noroute_matches_general(self) -> None:
        """NESTOR_DENSE_FULL_ATTN + NESTOR_DENSE_NO_ROUTER on the AttnRes path."""

        out_fast, out_general = self._forward_pair(full_attn=True, noroute=True)
        torch.testing.assert_close(out_fast, out_general, rtol=1e-5, atol=1e-5)


def _create_attn_res_config(residual_mode: str) -> NestorConfig:
    """Small 2-block config exercising the AttnRes paths (non-causal)."""

    return NestorConfig(
        input_dim=32,
        num_blocks=2,
        feature_dims=[48, 64],
        router_num_experts_per_group=[4, 1],
        router_num_shared_experts=[1, 1],
        pooling_strides=[2, 1],
        num_groups=[2, 1],
        num_groups_per_token=[2, 1],
        block_num_layers=[2, 1],
        block_num_heads=[2, 4],
        block_head_dims=[8, 8],
        block_mlp_hidden_dims=[[64], [128]],
        block_mlp_activations=["gelu", "gelu"],
        router_num_experts_per_group_per_token=[2, 1],
        router_score_mlp_hidden_dims=[16],
        router_score_mlp_activations="gelu",
        router_noise_mlp_hidden_dims=[16],
        router_noise_mlp_activations="gelu",
        residual_mode=residual_mode,
    )


class AttnResModelTest:
    """End-to-end forward/backward through both AttnRes residual modes."""

    def _run_forward_backward(self, residual_mode: str) -> Nestor:
        config = _create_attn_res_config(residual_mode)
        model = Nestor.from_config(config)
        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, config.input_dim)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

        output = model(x, pos, attn_mask)
        assert isinstance(output, ModelOutput)
        assert torch.isfinite(output.pred).all()
        output.pred.sum().backward()

        # DDP-readiness: every parameter must participate in the loss (DDP
        # runs without find_unused_parameters; an unused parameter deadlocks
        # the gradient reduction). A run failed exactly this way when the
        # last block's summary_out_proj had no reader.
        missing = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        assert not missing, f"Parameters without gradient: {missing}"
        return model

    def test_attn_res_forward_backward(self) -> None:
        model = self._run_forward_backward("attn_res")
        block = model.blocks[0]
        # The depth-attention parameters participate in the graph.
        assert block.out_query.weight.grad is not None
        assert block.layers[0].attn_query.weight.grad is not None
        assert block.layers[0].mlp_query.weight.grad is not None
        assert block.layers[0].attn_key_norm.weight.grad is not None

    def test_attn_res_xblock_forward_backward(self) -> None:
        model = self._run_forward_backward("attn_res_xblock")
        # Cross-block summary projections receive gradient via block 1.
        block0 = model.blocks[0]
        assert block0.summary_out_proj.weight.grad is not None
        block1 = model.blocks[1]
        assert block1.summary_in_proj.weight.grad is not None


class DepthAttentionBatchedReadTest:
    """The batched read must equal the per-source loop it replaced.

    The optimization removed ``H`` inverse-RMS multiplies, ``H`` alpha selects
    and ``H`` weighted-value adds per read; only the ``H`` mat-vecs remain
    per-source. Everything here is a pure regrouping of the same arithmetic, so
    fp32 must agree BIT for BIT -- these tests use zero tolerance deliberately.
    """

    def reference_per_source_loop(
        self,
        history: DepthHistory,
        query: ScaledLinear,
        key_norm: RMSNorm,
    ) -> torch.Tensor:
        """The pre-optimization read, verbatim.

        Scales each source's logit separately before the stack, slices the
        softmax weights per source, and accumulates the mixture with
        out-of-place multiplies and adds.

        Args:
            history (DepthHistory): Depth sources with cached inverse RMS.
            query (ScaledLinear): ``dim -> 1`` pseudo-query head.
            key_norm (RMSNorm): Gain-only key RMSNorm.

        Returns:
            torch.Tensor: Depth-attended mixture ``[batch, seq, dim]``.
        """

        values = history.values
        w_eff = (query.weight.squeeze(0) * key_norm.weight).to(values[0].dtype)
        logits = torch.stack(
            [
                torch.matmul(value, w_eff).float() * inv_rms
                for value, inv_rms in zip(values, history.inv_rms)
            ],
            dim=-1,
        )
        alpha = logits.softmax(dim=-1).to(values[0].dtype)
        mixed = values[0] * alpha[..., 0].unsqueeze(-1)
        for i in range(1, len(values)):
            mixed = mixed + values[i] * alpha[..., i].unsqueeze(-1)
        return mixed

    def _read_pair(
        self,
        num_sources: int,
        *,
        dim: int = 16,
        dtype: torch.dtype = torch.float32,
        widen_from: int | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]
    ]:
        """Run both reads on identical inputs; return outputs and leaves.

        Args:
            num_sources (int): Depth-history length ``H``.
            dim (int): Feature width.
            dtype (torch.dtype): Dtype of the depth sources.
            widen_from (int | None): If set, sources from this index on are
                fp32 while the earlier ones keep ``dtype`` -- the
                ``attn_res_xblock`` shape (bf16 summary projections, then an
                fp32 block input) that the old out-of-place adds promoted
                lazily.

        Returns:
            tuple: ``(production, reference, production_leaves,
            reference_leaves)``.
        """

        query = ScaledLinear(dim, 1, role="score", bias=False)
        key_norm = RMSNorm(dim, bias=False)
        with torch.no_grad():
            torch.manual_seed(7)
            query.weight.normal_(0.0, 1.0)
            key_norm.weight.uniform_(0.5, 1.5)

        def build() -> tuple[DepthHistory, list[torch.Tensor]]:
            generator = torch.Generator().manual_seed(11)
            leaves = []
            history = DepthHistory()
            for index in range(num_sources):
                source_dtype = (
                    torch.float32
                    if widen_from is not None and index >= widen_from
                    else dtype
                )
                leaf = torch.randn(
                    2, 5, dim, generator=generator, dtype=torch.float32
                ).to(source_dtype)
                leaf.requires_grad_(True)
                leaves.append(leaf)
                # Per-source rescale so the key RMSNorm actually matters.
                history.append(leaf * (0.5 + 0.25 * index))
            return history, leaves

        history, leaves = build()
        ref_history, ref_leaves = build()
        return (
            depth_attention(history, query, key_norm),
            self.reference_per_source_loop(ref_history, query, key_norm),
            leaves,
            ref_leaves,
        )

    def test_forward_matches_the_loop_bit_for_bit(self) -> None:
        """Every historically interesting width, fp32, zero tolerance."""

        for num_sources in (1, 2, 4, 20, 31, 35):
            out, ref, _, _ = self._read_pair(num_sources)
            torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)

    def test_gradients_match_the_loop_bit_for_bit(self) -> None:
        """The in-place accumulate must not perturb any gradient."""

        for num_sources in (2, 20, 35):
            out, ref, leaves, ref_leaves = self._read_pair(num_sources)
            grads = torch.autograd.grad((out * out).sum(), leaves)
            ref_grads = torch.autograd.grad((ref * ref).sum(), ref_leaves)
            for grad, ref_grad in zip(grads, ref_grads):
                torch.testing.assert_close(grad, ref_grad, rtol=0.0, atol=0.0)

    def test_query_and_key_norm_gradients_match_the_loop(self) -> None:
        """Gradients into the pseudo-query and the folded key gain too."""

        dim = 16
        query = ScaledLinear(dim, 1, role="score", bias=False)
        key_norm = RMSNorm(dim, bias=False)
        with torch.no_grad():
            torch.manual_seed(3)
            query.weight.normal_(0.0, 1.0)
            key_norm.weight.uniform_(0.5, 1.5)

        history, _ = _make_history(num_sources=20, dim=dim)
        out = depth_attention(history, query, key_norm)
        grads = torch.autograd.grad(
            (out * out).sum(), [query.weight, key_norm.weight]
        )

        ref_history, _ = _make_history(num_sources=20, dim=dim)
        ref = self.reference_per_source_loop(ref_history, query, key_norm)
        ref_grads = torch.autograd.grad(
            (ref * ref).sum(), [query.weight, key_norm.weight]
        )
        for grad, ref_grad in zip(grads, ref_grads):
            torch.testing.assert_close(grad, ref_grad, rtol=0.0, atol=0.0)

    def test_bfloat16_matches_the_loop_to_a_rounding_of_the_accumulator(
        self,
    ) -> None:
        """bf16 agrees to ~1 ulp of the mixture's own magnitude.

        The one place the regrouping is NOT bit-exact: ``addcmul_`` forms
        ``value * weight`` at the promoted compute dtype and rounds ONCE on the
        store, where ``mixed + value * weight`` rounded the product to bf16
        first and then rounded the sum. So the batched read drops one rounding
        per source -- strictly the more accurate of the two, but not the same
        bits. Measured divergence is 0.8-1.4 ulps of the accumulator across
        4, 20 and 35 sources; the bound below allows four. The logits path is
        unaffected (the fp32 tests above pin it bit for bit), and fp32 training
        runs are bit-identical.
        """

        bfloat16_eps = 2.0**-8
        for num_sources in (4, 20, 35):
            out, ref, _, _ = self._read_pair(
                num_sources, dtype=torch.bfloat16
            )
            assert out.dtype == torch.bfloat16
            torch.testing.assert_close(
                out,
                ref,
                rtol=0.0,
                atol=4.0 * bfloat16_eps * ref.abs().max().item(),
            )

    def test_widening_history_promotes_like_the_loop(self) -> None:
        """A history that widens mid-way keeps the loop's output dtype.

        Under ``attn_res_xblock`` + autocast the leading sources are bf16
        summary projections and the block input appended after them is fp32.
        The old chain of out-of-place adds promoted the accumulator lazily; the
        in-place accumulate has to promote up front instead.
        """

        with torch.autocast("cpu", dtype=torch.bfloat16):
            out, ref, _, _ = self._read_pair(
                6, dtype=torch.bfloat16, widen_from=3
            )
        assert ref.dtype == torch.float32
        assert out.dtype == torch.float32
        # Same one-rounding-per-source slack as the bf16 test above: the bf16
        # sources are what limit the agreement, not the promotion.
        torch.testing.assert_close(
            out, ref, rtol=0.0, atol=4.0 * 2.0**-8 * ref.abs().max().item()
        )

    def test_no_full_width_history_copy_is_retained(self) -> None:
        """No ``(H, batch, seq, dim)`` tensor exists in the read's graph.

        The docstring's memory promise: every stack the read builds is over
        ``(batch, seq)`` scalars. Walk the autograd graph of the mixture and
        assert nothing saved for backward has the history's full-width shape
        stacked along a depth axis.
        """

        num_sources, dim = 12, 16
        history, _ = _make_history(num_sources=num_sources, dim=dim)
        query = ScaledLinear(dim, 1, role="score", bias=False)
        key_norm = RMSNorm(dim, bias=False)
        out = depth_attention(history, query, key_norm)

        seen: set[int] = set()
        stack: list[torch.autograd.graph.Node] = [out.grad_fn]
        oversized: list[tuple[int, ...]] = []
        while stack:
            node = stack.pop()
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            for attribute in dir(node):
                if not attribute.startswith("_saved"):
                    continue
                try:
                    saved = getattr(node, attribute)
                except (AttributeError, RuntimeError):
                    # Some saved slots are only readable during backward.
                    continue
                if isinstance(saved, torch.Tensor) and saved.numel() >= (
                    num_sources * 2 * 5 * dim
                ):
                    oversized.append(tuple(saved.shape))
            stack.extend(parent for parent, _ in node.next_functions)
        assert not oversized, f"History-sized tensors retained: {oversized}"


class DepthAttentionCompileTest:
    """The read compiles again, at the source counts that once broke it.

    ``@torch.compiler.disable`` cost two graph breaks per layer AND kept every
    read in eager. It was there because the depth axis used to be stacked
    FIRST, which fused all ``H`` inverse-RMS reductions into one persistent
    reduction whose shared memory grew with ``H``; past ~20 sources no Triton
    config fit and the 31- and 35-source column arms died at step 0. The depth
    axis is last now, so these widths must codegen.

    A CPU-inductor run cannot reproduce a *Triton* shared-memory limit, so this
    is a codegen/fusion regression test, not a proof about H200; the GPU
    evidence is that the column reader already compiles at 31 and 35 sources
    with this same layout, and ``NESTOR_DEPTH_READ_EAGER=1`` remains the
    escape hatch.
    """

    def wide_attn_res_config(self, num_layers: int) -> NestorConfig:
        """A one-block AttnRes config whose reads span 1..``2 * L + 1`` sources.

        Args:
            num_layers (int): Layers in the single block.

        Returns:
            NestorConfig: Minimal dense AttnRes backbone.
        """

        return NestorConfig(
            input_dim=8,
            num_blocks=1,
            feature_dims=[8],
            router_num_experts_per_group=[1],
            router_num_shared_experts=[1],
            pooling_strides=[1],
            num_groups=[1],
            num_groups_per_token=[1],
            block_num_layers=[num_layers],
            block_num_heads=[1],
            block_head_dims=[8],
            block_mlp_hidden_dims=[[8]],
            block_mlp_activations=["gelu"],
            router_num_experts_per_group_per_token=[1],
            router_score_mlp_hidden_dims=[4],
            router_score_mlp_activations="gelu",
            router_noise_mlp_hidden_dims=[4],
            router_noise_mlp_activations="gelu",
            residual_mode="attn_res",
        )

    def test_wide_source_backbone_compiles_forward_backward(self) -> None:
        """A 17-layer block compiles: its reads are 1..34 sources, then 35.

        One backbone covers every historically fatal width at once. The block
        seeds its history with the block input and every layer appends two
        sublayer outputs, so the sublayer reads see 1, 2, ... 2 * L sources and
        the block's output read sees 2 * L + 1 -- with L = 17 that is 20, 31
        and 35 among them.
        """

        # Probe compile on a trivial function first so a missing CPU inductor
        # toolchain skips instead of failing, while real codegen regressions in
        # the read still fail loudly below.
        try:
            probe = torch.compile(lambda t: t * 2.0)
            probe(torch.ones(2))
        except Exception as error:  # pragma: no cover - environment-specific
            pytest.skip(f"torch.compile unavailable: {error}")

        num_layers = 17
        widths = set(range(1, 2 * num_layers + 1)) | {2 * num_layers + 1}
        assert {20, 31, 35} <= widths

        torch.manual_seed(5)
        config = self.wide_attn_res_config(num_layers)
        model = Nestor.from_config(config)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "query.weight" in name:
                    param.normal_(0.0, 1.0)
                elif "key_norm.weight" in name:
                    param.uniform_(0.5, 1.5)
        model.eval()

        x = torch.randn(1, 2, config.input_dim)
        pos = torch.arange(2).unsqueeze(0)
        attn_mask = torch.ones(1, 2, 2, dtype=torch.bool)
        with torch.no_grad():
            eager = model(x, pos, attn_mask).pred

        compiled = torch.compile(model, fullgraph=False)
        pred = compiled(x, pos, attn_mask).pred
        # CPU inductor reorders reductions; a loose tolerance is expected.
        torch.testing.assert_close(pred, eager, rtol=1e-4, atol=1e-4)

        pred.square().mean().backward()
        block = model.blocks[0]
        assert block.out_query.weight.grad is not None
        for layer in block.layers:
            assert layer.attn_query.weight.grad is not None
            assert layer.mlp_query.weight.grad is not None


class DepthReadEagerFlagTest:
    """``NESTOR_DEPTH_READ_EAGER=1`` puts the read back outside torch.compile.

    The flag is read once at import (as in ``attention_residual``), so these run
    the real module in a subprocess rather than reaching into its state.
    """

    def run_probe(self, flag: str) -> str:
        """Report whether ``depth_attention`` is dynamo-disabled.

        Args:
            flag (str): Value for ``NESTOR_DEPTH_READ_EAGER``.

        Returns:
            str: ``"True"`` or ``"False"``.
        """

        probe = (
            "from nestor.backbones.nestor import depth_attention\n"
            "print(getattr(depth_attention, '_torchdynamo_disable', False))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env={**os.environ, "NESTOR_DEPTH_READ_EAGER": flag},
            check=True,
        )
        return result.stdout.strip()

    def test_default_compiles_the_read(self) -> None:
        """Unset, the read is traced with the rest of the block."""

        assert self.run_probe("0") == "False"

    def test_flag_graph_breaks_the_read(self) -> None:
        """Set, the read is a ``torch.compiler.disable``d callable again."""

        assert self.run_probe("1") == "True"
