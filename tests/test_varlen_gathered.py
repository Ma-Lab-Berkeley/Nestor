"""Equivalence tests for packed variable-length gathered attention.

:meth:`~nestor.backbones.nestor.GroupRoutedAttention.forward_gathered_varlen`
attends over each image's members packed end-to-end with ``cu_seqlens``
boundaries, instead of padding every image to ``cap`` = the max member count in
the batch. The padded path is the reference implementation throughout: the two
must agree, because the packed form changes only WHICH slots the attention core
touches, never the math.

This matters because the padded path is wasteful exactly when the router is
imbalanced. On the trained 500m router the median image routes 544 tokens to a
group but the max is 4096, so a single saturated image forces every image to the
full sequence -- 4.67x the attention the routed tokens require. The tests below
therefore include an imbalanced case, which is the case that motivated the path.

Runs on CUDA only: ``varlen_attn`` is a flash-attention kernel with no CPU
implementation.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="varlen_attn is CUDA-only"
)


def _build_group(dim: int = 64, num_heads: int = 4, head_dim: int = 16):
    """Build one GroupRoutedAttention with the compact-causal path verified.

    Args:
        dim (int): Model width.
        num_heads (int): Heads in the group.
        head_dim (int): Per-head width.

    Returns:
        GroupRoutedAttention: Module on CUDA in eval mode, with
        ``_gathered_causal_flash_ok`` forced True so ``forward_gathered`` takes
        the maskless compact-causal branch -- the branch production uses and the
        one the packed path must match.
    """

    from nestor.backbones.nestor import GroupRoutedAttention

    torch.manual_seed(0)
    group = GroupRoutedAttention(
        dim=dim, num_heads=num_heads, head_dim=head_dim
    ).cuda()
    group._gathered_causal_flash = True
    group._gathered_causal_flash_ok = True
    return group.eval()


def _members(batch_size: int, seq_len: int, counts: list[int]) -> torch.Tensor:
    """Membership mask routing a prescribed number of tokens per image.

    Members are the FIRST ``counts[b]`` positions, so the compact order is the
    identity and any disagreement is attributable to the attention, not to the
    ordering.

    Args:
        batch_size (int): Images.
        seq_len (int): Sequence length.
        counts (list[int]): Members per image.

    Returns:
        Bool[Tensor, "batch_size seq_len"]: Membership mask.
    """

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device="cuda")
    for b, c in enumerate(counts):
        mask[b, :c] = True
    return mask


class VarlenGatheredEquivalenceTest:
    """Packed varlen attention must reproduce the padded path exactly."""

    @pytest.mark.parametrize(
        "head_dim",
        [
            16,  # multiple of 8: the kernel's native case
            60,  # the PRODUCTION 500m head_dim -- NOT a multiple of 8, so
                 # it exercises the zero-pad-to-64 path and its adjusted scale.
                 # The first integration attempt failed exactly here.
        ],
    )
    @pytest.mark.parametrize(
        "counts",
        [
            [8, 8, 8, 8],       # balanced
            [1, 16, 3, 32],     # ragged
            [32, 4, 4, 4],      # ONE saturated image sets cap for all -- the
                                # case that costs 4.67x in production
            [0, 12, 0, 7],      # empty groups (must not produce NaN)
        ],
    )
    def test_matches_padded_path(self, counts: list[int], head_dim: int) -> None:
        """Outputs agree with ``forward_gathered`` on every routing shape."""

        seq_len, dim = 32, 64
        group = _build_group(dim=dim, head_dim=head_dim)
        members = _members(len(counts), seq_len, counts)
        torch.manual_seed(1)
        x = torch.randn(len(counts), seq_len, dim, device="cuda")
        pos = torch.arange(seq_len, device="cuda").expand(len(counts), seq_len)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = group.forward_gathered(x, None, pos, members)
            got = group.forward_gathered_varlen(x, pos, members)

        assert torch.isfinite(got).all(), "packed path produced NaN/Inf"
        torch.testing.assert_close(
            got.float(), ref.float(), rtol=2e-2, atol=2e-2
        )

    def test_non_members_stay_zero(self) -> None:
        """Only member positions may be written; the rest must stay exactly 0.

        The group's output is summed into a residual stream, so a non-zero
        non-member position would silently corrupt tokens the group never routed.
        """

        seq_len, dim, counts = 32, 64, [5, 0, 17, 31]
        group = _build_group(dim=dim)
        members = _members(len(counts), seq_len, counts)
        torch.manual_seed(2)
        x = torch.randn(len(counts), seq_len, dim, device="cuda")
        pos = torch.arange(seq_len, device="cuda").expand(len(counts), seq_len)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            got = group.forward_gathered_varlen(x, pos, members)

        assert (got[~members].abs() == 0).all(), "wrote to a non-member position"

    def test_flag_dispatch_after_verification(self) -> None:
        """`NESTOR_GATHERED_VARLEN=1` must route through the packed path only
        AFTER the compact-causal verification, and produce the same output.

        Dispatch is a pure function of the ARGUMENTS (``attn_mask is None``),
        never of module state: a state-keyed branch flips inside the verifying
        forward, so an activation-checkpoint recompute takes a different branch
        than the forward it replays -- the exact CheckpointError this caught in
        the first real training run. The first call passes the mask (padded +
        verification); the second passes ``None`` exactly as production callers
        do once verified, and must take the packed path.
        """

        import os

        seq_len, dim, counts = 32, 64, [16, 4, 9, 32]
        members = _members(len(counts), seq_len, counts)
        torch.manual_seed(4)
        x = torch.randn(len(counts), seq_len, dim, device="cuda")
        pos = torch.arange(seq_len, device="cuda").expand(len(counts), seq_len)
        # A causal mask restricted to members, for the verification call.
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda")
        )
        mask = causal[None] & members[:, :, None] & members[:, None, :]

        previous = os.environ.get("NESTOR_GATHERED_VARLEN")
        os.environ["NESTOR_GATHERED_VARLEN"] = "1"
        try:
            from nestor.backbones.nestor import GroupRoutedAttention

            torch.manual_seed(0)
            group = GroupRoutedAttention(dim=dim, num_heads=4, head_dim=16).cuda()
            group._gathered_causal_flash = True
            group = group.eval()
            assert group._gathered_varlen, "flag not picked up at __init__"

            with torch.autocast("cuda", dtype=torch.bfloat16):
                first = group.forward_gathered(x, mask, pos, members)
                assert group._gathered_causal_flash_ok is True
                # Same args again -> must take the SAME (padded) branch, which
                # is what makes the dispatch checkpoint-recompute-safe.
                replay = group.forward_gathered(x, mask, pos, members)
                # Mask omitted (legal only once verified) -> packed path.
                second = group.forward_gathered(x, None, pos, members)
            torch.testing.assert_close(replay, first, rtol=0, atol=0)
        finally:
            if previous is None:
                os.environ.pop("NESTOR_GATHERED_VARLEN", None)
            else:
                os.environ["NESTOR_GATHERED_VARLEN"] = previous

        ref_group = _build_group(dim=dim)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = ref_group.forward_gathered(x, None, pos, members)

        torch.testing.assert_close(
            first.float(), expected.float(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            second.float(), expected.float(), rtol=2e-2, atol=2e-2
        )

    def test_backward_finite_with_dirty_allocator(self) -> None:
        """Weight gradients stay finite when the allocator hands back garbage.

        Packed rows past ``cu[-1]`` belong to no segment unless the tail is
        chunked into segments too; the varlen kernel then never writes their
        backward dq/dk/dv, and whatever inf/NaN the caching allocator left in
        that memory poisons the weight gradient through ``0 * inf = NaN``.
        Fresh CUDA allocations are zero pages, which is why the other tests
        could not see it -- a production run trained with loss=NaN from
        step 20. Poison the allocator cache with inf blocks first so freed
        memory reuse is garbage, as in a real training loop.
        """

        seq_len, dim, counts = 64, 64, [3, 40, 0, 17]
        group = _build_group(dim=dim)
        members = _members(len(counts), seq_len, counts)
        torch.manual_seed(5)
        x = torch.randn(
            len(counts), seq_len, dim, device="cuda", requires_grad=True
        )
        pos = torch.arange(seq_len, device="cuda").expand(len(counts), seq_len)

        # A spread of block sizes likely to be reused for the kernel's
        # internal packed q/k/v/out and gradient buffers.
        rows = len(counts) * seq_len
        for numel in (rows * dim, rows * 64, rows * 16, 2 * rows * dim):
            poison = torch.full(
                (numel,), float("inf"), device="cuda", dtype=torch.bfloat16
            )
            del poison

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = group.forward_gathered_varlen(x, pos, members)
        out.float().pow(2).sum().backward()

        assert torch.isfinite(x.grad).all(), "non-finite input grad"
        for name, param in group.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"non-finite grad in {name}"
                )

    @pytest.mark.parametrize("head_dim", [16, 60])
    def test_gradients_match_padded_path(self, head_dim: int) -> None:
        """Backward agrees too -- an equal forward with a wrong backward would
        train subtly differently and be far harder to notice."""

        seq_len, dim, counts = 32, 64, [16, 4, 9, 32]
        group = _build_group(dim=dim, head_dim=head_dim)
        members = _members(len(counts), seq_len, counts)
        torch.manual_seed(3)
        base = torch.randn(len(counts), seq_len, dim, device="cuda")
        pos = torch.arange(seq_len, device="cuda").expand(len(counts), seq_len)

        grads = []
        for fn in ("forward_gathered", "forward_gathered_varlen"):
            x = base.clone().requires_grad_(True)
            group.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = (
                    group.forward_gathered(x, None, pos, members)
                    if fn == "forward_gathered"
                    else group.forward_gathered_varlen(x, pos, members)
                )
            out.float().pow(2).sum().backward()
            grads.append(x.grad.detach().float().clone())

        torch.testing.assert_close(grads[1], grads[0], rtol=5e-2, atol=5e-2)
