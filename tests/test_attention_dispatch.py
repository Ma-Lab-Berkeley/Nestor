"""Equivalence tests for gather-based ("token dispatch") grouped attention.

``GroupRoutedAttention.forward_gathered`` projects only a group's member tokens
instead of the full sequence, so attention-projection FLOPs scale with
groups-per-token rather than total groups. It must be numerically identical to
the masked ``forward`` path for the tokens that belong to the group.

These run on CPU, where both paths use ``F.scaled_dot_product_attention``. Note a
CPU-only caveat: SDPA returns NaN for a fully-masked query row, so the masked
``forward`` produces NaN at *non-member* rows (the runtime flex kernel returns 0
there, and those rows are zeroed by the router gate anyway). We therefore compare
member rows for the partition case, and full tensors only when every token is a
member of every group (no fully-masked rows).
"""

import torch

from nestor.backbones import Nestor
from nestor.backbones.nestor import GroupRoutedAttention
from tests.test_nestor import create_minimal_nestor_config


def _make_attn() -> GroupRoutedAttention:
    torch.manual_seed(0)
    attn = GroupRoutedAttention(dim=48, num_heads=3, head_dim=16)
    attn.eval()
    return attn


def _inputs(batch_size: int = 2, seq_len: int = 7):
    x = torch.randn(batch_size, seq_len, 48)
    pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    base = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)
    return x, pos, base


def test_full_membership_matches_forward() -> None:
    """Every token a member -> gathered path equals masked path exactly."""
    attn = _make_attn()
    x, pos, base = _inputs()
    members = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool)
    gmask = base & members[:, :, None] & members[:, None, :]
    with torch.no_grad():
        out_masked = attn(x, gmask, pos)
        out_gather = attn.forward_gathered(x, gmask, pos, members)
    assert torch.isfinite(out_masked).all()
    assert torch.allclose(out_masked, out_gather, atol=1e-5, rtol=1e-4)


def test_partition_member_rows_match_forward() -> None:
    """Random membership -> member rows identical, non-member rows zeroed."""
    attn = _make_attn()
    x, pos, base = _inputs()
    torch.manual_seed(1)
    members = torch.rand(x.shape[0], x.shape[1]) > 0.4
    gmask = base & members[:, :, None] & members[:, None, :]
    with torch.no_grad():
        out_masked = attn(x, gmask, pos)
        out_gather = attn.forward_gathered(x, gmask, pos, members)

    member = members.unsqueeze(-1).expand_as(out_gather)
    # Member outputs are well-defined in both paths and must match.
    assert torch.isfinite(out_masked[member]).all()
    assert torch.allclose(out_masked[member], out_gather[member], atol=1e-5,
                          rtol=1e-4)
    # The gathered path leaves non-members exactly zero (no NaN leakage).
    assert torch.count_nonzero(out_gather[~member]) == 0


def test_empty_group_is_zero_and_finite() -> None:
    """A group with no members returns zeros without NaNs (FSDP-safe guard)."""
    attn = _make_attn()
    x, pos, base = _inputs()
    members = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool)
    gmask = base & members[:, :, None] & members[:, None, :]
    with torch.no_grad():
        out_gather = attn.forward_gathered(x, gmask, pos, members)
    assert torch.isfinite(out_gather).all()
    assert torch.count_nonzero(out_gather) == 0


def test_model_forward_equivalence() -> None:
    """End-to-end: toggling dispatch leaves model outputs unchanged.

    The minimal config uses num_groups_per_token == num_groups in its routed
    block, so every token is a member of every group -> no fully-masked rows ->
    the masked path is NaN-free on CPU and full-tensor equality holds.
    """
    config = create_minimal_nestor_config()
    torch.manual_seed(0)
    model = Nestor.from_config(config)
    model.eval()

    batch_size, seq_len = 4, 8
    x = torch.randn(batch_size, seq_len, config.input_dim)
    pos = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

    with torch.no_grad():
        out_masked = model(x, pos, attn_mask).pred
    # Flip the implementation flag in place (same weights, same routing).
    for block in model.blocks:
        for layer in block.layers:
            layer.attention_token_dispatch = True
    with torch.no_grad():
        out_gather = model(x, pos, attn_mask).pred

    assert torch.isfinite(out_masked).all()
    assert torch.allclose(out_masked, out_gather, atol=1e-5, rtol=1e-4)


def test_config_roundtrips_flag() -> None:
    """attention_token_dispatch survives config expansion and block round-trip."""
    config = create_minimal_nestor_config()
    config.attention_token_dispatch = True
    model = Nestor.from_config(config)
    for block in model.blocks:
        assert block.attention_token_dispatch is True
        assert block.to_config()["attention_token_dispatch"] is True
        for layer in block.layers:
            assert layer.attention_token_dispatch is True


if __name__ == "__main__":
    tests = [
        test_full_membership_matches_forward,
        test_partition_member_rows_match_forward,
        test_empty_group_is_zero_and_finite,
        test_model_forward_equivalence,
        test_config_roundtrips_flag,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("all attention-dispatch equivalence tests passed")
