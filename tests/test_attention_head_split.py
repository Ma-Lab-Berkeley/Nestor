"""Auto head-split: heads/group = round(full_heads / num_groups_per_token).

The attention analogue of mlp_auto_dense_ratio. Splitting a block's full head
count (feature_dims/head_dim) across the active groups keeps active attention
compute at a dense block for any top_k (= num_groups_per_token). At top_k=1 it is
a no-op (each group keeps full heads).
"""

from nestor.backbones import Nestor, NestorConfig


def _heads_per_block(config: NestorConfig) -> list[int]:
    return [bc["block_num_heads"] for bc in config.to_dict()["block_configs"]]


def _cfg(feature_dims, num_groups, top_k, *, head_split, num_heads=None):
    n = len(feature_dims)
    return NestorConfig(
        input_dim=feature_dims[0],
        num_blocks=n,
        feature_dims=list(feature_dims),
        num_groups=num_groups,
        num_groups_per_token=top_k,
        block_num_layers=[1] * n,
        # full heads by default (feature_dims/64); overridden when head_split on
        block_num_heads=num_heads or [d // 64 for d in feature_dims],
        block_head_dims=[64] * n,
        router_num_experts_per_group=[1] * n,
        router_num_shared_experts=[1] * n,
        router_num_experts_per_group_per_token=[1] * n,
        block_mlp_hidden_dims=[[256]] * n,
        block_mlp_activations=["gelu"] * n,
        router_score_mlp_hidden_dims=[[128]] * n,
        router_score_mlp_activations=["gelu"] * n,
        router_noise_mlp_hidden_dims=[[128]] * n,
        router_noise_mlp_activations=["gelu"] * n,
        attention_auto_head_split=head_split,
    )


def test_head_split_divides_by_top_k() -> None:
    # d=512 -> full 8 heads. top_k = [1, 2, 4] -> 8, 4, 2 heads/group.
    config = _cfg([512, 512, 512], [4, 4, 4], [1, 2, 4], head_split=True)
    assert _heads_per_block(config) == [8, 4, 2]


def test_head_split_noop_at_top_k_1() -> None:
    # top_k=1 everywhere -> full heads, identical to no split.
    dims = [256, 384, 512]
    on = _cfg(dims, [2, 2, 1], [1, 1, 1], head_split=True)
    off = _cfg(dims, [2, 2, 1], [1, 1, 1], head_split=False)
    assert _heads_per_block(on) == _heads_per_block(off) == [4, 6, 8]


def test_head_split_off_keeps_explicit_heads() -> None:
    # With the flag off, block_num_heads is used verbatim (current behavior).
    config = _cfg([512, 512], [2, 1], [2, 1], head_split=False, num_heads=[8, 8])
    assert _heads_per_block(config) == [8, 8]


def test_head_split_rounds_when_not_divisible() -> None:
    # d=384 -> full 6 heads; top_k=4 -> round(6/4)=2 (not 1).
    config = _cfg([384], [4], [4], head_split=True)
    assert _heads_per_block(config) == [2]


def test_model_builds_with_head_split() -> None:
    # End-to-end: block-0 has 2 active groups, each with full/2 = 2 heads.
    config = _cfg([256, 256, 256], [2, 1, 1], [2, 1, 1], head_split=True)
    model = Nestor.from_config(config)
    group_attn = model.blocks[0].layers[0].attention_groups[0]
    assert group_attn.num_heads == 2  # round((256/64)/2) = 2


if __name__ == "__main__":
    tests = [
        test_head_split_divides_by_top_k,
        test_head_split_noop_at_top_k_1,
        test_head_split_off_keeps_explicit_heads,
        test_head_split_rounds_when_not_divisible,
        test_model_builds_with_head_split,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print("all attention head-split tests passed")
