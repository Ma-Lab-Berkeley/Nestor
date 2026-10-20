"""Tests for ``NESTOR_ACTIVATION_CHECKPOINT``.

Checkpointing must be numerically invisible: the same forward values, the same
gradients, and the same per-step side effects (the router's noise-anneal
counter). The interesting cases are the depth-read modes, where the backbone
threads a GROWING source list through the blocks -- a recomputed early block
must see the list as it stood at that block's entry, not as it stands during
the backward.
"""

from __future__ import annotations

import os

import pytest
import torch

from nestor.backbones import Nestor, NestorConfig


def _checkpointing(enabled: bool):
    """Context manager setting ``NESTOR_ACTIVATION_CHECKPOINT``.

    The flag is read per layer call rather than cached at import, so flipping
    the environment is enough and no module reload is needed.

    Args:
        enabled (bool): Whether checkpointing should be on inside the block.

    Returns:
        Any: A context manager restoring the previous value on exit.
    """

    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        """Set the flag, yield, then restore whatever was there before."""

        key = "NESTOR_ACTIVATION_CHECKPOINT"
        previous = os.environ.get(key)
        os.environ[key] = "1" if enabled else "0"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    return _ctx()


def _config(block_residual_mode: str) -> NestorConfig:
    """Build a small multi-block causal config in the requested residual mode.

    Args:
        block_residual_mode (str): ``"projection"``, ``"attn_res_kimi"`` or
            ``"attn_res_column"``.

    Returns:
        NestorConfig: Three blocks of two layers at width 16, so there are
        several checkpoint segments and a genuinely growing depth-source list.
    """

    num_blocks = 3
    groups = [4] * num_blocks
    per_token = [1, 2, 4] if block_residual_mode == "attn_res_column" else [1] * num_blocks
    return NestorConfig(
        input_dim=16,
        num_blocks=num_blocks,
        feature_dims=[16] * num_blocks,
        pooling_strides=[1] * num_blocks,
        causal=True,
        block_num_layers=[2] * num_blocks,
        num_groups=groups,
        num_groups_per_token=per_token,
        block_num_heads=[1] * num_blocks,
        block_head_dims=[8] * num_blocks,
        block_mlp_hidden_dims=[[16]] * num_blocks,
        router_num_experts_per_group=[1] * num_blocks,
        router_num_shared_experts=[1] * num_blocks,
        router_num_experts_per_group_per_token=[1] * num_blocks,
        router_score_mlp_hidden_dims=[8],
        router_noise_mlp_hidden_dims=[8],
        residual_mode="convex",
        block_residual_mode=block_residual_mode,
    )


def _run(model: Nestor, seed: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one forward/backward and return the output and every gradient.

    Args:
        model (Nestor): Backbone to run, left in eval mode so the only source
            of variation is the code path under test.
        seed (int): Seed drawn before the input, so both arms see one batch.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: Output and a name ->
        gradient mapping including the input's.
    """

    torch.manual_seed(seed)
    x = torch.randn(2, 6, 16, requires_grad=True)
    pos = torch.arange(6).expand(2, -1)
    attn_mask = torch.ones(2, 6, 6, dtype=torch.bool)

    out = model(x, pos, attn_mask)
    torch.manual_seed(seed + 1)
    out.pred.mul(torch.randn_like(out.pred)).sum().backward()

    grads = {"input": x.grad}
    grads.update(
        {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
    )
    return out.pred, grads


class ActivationCheckpointEquivalenceTest:
    """Checkpointing must not change values, gradients, or side effects."""

    @pytest.mark.parametrize(
        "mode", ["projection", "attn_res_kimi", "attn_res_column"]
    )
    def test_forward_and_gradients_match_uncheckpointed(self, mode: str) -> None:
        """Recomputed sublayers reproduce the retained ones, in every mode.

        The depth-read modes are the ones at risk: the backbone extends one
        source list per block, so a recomputed block sees the live list unless
        the block snapshots it at entry.
        """

        torch.manual_seed(7)
        config = _config(mode)
        plain = Nestor(config)
        torch.manual_seed(7)
        checkpointed = Nestor(config)
        plain.eval()
        checkpointed.eval()

        with _checkpointing(False):
            plain_out, plain_grads = _run(plain, seed=11)
        with _checkpointing(True):
            ckpt_out, ckpt_grads = _run(checkpointed, seed=11)

        torch.testing.assert_close(ckpt_out, plain_out, rtol=1e-5, atol=1e-6)
        assert plain_grads.keys() == ckpt_grads.keys()
        for name, plain_grad in plain_grads.items():
            torch.testing.assert_close(
                ckpt_grads[name],
                plain_grad,
                rtol=1e-4,
                atol=1e-6,
                msg=lambda text, name=name: f"{name}: {text}",
            )

    def test_depth_key_cache_is_disabled_under_checkpointing(self) -> None:
        """The cache and recompute cannot both be on -- the cache grows.

        Its source list is extended between blocks, so a recomputed early block
        would be handed later blocks' sources; the read rejects that by
        identity. :class:`Nestor` must turn the cache off itself, and must do
        it per FORWARD rather than at construction -- a model built before the
        flag was set is the ordinary case, and the pairing has to be safe
        anyway.
        """

        torch.manual_seed(3)
        model = Nestor(_config("attn_res_column"))  # built with the flag OFF
        model.eval()
        assert model.depth_key_cache_enabled

        caches: list[object] = []
        original = model.blocks[0].forward

        def record(*args, **kwargs):
            """Capture the cache the backbone handed this block."""

            caches.append(kwargs.get("depth_key_cache"))
            return original(*args, **kwargs)

        model.blocks[0].forward = record
        with _checkpointing(True):
            _run(model, seed=5)
        with _checkpointing(False):
            _run(model, seed=5)
        model.blocks[0].forward = original

        assert caches[0] is None, "checkpointed forward must not build a cache"
        assert caches[1] is not None, "the cache must return when it is safe"

    def test_noise_anneal_counter_advances_once_per_step(self) -> None:
        """Recompute must not double-advance the router's noise schedule.

        The counter lives in the router's forward, which a checkpoint re-enters
        during the backward. Advancing it twice per step halves the annealing
        horizon silently -- visible only as a routing-quality regression, so it
        is worth pinning.
        """

        previous = os.environ.get("NESTOR_ROUTER_NOISE_ANNEAL_STEPS")
        os.environ["NESTOR_ROUTER_NOISE_ANNEAL_STEPS"] = "1000"
        try:
            torch.manual_seed(9)
            model = Nestor(_config("projection"))
            model.train()
            routers = [
                block.group_router
                for block in model.blocks
                if hasattr(block, "group_router")
            ]
            assert routers, "config must build real group routers"
            before = int(routers[0]._noise_steps.item())
            with _checkpointing(True):
                _run(model, seed=13)
            after = int(routers[0]._noise_steps.item())
        finally:
            if previous is None:
                os.environ.pop("NESTOR_ROUTER_NOISE_ANNEAL_STEPS", None)
            else:
                os.environ["NESTOR_ROUTER_NOISE_ANNEAL_STEPS"] = previous

        assert after - before == 1, (
            f"noise schedule advanced {after - before} times in one step; "
            "the backward recompute is being counted as a step"
        )


class DistributedCheckpointingTest:
    """Checkpointing must not break DDP's gradient reducer.

    Recompute moves when a sublayer's gradients are produced, and DDP's default
    reducer rejects that with "Expected to have finished reduction in the prior
    iteration". :func:`nestor.utils.distributed.wrap_model_for_distributed`
    pairs checkpointing with ``static_graph=True`` for exactly this reason; the
    combination is easy to break and only shows up on a multi-GPU run, so it is
    worth pinning on gloo where it costs seconds.
    """

    def test_ddp_backward_survives_two_steps(self) -> None:
        """Two steps, because DDP validates its graph on the second.

        Two steps is the whole point: a single step passes even with the wrong
        reducer settings, and the mismatch only surfaces when the second
        iteration checks the first one's bookkeeping.
        """

        import torch.multiprocessing as mp

        if not torch.distributed.is_available():
            pytest.skip("torch.distributed unavailable")

        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(_ddp_worker, args=(2, results), nprocs=2, join=True)
        assert results.get("status") == "OK", results.get("status")


def _ddp_worker(rank: int, world_size: int, results: dict) -> None:
    """Run two DDP steps with checkpointing on and record the outcome.

    Module-level so it is picklable by ``mp.spawn``.

    Args:
        rank (int): This process's rank.
        world_size (int): Total processes.
        results (dict): Manager dict receiving ``"status"``.
    """

    import os as _os

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29531")
    _os.environ["NESTOR_ACTIVATION_CHECKPOINT"] = "1"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        model = DDP(
            Nestor(_config("attn_res_column")),
            gradient_as_bucket_view=True,
            # What wrap_model_for_distributed sets when checkpointing is on.
            static_graph=True,
        )
        # Enough tokens that every attention group receives some: with fewer
        # tokens than groups a group is genuinely unused and static_graph
        # rejects the run, which is a real constraint rather than a bug.
        for _ in range(2):
            x = torch.randn(4, 128, 16, requires_grad=True)
            pos = torch.arange(128).expand(4, -1)
            attn_mask = torch.ones(4, 128, 128, dtype=torch.bool)
            model(x, pos, attn_mask).pred.sum().backward()
            model.zero_grad(set_to_none=True)
        results["status"] = "OK"
    except Exception as error:  # noqa: BLE001
        results["status"] = f"{type(error).__name__}: {error}"
    finally:
        dist.destroy_process_group()
