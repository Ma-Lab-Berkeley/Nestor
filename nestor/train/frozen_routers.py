"""Frozen routers for continued training (``NESTOR_FREEZE_ROUTERS=1``).

The domain-adaptation recipe of the sparse models (Nestor and the MoE
control): every router's score and noise networks keep their pretrained
values, the loss-free-balancing choice biases stop moving, and routing runs
without exploration noise, while everything else -- the attention
projections, the experts, the norms, the embeddings, the head, the depth
reads and the hybrid pool's path gate -- trains with the recipe's weight
decay. Routes are NOT cached: a frozen router still maps the current hidden
state to selections and gates, so a token's route can change as the trainable
layers below it move. What is held is the map, not its outputs.

The flag is applied in three places, each through one helper here:

* :func:`freeze_routers` -- BEFORE the distributed wrap
  (``nestor.pipelines.train`` ``_build_model``): DDP registers only the
  parameters that require gradients at wrap time, so freezing after the wrap
  would leave the reducer waiting for gradients that never come.
* :func:`hold_routing_in_eval_mode` -- after every ``model.train()`` in the
  trainer: routers, expert pools and routed layers are put back into eval
  mode (noise off) while the rest of the model stays in train mode (the fused
  head-CE path, for one, is gated on the top-level module's train flag).
* The trainer skips ``update_detached_parameters`` (the ``choice_bias``
  nudges) whenever the flag is set.

A parameter the model already froze at construction (the no-op routers of
``NESTOR_DENSE_NO_ROUTER=1``) stays frozen; nothing here re-enables gradients.
"""

from __future__ import annotations

import dataclasses
import os

from torch import nn

FREEZE_ROUTERS_ENV: str = "NESTOR_FREEZE_ROUTERS"
"""str: Environment flag; ``1`` freezes the routers for the whole run."""


@dataclasses.dataclass(frozen=True)
class FrozenRoutersReport:
    """What :func:`freeze_routers` did.

    Attributes:
        num_frozen (int): Router parameters (tensors) frozen by the call.
        num_trainable (int): Parameters (tensors) left trainable.
        num_total (int): Parameters (tensors) in the model.
        frozen_elements (int): Scalars frozen by the call.
        total_elements (int): Scalars in the model.
        frozen_names (tuple[str, ...]): Names of the frozen router parameters.
    """

    num_frozen: int
    num_trainable: int
    num_total: int
    frozen_elements: int
    total_elements: int
    frozen_names: tuple[str, ...]

    def describe(self) -> str:
        """One log line summarising the freeze."""

        return (
            f"frozen routers: {self.num_frozen} router parameters "
            f"({self.frozen_elements / 1e6:.2f}M of {self.total_elements / 1e6:.1f}M elements) "
            f"frozen; {self.num_trainable}/{self.num_total} parameters trainable"
        )


def freeze_routers_enabled() -> bool:
    """Whether ``NESTOR_FREEZE_ROUTERS=1`` is set.

    Returns:
        bool: ``True`` when the environment asks for frozen routers.
    """

    return os.environ.get(FREEZE_ROUTERS_ENV, "").strip() == "1"


def router_modules(model: nn.Module) -> list[nn.Module]:
    """Every router score network of a model.

    A router is recognised structurally (no backbone import): it owns a
    ``score_mlp`` and a ``choice_bias``. That covers the block-level column
    router, each routed layer's expert-bank router, and an expert pool's
    public and private routers.

    Args:
        model (nn.Module): Any module, wrapped or not.

    Returns:
        list[nn.Module]: The routers in ``model.modules()`` order.
    """

    return [
        module
        for module in model.modules()
        if hasattr(module, "score_mlp") and hasattr(module, "choice_bias")
    ]


def router_parameter_ids(model: nn.Module) -> set[int]:
    """Ids of every router parameter (score and noise networks).

    Args:
        model (nn.Module): The (unwrapped) model.

    Returns:
        set[int]: ``id(param)`` of every parameter under a router module.
    """

    return {id(p) for router in router_modules(model) for p in router.parameters()}


def routing_modules(model: nn.Module) -> list[nn.Module]:
    """Modules whose train-mode behaviour is routing exploration.

    Three kinds, all identified without importing the backbone: a router
    (owns ``score_mlp`` and ``choice_bias``, adds noise when training), an
    expert pool (owns ``public_router`` / ``private_routers`` and passes its
    own train flag to the private routing), and a routed layer
    (``draw_router_noise`` is gated on the layer's train flag).

    Args:
        model (nn.Module): Any module, wrapped or not.

    Returns:
        list[nn.Module]: The routing modules in ``model.modules()`` order.
    """

    routers = set(map(id, router_modules(model)))
    found: list[nn.Module] = []
    for module in model.modules():
        is_pool = hasattr(module, "public_router") or hasattr(module, "private_routers")
        is_routed_layer = callable(getattr(module, "draw_router_noise", None))
        if id(module) in routers or is_pool or is_routed_layer:
            found.append(module)
    return found


def freeze_routers(model: nn.Module) -> FrozenRoutersReport:
    """Set ``requires_grad=False`` on every router parameter of ``model``.

    Must run BEFORE any DDP / FSDP wrap: the wrappers register only the
    parameters that require gradients at wrap time. Parameters outside the
    routers are left exactly as they are (a construction-time freeze is kept).

    Args:
        model (nn.Module): The unwrapped model.

    Returns:
        FrozenRoutersReport: What was frozen.

    Raises:
        ValueError: When the model has no router parameters (the flag would be
            a silent no-op on a dense model built without routers).
    """

    frozen_ids = router_parameter_ids(model)
    if not frozen_ids:
        raise ValueError(
            f"{FREEZE_ROUTERS_ENV}=1 but the model has no router parameters to freeze"
        )
    frozen_names: list[str] = []
    frozen_elements = 0
    total_elements = 0
    num_total = 0
    num_trainable = 0
    for name, param in model.named_parameters():
        num_total += 1
        total_elements += param.numel()
        if id(param) in frozen_ids:
            param.requires_grad_(False)
            frozen_names.append(name)
            frozen_elements += param.numel()
        elif param.requires_grad:
            num_trainable += 1
    return FrozenRoutersReport(
        num_frozen=len(frozen_names),
        num_trainable=num_trainable,
        num_total=num_total,
        frozen_elements=frozen_elements,
        total_elements=total_elements,
        frozen_names=tuple(frozen_names),
    )


def hold_routing_in_eval_mode(model: nn.Module) -> int:
    """Put the routing modules into eval mode (exploration noise off).

    Sets ``training = False`` on each routing module *without* recursing, so
    the experts and attention below a routed layer keep the top-level train
    flag (they have no train-mode behaviour of their own). Call after every
    ``model.train()``.

    Args:
        model (nn.Module): The model, wrapped or not.

    Returns:
        int: Number of modules switched.
    """

    modules = routing_modules(model)
    for module in modules:
        module.training = False
    return len(modules)
