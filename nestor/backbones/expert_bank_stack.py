"""Stacked ownership of expert banks (``NESTOR_STACKED_EXPERT_BANKS=1``).

An :class:`~nestor.backbones.expert_pools.ExpertPoolMLP` layer of the 500m
column arms owns ``8 + 8 * 8 = 72`` bias-free SwiGLU experts as ``72 * 3 = 216``
separate 2-D parameters; the grouped-GEMM path then re-materializes every
bank each micro-step as ``stack -> cast -> transpose -> contiguous`` (three
full passes over the weights) and, worse, its backward accumulates ONE tiny
``copy_`` per parameter into the DDP bucket views -- ~9,300 kernels of ~10 us
per step on the 27-layer 500m column arm, about 15% of the
micro-step in launches and copies that do no arithmetic.

:class:`StackedExpertBank` owns a bank as parameters in exactly the layout
the grouped GEMM consumes -- ``w_up`` / ``w_gate`` ``[E, d_in, hidden]``,
``w_down`` ``[E, hidden, d_out]`` and, for a biased bank, ``b_up``
``[E, hidden]`` / ``b_down`` ``[E, d_out]`` -- built ONCE from the already
initialized per-expert modules (identical values, identical RNG stream), with
the per-expert ``ScaledLinear.weight`` / ``.bias`` attributes re-pointed at
VIEWS of the owners so every other reader (the sparse CPU fallback, probes,
tests) keeps working and its gradients land on the stacked parameter. The
grouped path then pays one ``bf16`` cast per bank per micro-step and its
backward is one gradient tensor per bank.

The same disease lives in the NATIVE shared / group-specific bank of
``RoutedTransformerLayer`` (the MoE baselines and shared-bank column arms),
whose SwiGLU experts carry up/down biases -- so the bank supports biased and
non-gated experts too, and the layer declares its combined
``[shared | group 0 | group 1 | ...]`` bank as ``stacked_moe_bank``.

Same math: every expert's weights are the same numbers in the same GEMMs.
Muon orthogonalizes a ``[E, m, n]`` parameter slice by slice (the batched
Newton--Schulz), so the update is the per-expert update. The muP scaling tag
is copied from the experts (all experts of a bank share it); bias banks carry
the per-expert bias tag (AdamW), so a 2-D ``[E, hidden]`` bias bank never
reaches Muon.

FRESH RUNS ONLY: the checkpoint parameter set changes (like
``NESTOR_DENSE_NO_ROUTER``); an old checkpoint loads only non-strictly and
would leave the banks at init.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from nestor.scaling.blocks import get_scaling, set_scaling


def stacked_banks_requested() -> bool:
    """Whether ``NESTOR_STACKED_EXPERT_BANKS=1`` is set (read at construction)."""

    return os.environ.get("NESTOR_STACKED_EXPERT_BANKS") == "1"


class StackedExpertBank(nn.Module):
    """One bank of shape-identical experts owned as stacked parameters.

    Covers both expert kinds the grouped path dispatches: SwiGLU
    (:class:`~nestor.scaling.blocks.GatedScaledMLP`, ``gated``) and plain
    2-layer :class:`~nestor.scaling.blocks.ScaledMLP` experts, with or
    without up/down biases (the native shared / group-specific bank of
    ``RoutedTransformerLayer`` carries biases; the expert pools do not). A
    SwiGLU gate never carries a bias (standard SwiGLU), and every expert of a
    bank must agree on gatedness and bias presence.

    Attributes:
        num_experts (int): ``E``.
        gated (bool): Whether the experts are SwiGLU (own a ``gate``).
        w_up (nn.Parameter): ``[E, d_in, hidden]`` up projections (the
            grouped GEMM's ``x @ w_up[e]`` layout, i.e. each expert's
            ``layers[0].weight.t()``).
        w_gate (nn.Parameter | None): ``[E, d_in, hidden]`` gate projections
            (SwiGLU banks only).
        w_down (nn.Parameter): ``[E, hidden, d_out]`` down projections.
        b_up (nn.Parameter | None): ``[E, hidden]`` up biases, or None for a
            bias-free bank. Tagged from the per-expert bias (AdamW), so a 2-D
            bias bank never reaches Muon.
        b_down (nn.Parameter | None): ``[E, d_out]`` down biases, or None.
    """

    def __init__(self, experts: Sequence[nn.Module]) -> None:
        """Consolidate ``experts`` into stacked owners.

        Args:
            experts (Sequence[nn.Module]): Shape-identical 2-layer experts
                with ``layers[0]`` (up) and ``layers[1]`` (down)
                :class:`~nestor.scaling.blocks.ScaledLinear` projections, plus
                a bias-free ``gate`` when SwiGLU. Bias presence and gatedness
                must be uniform across the bank.
        """

        super().__init__()
        first = experts[0]
        gated = bool(getattr(first, "gated", False))
        if len(first.layers) != 2:
            raise ValueError(
                "StackedExpertBank expects 2-layer (single-hidden) experts."
            )
        biased = first.layers[0].bias is not None
        for e in experts:
            if bool(getattr(e, "gated", False)) != gated or len(e.layers) != 2:
                raise ValueError(
                    "StackedExpertBank expects a structurally uniform bank."
                )
            if (e.layers[0].bias is not None) != biased or (
                e.layers[1].bias is not None
            ) != biased:
                raise ValueError(
                    "StackedExpertBank expects uniform bias presence."
                )
            if gated and e.gate.bias is not None:
                raise ValueError(
                    "StackedExpertBank expects bias-free gates (standard SwiGLU)."
                )
        self.num_experts = len(experts)
        self.gated = gated
        with torch.no_grad():
            self.w_up = nn.Parameter(
                torch.stack([e.layers[0].weight.t() for e in experts]).contiguous()
            )
            self.w_gate = (
                nn.Parameter(
                    torch.stack([e.gate.weight.t() for e in experts]).contiguous()
                )
                if gated
                else None
            )
            self.w_down = nn.Parameter(
                torch.stack([e.layers[1].weight.t() for e in experts]).contiguous()
            )
            self.b_up = (
                nn.Parameter(
                    torch.stack([e.layers[0].bias for e in experts]).contiguous()
                )
                if biased
                else None
            )
            self.b_down = (
                nn.Parameter(
                    torch.stack([e.layers[1].bias for e in experts]).contiguous()
                )
                if biased
                else None
            )
        sources = [
            (self.w_up, first.layers[0].weight),
            (self.w_down, first.layers[1].weight),
        ]
        if gated:
            sources.append((self.w_gate, first.gate.weight))
        if biased:
            sources.append((self.b_up, first.layers[0].bias))
            sources.append((self.b_down, first.layers[1].bias))
        for owner, source in sources:
            scaling = get_scaling(source)
            if scaling is not None:
                set_scaling(owner, scaling)
        self.install_views(experts)

    def install_views(self, experts: Sequence[nn.Module]) -> None:
        """Point every expert's projections (and biases) at views of the owners.

        The per-expert ``weight`` / ``bias`` entries leave ``_parameters`` (so
        the model exposes each bank once) and become plain attributes that
        view the stacked parameter: forward through them is unchanged and
        their gradients flow to the owner. Must be re-run after any
        ``_apply`` (``.to()`` / ``.cuda()``), which rebuilds the owners.

        Args:
            experts (Sequence[nn.Module]): The experts consolidated in
                :meth:`__init__`, in the same order.
        """

        for i, e in enumerate(experts):
            weight_pairs = [
                (e.layers[0], self.w_up),
                (e.layers[1], self.w_down),
            ]
            if self.gated:
                weight_pairs.append((e.gate, self.w_gate))
            for linear, owner in weight_pairs:
                if "weight" in linear._parameters:
                    del linear._parameters["weight"]
                linear.weight = owner[i].t()  # [out, in] view of the owner
            if self.b_up is not None:
                for linear, owner in (
                    (e.layers[0], self.b_up),
                    (e.layers[1], self.b_down),
                ):
                    if "bias" in linear._parameters:
                        del linear._parameters["bias"]
                    linear.bias = owner[i]  # [out] view of the owner

    def cast(self, dtype: torch.dtype) -> tuple[Tensor, Tensor | None, Tensor]:
        """The weight banks in the GEMM dtype: ``(w_up, w_gate, w_down)``.

        Args:
            dtype (torch.dtype): Autocast (GEMM) dtype.

        Returns:
            tuple[Tensor, Tensor | None, Tensor]: ``[E, d_in, hidden]``,
            ``[E, d_in, hidden]`` (None for a non-gated bank),
            ``[E, hidden, d_out]`` -- one cast each, already contiguous.
        """

        return (
            self.w_up.to(dtype),
            self.w_gate.to(dtype) if self.w_gate is not None else None,
            self.w_down.to(dtype),
        )


class StackedBanksMixin:
    """Owner-side plumbing for a module whose expert banks may be stacked.

    The host module (``ExpertPoolMLP``) DECLARES its banks in ``__init__``
    (:meth:`_declare_expert_banks`: the attributes start as ``None``) and the
    model builder consolidates them LAST, through
    :func:`consolidate_stacked_expert_banks`, after every initialization and
    tagging pass has run on the per-expert parameters -- the reference-muP
    pass (``configure_reference_mup_model``) re-initializes and re-tags each
    expert's ``ScaledLinear`` after construction, and consolidating before it
    would hand that pass transposed VIEWS (a different element order for the
    same random draws, tags on the view instead of the owner). This mixin's
    ``_apply`` re-installs the expert views after device/dtype moves.

    Attributes:
        _stacked_bank_specs (list[tuple[str, list[nn.Module]]]): Declared
            ``(attribute_name, experts)`` banks awaiting consolidation.
        _stacked_view_sources (list[tuple[StackedExpertBank, list[nn.Module]]]):
            Every stacked bank with the experts that view it.
    """

    _stacked_bank_specs: list[tuple[str, list[nn.Module]]]
    _stacked_view_sources: list[tuple[StackedExpertBank, list[nn.Module]]]

    def _declare_expert_banks(
        self, banks: Sequence[tuple[str, Sequence[nn.Module] | None]]
    ) -> None:
        """Register the banks that :func:`consolidate_stacked_expert_banks` may stack.

        Args:
            banks (Sequence[tuple[str, Sequence[nn.Module] | None]]):
                ``(attribute_name, experts)`` pairs; ``None`` experts (an
                absent public pool) are skipped. Each attribute is set to
                ``None`` now (the unstacked state).
        """

        self._stacked_bank_specs = []
        self._stacked_view_sources = []
        for name, experts in banks:
            setattr(self, name, None)
            if experts:
                self._stacked_bank_specs.append((name, list(experts)))

    def _consolidate_expert_banks(self) -> int:
        """Stack every declared bank not yet stacked; returns how many were.

        Idempotent. Called by :func:`consolidate_stacked_expert_banks` when
        ``NESTOR_STACKED_EXPERT_BANKS=1``.
        """

        done = 0
        for name, experts in getattr(self, "_stacked_bank_specs", []):
            if getattr(self, name, None) is not None:
                continue
            stacked = StackedExpertBank(experts)
            setattr(self, name, stacked)
            self._stacked_view_sources.append((stacked, experts))
            done += 1
        return done

    def _apply(self, fn, recurse=True):  # noqa: ANN001, ANN202
        """Re-derive the expert weight views after ``.to()`` / ``.cuda()``."""

        module = super()._apply(fn, recurse)  # type: ignore[misc]
        for stacked, experts in getattr(self, "_stacked_view_sources", []):
            stacked.install_views(experts)
        return module


def consolidate_stacked_expert_banks(model: nn.Module, *, force: bool = False) -> int:
    """Stack the expert banks of every declaring module in ``model``.

    Call once the model is FULLY initialized (after ``configure_reference_mup_model``
    and any other pass that writes or tags per-expert parameters) and before
    it is moved, wrapped (DDP / compile) or handed to an optimizer.

    Args:
        model (nn.Module): The model (or any submodule tree).
        force (bool): Consolidate even when ``NESTOR_STACKED_EXPERT_BANKS`` is
            unset (tests).

    Returns:
        int: Number of banks stacked (0 when the flag is off).
    """

    if not (force or stacked_banks_requested()):
        return 0
    return sum(
        m._consolidate_expert_banks()
        for m in model.modules()
        if isinstance(m, StackedBanksMixin)
    )
