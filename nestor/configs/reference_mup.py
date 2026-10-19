"""Configuration for reproducing the variable-width-transformers muP recipe."""

from __future__ import annotations

import dataclasses


REFERENCE_MUP_COMPONENTS = (
    "architecture",
    "residual",
    "embedding",
    "logits",
    "initialization",
    "optimizer",
    "router",
)


@dataclasses.dataclass(kw_only=True)
class ReferenceMupConfig:
    """Reference muP transformations applied after model construction.

    The transformations are opt-in because Nestor's native scaling and residual
    rules are intentional. A component tuple also permits cumulative ablations
    without maintaining several nearly identical model implementations.

    Attributes:
        components (tuple[str, ...]): Enabled transformations. Valid names are
            listed in :data:`REFERENCE_MUP_COMPONENTS`.
        matrix_optimizer (str): Which optimizer the ``"optimizer"`` component
            targets for the in-stream matrices, ``"adamw"`` or ``"muon"``.
            ``"adamw"`` (the reference recipe) retags every matrix into the
            AdamW ``lr / m_width`` group. ``"muon"`` keeps the matrices' native
            spectral Muon tags (so :class:`MuonAdamW` owns them) and places the
            softmax-read score heads (routers, expert-pool path gates) in the
            AdamW group at the BASE rate — their native ``1/d_in`` multiplier
            would leave a deterministic single-matrix router effectively frozen
            under Muon's small companion AdamW learning rate.
        group_router_lr (str): Learning-rate treatment of the block GROUP
            (attention-column) router's score/noise read-out layers.
            ``"native"`` (default, every arm on record) keeps their native
            ``score_lr_mul / d_in`` multiplier -- under the Muon pairing that
            is ``lr_adamw * M_l^-0.5 / 128 ~ 1e-6``, i.e. the read-outs are
            effectively frozen and only the Muon-owned hidden layer adapts.
            ``"base"`` gives them the same base-rate AdamW treatment the
            expert/pool routers receive (``lr_mul = 1``).
        initializer_range (float): Base normal initialization standard deviation.
        width_multiplier (float): muP width multiplier ``m_width``.
        embedding_multiplier (float): Input embedding multiplier ``m_emb``.
        residual_multiplier (float): Additive branch multiplier ``m_residual``.
        norm_eps (float): RMSNorm numerical epsilon.
        num_layers (int): Transformer depth used for depth-scaled output init.
    """

    components: tuple[str, ...] = REFERENCE_MUP_COMPONENTS
    matrix_optimizer: str = "adamw"
    group_router_lr: str = "native"
    initializer_range: float = 0.1
    width_multiplier: float = 8.0
    embedding_multiplier: float = 12.0
    residual_multiplier: float = 0.22
    norm_eps: float = 1e-5
    num_layers: int = 40

    def __post_init__(self) -> None:
        """Validate component names and positive scalar settings."""

        unknown = set(self.components) - set(REFERENCE_MUP_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown reference muP components: {sorted(unknown)}")
        if self.matrix_optimizer not in ("adamw", "muon"):
            raise ValueError(
                "matrix_optimizer must be 'adamw' or 'muon', got "
                f"{self.matrix_optimizer!r}."
            )
        if self.group_router_lr not in ("native", "base"):
            raise ValueError(
                "group_router_lr must be 'native' or 'base', got "
                f"{self.group_router_lr!r}."
            )
        if self.width_multiplier <= 0 or self.num_layers <= 0:
            raise ValueError("width_multiplier and num_layers must be positive.")

    def enabled(self, component: str) -> bool:
        """Return whether one transformation is enabled.

        Args:
            component (str): Component name from ``REFERENCE_MUP_COMPONENTS``.

        Returns:
            bool: Whether the component is enabled.
        """

        return component in self.components
