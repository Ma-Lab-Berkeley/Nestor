"""Loss functions."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from jaxtyping import Float

from nestor.configs.losses import (
    LossConfig,
    CausalLMLossConfig,
    RouterBalanceLossConfig,
    ReferenceMoEAuxLossConfig,
    CompositeLossConfig,
)
from nestor.constants import PADDING_TOKEN_VALUE


@dataclasses.dataclass
class LossOutput:
    """
    Output from loss computation.

    Attributes:
        total (Tensor): Total loss value (scalar).
        components (dict[str, Tensor]): Individual loss components for logging.
    """

    total: Tensor
    components: dict[str, Tensor]


class Loss(nn.Module):
    """
    Base loss class.

    Attributes:
        config (LossConfig): Loss configuration.
    """

    def __init__(self, config: LossConfig):
        """
        Initialize the loss.

        Args:
            config (LossConfig): Loss configuration.
        """
        super().__init__()
        self.config: LossConfig = config

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        aux: dict[str, Any] | None = None,
    ) -> LossOutput:
        """
        Compute the loss.

        Args:
            pred (Tensor): Model predictions.
            target (Tensor): Ground truth targets.
            aux (dict | None): Auxiliary model outputs (e.g., router info).

        Returns:
            LossOutput: Loss value and components.
        """
        raise NotImplementedError

    @staticmethod
    def from_config(config: LossConfig) -> "Loss":
        """
        Create a loss from configuration.

        Args:
            config (LossConfig): Loss configuration.

        Returns:
            Loss: The constructed loss.
        """
        if isinstance(config, CausalLMLossConfig):
            return CausalLMLoss(config)
        elif isinstance(config, RouterBalanceLossConfig):
            return RouterBalanceLoss(config)
        elif isinstance(config, ReferenceMoEAuxLossConfig):
            return ReferenceMoEAuxLoss(config)
        elif isinstance(config, CompositeLossConfig):
            return CompositeLoss(config)
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")


class CausalLMLoss(Loss):
    """
    Next-token prediction loss for autoregressive language modeling.

    Given per-token logits ``(batch_size, seq_len, vocab_size)`` and the input
    token ids ``(batch_size, seq_len)``, the logits and targets are shifted by
    one position so that position ``t`` predicts the token at ``t + 1``. The
    loss is the mean cross-entropy over the predicted positions; padded targets
    (``PADDING_TOKEN_VALUE``) are ignored.

    Attributes:
        config (CausalLMLossConfig): Loss configuration.
    """

    config: CausalLMLossConfig

    def __init__(self, config: CausalLMLossConfig):
        """
        Initialize the causal language-modeling loss.

        Args:
            config (CausalLMLossConfig): Loss configuration.
        """
        super().__init__(config)

    @classmethod
    def sum_loss_and_tokens(
        cls,
        pred: Float[Tensor, "batch_size seq_len vocab_size"],
        target: Tensor,
        *,
        valid_token_mask: Tensor | None = None,
        ignore_index: int = PADDING_TOKEN_VALUE,
        label_smoothing: float = 0.0,
    ) -> tuple[Tensor, int]:
        """Compute sum-reduced causal LM loss and valid target-token count.

        Args:
            pred (Tensor): Per-token logits. Shape:
                ``(batch_size, seq_len, vocab_size)``.
            target (Tensor): Token ids. Shape: ``(batch_size, seq_len)``.
            valid_token_mask (Tensor | None): Optional mask for shifted target
                positions. Shape: ``(batch_size, seq_len - 1)``.
            ignore_index (int): Target value ignored by cross entropy.
            label_smoothing (float): Label smoothing passed to cross entropy.

        Returns:
            tuple[Tensor, int]: Sum-reduced loss and number of valid target
            tokens.

        Raises:
            ValueError: If ``valid_token_mask`` does not match shifted targets.
        """
        shift_logits = pred[:, :-1, :].reshape(-1, pred.shape[-1])
        shift_targets = target[:, 1:].reshape(-1)
        if valid_token_mask is not None:
            expected_shape = target[:, 1:].shape
            if valid_token_mask.shape != expected_shape:
                raise ValueError(
                    "valid_token_mask must have shape "
                    f"{tuple(expected_shape)}, got {tuple(valid_token_mask.shape)}."
                )
            shift_targets = shift_targets.masked_fill(
                ~valid_token_mask.reshape(-1),
                ignore_index,
            )

        num_tokens = int((shift_targets != ignore_index).sum().item())
        if num_tokens == 0:
            return pred.new_tensor(0.0), 0

        # NESTOR_CE_CHUNK_ROWS > 0: compute the cross entropy in row chunks,
        # each under activation checkpointing. One fused call retains the FULL
        # fp32 log-softmax for backward -- [B*(S-1), vocab] is ~13 GiB at the
        # 1b arm's per-rank 8, and that retention is what made per-rank 8 OOM
        # by ~20 MiB. Chunking retains one chunk's softmax at a time and
        # recomputes in backward; sum-reduction makes the split EXACT (the
        # same additions in a different association), not an approximation.
        import os

        chunk_rows = int(os.environ.get("NESTOR_CE_CHUNK_ROWS", "0"))
        if chunk_rows > 0 and shift_logits.requires_grad:
            from torch.utils.checkpoint import checkpoint

            def _chunk_ce(logits: Tensor, targets: Tensor) -> Tensor:
                """Sum-reduced CE for one row chunk (recomputed in backward)."""

                return F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=ignore_index,
                    label_smoothing=label_smoothing,
                    reduction="sum",
                )

            total = shift_logits.new_zeros(())
            for start in range(0, shift_logits.shape[0], chunk_rows):
                total = total + checkpoint(
                    _chunk_ce,
                    shift_logits[start : start + chunk_rows],
                    shift_targets[start : start + chunk_rows],
                    use_reentrant=False,
                )
            return total, num_tokens

        loss = F.cross_entropy(
            shift_logits,
            shift_targets,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="sum",
        )
        return loss, num_tokens

    def forward(
        self,
        pred: Float[Tensor, "batch_size seq_len vocab_size"],
        target: Tensor,
        aux: dict[str, Any] | None = None,
    ) -> LossOutput:
        """
        Compute the shifted next-token cross-entropy loss.

        Args:
            pred (Tensor): Per-token logits. Shape: (batch_size, seq_len, vocab_size).
            target (Tensor): Token ids. Shape: (batch_size, seq_len).
            aux (dict | None): Unused.

        Returns:
            LossOutput: Loss value and components (cross-entropy and perplexity).
        """
        head = (aux or {}).get("fused_head_module")
        if head is not None:
            # Fused head+CE (NESTOR_FUSED_HEAD_CE): ``pred`` is the HIDDEN
            # sequence, not logits. Shift first (the projection is per-token,
            # so it commutes with the shift), then project each row chunk to
            # logits and take its CE inside one checkpointed closure -- the
            # full [tokens, vocab] logits tensor is never materialized, and
            # backward recomputes one chunk's logits at a time. Head gradients
            # flow through the loss; parameter identity is unchanged, so
            # optimizer state and checkpoints are untouched.
            import os

            from torch.utils.checkpoint import checkpoint

            chunk_rows = int(os.environ.get("NESTOR_CE_CHUNK_ROWS", "0")) or 4096
            hidden = pred[:, :-1, :].reshape(-1, pred.shape[-1])
            targets = target[:, 1:].reshape(-1)
            num_tokens = int((targets != PADDING_TOKEN_VALUE).sum().item())
            if num_tokens == 0:
                raise ValueError(
                    "Causal LM loss received zero valid target tokens."
                )

            smoothing = self.config.label_smoothing

            def _project_ce(h: Tensor, t: Tensor) -> Tensor:
                """Logits for one chunk, then sum-CE (recomputed in bwd)."""

                logits = head.projection(h) * head.logit_multiplier
                return F.cross_entropy(
                    logits,
                    t,
                    ignore_index=PADDING_TOKEN_VALUE,
                    label_smoothing=smoothing,
                    reduction="sum",
                )

            loss_sum = hidden.new_zeros(())
            for start in range(0, hidden.shape[0], chunk_rows):
                loss_sum = loss_sum + checkpoint(
                    _project_ce,
                    hidden[start : start + chunk_rows],
                    targets[start : start + chunk_rows],
                    use_reentrant=False,
                )
        else:
            loss_sum, num_tokens = self.sum_loss_and_tokens(
                pred,
                target,
                label_smoothing=self.config.label_smoothing,
            )
        if num_tokens == 0:
            raise ValueError("Causal LM loss received zero valid target tokens.")
        loss = loss_sum / num_tokens
        return LossOutput(
            total=loss,
            components={"cross_entropy": loss, "perplexity": loss.detach().exp()},
        )


class RouterBalanceLoss(Loss):
    """
    Router load balancing loss for MoE models.

    Encourages balanced expert utilization.

    Attributes:
        config (RouterBalanceLossConfig): Loss configuration.
    """

    config: RouterBalanceLossConfig

    @staticmethod
    def _balance_term(
        router_aux: dict[str, Any],
        *,
        num_choices: int,
    ) -> Tensor:
        """Compute the standard auxiliary load-balancing term for one router."""
        scores = router_aux["scores"]
        topk_indices = router_aux["topk_indices"]
        valid_token_mask = router_aux.get(
            "valid_token_mask",
            torch.ones(
                scores.shape[:2],
                dtype=torch.bool,
                device=scores.device,
            ),
        )

        softmax_probs = F.softmax(scores, dim=-1)
        valid_float = valid_token_mask.float()
        num_valid_tokens = valid_float.sum().clamp_min(1.0)

        token_assignments = F.one_hot(topk_indices, num_classes=num_choices).float()
        token_assignments = token_assignments.sum(dim=-2)
        token_assignments = token_assignments * valid_float.unsqueeze(-1)

        assigned_fraction = token_assignments.sum(dim=(0, 1)) / num_valid_tokens
        router_fraction = (softmax_probs * valid_float.unsqueeze(-1)).sum(
            dim=(0, 1)
        ) / num_valid_tokens
        return num_choices * (assigned_fraction * router_fraction).sum()

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        aux: dict[str, Any] | None = None,
    ) -> LossOutput:
        """
        Compute router balance loss from auxiliary outputs.

        Args:
            pred (Tensor): Model predictions (unused).
            target (Tensor): Targets (unused).
            aux (dict | None): Must contain router information under "blocks".

        Returns:
            LossOutput: Balance loss value.
        """
        if aux is None or "blocks" not in aux:
            # No router info, return zero loss
            zero = torch.tensor(0.0, device=pred.device)
            return LossOutput(total=zero, components={"router_balance": zero})

        group_balance_losses: list[Tensor] = []
        expert_balance_losses: list[Tensor] = []

        for block_aux in aux["blocks"]:
            group_router_aux = block_aux.get("group_router")
            if isinstance(group_router_aux, dict):
                group_balance_losses.append(
                    self._balance_term(
                        group_router_aux,
                        num_choices=group_router_aux["scores"].shape[-1],
                    )
                )

            for layer_aux in block_aux.get("layers", []):
                if not isinstance(layer_aux, dict):
                    continue
                expert_router_aux = layer_aux.get("expert_router")
                if isinstance(expert_router_aux, dict):
                    expert_balance_losses.append(
                        self._balance_term(
                            expert_router_aux,
                            num_choices=expert_router_aux["scores"].shape[-1],
                        )
                    )

        if not group_balance_losses and not expert_balance_losses:
            zero = torch.tensor(0.0, device=pred.device)
            return LossOutput(total=zero, components={"router_balance": zero})

        components: dict[str, Tensor] = {}
        total_loss = torch.tensor(0.0, device=pred.device)

        if group_balance_losses:
            group_balance = (
                torch.stack(group_balance_losses).mean() * self.config.balance_weight
            )
            components["router_balance_group"] = group_balance
            total_loss = total_loss + group_balance
        if expert_balance_losses:
            expert_balance = (
                torch.stack(expert_balance_losses).mean() * self.config.balance_weight
            )
            components["router_balance_expert"] = expert_balance
            total_loss = total_loss + expert_balance

        components["router_balance"] = total_loss
        return LossOutput(total=total_loss, components=components)


class ReferenceMoEAuxLoss(Loss):
    """Exact per-layer switch plus router z-loss used by the reference run."""

    config: ReferenceMoEAuxLossConfig

    def __init__(self, config: ReferenceMoEAuxLossConfig) -> None:
        """Initialize the reference MoE auxiliary loss.

        Args:
            config (ReferenceMoEAuxLossConfig): Loss coefficients.
        """

        super().__init__(config)

    def _layer_term(self, router_aux: dict[str, Any]) -> Tensor:
        """Compute one layer's switch loss and router z-loss.

        Args:
            router_aux (dict[str, Any]): Router logits, top-k assignments, and
                token validity tensors.

        Returns:
            Tensor: Scalar unweighted layer auxiliary loss.
        """

        logits = router_aux["scores"]  # [batch, seq_len, num_experts]
        topk_indices = router_aux["topk_indices"]  # [batch, seq_len, top_k]
        valid = router_aux.get(
            "valid_token_mask",
            torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device),
        )
        valid_float = valid.to(dtype=logits.dtype)
        probs = F.softmax(logits, dim=-1)
        acc_probs = (probs * valid_float.unsqueeze(-1)).sum(dim=(0, 1))

        assignments = F.one_hot(
            topk_indices,
            num_classes=logits.shape[-1],
        ).sum(dim=-2)
        frequency = (
            assignments * valid_float.unsqueeze(-1)
        ).sum(dim=(0, 1)).detach()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(frequency)

        switch = logits.shape[-1] * (
            F.normalize(acc_probs, p=1, dim=0)
            * F.normalize(frequency, p=1, dim=0)
        ).sum()
        z_per_token = torch.logsumexp(logits, dim=-1).square()
        z_loss = (z_per_token * valid_float).sum() / valid_float.sum().clamp_min(1)
        return switch + self.config.z_loss_weight * z_loss

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        aux: dict[str, Any] | None = None,
    ) -> LossOutput:
        """Sum expert-router auxiliary losses over transformer layers.

        Args:
            pred (Tensor): Model predictions, used only for a zero tensor device.
            target (Tensor): Training targets (unused).
            aux (dict[str, Any] | None): Backbone router metadata.

        Returns:
            LossOutput: Weighted total and named auxiliary component.
        """

        del target
        terms: list[Tensor] = []
        if aux is not None:
            for block_aux in aux.get("blocks", []):
                for layer_aux in block_aux.get("layers", []):
                    router_aux = layer_aux.get("expert_router")
                    if isinstance(router_aux, dict):
                        terms.append(self._layer_term(router_aux))

        if terms:
            total = torch.stack(terms).sum() * self.config.coefficient
        else:
            total = pred.new_zeros(())
        return LossOutput(total=total, components={"reference_moe_aux": total})


class CompositeLoss(Loss):
    """
    Composite loss combining multiple loss functions.

    Attributes:
        config (CompositeLossConfig): Loss configuration.
        losses (list[Loss]): List of loss functions.
        weights (list[float]): Weights for each loss.
    """

    config: CompositeLossConfig

    def __init__(self, config: CompositeLossConfig):
        """
        Initialize composite loss.

        Args:
            config (CompositeLossConfig): Loss configuration.
        """
        super().__init__(config)
        self.losses = nn.ModuleList([Loss.from_config(c) for c in config.losses])
        self.weights: list[float] = (
            config.weights if config.weights else [1.0] * len(self.losses)
        )
        if len(self.weights) != len(self.losses):
            raise ValueError(
                f"CompositeLoss: len(weights)={len(self.weights)} != "
                f"len(losses)={len(self.losses)}"
            )

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        aux: dict[str, Any] | None = None,
    ) -> LossOutput:
        """
        Compute weighted sum of losses.

        Args:
            pred (Tensor): Model predictions.
            target (Tensor): Targets.
            aux (dict | None): Auxiliary model outputs.

        Returns:
            LossOutput: Combined loss value.
        """
        total = torch.tensor(0.0, device=pred.device)
        components: dict[str, Tensor] = {}

        for loss_fn, weight in zip(self.losses, self.weights):
            output = loss_fn(pred, target, aux)
            total = total + weight * output.total
            components.update(output.components)

        return LossOutput(total=total, components=components)
