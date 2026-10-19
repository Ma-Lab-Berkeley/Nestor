"""Evaluation metrics."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float


def compute_router_entropy(
    router_probs: Float[Tensor, "batch_size seq_len num_experts"],
) -> float:
    """
    Compute average routing entropy (measures routing diversity).

    Higher entropy means more diverse routing.

    Args:
        router_probs (Tensor): Router probabilities.
            Shape: (batch_size, seq_len, num_experts).

    Returns:
        float: Average entropy across all tokens.
    """
    # Compute entropy per token
    log_probs = (router_probs + 1e-10).log()
    entropy = -(router_probs * log_probs).sum(dim=-1)  # (batch_size, seq_len)

    return entropy.mean().item()


def compute_router_load_balance(
    router_probs: Float[Tensor, "batch_size seq_len num_experts"],
) -> float:
    """
    Compute router load balance coefficient.

    Returns 1.0 for perfectly balanced load, 0.0 for extremely imbalanced.

    Args:
        router_probs (Tensor): Router probabilities.
            Shape: (batch_size, seq_len, num_experts).

    Returns:
        float: Load balance coefficient in [0, 1].
    """
    # Mean probability per expert across all tokens
    mean_prob = router_probs.mean(dim=(0, 1))  # (num_experts,)
    num_experts = mean_prob.size(0)

    # Coefficient of variation (std / mean)
    # Lower is better, so we return 1 - normalized_cv
    std = mean_prob.std()
    mean = mean_prob.mean()
    cv = std / (mean + 1e-10)

    # Normalize: max CV for uniform mean is sqrt(n-1)/sqrt(n)
    max_cv = ((num_experts - 1) / num_experts) ** 0.5
    normalized_cv = cv / max_cv

    return max(0.0, 1.0 - normalized_cv.item())


def _router_choice_stats(
    router_aux: dict[str, Any],
    valid_example_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Per-choice hard assignment counts and soft routing-probability mass.

    Mirrors the two terms of the load-balancing loss
    (:class:`nestor.train.losses.RouterBalanceLoss`): the hard assignment
    ``f_i`` (a token counts toward a choice when that choice is in its top-k)
    and the soft routing probability ``P_i`` (the softmax mass the router puts
    on each choice). Trivial single-choice routers (dense / non-routed blocks,
    which are always 100%) return ``None`` since their stats carry no
    information.

    Args:
        router_aux (dict): One router's aux entry, with ``topk_indices``,
            ``scores``, optional ``probs`` (softmax of ``scores``), and optional
            ``valid_token_mask``.
        valid_example_mask (Tensor | None): Optional mask selecting real batch
            examples. Shape: ``(batch_size,)``. This excludes distributed
            sampler padding from routing metrics.

    Returns:
        tuple[Tensor, Tensor, Tensor] | None: ``(counts, prob_sum, total)``
        where ``counts`` is a ``[num_choices]`` tensor of how many valid tokens
        selected each choice, ``prob_sum`` is the summed softmax probability per
        choice over valid tokens, and ``total`` is the scalar count of valid
        tokens. Returns ``None`` for a single-choice router. For top-k > 1
        routers ``counts.sum() == k * total``; ``prob_sum.sum() == total``.
    """
    topk_indices = router_aux.get("topk_indices")
    scores = router_aux.get("scores")
    if topk_indices is None or scores is None:
        return None
    num_choices = int(scores.shape[-1])
    if num_choices <= 1:
        return None

    valid_token_mask = router_aux.get("valid_token_mask")
    if valid_token_mask is None:
        valid_token_mask = torch.ones(
            topk_indices.shape[:2], dtype=torch.bool, device=topk_indices.device
        )
    if valid_example_mask is not None:
        valid_token_mask = valid_token_mask & valid_example_mask.to(
            device=valid_token_mask.device,
            dtype=torch.bool,
        ).unsqueeze(-1)
    valid_float = valid_token_mask.to(dtype=torch.float32).unsqueeze(-1)

    assignments = F.one_hot(topk_indices, num_classes=num_choices).to(torch.float32)
    assignments = assignments.sum(dim=-2)  # [B, S, num_choices]
    counts = (assignments * valid_float).sum(dim=(0, 1))  # [num_choices]

    # Soft routing probability P_i: prefer the stored softmax `probs`, else
    # recompute from scores (matches the loss; deterministic at eval, no noise).
    probs = router_aux.get("probs")
    if probs is None:
        probs = F.softmax(scores, dim=-1)
    prob_sum = (probs.to(torch.float32) * valid_float).sum(dim=(0, 1))  # [num_choices]

    total = valid_float.sum()
    return counts, prob_sum, total


def accumulate_routing_stats(
    aux: dict[str, Any] | None,
    accumulator: dict[str, dict[str, Any]],
    valid_example_mask: Tensor | None = None,
) -> None:
    """Accumulate per-router token-assignment counts from one model forward.

    Walks the Nestor hierarchical routing aux (``aux["blocks"]``), adding the
    per-choice assignment counts for every attention-group router and MoE
    expert router into ``accumulator`` (keyed by a stable router id), so the
    caller can divide by the running total after a full eval pass. No-op when
    ``aux`` carries no multi-choice routing (e.g. a plain ViT or a dense
    no-router block), so it is safe to call for any model.

    Args:
        aux (dict | None): A model's ``ModelOutput.aux`` for one forward.
        accumulator (dict): Mutated in place; maps router id to an entry with
            ``kind`` (``"group"`` / ``"expert"``), running ``counts`` (hard
            assignments), running ``prob_sum`` (soft routing probability), and
            running ``total``.
        valid_example_mask (Tensor | None): Optional mask selecting real batch
            examples. Shape: ``(batch_size,)``.
    """
    if not isinstance(aux, dict):
        return
    blocks = aux.get("blocks")
    if not isinstance(blocks, list):
        return

    def _add(key: str, kind: str, router_aux: Any) -> None:
        if not isinstance(router_aux, dict):
            return
        result = _router_choice_stats(router_aux, valid_example_mask)
        if result is None:
            return
        counts, prob_sum, total = result
        entry = accumulator.get(key)
        if entry is None:
            accumulator[key] = {
                "kind": kind,
                "counts": counts.detach().double(),
                "prob_sum": prob_sum.detach().double(),
                "total": total.detach().double(),
            }
        else:
            entry["counts"] = entry["counts"] + counts.detach().double()
            entry["prob_sum"] = entry["prob_sum"] + prob_sum.detach().double()
            entry["total"] = entry["total"] + total.detach().double()

    for block_idx, block_aux in enumerate(blocks):
        if not isinstance(block_aux, dict):
            continue
        _add(f"block{block_idx}/group", "group", block_aux.get("group_router"))
        layers = block_aux.get("layers")
        if not isinstance(layers, list):
            continue
        for layer_idx, layer_aux in enumerate(layers):
            if isinstance(layer_aux, dict):
                _add(
                    f"block{block_idx}/layer{layer_idx}/expert",
                    "expert",
                    layer_aux.get("expert_router"),
                )


def routing_stats_to_metrics(
    accumulator: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Convert accumulated routing counts into a flat metrics dict.

    For each router emits, per choice ``i``:
      - ``routing/<router>/frac/c<i>`` -- fraction of valid tokens whose top-k
        selected choice ``i`` (hard assignment ``f_i``; sums to top-k).
      - ``routing/<router>/prob/c<i>`` -- mean softmax routing probability the
        router placed on choice ``i`` (soft ``P_i``; sums to 1). This is the MoE
        MLP routing probability; it reveals how the router *weights* experts
        independent of the top-k cutoff.

    Plus per-router summaries ``max_frac``, ``max_prob`` (most-preferred
    choice's hard / soft mass) and a ``balance`` coefficient in [0, 1] where 1.0
    is perfectly uniform load. Headline averages over all ``group`` / ``expert``
    routers: ``routing/<kind>/mean_balance``, ``mean_max_frac``, ``mean_max_prob``.

    Args:
        accumulator (dict): Output of :func:`accumulate_routing_stats`.

    Returns:
        dict[str, float]: Metric name -> value. Empty when no routing was seen.
    """
    metrics: dict[str, float] = {}
    per_kind_balance: dict[str, list[float]] = {}
    per_kind_maxfrac: dict[str, list[float]] = {}
    per_kind_maxprob: dict[str, list[float]] = {}

    for key, entry in accumulator.items():
        total = float(entry["total"].item())
        if total <= 0:
            continue
        counts: Tensor = entry["counts"]
        prob_sum: Tensor = entry["prob_sum"]
        num_choices = int(counts.numel())

        # Hard assignment: fraction of valid tokens routed to each choice.
        fractions = (counts / total).tolist()
        for choice_idx, fraction in enumerate(fractions):
            metrics[f"routing/{key}/frac/c{choice_idx}"] = float(fraction)
        max_frac = max(fractions)
        metrics[f"routing/{key}/max_frac"] = float(max_frac)

        # Soft routing probability P_i: mean softmax mass per choice (sums to 1).
        probs = (prob_sum / total).tolist()
        for choice_idx, prob in enumerate(probs):
            metrics[f"routing/{key}/prob/c{choice_idx}"] = float(prob)
        max_prob = max(probs)
        metrics[f"routing/{key}/max_prob"] = float(max_prob)

        # Balance of the per-choice load (normalized to a distribution), via the
        # normalized coefficient of variation: 1.0 = uniform, 0.0 = collapsed.
        load = counts / counts.sum().clamp_min(1.0)
        mean = load.mean()
        std = load.std(unbiased=False)
        cv = float((std / mean.clamp_min(1e-10)).item())
        max_cv = ((num_choices - 1) / num_choices) ** 0.5
        balance = max(0.0, 1.0 - cv / max_cv) if max_cv > 0 else 1.0
        metrics[f"routing/{key}/balance"] = float(balance)

        kind = str(entry["kind"])
        per_kind_balance.setdefault(kind, []).append(balance)
        per_kind_maxfrac.setdefault(kind, []).append(max_frac)
        per_kind_maxprob.setdefault(kind, []).append(max_prob)

    for kind, values in per_kind_balance.items():
        metrics[f"routing/{kind}/mean_balance"] = float(sum(values) / len(values))
    for kind, values in per_kind_maxfrac.items():
        metrics[f"routing/{kind}/mean_max_frac"] = float(sum(values) / len(values))
    for kind, values in per_kind_maxprob.items():
        metrics[f"routing/{kind}/mean_max_prob"] = float(sum(values) / len(values))

    return metrics
