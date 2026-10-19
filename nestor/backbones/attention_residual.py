"""Pointwise attention residuals across heterogeneous block widths.

This module implements the block-boundary equation from the Nestor writeup:
each target block uses a learned query to attend over per-token keys from all
earlier block outputs, then mixes their separately projected values.

It also holds :class:`DepthAttentionRead`, the muP-adapted Kimi read
(arXiv:2603.15031 Eq. 3-6) used by ``block_residual_mode="attn_res_kimi"``:
weight-free RMSNorm keys, gain-scaled unit-RMS values, ``1/sqrt(dim)`` depth
logits, and one zero-init pseudo-query -- plus :class:`DepthKeyCache`, which
shares the (weight-free, therefore read-independent) keys of the sources that
stay fixed across a backbone's many reads.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn

from nestor.scaling.blocks import ParamScaling, ScaledLinear, set_scaling


def _depth_softmax(
    depth_logits: Tensor,  # [batch, seq, n]
    gains: Tensor,  # [n]
) -> Tensor:
    """Softmax over the depth axis in fp32, scaled by per-source gains.

    The depth axis ``n`` is the LAST axis throughout the read, deliberately.
    See the module note on depth-axis placement.

    Args:
        depth_logits (Tensor): ``[batch, seq, n]`` temperature-scaled logits.
        gains (Tensor): ``[n]`` learned per-source value gains.

    Returns:
        Tensor: ``[batch, seq, n]`` fp32 mixture weights.
    """

    return depth_logits.float().softmax(dim=-1) * gains.float()


# WHY THE DEPTH AXIS IS LAST EVERYWHERE IN THIS MODULE
#
# The read builds n per-source quantities and combines them. Laying those out as
# [n, batch, seq] -- n slowest -- makes Inductor fuse all n weight-free RMS
# reductions and their stack into ONE persistent reduction kernel whose shared
# memory grows with n. Past roughly 20 sources there is no valid Triton config
# at all and compilation dies outright:
#
#   triton_per_fused_add_div_expand_mul_pow_squeeze_stack_sum_285
#   Required: 311432   Hardware limit: 232448
#
# which the retry wrapper cannot rescue because it is deterministic. It killed
# the 31- and 35-source column arms of the schedule series at step 0.
#
# Stacking along the LAST axis instead ([batch, seq, n]) makes n the
# fastest-varying axis: each (batch, seq) position writes n contiguous values,
# the reduction axis stays ``dim``, and n becomes a short unrolled loop rather
# than a tile multiplier. The tensors involved are tiny either way
# (n * batch * seq fp32, a few MB against the [batch, seq, dim] sources).
#
# Do not "simplify" this back to dim=0 stacking.
#
# Wide reads still exhaust Inductor's shared memory even with the trailing
# axis: at 40 sources the fused reduction asks 352,392 B against the H200's
# 232,448, deterministically. The escape used to be global --
# NESTOR_DEPTH_READ_EAGER=1 wrapped the softmax for EVERY read, so a head-only
# backbone paid 33 graph breaks when only its one 40-source head read needed
# one. The break is now WIDTH-CONDITIONAL: reads at or under
# NESTOR_DEPTH_READ_COMPILE_MAX sources (default 24, comfortably above every
# measured compiling width and below every measured failing one) take the
# compiled softmax, wider reads take the eager twin. NESTOR_DEPTH_READ_EAGER=1
# still forces eager for everything (the old behavior, kept for A/Bs).
_depth_softmax_eager = torch.compiler.disable(_depth_softmax)
_COMPILE_MAX_SOURCES = (
    0
    if os.environ.get("NESTOR_DEPTH_READ_EAGER", "0") == "1"
    else int(os.environ.get("NESTOR_DEPTH_READ_COMPILE_MAX", "24"))
)


def _dispatch_depth_softmax(
    depth_logits: Tensor,  # [batch, seq, n]
    gains: Tensor,  # [n]
) -> Tensor:
    """Route the depth softmax to its compiled or eager form by width.

    Args:
        depth_logits (Tensor): ``[batch, seq, n]`` temperature-scaled logits.
        gains (Tensor): ``[n]`` learned per-source value gains.

    Returns:
        Tensor: ``[batch, seq, n]`` fp32 mixture weights.
    """

    if depth_logits.shape[-1] <= _COMPILE_MAX_SOURCES:
        return _depth_softmax(depth_logits, gains)
    return _depth_softmax_eager(depth_logits, gains)


# The read's output dtype is the promotion over its sources, and the real
# backbone hands it one fp32 source (the embedding, which autocast leaves
# alone) beside bf16 block states -- so every mixture comes out fp32 even
# though it feeds a pre-norm and then a bf16 matmul, which discard the extra
# precision a few ops later. NESTOR_DEPTH_READ_AUTOCAST=1 mixes at the autocast
# dtype instead.
#
# Worth much less than the source count suggests: measured on one H200 at batch
# 4 / seq 4096, peak activation memory moves 67.11 -> 66.46 GiB on
# sched_lin8_colres and 95.05 -> 94.39 GiB on sched_lin16_colres, i.e. ~0.65 GiB
# either way, with no step-time change. The per-read mixtures are evidently
# freed rather than held to the backward, so narrowing them buys little. Off by
# default, and on this evidence not worth the numerics change unless something
# else makes the mixture dtype matter -- it changes training results, so FRESH
# runs only.
_MIX_AT_AUTOCAST_DTYPE = os.environ.get("NESTOR_DEPTH_READ_AUTOCAST", "0") == "1"


def unit_rms_scale(source: Tensor, *, eps: float) -> Tensor:
    """Return the scalar factor of the weight-free Kimi key map.

    The key of a depth source is ``source * inv_rms``, and this is the
    ``inv_rms`` half: a ``[..., 1]`` per-token scalar rather than a full-width
    tensor. :class:`DepthAttentionRead` never materializes the product (see its
    ``forward``), so this scalar is the only part of the key map anything
    retains -- ``dim`` times smaller than the key itself.

    Args:
        source (Tensor): Depth source of shape ``[batch_size, seq_len, dim]``
            (any leading shape works; only the last axis is reduced).
        eps (float): RMSNorm numerical epsilon.

    Returns:
        Tensor: fp32 inverse RMS of shape ``source.shape[:-1] + (1,)``.
    """

    return torch.rsqrt(source.float().square().mean(dim=-1, keepdim=True) + eps)


def unit_rms_key(
    source: Tensor,
    *,
    eps: float,
    dtype: torch.dtype,
) -> Tensor:
    """Normalize one depth source to unit RMS, the weight-free Kimi key map.

    The explicit key. :class:`DepthAttentionRead` does not call this -- it
    factors the scalar out instead (:func:`unit_rms_scale`) so no full-width key
    is ever built or retained -- but the definition stays here as the reference
    the factored form is checked against, and as the readable statement of what
    the read's key IS.

    Args:
        source (Tensor): Depth source of shape ``[batch_size, seq_len, dim]``
            (any leading shape works; only the last axis is reduced).
        eps (float): RMSNorm numerical epsilon.
        dtype (torch.dtype): Dtype the read mixes in; the key is cast to it.

    Returns:
        Tensor: Unit-RMS key with ``source``'s shape and ``dtype``.
    """

    return (source * unit_rms_scale(source, eps=eps)).to(dtype)


class DepthKeyCache:
    """Key scales of depth sources that are fixed across many reads.

    A depth-read backbone re-reads the SAME source list many times: under
    ``block_residual_mode='attn_res_column'`` every one of a block's ``2 *
    num_layers`` reads sees the embedding plus every completed block's exported
    column streams, and only the tail (this block's ``k`` partial streams)
    changes between reads. Because the key map is weight-FREE (RMSNorm with no
    learnable gain), a source's key scale is read-INDEPENDENT: every read may
    share it, applying only its own query (to form logits) and its own value
    gains (to form the mixture) afterwards. This cache is that shared prefix --
    pure common-subexpression elimination, exact to the bit.

    What is cached is the ``[batch, seq, 1]`` scalar ``inv_rms``, not the
    full-width key: :meth:`DepthAttentionRead.forward` folds the scalar into
    its depth weights and mixes the RAW sources, so a key tensor is never built.
    The cache is therefore ``dim`` times smaller than it looks, and the saving
    is now compute rather than memory -- a squared-mean reduction over
    ``[batch, seq, dim]`` per source per read, which on the 4-block
    ``(3, 5, 7, 1)`` ladder drops from 369 to 129.

    Scales are computed lazily and the source list may grow between reads (the
    backbone appends each block's exports); a later read re-uses the earlier
    scales and reduces only the appended sources.

    Attributes:
        sources (list[Tensor]): Cached sources in depth order, each of shape
            ``[batch_size, seq_len, dim]``. Callers must pass these SAME tensor
            objects as the leading sources of every read using this cache;
            :meth:`DepthAttentionRead.forward` verifies it by identity.
        eps (float): RMSNorm epsilon the scales were built with. Must equal the
            consuming read's ``eps``.
        _scales (list[Tensor]): fp32 ``[batch_size, seq_len, 1]`` inverse RMS
            per source, aligned with a prefix of ``sources``.
    """

    sources: list[Tensor]
    eps: float
    _scales: list[Tensor]

    def __init__(
        self,
        sources: Sequence[Tensor] = (),
        *,
        eps: float = 1e-6,
    ) -> None:
        """Initialize a depth key-scale cache.

        Args:
            sources (Sequence[Tensor]): Initial fixed sources in depth order,
                each of shape ``[batch_size, seq_len, dim]``.
            eps (float): RMSNorm epsilon; must match the consuming reads.
        """

        self.sources = list(sources)
        self.eps = float(eps)
        self._scales = []

    @property
    def num_sources(self) -> int:
        """int: Number of cached sources, i.e. the covered read prefix."""

        return len(self.sources)

    def extend(self, sources: Iterable[Tensor]) -> None:
        """Append newly completed sources; their scales stay lazy.

        Args:
            sources (Iterable[Tensor]): New fixed sources in depth order, each
                of shape ``[batch_size, seq_len, dim]``. The caller must hand
                the SAME tensor objects to the reads that consume them.
        """

        self.sources.extend(sources)

    def scales(self) -> list[Tensor]:
        """Return every cached source's key scale, computing new ones.

        Returns:
            list[Tensor]: ``num_sources`` fp32 inverse-RMS tensors of shape
            ``[batch_size, seq_len, 1]``, in depth order.
        """

        for source in self.sources[len(self._scales):]:
            self._scales.append(unit_rms_scale(source, eps=self.eps))
        return self._scales


class BlockAttentionResidual(nn.Module):
    """Mix earlier block outputs into one target-width residual stream.

    Every source owns a separate key and value projection into ``target_width``.
    A zero-initialized target query scores the projected keys independently at
    every token position, and a softmax across sources produces a convex value
    mixture. Source tensors may have any number of leading dimensions, but all
    sources must share those dimensions.

    Attributes:
        source_widths (tuple[int, ...]): Last-axis widths of the source tensors.
        target_width (int): Last-axis width of the returned tensor.
        key_projections (nn.ModuleList): Per-source scaled semi-orthogonal key
            maps from ``[..., source_width]`` to ``[..., target_width]``.
        value_projections (nn.ModuleList): Per-source scaled semi-orthogonal
            value maps from ``[..., source_width]`` to ``[..., target_width]``.
        query (ScaledLinear): Zero-initialized learned query represented as a
            score map from ``[..., target_width]`` to ``[..., 1]``.
    """

    source_widths: tuple[int, ...]
    target_width: int
    key_projections: nn.ModuleList
    value_projections: nn.ModuleList
    query: ScaledLinear

    def __init__(
        self,
        source_widths: Sequence[int],
        target_width: int,
    ) -> None:
        """Initialize a block attention residual.

        Args:
            source_widths (Sequence[int]): Positive width of every retained
                source, in depth order.
            target_width (int): Positive width required by the target block.

        Raises:
            ValueError: If there are no sources or any width is non-positive.
        """

        super().__init__()
        self.source_widths = tuple(int(width) for width in source_widths)
        self.target_width = int(target_width)
        if not self.source_widths:
            raise ValueError("BlockAttentionResidual needs at least one source.")
        if any(width < 1 for width in self.source_widths):
            raise ValueError("All source widths must be positive.")
        if self.target_width < 1:
            raise ValueError("target_width must be positive.")

        self.key_projections = nn.ModuleList(
            [
                ScaledLinear(
                    source_width,
                    self.target_width,
                    role="projection",
                    bias=False,
                )
                for source_width in self.source_widths
            ]
        )
        self.value_projections = nn.ModuleList(
            [
                ScaledLinear(
                    source_width,
                    self.target_width,
                    role="projection",
                    bias=False,
                )
                for source_width in self.source_widths
            ]
        )
        self.query = ScaledLinear(
            self.target_width,
            1,
            role="score",
            bias=False,
            zero_init=True,
        )

    def forward(self, sources: Sequence[Tensor]) -> Tensor:
        """Attend over retained block outputs at every token independently.

        Args:
            sources (Sequence[Tensor]): Tensors in depth order. Source ``i`` has
                shape ``[..., source_widths[i]]``; all leading shapes must be
                equal. For sequence models this is normally
                ``[batch_size, seq_len, source_widths[i]]``.

        Returns:
            Tensor: Per-position convex mixture with shape
            ``[..., target_width]``.

        Raises:
            ValueError: If the source count or any source shape is incorrect.
        """

        if len(sources) != len(self.source_widths):
            raise ValueError(
                f"Expected {len(self.source_widths)} sources, got {len(sources)}."
            )

        leading_shape = sources[0].shape[:-1]
        projected_values: list[Tensor] = []
        logits: list[Tensor] = []
        for source_idx, (source, expected_width, key_proj, value_proj) in enumerate(
            zip(
                sources,
                self.source_widths,
                self.key_projections,
                self.value_projections,
            )
        ):
            if source.shape[:-1] != leading_shape or source.shape[-1] != expected_width:
                raise ValueError(
                    f"Source {source_idx} must have shape {leading_shape + (expected_width,)}, "
                    f"got {tuple(source.shape)}."
                )
            key = key_proj(source)  # [..., target_width]
            projected_values.append(value_proj(source))
            logits.append(self.query(key).squeeze(-1))  # [...]

        # Compute the depth softmax in fp32 for stable low-precision training,
        # then return to the values' dtype for the weighted sum.
        depth_logits = torch.stack(logits, dim=0)  # [num_sources, ...]
        weights = (
            depth_logits.float().softmax(dim=0).to(dtype=projected_values[0].dtype)
        )
        values = torch.stack(
            projected_values, dim=0
        )  # [num_sources, ..., target_width]
        return (values * weights.unsqueeze(-1)).sum(dim=0)


class DepthAttentionRead(nn.Module):
    """muP-adapted Kimi depth attention over same-width sources.

    Implements the read of *Attention Residuals* (arXiv:2603.15031 Eq. 3-6):
    every source acts as both key and value, keys are RMSNorm-ed WITHOUT a
    learnable gain, there are no projections, and a single learned pseudo-query
    scores the keys. The pseudo-query is zero-initialized so the initial mixture
    is exactly uniform. The softmax runs over the depth axis only -- token
    positions never mix.

    Two deliberate deviations from the paper make it work under this repo's
    reference muP, where the paper-verbatim read measured materially worse
    (held-out CE 3.5007 vs 3.3576 for this one). First, values are the unit-RMS
    (RMSNorm-ed) sources scaled by learned per-source gains rather than the raw
    sources: the ``m_emb``-scaled embedding (RMS ~1.2) would otherwise dominate
    a raw-value mixture of depth-scaled block sums (RMS ~0.01), and since every
    consumer of the mixture is pre-norm that magnitude gap is pure init
    artifact. Second, the depth logits carry the canonical ``1/sqrt(dim)``
    temperature: ``|RMSNorm(k)| = sqrt(dim)``, so untempered logits train the
    depth-attention alpha ~``sqrt(dim)`` too hot at the muP base learning rate.

    Attributes:
        dim (int): Feature width shared by every source.
        eps (float): RMSNorm numerical epsilon.
        max_sources (int): Upper bound on ``len(sources)``, sizing the gains.
        query (nn.Parameter): Zero-init pseudo-query of shape ``[dim]``,
            tagged for base-rate AdamW without weight decay (like norm gains).
        value_gains (nn.Parameter): Per-source value gains of shape
            ``[max_sources]``, unit-init and tagged like ``query``.
    """

    dim: int
    eps: float
    max_sources: int
    per_source_queries: bool
    query: nn.Parameter
    value_gains: nn.Parameter

    def __init__(
        self,
        dim: int,
        *,
        max_sources: int,
        eps: float = 1e-6,
        per_source_queries: bool = False,
    ) -> None:
        """Initialize a depth attention read.

        Args:
            dim (int): Positive feature width of every source tensor.
            max_sources (int): Positive upper bound on ``len(sources)`` so the
                per-source value gains can be allocated.
            eps (float): RMSNorm numerical epsilon.
            per_source_queries (bool): One query vector PER SOURCE instead of
                the paper's single shared pseudo-query. Every source is then
                scored by its own direction detector -- ``z_i = q_i . k_i``
                rather than ``z_i = q . k_i`` -- which removes the constraint
                that made both measured widenings of this read (columns, and
                columns+aggregate at unit gains) lose to merged-only: with one
                shared query, promoting one source's logit direction demotes
                every source not aligned with it. Runtime-identical (one GEMV
                per source either way); zero-init keeps the step-0 mixture
                uniform exactly as the shared query does.

        Raises:
            ValueError: If ``dim`` or ``max_sources`` is not positive.
        """

        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive.")
        if max_sources < 1:
            raise ValueError("max_sources must be positive.")
        self.dim = dim
        self.eps = eps
        self.max_sources = max_sources
        self.per_source_queries = per_source_queries
        # Zero init => uniform depth weights at step 0 (paper-critical).
        self.query = set_scaling(
            nn.Parameter(
                torch.zeros(max_sources, dim)
                if per_source_queries
                else torch.zeros(dim)
            ),
            ParamScaling("adamw", lr_mul=1.0, weight_decay=False),
        )
        # Unit init: the mixture starts as the uniform mean of unit-RMS source
        # directions; gains re-learn magnitude preferences.
        self.value_gains = set_scaling(
            nn.Parameter(torch.ones(max_sources)),
            ParamScaling("adamw", lr_mul=1.0, weight_decay=False),
        )

    def forward(
        self,
        sources: Sequence[Tensor],
        *,
        key_cache: DepthKeyCache | None = None,
    ) -> Tensor:
        """Mix gain-scaled unit-RMS sources with softmax-over-depth weights.

        Never materializes a key. Writing the key map as
        ``k_i = r_i * s_i`` with the ``[batch, seq, 1]`` scalar
        ``r_i = rsqrt(mean(s_i^2) + eps)``, both places a key appears factor
        through that scalar:

        - the logit is ``(k_i . q) / sqrt(d) == r_i * (s_i . q) / sqrt(d)``, so
          the pseudo-query can be dotted with the RAW source;
        - the mixture is ``sum_i a_i k_i == sum_i (a_i * r_i) s_i``, so folding
          ``r_i`` into the depth weight alongside the value gain mixes the RAW
          sources too.

        That matters because keys previously doubled as values and so were
        retained for the backward pass. A depth-``N`` backbone runs this read
        ``O(N^2)`` times with up to ``N + 1`` sources, and every one of those
        source-slots kept a ``[batch, seq, dim]`` key alive until backward --
        on the 4-block ``(3, 5, 7, 1)`` ladder, 369 of them (129 with a
        ``key_cache``), which is what forced the column arms onto a quarter of
        the micro-batch. The factored form retains ``[batch, seq]`` weights
        instead, ``dim`` times smaller, and the raw sources it mixes are the
        residual streams the graph was holding anyway. It also drops a full
        ``[batch, seq, dim]`` write and re-read per source-slot, since no key is
        written to be read back.

        Three further traffic cuts, all algebraically exact:

        - logits are a GEMV (``[batch*seq, dim] @ [dim]``) rather than an
          ``einsum`` that expands to a full-width product plus a reduction;
        - the per-source value gains ride the same ``[n, batch, seq]`` weights
          as ``r_i`` instead of scaling every ``[batch, seq, dim]`` value;
        - the mixture accumulates in place, so there is no per-source weighted
          value temporary.

        ``key_cache`` shares the remaining per-source work: because the key map
        carries no weights, ``r_i`` is read-INDEPENDENT, so sources fixed across
        a run of reads (the embedding plus every completed block, re-read by
        every sublayer) reduce once and every later read reuses the result.

        Output dtype is unchanged -- the promotion ``torch.stack(sources)``
        would have produced, so mixed fp32/bf16 source lists under autocast
        still yield fp32. Folding ``r_i`` into the fp32 depth weights rather
        than into the source keeps the scalar at full precision one step longer
        than the explicit key did; the reassociation can still move the last bit
        in low precision.

        Args:
            sources (Sequence[Tensor]): Depth sources, each of shape
                ``[batch_size, seq_len, dim]`` (all shapes must be equal).
                Dtypes may differ; the mixture runs in their promoted dtype.
            key_cache (DepthKeyCache | None): Pre-computed key scales for the
                first ``key_cache.num_sources`` entries of ``sources``, which
                must be the SAME tensor objects the cache holds (verified by
                identity). The remaining sources are reduced here as usual.
                None reduces everything.

        Returns:
            Tensor: Depth-attended mixture of shape ``[batch_size, seq_len, dim]``.

        Raises:
            ValueError: If ``sources`` is empty, or if ``key_cache`` covers more
                sources than were passed, disagrees on ``eps``, or does not hold
                the very tensors at the head of ``sources``.
        """

        if len(sources) == 0:
            raise ValueError("DepthAttentionRead needs at least one source.")

        # Dtype ``torch.stack(sources)`` would have produced.
        mix_dtype = sources[0].dtype
        for source in sources[1:]:
            mix_dtype = torch.promote_types(mix_dtype, source.dtype)
        if _MIX_AT_AUTOCAST_DTYPE:
            # Narrow to the dtype the rest of the model runs in, so a single
            # fp32 source cannot promote the whole mixture (see the flag's
            # comment above). Only ever narrows: an all-fp32 source list outside
            # autocast is left alone.
            device_type = sources[0].device.type
            if torch.is_autocast_enabled(device_type):
                autocast_dtype = torch.get_autocast_dtype(device_type)
                if mix_dtype.itemsize > autocast_dtype.itemsize:
                    mix_dtype = autocast_dtype

        num_cached = 0
        cached_scales: Sequence[Tensor] = ()
        if key_cache is not None:
            num_cached = key_cache.num_sources
            if num_cached > len(sources):
                raise ValueError(
                    f"key_cache covers {num_cached} sources but only "
                    f"{len(sources)} were passed."
                )
            if key_cache.eps != self.eps:
                raise ValueError(
                    f"key_cache eps {key_cache.eps} does not match this read's "
                    f"eps {self.eps}; the cached scales would differ."
                )
            if any(
                source is not cached
                for source, cached in zip(sources, key_cache.sources)
            ):
                raise ValueError(
                    "key_cache must hold the leading sources of this read; got "
                    "different tensors (the caller's source list and the cache "
                    "have drifted apart)."
                )
            cached_scales = key_cache.scales()

        # Sources may carry different dtypes (an fp32 embedding beside bf16
        # block states), and the GEMV below runs in each source's OWN dtype to
        # avoid a full-width upcast; memoize the query cast per dtype so that
        # costs one tiny copy per distinct dtype rather than one per source.
        queries: dict[torch.dtype, Tensor] = {}
        scales: list[Tensor] = []  # per ENTRY, [batch, seq, width] fp32
        logits: list[Tensor] = []  # per ENTRY, [batch, seq, width]
        # Depth index each entry starts at: an entry may carry several sources,
        # so this is what indexes ``query``/``value_gains`` rows.
        depth_start = 0
        for source_idx, source in enumerate(sources):
            # A GROUP entry is [batch, seq, width, dim]: ``width`` sources that
            # are already contiguous on the source axis. Every step below then
            # runs ONCE for the whole group instead of once per source, which is
            # the whole point -- see the grouped-source note in the docstring.
            width = source.shape[2] if source.dim() == 4 else 1
            # Weight-free key RMSNorm scalar in fp32, from the shared cache for
            # the fixed prefix (see :func:`unit_rms_scale`).
            scale = (
                cached_scales[source_idx]
                if source_idx < num_cached
                else unit_rms_scale(source, eps=self.eps)
            )  # [batch, seq, 1] or [batch, seq, width, 1]
            scales.append(scale.squeeze(-1).reshape(*source.shape[:2], width))
            # Matrix-vector product on the RAW source: the key's scalar is
            # applied to the [batch, seq] result instead of the [batch, seq,
            # dim] input. [batch*seq*width, dim] @ [dim] -> [batch*seq*width].
            query = queries.get(source.dtype)  # [dim] or [max_sources, dim]
            if query is None:
                query = queries.setdefault(
                    source.dtype, self.query.to(source.dtype)
                )
            if self.per_source_queries:
                # Row i scores source i. Across a group the rows differ, so this
                # is a per-source contraction rather than one shared GEMV.
                rows = query[depth_start : depth_start + width]  # [width, dim]
                projected = (
                    torch.einsum("bswd,wd->bsw", source, rows)
                    if width > 1
                    else (source.reshape(-1, self.dim) @ rows[0]).view(
                        *source.shape[:2], 1
                    )
                )
            else:
                # Shared pseudo-query: one GEMV over the flattened rows, so a
                # group costs the same launch as a single source.
                projected = (source.reshape(-1, self.dim) @ query).view(
                    *source.shape[:2], width
                )  # [batch, seq, width]
            # Scale first, cast after: the fp32 ``scale`` must not truncate a
            # higher-precision ``projected`` before the product. The softmax
            # runs in fp32 either way, so the cast here only pins the stack's
            # dtype when sources disagree.
            logits.append((projected * scales[-1]).float())
            depth_start += width

        # Canonical scaled-dot-product temperature: |RMSNorm(k)| = sqrt(d), so
        # unscaled logits move ~sqrt(d) faster per unit query change -- too hot
        # under the reference muP base LR.
        # Depth axis LAST -- see the module note on depth-axis placement; a
        # leading n makes Inductor emit a persistent reduction that runs out of
        # shared memory past ~20 sources.
        # Entries already carry a trailing source axis (width 1 for a plain
        # source), so this is a cat rather than a stack -- a group contributes
        # its ``width`` columns in depth order.
        depth_logits = torch.cat(logits, dim=-1) / (
            self.dim ** 0.5
        )  # [batch, seq, n]
        num_depth = depth_logits.shape[-1]
        if num_depth > self.max_sources:
            raise ValueError(
                f"read got {num_depth} depth sources (counting group widths) "
                f"but was built for max_sources={self.max_sources}; the gains "
                "and per-source queries are sized by that bound."
            )
        # Depth softmax in fp32 for stable low-precision training, folding in
        # the per-source gains. Directional mixture: unit-RMS source x learned
        # per-source gain -- ``(g_i * k_i) * a_i == k_i * (g_i * a_i)``, so
        # scaling the tiny [batch, seq, n] weights replaces n full-width
        # multiplies. Factored out so NESTOR_DEPTH_READ_EAGER=1 can graph-break
        # here if a read is ever wide enough to defeat even this layout.
        weights = _dispatch_depth_softmax(
            depth_logits, self.value_gains[:num_depth]
        )  # [batch, seq, n]
        # Fold the key scales in too, so the mixture below reads raw sources.
        # The product carries the key's whole contribution to the mixture, so
        # it runs at no less than fp32 (and at the mixture's precision when
        # that is wider) before being cast down for the accumulate. The scales
        # are [batch, seq, 1] each, so a cat along -1 is the depth stack.
        weight_dtype = torch.promote_types(torch.float32, mix_dtype)
        alpha = (
            weights.to(weight_dtype)
            * torch.cat(scales, dim=-1).to(weight_dtype)
        ).to(mix_dtype)  # [batch, seq, n]

        mixture: Tensor | None = None
        depth_start = 0
        for source in sources:
            # Branch on RANK, not on width: a k == 1 block exports a genuine
            # group of shape [batch, seq, 1, dim], which is still 4-D and must
            # take the contraction path even though it carries one source.
            is_group = source.dim() == 4
            width = source.shape[2] if is_group else 1
            weight = alpha[..., depth_start : depth_start + width]
            depth_start += width
            if is_group:
                # One batched contraction for the whole group, replacing
                # ``width`` separate weighted accumulates over strided views.
                term = torch.einsum("bsw,bswd->bsd", weight, source)
                mixture = term if mixture is None else mixture + term
            elif mixture is None:
                mixture = source * weight  # [batch, seq, dim]
            else:
                # In-place accumulate: no per-source weighted-value temporary.
                mixture.addcmul_(source, weight)
        assert mixture is not None  # sources is non-empty (checked above)
        return mixture.to(mix_dtype)
