"""Scaling-aware building blocks and the Muon/AdamW optimizer.

Every learnable tensor produced here is tagged with a :class:`ParamScaling`
describing its initialization, per-parameter learning-rate multiplier,
optimizer assignment, and (for Muon) the spectral update scale
``sqrt(fan_out / fan_in)``. The optimizer builder reads these tags to split
the parameters between Muon (in-stream matrices) and AdamW (embeddings, score
heads, norm gains, biases).
"""

from nestor.scaling.blocks import (
    ParamScaling,
    RMSNorm,
    RoPE,
    ScaledEmbedding,
    ScaledLinear,
    ScaledMLP,
    convex_residual,
    get_scaling,
    set_scaling,
)
from nestor.scaling.muon import MuonAdamW, zeropower_via_newtonschulz5

__all__ = [
    "MuonAdamW",
    "ParamScaling",
    "RMSNorm",
    "RoPE",
    "ScaledEmbedding",
    "ScaledLinear",
    "ScaledMLP",
    "convex_residual",
    "get_scaling",
    "set_scaling",
    "zeropower_via_newtonschulz5",
]
