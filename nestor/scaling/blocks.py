"""Scaling-aware building blocks for the Nestor backbone.

Every parameter-bearing module here bakes in the initialization, per-parameter
learning-rate multiplier, and optimizer assignment prescribed in
``paper/nestor.tex`` (Section ``sec:scaling`` and Table
``tab:scaling-prescription``). The rules are *absolute* functions of the current
shapes -- there is no base model and no width or depth exponent to tune.

The three operator families are:

* **In-stream maps** (attention ``W_Q,W_K,W_V,W_O`` and the expert MLP matrices)
  obey the spectral condition ``||W||_* = Theta(sqrt(d_out/d_in))``. They are
  initialized fan-in ``N(0, 1/d_in)`` and updated by Muon, whose orthogonalized
  step is scaled by ``sqrt(d_out/d_in)``.
* **Score heads** (the group/expert routers and the task head) are read in the
  max norm a softmax is Lipschitz in, so they hold ``||W||_{rms->inf} = Theta(1)``:
  entry variance ``1/d_in^2`` (a factor ``d_in`` below the in-stream fan-in) and
  AdamW with learning rate ``eta_A / d_in``.
* **Embeddings / norms / biases** contract over a finite axis (or carry no width
  at all), so they use base-rate AdamW with no width factor.

Each tensor is tagged with a :class:`ParamScaling`; :func:`get_scaling` lets the
optimizer builder recover the tag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "tanh": torch.tanh,
}


@dataclass(frozen=True)
class ParamScaling:
    """Scaling metadata attached to a single parameter.

    Attributes:
        optimizer (str): Owning optimizer, ``"muon"`` or ``"adamw"``.
        lr_mul (float): Multiplier on the owning optimizer's base learning rate.
            For AdamW score heads this is ``1/d_in`` (times any shared-parameter
            factor); for Muon parameters it is ``1`` (the shape scale lives in
            ``muon_scale``).
        muon_scale (float): Spectral update scale ``sqrt(d_out/d_in)`` Muon
            applies to the orthogonalized step. Unused for AdamW parameters.
        weight_decay (bool): Whether weight decay applies (``False`` for norm
            gains, biases, and embeddings).
    """

    optimizer: str
    lr_mul: float = 1.0
    muon_scale: float = 1.0
    weight_decay: bool = True


_SCALING_ATTR = "_nestor_scaling"


def set_scaling(param: nn.Parameter, scaling: ParamScaling) -> nn.Parameter:
    """Tag ``param`` with its scaling metadata.

    Args:
        param (nn.Parameter): Parameter to tag.
        scaling (ParamScaling): Metadata to attach.

    Returns:
        nn.Parameter: The same parameter, tagged in place.
    """

    setattr(param, _SCALING_ATTR, scaling)
    return param


def get_scaling(param: object) -> ParamScaling | None:
    """Return the scaling metadata of ``param`` if it was tagged.

    Args:
        param (object): A parameter (or any object).

    Returns:
        ParamScaling | None: The attached metadata, or ``None`` if untagged.
    """

    scaling = getattr(param, _SCALING_ATTR, None)
    return scaling if isinstance(scaling, ParamScaling) else None


class ScaledLinear(nn.Module):
    """A linear map carrying its own init and optimizer-scaling prescription.

    Args:
        in_features (int): Fan-in ``d_in``.
        out_features (int): Fan-out ``d_out``.
        role (str): One of ``"in_stream"``, ``"score"``, or ``"projection"``.
        bias (bool): Whether to include a bias (zero-init, base-rate AdamW).
        lr_mul (float): Extra learning-rate multiplier applied to a score head
            (e.g. ``1/sqrt(M_l)`` for a router shared across ``M_l`` layers).
            Ignored for non-score roles.
        zero_init (bool): For a score head, initialize the weight at zero so
            routing/prediction begin exactly uniform.
    """

    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        role: str = "in_stream",
        bias: bool = True,
        lr_mul: float = 1.0,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        if role not in ("in_stream", "score", "projection"):
            raise ValueError(
                f"Unknown role '{role}'. Expected in_stream/score/projection."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.role = role
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters(lr_mul=lr_mul, zero_init=zero_init)

    def reset_parameters(
        self, *, lr_mul: float = 1.0, zero_init: bool = False
    ) -> None:
        """Initialize and (re)tag the weight and bias for the chosen role."""

        d_in, d_out = self.in_features, self.out_features
        spectral = math.sqrt(d_out / d_in)
        with torch.no_grad():
            if self.role == "in_stream":
                nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(d_in))
                set_scaling(
                    self.weight, ParamScaling("muon", muon_scale=spectral)
                )
            elif self.role == "score":
                if zero_init:
                    nn.init.zeros_(self.weight)
                else:
                    # Entry variance 1/d_in^2 holds ||W||_{rms->inf} = Theta(1).
                    nn.init.normal_(self.weight, mean=0.0, std=1.0 / d_in)
                set_scaling(
                    self.weight, ParamScaling("adamw", lr_mul=lr_mul / d_in)
                )
            else:  # projection: scaled semi-orthogonal, ||W||_* = sqrt(d_out/d_in)
                nn.init.orthogonal_(self.weight)
                self.weight.mul_(spectral)
                set_scaling(
                    self.weight, ParamScaling("muon", muon_scale=spectral)
                )
            if self.bias is not None:
                nn.init.zeros_(self.bias)
                set_scaling(
                    self.bias, ParamScaling("adamw", weight_decay=False)
                )

    def forward(
        self, x: Float[Tensor, "... d_in"]
    ) -> Float[Tensor, "... d_out"]:
        """Apply the linear map to the last axis."""

        return F.linear(x, self.weight, self.bias)


class ScaledMLP(nn.Module):
    """A two-or-more layer MLP built from :class:`ScaledLinear`.

    Hidden layers are always in-stream maps. The final layer's role is
    configurable: ``"in_stream"`` for an expert MLP (output read in RMS) or
    ``"score"`` for a router/readout head (output read by a softmax).

    Args:
        input_dim (int): Input width.
        hidden_dims (list[int]): Hidden widths (may be empty).
        output_dim (int): Output width.
        activation (str): Named activation between layers.
        final_role (str): Role of the final layer, ``"in_stream"`` or ``"score"``.
        final_lr_mul (float): Extra learning-rate multiplier for a score head.
        bias (bool): Whether linear layers carry biases.
        zero_init_final (bool): Zero-init the final layer (score heads only).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "gelu",
        *,
        final_role: str = "in_stream",
        final_lr_mul: float = 1.0,
        bias: bool = True,
        zero_init_final: bool = False,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Invalid activation '{activation}'. "
                f"Allowed: {sorted(_ACTIVATIONS)}."
            )
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation = activation

        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[ScaledLinear] = []
        last = len(dims) - 2
        for i in range(len(dims) - 1):
            is_final = i == last
            role = final_role if is_final else "in_stream"
            layers.append(
                ScaledLinear(
                    dims[i],
                    dims[i + 1],
                    role=role,
                    bias=bias,
                    lr_mul=final_lr_mul if is_final else 1.0,
                    zero_init=zero_init_final and is_final and role == "score",
                )
            )
        self.layers = nn.ModuleList(layers)
        self._activation_fn = _ACTIVATIONS[activation]

    def forward(
        self, x: Float[Tensor, "... input_dim"]
    ) -> Float[Tensor, "... output_dim"]:
        """Apply the MLP to the last axis."""

        for layer in self.layers[:-1]:
            x = self._activation_fn(layer(x))
        return self.layers[-1](x)


class GatedScaledMLP(nn.Module):
    """A SwiGLU (gated-linear-unit) expert MLP built from :class:`ScaledLinear`.

    Computes ``down(act(gate(x)) * up(x))`` -- the gated FFN used by
    Llama/DeepSeek and the variable-width-transformers reference (``swiglu``),
    vs :class:`ScaledMLP`'s pointwise ``down(act(up(x)))``. Three in-stream
    projections (gate, up, down), each Muon-tagged exactly like a ScaledMLP
    layer. ``up`` and ``down`` are exposed as ``layers[0]`` / ``layers[1]`` so the
    backbone's grouped-GEMM expert path can stack them like a 2-layer ScaledMLP;
    the extra ``gate`` is stacked separately. The gate carries no bias (standard
    SwiGLU), so the grouped path's per-expert bias bookkeeping is unchanged.

    Args:
        input_dim (int): Input width.
        hidden_dims (list[int]): Single hidden width (SwiGLU intermediate).
        output_dim (int): Output width.
        activation (str): Gate activation (``"silu"`` for true SwiGLU).
        bias (bool): Whether ``up``/``down`` carry biases (gate never does).
    """

    gated: bool

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "silu",
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 1:
            raise ValueError(
                f"GatedScaledMLP (SwiGLU) needs exactly one hidden width, got "
                f"{hidden_dims}."
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Invalid activation '{activation}'. Allowed: {sorted(_ACTIVATIONS)}."
            )
        hidden = hidden_dims[0]
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.activation = activation
        self.gated = True
        # up + down are exposed as .layers so the grouped-GEMM path stacks
        # layers[0]/[1] exactly as for a 2-layer ScaledMLP; gate is separate.
        up = ScaledLinear(input_dim, hidden, role="in_stream", bias=bias)
        down = ScaledLinear(hidden, output_dim, role="in_stream", bias=bias)
        self.layers = nn.ModuleList([up, down])
        self.gate = ScaledLinear(input_dim, hidden, role="in_stream", bias=False)
        self._activation_fn = _ACTIVATIONS[activation]

    def forward(
        self, x: Float[Tensor, "... input_dim"]
    ) -> Float[Tensor, "... output_dim"]:
        """Apply ``down(act(gate(x)) * up(x))`` to the last axis."""

        up, down = self.layers[0], self.layers[1]
        return down(self._activation_fn(self.gate(x)) * up(x))


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization with an affine gain and bias.

    The gain and bias carry no width factor and are trained with base-rate AdamW
    and no weight decay.

    Args:
        dim (int): Feature width.
        eps (float): Numerical floor on the RMS.
        bias (bool): Whether to include an affine bias.
    """

    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(self, dim: int, *, eps: float = 1e-6, bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = set_scaling(
            nn.Parameter(torch.ones(dim)),
            ParamScaling("adamw", weight_decay=False),
        )
        if bias:
            self.bias = set_scaling(
                nn.Parameter(torch.zeros(dim)),
                ParamScaling("adamw", weight_decay=False),
            )
        else:
            self.bias = None

    def forward(
        self, x: Float[Tensor, "... dim"]
    ) -> Float[Tensor, "... dim"]:
        """Normalize the last axis to unit RMS, then apply the affine map."""

        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x32.to(dtype) * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ScaledEmbedding(nn.Module):
    """A lookup embedding with ``Theta(1)`` entries and base-rate AdamW.

    Embeddings contract over a finite vocabulary axis, so width does not enter
    the init or the learning rate.

    Args:
        num_embeddings (int): Vocabulary size.
        embedding_dim (int): Output width.
    """

    weight: nn.Parameter

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        weight = torch.empty(num_embeddings, embedding_dim)
        nn.init.normal_(weight, mean=0.0, std=1.0)
        self.weight = set_scaling(
            nn.Parameter(weight), ParamScaling("adamw", weight_decay=False)
        )

    def forward(
        self, idx: Int[Tensor, "..."]
    ) -> Float[Tensor, "... embedding_dim"]:
        """Look up embeddings for integer indices."""

        return F.embedding(idx, self.weight)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the halves of the last axis: ``[x1, x2] -> [-x2, x1]``."""

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """Rotary position embedding applied to attention queries and keys.

    The block supplies the integer token positions and a per-block frequency
    base ``theta`` (``theta_l^rope`` in the paper). Positions may be the rounded
    mean window positions produced by pooling, or padding sentinels (clamped to
    zero here and masked out downstream).

    Args:
        head_dim (int): Per-head width; must be even.
        base (float): RoPE frequency base ``theta``.
    """

    inv_freq: Tensor

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "batch heads seq head_dim"],
        pos: Int[Tensor, "batch seq"],
    ) -> Float[Tensor, "batch heads seq head_dim"]:
        """Apply rotary embeddings to ``x`` at integer positions ``pos``."""

        angles = pos.clamp_min(0).to(torch.float32).unsqueeze(-1) * self.inv_freq
        emb = torch.cat((angles, angles), dim=-1)  # (batch, seq, head_dim)
        cos = emb.cos().unsqueeze(1)  # (batch, 1, seq, head_dim)
        sin = emb.sin().unsqueeze(1)
        x32 = x.float()
        out = x32 * cos + _rotate_half(x32) * sin
        return out.to(x.dtype)


def convex_residual(
    h: Float[Tensor, "... dim"],
    update: Float[Tensor, "... dim"],
    depth_scale: float,
) -> Float[Tensor, "... dim"]:
    """Combine a residual stream with a branch update by the convex depth rule.

    Implements ``(1 - 1/D) h + (1/D) f(h)`` with ``depth_scale = 1/D`` and
    ``D = 2 * sum_l M_l`` the total number of attention/feed-forward branches.
    A convex combination of two ``Theta(1)``-Lipschitz maps is
    ``Theta(1)``-Lipschitz, so the depth-``D`` composition stays
    sensitivity-``Theta(1)`` and no learning rate carries a depth factor.

    Args:
        h (Tensor): Incoming residual stream.
        update (Tensor): Branch output ``f(h)``.
        depth_scale (float): ``1/D``.

    Returns:
        Tensor: The updated residual stream.
    """

    return (1.0 - depth_scale) * h + depth_scale * update
