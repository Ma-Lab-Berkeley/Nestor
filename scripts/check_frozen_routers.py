#!/usr/bin/env python3
"""Verify a frozen-router run against its starting checkpoint.

Loads the two ``model_*.pt`` files (memory-mapped, CPU) and checks what
``NESTOR_FREEZE_ROUTERS=1`` promises:

* every ROUTER tensor -- the score / noise networks of the column routers,
  expert-bank routers and pool routers, and their ``choice_bias`` balancing
  buffers -- is bit-identical to the start;
* every other floating-point tensor moved, except tensors that receive no
  gradient BY CONSTRUCTION (``--allow-unchanged``): by default the first
  layer's depth-read pseudo-query, which scores a single source (the
  embedding), so its softmax is the constant 1 whatever the query.

Router keys are recognised by name (``group_router`` / ``expert_router`` /
``public_router`` / ``private_routers.<i>``); a ``_orig_mod.`` prefix (a
compiled checkpoint) is stripped. A run root resolves to its newest
``model_final.pt``. Exit status 0 on PASS, 1 otherwise.

::

    python scripts/check_frozen_routers.py --init <32k run root or model_final.pt> \\
        --final <adaptation run root or model_final.pt>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from nestor.continual.extension import resolve_checkpoint

ROUTER_KEY: re.Pattern[str] = re.compile(
    r"(^|\.)(group_router|expert_router|public_router|private_routers\.\d+)\."
)
"""re.Pattern[str]: Parameter / buffer names that belong to a router."""

DEFAULT_ALLOW_UNCHANGED: str = r"^backbone\.blocks\.0\.layers\.0\.kimi_attn_read\.query$"
"""str: Non-router tensors with zero gradient by construction (see the module docstring)."""


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    """The ``model`` state dict of a trainer checkpoint, prefix-stripped.

    Args:
        path (Path): The checkpoint file.

    Returns:
        dict[str, torch.Tensor]: Tensor name -> tensor (memory-mapped).
    """

    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def compare(
    init: dict[str, torch.Tensor],
    final: dict[str, torch.Tensor],
    allow_unchanged: re.Pattern[str] | None = None,
) -> tuple[list[str], list[str], list[str], int]:
    """Classify every tensor of the pair.

    Args:
        init (dict[str, torch.Tensor]): The starting checkpoint's model state.
        final (dict[str, torch.Tensor]): The final checkpoint's model state.
        allow_unchanged (re.Pattern[str] | None): Non-router keys that may stay
            unchanged (gradient-free by construction).

    Returns:
        tuple[list[str], list[str], list[str], int]: Router tensors that
        CHANGED (must be empty), non-router floating tensors that did NOT
        change, keys present in one file only or of another shape, and the
        number of non-router tensors that moved.
    """

    router_changed: list[str] = []
    other_unchanged: list[str] = []
    other_moved = 0
    mismatched = sorted(set(init) ^ set(final))
    for key in sorted(set(init) & set(final)):
        a, b = init[key], final[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatched.append(f"{key} (shape/dtype)")
            continue
        same = torch.equal(a, b)
        if ROUTER_KEY.search(key):
            if not same:
                router_changed.append(key)
        elif a.is_floating_point():
            if same:
                if allow_unchanged is None or not allow_unchanged.search(key):
                    other_unchanged.append(key)
            else:
                other_moved += 1
    return router_changed, other_unchanged, mismatched, other_moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument(
        "--allow-unchanged", default=DEFAULT_ALLOW_UNCHANGED,
        help="regex of non-router keys allowed to stay unchanged ('' = none)",
    )
    args = parser.parse_args()
    allow = re.compile(args.allow_unchanged) if args.allow_unchanged else None
    init_path, final_path = resolve_checkpoint(args.init), resolve_checkpoint(args.final)
    init, final = load_model_state(init_path), load_model_state(final_path)
    router_keys = [k for k in init if ROUTER_KEY.search(k)]
    bias_keys = [k for k in router_keys if k.endswith("choice_bias")]
    router_changed, other_unchanged, mismatched, other_moved = compare(init, final, allow)
    print(f"[frozen-check] init  = {init_path}")
    print(f"[frozen-check] final = {final_path}")
    print(
        f"[frozen-check] router tensors {len(router_keys)} ({len(bias_keys)} choice_bias buffers): "
        f"{len(router_keys) - len(router_changed)} unchanged, {len(router_changed)} CHANGED"
    )
    print(f"[frozen-check] non-router floating tensors: {other_moved} moved, {len(other_unchanged)} unchanged")
    for key in router_changed:
        print(f"[frozen-check]   ROUTER CHANGED: {key}")
    for key in other_unchanged:
        print(f"[frozen-check]   non-router unchanged: {key}")
    for key in mismatched:
        print(f"[frozen-check]   key mismatch: {key}")
    ok = not router_changed and not other_unchanged and not mismatched and other_moved > 0
    print(f"[frozen-check] RESULT {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
