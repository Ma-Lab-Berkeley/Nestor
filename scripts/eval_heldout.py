#!/usr/bin/env python3
"""Held-out next-token cross-entropy / perplexity of a trained checkpoint.

The paper's reported metric: the token-weighted mean of the shifted next-token
cross-entropy over a disjoint DCLM token ``.bin`` (preprocess one from global
shards the training run never read).

Example:
    python scripts/eval_heldout.py \
        --experiment nestor_200m \
        --checkpoint outputs/nestor_200m/<run>/checkpoints/model_final.pt \
        --val-bin data/dclm/dclm_val.bin --max-tokens 15000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nestor.configs.data import DclmDatasetConfig
from nestor.data.datasets import DclmTokenDataset
from nestor.eval.checkpoint import load_experiment_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="Registered experiment whose model to rebuild")
    ap.add_argument("--checkpoint", required=True, help="Path to model_*.pt")
    ap.add_argument("--val-bin", required=True, help="Held-out token .bin (disjoint from training)")
    ap.add_argument("--block-size", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=15_000_000, help="Cap eval tokens for speed")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Tolerate checkpoint PARAMETERS missing from the rebuilt model. "
            "Off by default: a registry/env mismatch would otherwise evaluate "
            "randomly initialised weights with only a warning."
        ),
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = load_experiment_checkpoint(
        args.experiment, Path(args.checkpoint), device=device, allow_partial=args.allow_partial
    )
    if info["missing"]:
        print(f"[eval] WARNING missing keys: {len(info['missing'])} (e.g. {info['missing'][:3]})")
    if info["unexpected"]:
        print(f"[eval] WARNING unexpected keys: {len(info['unexpected'])} (e.g. {info['unexpected'][:3]})")
    print(
        f"[eval] loaded {args.checkpoint} | step={info['step']} | "
        f"params={info['num_params']/1e6:.1f}M"
    )

    # The paper's protocol scores the held-out windows as tokenized, without the
    # training-time end-of-text remap; keep it that way so numbers match.
    ds = DclmTokenDataset(
        DclmDatasetConfig(data_path=Path(args.val_bin), block_size=args.block_size)
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    print(f"[eval] held-out windows={len(ds)} block_size={args.block_size}")

    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    with torch.no_grad():
        for inputs, _targets in loader:
            inputs = inputs.to(device)
            # fp32 forward; the gap vs the bf16 training forward is ~1e-3 nats.
            logits = model(inputs).pred  # (B, S, V)
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1]).float()
            shift_targets = inputs[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_targets, reduction="sum"
            )
            total_loss += float(loss.item())
            total_tokens += int(shift_targets.numel())
            if total_tokens >= args.max_tokens:
                break

    ce = total_loss / max(total_tokens, 1)
    ppl = float(torch.tensor(ce).exp())
    print(
        f"[eval] DONE: tokens={total_tokens:,} | cross_entropy={ce:.4f} nats | "
        f"perplexity={ppl:.3f} | {time.time()-t0:.0f}s"
    )
    print(f"[eval] RESULT cross_entropy={ce:.4f}")


if __name__ == "__main__":
    main()
