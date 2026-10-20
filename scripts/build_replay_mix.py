#!/usr/bin/env python3
"""Build a target-domain / DCLM replay mixture for domain adaptation.

The replay runs of the domain-adaptation experiments draw ``--replay`` percent
of their 0.5B tokens from the DCLM sequences the model saw during its 32k
context extension and the rest from the target corpus; both parts are
prefixes of the trainer's own permutation of their source, so the target part
is a subset of the no-replay run's sequences and the DCLM part is true replay
(``nestor.data.seen_subset.build_replay_mix``). The mixture is sized at 504M
tokens; the 477-step run consumes ~99 % of it once, in a fresh shuffle.

::

    python scripts/build_replay_mix.py --corpus pg19 --replay 25 --data-dir data/longctx

writes ``data/longctx/pg19_replay25.bin`` from ``pg19_train.bin`` and
``dclm_longdoc_ge32768_train.bin``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nestor.continual.adaptation import (
    ADAPTATION_LENGTH,
    CORPORA,
    MIX_TOKENS,
    REPLAY_SOURCE,
    replay_mix_name,
)
from nestor.continual.extension import TOKENS_PER_STEP
from nestor.data.seen_subset import build_replay_mix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, choices=tuple(CORPORA))
    parser.add_argument("--replay", type=int, required=True, help="percent of the budget replayed from DCLM (1-99)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/longctx"))
    parser.add_argument("--mix-tokens", type=int, default=MIX_TOKENS)
    parser.add_argument("--seed", type=int, default=0, help="the trainer's sampler seed")
    args = parser.parse_args()

    out = args.data_dir / replay_mix_name(args.corpus, args.replay)
    meta = build_replay_mix(
        args.data_dir / CORPORA[args.corpus].train_bin,
        args.data_dir / REPLAY_SOURCE,
        out,
        block_size=ADAPTATION_LENGTH,
        target_share_pct=100 - args.replay,
        mix_tokens=args.mix_tokens,
        tokens_per_step=TOKENS_PER_STEP,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))
    print(f"[mix] wrote {out}")


if __name__ == "__main__":
    main()
