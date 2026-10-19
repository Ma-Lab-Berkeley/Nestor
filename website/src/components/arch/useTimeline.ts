import { useCallback, useEffect, useState } from "react";

/** The four stages drawn for each block, in order. */
export const PHASES = ["route", "attend", "experts", "combine"] as const;
export type Phase = (typeof PHASES)[number];

/** Milliseconds per phase at speed 1. */
const DURATION: Record<Phase, number> = { route: 1500, attend: 1500, experts: 1100, combine: 1200 };

export type Timeline = {
  block: number;
  phase: Phase;
  /** Index of `phase` in PHASES; stage s is visible when s <= phaseIdx. */
  phaseIdx: number;
  playing: boolean;
  speed: number;
  setSpeed: (s: number) => void;
  toggle: () => void;
  next: () => void;
  prev: () => void;
  goto: (block: number) => void;
};

/**
 * Advances through blocks and phases on a timer.
 *
 * Args:
 *   numBlocks: blocks to cycle through before wrapping to the first.
 *   autoplay: whether to start running (false under reduced motion).
 */
export function useTimeline(numBlocks: number, autoplay: boolean): Timeline {
  const [block, setBlock] = useState(0);
  const [phaseIdx, setPhaseIdx] = useState(0);
  const [playing, setPlaying] = useState(autoplay);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => {
      if (phaseIdx < PHASES.length - 1) {
        setPhaseIdx(phaseIdx + 1);
      } else {
        setPhaseIdx(0);
        setBlock((block + 1) % numBlocks);
      }
    }, DURATION[PHASES[phaseIdx]] / speed);
    return () => window.clearTimeout(id);
  }, [playing, phaseIdx, block, speed, numBlocks]);

  const goto = useCallback(
    (b: number) => {
      setBlock(((b % numBlocks) + numBlocks) % numBlocks);
      setPhaseIdx(0);
    },
    [numBlocks],
  );

  return {
    block,
    phase: PHASES[phaseIdx],
    phaseIdx,
    playing,
    speed,
    setSpeed,
    toggle: () => setPlaying((p) => !p),
    next: () => goto(block + 1),
    prev: () => goto(block - 1),
    goto,
  };
}
