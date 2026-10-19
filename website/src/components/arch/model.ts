/**
 * A small, deterministic stand-in for Nestor's routing, used only to drive the
 * architecture animation. Shapes follow the paper's 200M configuration
 * (C = 8 columns, K_c = 3,4,5,6,6,7,8 over seven blocks); the assignments and
 * weights are synthetic but balanced, so every column receives the same number
 * of tokens, as the bias-balanced routers do in training.
 */

/** Columns per block. */
export const C = 8;
/** Tokens drawn in the animation. */
export const S = 8;
/** Experts drawn per column layer (the paper's models use more). */
export const E = 4;
/** Active experts per column layer. */
export const KE = 2;

export type BlockSpec = {
  /** Active columns per token. */
  kc: number;
  /** Layers in the block (M). */
  layers: number;
  /** Heads per column, round(16 / K_c). */
  heads: number;
};

/** The seven blocks of the 200M model. */
export const BLOCKS: readonly BlockSpec[] = [
  { kc: 3, layers: 2, heads: 5 },
  { kc: 4, layers: 2, heads: 4 },
  { kc: 5, layers: 2, heads: 3 },
  { kc: 6, layers: 3, heads: 3 },
  { kc: 6, layers: 3, heads: 3 },
  { kc: 7, layers: 3, heads: 2 },
  { kc: 8, layers: 2, heads: 2 },
];

/** `[0, 1, ..., n-1]`. */
export const range = (n: number): number[] => Array.from({ length: n }, (_, i) => i);

/** A tiny LCG so the routing is identical on every visit. Returns draws in [0, 1). */
function lcg(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return (s >>> 8) / 0x01000000;
  };
}

/** A seeded Fisher–Yates permutation of `[0, n)`. */
function permutation(n: number, seed: number): number[] {
  const rnd = lcg(seed);
  const p = range(n);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  return p;
}

const COLUMN_PERMS = BLOCKS.map((_, b) => permutation(C, 101 + 7 * b));

/**
 * The K_c columns token `token` is routed to in block `block`, ascending.
 *
 * Token t takes a window of K_c consecutive entries of a per-block permutation
 * starting at t·K_c (mod C). With S = C this gives every column exactly K_c
 * members, i.e. a perfectly balanced assignment.
 */
export function columnsOf(block: number, token: number): number[] {
  const { kc } = BLOCKS[block];
  const perm = COLUMN_PERMS[block];
  return range(kc)
    .map((i) => perm[(token * kc + i) % C])
    .sort((a, b) => a - b);
}

/** The tokens routed to `column` in `block`, in sequence order. */
export function membersOf(block: number, column: number): number[] {
  return range(S).filter((t) => columnsOf(block, t).includes(column));
}

/** The K_e of E experts token `token` uses in layer `layer` of `column` in `block`. */
export function expertsOf(block: number, token: number, column: number, layer: number): number[] {
  const seed = 7919 * block + 613 * column + 97 * layer + 13 * token + 5;
  return permutation(E, seed)
    .slice(0, KE)
    .sort((a, b) => a - b);
}

/**
 * Illustrative depth-read weights for the input of `block`: a softmax over
 * the token embedding (index 0) and the outputs of blocks 0..block-1
 * (indices 1..block), tilted towards the most recent blocks. Length block + 1.
 */
export function readWeights(block: number): number[] {
  const logits = [-0.6, ...range(block).map((i) => 0.35 * i - 0.25 * (block - 1 - i))];
  const m = Math.max(...logits);
  const ex = logits.map((l) => Math.exp(l - m));
  const z = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / z);
}
