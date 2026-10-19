import { useState, type ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";
import { BLOCKS, C, E, S, columnsOf, expertsOf, membersOf, range, readWeights } from "./model";
import { PHASES, useTimeline } from "./useTimeline";

/* ----------------------------------------------------------------------------
 * Geometry of the 960 x 600 canvas. Panel A (top) is the whole network, panel B
 * (bottom) magnifies the block the playhead is on. All numbers are viewBox units.
 * ------------------------------------------------------------------------- */
const VW = 960;
const VH = 604;

const A = {
  blockY: 104,
  blockH: 92,
  blockW: 72,
  pitch: 116,
  x0: 112,
  inX: 56,
  outX: 918,
};
const aBlockX = (b: number) => A.x0 + b * A.pitch;
const aMidY = A.blockY + A.blockH / 2;

const B = {
  top: 266,
  tokX: 56,
  tokY0: 334,
  tokPitch: 28,
  routerX: 104,
  routerW: 92,
  routerY: 420,
  routerH: 28,
  laneX0: 236,
  laneX1: 750,
  laneY0: 300,
  lanePitch: 34,
  laneH: 30,
  memX0: 284,
  memPitch: 10,
  memSize: 7,
  cellX0: 384,
  cellX1: 742,
  sumX: 800,
};
const bTokY = (t: number) => B.tokY0 + t * B.tokPitch;
const bLaneY = (c: number) => B.laneY0 + c * B.lanePitch;
const bLaneMid = (c: number) => bLaneY(c) + B.laneH / 2;
const bRouterMid = B.routerY + B.routerH / 2;
const bMemX = (i: number) => B.memX0 + i * B.memPitch;

/** Cubic path from (x0,y0) to (x1,y1) with horizontal tangents. */
const sCurve = (x0: number, y0: number, x1: number, y1: number): string => {
  const mx = (x0 + x1) / 2;
  return `M ${x0} ${y0} C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1}`;
};

const EASE = [0.4, 0, 0.2, 1] as const;

/** The walkthrough of a token's path through the network, block by block. */
export function ArchFlow() {
  const reduce = useReducedMotion() ?? false;
  const tl = useTimeline(BLOCKS.length, !reduce);
  const [token, setToken] = useState(3);

  const { block, phaseIdx } = tl;
  const spec = BLOCKS[block];
  const cols = columnsOf(block, token);
  const activeSet = new Set(cols);
  const show = (stage: number) => phaseIdx >= stage; // 0 route, 1 attend, 2 experts, 3 combine
  const dur = (ms: number) => (reduce ? 0 : ms / 1000 / tl.speed);
  const weights = readWeights(block);

  const cellW = (B.cellX1 - B.cellX0 - (spec.layers - 1) * 8) / spec.layers;
  const cellX = (l: number) => B.cellX0 + l * (cellW + 8);
  const contentW = 38 + 10 + E * 12 - 3;
  const contentOff = (cellW - contentW) / 2;

  const status = describe(block, token, cols, phaseIdx);

  return (
    <div className="arch">
      <div className="arch__bar">
        <div className="arch__controls" role="group" aria-label="Playback">
          <button type="button" className="btn btn--sm" onClick={tl.prev} aria-label="Previous block">
            ‹
          </button>
          <button type="button" className="btn btn--sm btn--primary" onClick={tl.toggle} aria-pressed={tl.playing}>
            {tl.playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="btn btn--sm" onClick={tl.next} aria-label="Next block">
            ›
          </button>
          <label className="arch__speed">
            speed
            <select value={tl.speed} onChange={(e) => tl.setSpeed(Number(e.target.value))}>
              <option value={0.5}>0.5×</option>
              <option value={1}>1×</option>
              <option value={2}>2×</option>
            </select>
          </label>
        </div>
        <ol className="arch__scrub" aria-label="Blocks">
          {BLOCKS.map((bk, i) => (
            <li key={i}>
              <button
                type="button"
                className={`arch__dot${i === block ? " is-current" : i < block ? " is-done" : ""}`}
                onClick={() => tl.goto(i)}
                aria-current={i === block ? "step" : undefined}
                title={`Block ${i + 1}: K_c = ${bk.kc}`}
              >
                {i + 1}
              </button>
            </li>
          ))}
        </ol>
      </div>

      <div className="arch__canvas">
        <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label={status.replace(/K_([ce])/g, "K$1")}>
          <defs>
            <marker id="arch-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0 0.6 L7.4 4 L0 7.4 Z" fill="var(--muted)" />
            </marker>
            <marker id="arch-arrow-accent" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0 0.6 L7.4 4 L0 7.4 Z" fill="var(--accent)" />
            </marker>
          </defs>

          {/* ------------------------------ panel A ------------------------------ */}
          <text x={A.inX - 32} y={22} className="t-title">
            The network
          </text>
          <text x={A.inX + 90} y={22} className="t-muted">
            seven blocks · C = {C} columns each · a token uses <Sub base="K" sub="c" after=" of them, more as it goes deeper" />
          </text>

          {/* depth read arcs into the current block */}
          {block > 0 && (
            <g key={`read-${block}`}>
              {range(block + 1).map((src) => {
                const x0 = src === 0 ? A.inX : aBlockX(src - 1) + A.blockW / 2;
                const y0 = src === 0 ? aMidY - 10 : A.blockY;
                const x1 = aBlockX(block) + A.blockW / 2;
                const dist = block - src + 1;
                const cy = A.blockY - 16 - 8 * dist;
                const d = `M ${x0} ${y0} Q ${(x0 + x1) / 2} ${cy} ${x1} ${A.blockY}`;
                return (
                  <motion.path
                    key={src}
                    d={d}
                    fill="none"
                    stroke="var(--house-blue)"
                    strokeWidth={0.8 + 5 * weights[src]}
                    strokeLinecap="round"
                    opacity={0.85}
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{ pathLength: 1, opacity: 0.85 }}
                    transition={{ duration: dur(900), ease: EASE, delay: dur(60 * src) }}
                  />
                );
              })}
              <text x={aBlockX(block) + A.blockW / 2} y={A.blockY - 24 - 8 * (block + 1)} className="t-blue" textAnchor="middle">
                depth read
              </text>
            </g>
          )}

          {/* input token and residual stream */}
          <circle cx={A.inX} cy={aMidY} r={9} fill="var(--accent)" />
          <text x={A.inX} y={aMidY + 3.5} className="t-onaccent" textAnchor="middle">
            {token + 1}
          </text>
          <text x={A.inX} y={aMidY + 30} className="t-muted" textAnchor="middle">
            <Sub base="token x" sub={String(token + 1)} after=" " />
          </text>
          <path d={`M ${A.inX + 11} ${aMidY} H ${aBlockX(0) - 2}`} className="stream" markerEnd="url(#arch-arrow)" />
          {BLOCKS.slice(0, -1).map((_, b) => (
            <path key={b} d={`M ${aBlockX(b) + A.blockW} ${aMidY} H ${aBlockX(b + 1) - 2}`} className="stream" markerEnd="url(#arch-arrow)" />
          ))}
          <path d={`M ${aBlockX(BLOCKS.length - 1) + A.blockW} ${aMidY} H ${A.outX - 30}`} className="stream" markerEnd="url(#arch-arrow)" />
          <text x={A.outX} y={aMidY + 3.5} className="t-muted" textAnchor="middle">
            logits
          </text>

          {/* the blocks with their column bars */}
          {BLOCKS.map((bk, b) => {
            const x = aBlockX(b);
            const state = b < block ? "done" : b === block ? "current" : "ahead";
            const bcols = new Set(columnsOf(b, token));
            return (
              <g key={b}>
                <motion.rect
                  x={x}
                  y={A.blockY}
                  width={A.blockW}
                  height={A.blockH}
                  rx={8}
                  className={`ablock ablock--${state}`}
                  animate={{ opacity: state === "ahead" ? 0.55 : 1 }}
                  transition={{ duration: dur(300) }}
                  onClick={() => tl.goto(b)}
                  style={{ cursor: "pointer" }}
                />
                {range(C).map((c) => {
                  const lit = bcols.has(c) && (state === "done" || (state === "current" && show(0)));
                  return (
                    <motion.rect
                      key={c}
                      x={x + 6.5 + c * 7.5}
                      y={A.blockY + 10}
                      width={5}
                      height={A.blockH - 20}
                      rx={1.5}
                      initial={false}
                      animate={{ fill: lit ? "var(--accent)" : "var(--bar-off)" }}
                      transition={{ duration: dur(350), delay: lit && state === "current" ? dur(250) : 0 }}
                      style={{ pointerEvents: "none" }}
                    />
                  );
                })}
                <text x={x + A.blockW / 2} y={A.blockY + A.blockH + 17} className={state === "current" ? "t-strong" : "t-muted"} textAnchor="middle">
                  block {b + 1}
                </text>
                <text x={x + A.blockW / 2} y={A.blockY + A.blockH + 31} className="t-small" textAnchor="middle">
                  <Sub base="K" sub="c" after={` = ${bk.kc} · ${bk.layers} layers`} />
                </text>
              </g>
            );
          })}

          {/* divider */}
          <line x1={24} y1={B.top - 14} x2={VW - 24} y2={B.top - 14} className="divider" />

          {/* ------------------------------ panel B ------------------------------ */}
          <text x={A.inX - 32} y={B.top + 10} className="t-title">
            Inside block {block + 1}
          </text>
          <text x={A.inX + 128} y={B.top + 10} className="t-muted">
            <Sub base="K" sub="c" after={` = ${spec.kc} of C = ${C} columns · M = ${spec.layers} layers · ${spec.heads} heads per column · `} />
            <Sub base="K" sub="e" after=" = 2 of E experts per layer" />
          </text>

          {/* tokens */}
          <text x={B.tokX} y={B.tokY0 - 22} className="t-small" textAnchor="middle">
            tokens
          </text>
          {range(S).map((t) => {
            const sel = t === token;
            return (
              <g key={t} onClick={() => setToken(t)} style={{ cursor: "pointer" }} role="button" aria-label={`Follow token ${t + 1}`}>
                <circle cx={B.tokX} cy={bTokY(t)} r={9} className={sel ? "tok tok--sel" : "tok"} />
                <text x={B.tokX} y={bTokY(t) + 3.5} className={sel ? "t-onaccent" : "t-small"} textAnchor="middle" style={{ pointerEvents: "none" }}>
                  {t + 1}
                </text>
              </g>
            );
          })}
          <text x={B.tokX} y={bTokY(S - 1) + 28} className="t-small" textAnchor="middle">
            click to follow
          </text>

          {/* router */}
          <motion.rect
            x={B.routerX}
            y={B.routerY}
            width={B.routerW}
            height={B.routerH}
            rx={6}
            className="router"
            animate={{ stroke: show(0) ? "var(--accent)" : "var(--line-2)" }}
            transition={{ duration: dur(300) }}
          />
          <text x={B.routerX + B.routerW / 2} y={bRouterMid + 3.5} className="t-strong" textAnchor="middle">
            column router
          </text>
          <text x={B.routerX + B.routerW / 2} y={B.routerY + B.routerH + 14} className="t-small" textAnchor="middle">
            top-{spec.kc} of {C}
          </text>

          <g key={`wires-${block}-${token}`}>
            {/* token -> router */}
            <motion.path
              d={sCurve(B.tokX + 10, bTokY(token), B.routerX, bRouterMid)}
              className="wire"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 1 }}
              transition={{ duration: dur(450), ease: EASE }}
            />
            {/* router -> columns */}
            {cols.map((c, i) => (
              <motion.path
                key={c}
                d={sCurve(B.routerX + B.routerW, bRouterMid, B.laneX0, bLaneMid(c))}
                className="wire"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: dur(550), ease: EASE, delay: dur(350 + 70 * i) }}
              />
            ))}
          </g>

          {/* lanes = columns */}
          {range(C).map((c) => {
            const active = activeSet.has(c);
            const y = bLaneY(c);
            const members = membersOf(block, c);
            const memberSet = new Set(members);
            return (
              <g key={c}>
                <motion.rect
                  x={B.laneX0}
                  y={y}
                  width={B.laneX1 - B.laneX0}
                  height={B.laneH}
                  rx={6}
                  className="lane"
                  initial={false}
                  animate={{
                    fill: active ? "var(--accent-soft)" : "var(--panel)",
                    stroke: active ? "var(--accent-mid)" : "var(--line)",
                  }}
                  transition={{ duration: dur(350), delay: active ? dur(600) : 0 }}
                />
                <text x={B.laneX0 + 8} y={y + B.laneH / 2 + 3.5} className={active ? "t-accent" : "t-small"}>
                  col {c + 1}
                </text>

                {/* membership strip: which tokens this column holds */}
                <g key={`mem-${block}-${token}`} opacity={active ? 1 : 0.4}>
                  {range(S).map((i) => {
                    const isMember = memberSet.has(i);
                    const visible = show(1);
                    const isSelf = i === token && active;
                    return (
                      <motion.rect
                        key={i}
                        x={bMemX(i)}
                        y={y + (B.laneH - B.memSize) / 2}
                        width={B.memSize}
                        height={B.memSize}
                        rx={1.5}
                        className="mem"
                        initial={false}
                        animate={{
                          fill: visible && isMember ? (isSelf ? "var(--accent)" : "var(--accent-mid)") : "var(--mem-off)",
                          stroke: isSelf && visible ? "var(--accent-ink)" : "transparent",
                        }}
                        transition={{ duration: dur(250), delay: visible ? dur(40 * i) : 0 }}
                      />
                    );
                  })}
                  {/* causal attention arcs from the followed token to earlier members of its column */}
                  {active &&
                    show(1) &&
                    members
                      .filter((i) => i < token)
                      .map((i, k) => {
                        const xj = bMemX(token) + B.memSize / 2;
                        const xi = bMemX(i) + B.memSize / 2;
                        const yy = y + (B.laneH - B.memSize) / 2 - 1;
                        const h = 4 + 1.2 * (token - i);
                        return (
                          <motion.path
                            key={i}
                            d={`M ${xj} ${yy} Q ${(xi + xj) / 2} ${yy - h} ${xi} ${yy}`}
                            className="arc"
                            markerEnd="url(#arch-arrow-accent)"
                            initial={{ pathLength: 0, opacity: 0 }}
                            animate={{ pathLength: 1, opacity: 1 }}
                            transition={{ duration: dur(500), ease: EASE, delay: dur(350 + 90 * k) }}
                          />
                        );
                      })}
                </g>

                {/* layer cells: attention + experts */}
                {range(spec.layers).map((l) => {
                  const cx = cellX(l) + contentOff;
                  const chosen = new Set(active ? expertsOf(block, token, c, l) : []);
                  return (
                    <g key={`${block}-${l}`} opacity={active ? 1 : 0.4}>
                      <rect x={cellX(l)} y={y + 3} width={cellW} height={B.laneH - 6} rx={4} className="cell" />
                      <motion.rect
                        x={cx}
                        y={y + 7}
                        width={38}
                        height={16}
                        rx={3}
                        className="chip"
                        initial={false}
                        animate={{
                          fill: active && show(1) ? "var(--accent)" : "var(--chip-off)",
                        }}
                        transition={{ duration: dur(300), delay: active && show(1) ? dur(300 + 120 * l) : 0 }}
                      />
                      {/* the label's colour is set by a class, not animated: the stylesheet's `svg text` rule
                          would override an animated fill attribute and leave dark text on the purple chip */}
                      <text x={cx + 19} y={y + 18.5} textAnchor="middle" className={active && show(1) ? "t-chip t-chip--on" : "t-chip"}>
                        attn
                      </text>
                      {range(E).map((e) => {
                        const lit = active && show(2) && chosen.has(e);
                        return (
                          <motion.rect
                            key={e}
                            x={cx + 48 + e * 12}
                            y={y + 10.5}
                            width={9}
                            height={9}
                            rx={2}
                            className="expert"
                            initial={false}
                            animate={{ fill: lit ? "var(--accent)" : "var(--mem-off)" }}
                            transition={{ duration: dur(250), delay: lit ? dur(120 * l + 60 * e) : 0 }}
                          />
                        );
                      })}
                    </g>
                  );
                })}
              </g>
            );
          })}
          <text x={bMemX(0)} y={B.laneY0 - 8} className="t-small">
            tokens in the column
          </text>
          <text x={B.cellX0} y={B.laneY0 - 8} className="t-small">
            {spec.layers} layers · attention over the column's tokens · <Sub base="K" sub="e" after=" = 2 of E experts" />
          </text>

          {/* combine */}
          <g key={`sum-${block}-${token}`}>
            {show(3) &&
              cols.map((c, i) => (
                <motion.path
                  key={c}
                  d={sCurve(B.laneX1, bLaneMid(c), B.sumX - 13, bRouterMid)}
                  className="wire"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: dur(500), ease: EASE, delay: dur(60 * i) }}
                />
              ))}
            <motion.circle
              cx={B.sumX}
              cy={bRouterMid}
              r={13}
              className="sum"
              initial={false}
              animate={{ stroke: show(3) ? "var(--accent)" : "var(--line-2)" }}
              transition={{ duration: dur(300), delay: show(3) ? dur(400) : 0 }}
            />
            <text x={B.sumX} y={bRouterMid + 4.5} className="t-strong" textAnchor="middle">
              Σ
            </text>
            <text x={B.sumX} y={bRouterMid + 30} className="t-small" textAnchor="middle">
              <Sub base="gated by w" sub="jc" after=" " />
            </text>
            {show(3) && (
              <motion.path
                d={`M ${B.sumX + 14} ${bRouterMid} H ${VW - 40}`}
                className="wire"
                markerEnd="url(#arch-arrow-accent)"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: dur(400), ease: EASE, delay: dur(600) }}
              />
            )}
            <text x={(B.sumX + 14 + VW - 40) / 2} y={bRouterMid - 10} className="t-small" textAnchor="middle">
              {block < BLOCKS.length - 1 ? `to block ${block + 2}` : "to the logits"}
            </text>
          </g>
        </svg>
      </div>

      <p className="arch__status" aria-live="polite">
        <span className="arch__phase">
          {PHASES.map((p, i) => (
            <span key={p} className={i === phaseIdx ? "is-on" : i < phaseIdx ? "is-past" : undefined}>
              {p}
            </span>
          ))}
        </span>
        <span className="arch__text">
          {/* Every sentence the walkthrough can show, stacked in one cell, so the box keeps the height of the longest. */}
          {range(S).flatMap((t) =>
            BLOCKS.flatMap((_, b) =>
              PHASES.map((_, p) => {
                const current = t === token && b === block && p === phaseIdx;
                return (
                  <span key={`${t}-${b}-${p}`} className={current ? "is-current" : undefined} aria-hidden={!current}>
                    {withSubs(describe(b, t, columnsOf(b, t), p))}
                  </span>
                );
              }),
            ),
          )}
        </span>
      </p>
    </div>
  );
}

/** One sentence per phase, read out under the canvas and as the SVG label. */
function describe(block: number, token: number, cols: number[], phaseIdx: number): string {
  const spec = BLOCKS[block];
  const j = token + 1;
  const list = cols.map((c) => c + 1).join(", ");
  const last = block === BLOCKS.length - 1;
  switch (phaseIdx) {
    case 0:
      return block === 0
        ? `Block 1 reads the token embedding. Its column router sends token ${j} to columns ${list}: K_c = ${spec.kc} of the C = ${C} columns.`
        : `The depth read mixes the token embedding and the outputs of blocks 1–${block} into the input of block ${block + 1}. Its column router sends token ${j} to columns ${list}: K_c = ${spec.kc} of ${C}.`;
    case 1:
      return spec.kc === C
        ? `Every token is in every column, so block ${block + 1} computes full attention over the whole sequence, with ${spec.heads} heads per column for ${spec.layers} layers.`
        : `In each of its ${spec.kc} columns, token ${j} attends only to the earlier tokens routed to that column, with ${spec.heads} heads per column, for ${spec.layers} layers.`;
    case 2:
      return `In every layer of each active column, an expert router picks K_e = 2 of the column's own experts for token ${j}; a few shared experts are tied across columns.`;
    default:
      return `The ${spec.kc} column outputs are weighted by the router's gates and summed into the residual stream${last ? ", which feeds the output head." : `, which flows on to block ${block + 2}.`}`;
  }
}

/** `base` followed by a subscript and the rest of the label, for use inside an SVG <text>. */
function Sub({ base, sub, after }: { base: string; sub: string; after: string }) {
  return (
    <>
      {base}
      <tspan dy={3} fontSize="0.78em">
        {sub}
      </tspan>
      <tspan dy={-3}>{after}</tspan>
    </>
  );
}

/** Renders the `K_c` / `K_e` of a status sentence as real subscripts. */
function withSubs(text: string): ReactNode[] {
  return text.split(/(K_[ce])/g).map((part, i) =>
    part === "K_c" || part === "K_e" ? (
      <span key={i}>
        <i>K</i>
        <sub>{part[2]}</sub>
      </span>
    ) : (
      part
    ),
  );
}
