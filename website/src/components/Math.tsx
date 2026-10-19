import katex from "katex";
import "katex/dist/katex.min.css";

/** The paper's notation macros, so equations can be pasted from the LaTeX source. */
const MACROS: Record<string, string> = {
  "\\vx": "\\mathbf{x}",
  "\\va": "\\mathbf{a}",
  "\\vb": "\\mathbf{b}",
  "\\ve": "\\mathbf{e}",
  "\\vf": "\\mathbf{f}",
  "\\vm": "\\mathbf{m}",
  "\\vq": "\\mathbf{q}",
  "\\vr": "\\mathbf{r}",
  "\\vs": "\\mathbf{s}",
  "\\vu": "\\mathbf{u}",
  "\\vy": "\\mathbf{y}",
  "\\vz": "\\mathbf{z}",
  "\\vA": "\\mathbf{A}",
  "\\vB": "\\mathbf{B}",
  "\\vG": "\\mathbf{G}",
  "\\vI": "\\mathbf{I}",
  "\\vzero": "\\mathbf{0}",
  "\\vV": "\\mathbf{V}",
  "\\vW": "\\mathbf{W}",
  "\\vX": "\\mathbf{X}",
  "\\veps": "\\boldsymbol{\\varepsilon}",
  "\\vXi": "\\boldsymbol{\\Xi}",
  "\\vSigma": "\\boldsymbol{\\Sigma}",
  "\\cC": "\\mathcal{C}",
  "\\cE": "\\mathcal{E}",
  "\\cI": "\\mathcal{I}",
  "\\cM": "\\mathcal{M}",
  "\\cU": "\\mathcal{U}",
  "\\R": "\\mathbb{R}",
  "\\Ex": "\\mathbb{E}",
  "\\TopK": "\\operatorname{TopK}",
  "\\MLP": "\\operatorname{\\mathbf{MLP}}",
  "\\Attn": "\\operatorname{\\mathbf{Attn}}",
  "\\AttnRes": "\\operatorname{\\mathbf{AttnRes}}",
  "\\rms": "\\operatorname{rms}",
  "\\rank": "\\operatorname{rank}",
  "\\argmin": "\\operatorname*{arg\\,min}",
  "\\row": "\\operatorname{row}",
  "\\sign": "\\operatorname{sign}",
  "\\Kc": "K_{\\mathrm{c}}",
  "\\Ke": "K_{\\mathrm{e}}",
  "\\dN": "d_{\\mathrm{N}}",
  "\\dF": "d_{\\mathrm{F}}",
  "\\norm": "\\left\\lVert #1 \\right\\rVert",
  "\\ip": "\\langle #1, #2 \\rangle",
};

function render(tex: string, displayMode: boolean): string {
  return katex.renderToString(tex, { displayMode, macros: MACROS, throwOnError: false, strict: "ignore", trust: false });
}

/** Inline math. */
export function M({ children }: { children: string }) {
  return <span className="math-inline" dangerouslySetInnerHTML={{ __html: render(children, false) }} />;
}

type EqProps = {
  /** The LaTeX source of the equation. */
  children: string;
  /** Optional right-hand tag, e.g. "1". */
  tag?: string;
};

/** A display equation, centred on the text column, with an optional equation tag. */
export function Eq({ children, tag }: EqProps) {
  return (
    <div className="eq" role="math">
      <div className="eq__body" dangerouslySetInnerHTML={{ __html: render(children, true) }} />
      {tag && <span className="eq__tag">({tag})</span>}
    </div>
  );
}
