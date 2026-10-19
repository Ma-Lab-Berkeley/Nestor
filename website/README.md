# Nestor project page

The paper's website, served by GitHub Pages at
<https://ma-lab-berkeley.github.io/Nestor/>. A static Vite + React +
TypeScript site laid out as a long-form article: a centred serif text column,
a sticky table of contents on wide screens, figures and tables slightly wider
than the text, equations set with [KaTeX](https://katex.org) in the paper's
notation (`src/components/Math.tsx` holds the macros), and an SVG walkthrough
of the architecture driven by [`motion`](https://motion.dev). Colours follow
the paper's house palette (`:root` tokens in `src/styles.css`).

```bash
pnpm install
pnpm dev        # http://localhost:5173/Nestor/
pnpm build      # type-check + bundle into dist/
pnpm preview    # serve dist/ the way Pages will
```

`.github/workflows/pages.yml` builds and deploys `website/dist` on every push
to `main` that touches `website/`. Pages must be set to the "GitHub Actions"
source once (Settings → Pages), which the workflow also attempts itself.

## Favicon

`public/favicon.svg` is the mark: one block of four columns on a purple tile, the
token's two active columns drawn as stacks of layers. `favicon-32.png` and the
full-bleed `apple-touch-icon.png` are rasterised from it (headless Chrome).

## Figures

`public/figures/*.svg` are the paper's own figures, compiled from the
manuscript's TikZ/pgfplots sources with `pdflatex` under `\iclrfinalcopy`
(no line numbers), cropped with the `preview` package and converted with
`pdftocairo -svg`. Re-export them from the manuscript when a figure changes.
The schematic figures (`arch.svg`, `theory.svg`) use the same named house
colours as the plots (`figures/exps/exps_style.tex` in the manuscript), so
the page, the plots and the schematics share one palette.

| file | paper figure |
| --- | --- |
| `arch.svg` | Figure 1, nested sparsity in one block |
| `scaling.svg` | Figure 2, held-out loss scaling laws |
| `domain.svg` | Figure 3, domain adaptation and forgetting relative to the MoE |
| `cpt.svg` | Figure 4, long-context extension ladder |
| `theory.svg` | Figure 5, the linear model and the two predictors |
| `flops.svg` | Appendix figure, FLOPs per token against context length |
