import { paper, quickstart, toc } from "./content/paper";
import { Hero } from "./components/Hero";
import { Section } from "./components/Section";
import { Figure } from "./components/Figure";
import { CodeBlock } from "./components/CodeBlock";
import { BibTeX } from "./components/BibTeX";
import { Toc } from "./components/Toc";
import { Callout } from "./components/Callout";
import { Eq, M } from "./components/Math";
import { eq } from "./content/equations";
import { ArchFlow } from "./components/arch/ArchFlow";

export function App() {
  return (
    <>
      <nav className="topnav" aria-label="Site">
        <a className="topnav__brand" href="#top">
          {paper.shortName}
        </a>
        <div className="topnav__links">
          <a href="#architecture">Architecture</a>
          <a href="#pretraining">Results</a>
          <a href={paper.links.code} rel="noopener">
            Code
          </a>
          <a href="#bibtex">Cite</a>
        </div>
      </nav>

      <main id="top">
        <Hero />

        <div className="column">
          <Figure
            src="arch.svg"
            number={1}
            width={533}
            height={216}
            size="column"
            eager
            alt="The Nestor architecture at three levels of detail, and the attention map computed by each column."
            caption={
              <>
                <em>(a)</em> A token is routed to 2 of 4 columns. In each column, it uses 2 of 4 experts per layer. It
                uses only the coloured parameters. <em>(b)</em> Each column attends only among the tokens routed to it.
              </>
            }
          />
        </div>

        <div className="layout">
          <Toc entries={toc} />

          <article className="article">
            <Section id="summary" title="Summary">
              <p>
                Nestor is a transformer architecture with <em>nested sparsity</em>, which generalizes mixture-of-experts
                (MoE) transformers, outperforms them at equal compute in language model pretraining, and is especially
                well-suited for continual pretraining.
              </p>
              <p>
                Below, we summarize the architecture, our pretraining and continual pretraining experiments, and a
                theoretical account of why nested sparsity helps. The paper contains the full details.
              </p>
            </Section>

            <Section id="architecture" title="What is Nestor?">
              <p>
                In an MoE transformer, each token uses only a few of many expert MLPs in each layer, so the number of
                parameters can grow while the compute per token stays fixed. This conditional computation is confined to
                the MLP, however; the attention operator remains dense. Nestor applies the same principle to attention.
                Each block contains several attention operators, which we call <em>columns</em>, and a router sends each
                token to a few of them. Each column has its own experts, and a second router selects a few of these for
                each token, so the two forms of sparsity nest, i.e., compose hierarchically. This is depicted in the
                figure at the top of the page.
              </p>
              <p>Overall, Nestor is similar to an MoE transformer, with two differences:</p>
              <ul>
                <li>
                  the attention operator is replaced by several columns, and each token attends only among the tokens
                  routed to the same column;
                </li>
                <li>and each column has its own experts, selected by its own router.</li>
              </ul>
              <p>
                With a single column, Nestor is exactly an MoE transformer, and with a single expert, a dense transformer.
                In all of our experiments, each block has 8 columns, and the number of columns a token uses grows with
                depth, from 3 in the first block to all 8 in the last. Below, we follow one token through our 200M model.
              </p>
              <div className="breakout">
                <ArchFlow />
              </div>
              <p className="arch__caption">Click a token to follow it, or a block to jump to it.</p>
            </Section>

            <Section id="pretraining" title="Compute-efficient pretraining">
              <p>
                We pretrain dense, MoE, and Nestor language models at four scales, from 100M to 1B active parameters. At
                each scale, Nestor stores about as many parameters as the MoE and uses no more training compute than
                either baseline. As shown below, Nestor achieves the lowest held-out loss at every scale.
              </p>
              <Figure
                src="scaling.svg"
                number={2}
                width={519}
                height={194}
                alt="Held-out loss against training compute for Nestor, MoE and dense models, with fitted scaling laws."
                caption={
                  <>
                    Held-out loss against training compute, with fitted scaling laws. Ticks: the compute each baseline
                    needs to match Nestor.
                  </>
                }
              />
              <p>
                By fitting a scaling law to each architecture, we estimate that the MoE needs 1.13–1.29× as much compute
                as Nestor to reach the same loss, and the dense transformer 3.1–4.1×. Neither factor shrinks as the scale
                grows.
              </p>
            </Section>

            <Section id="domain-adaptation" title="Domain adaptation">
              <p>
                A pretrained model is often trained further on new data, and it should keep what it has already learned.
                We adapt comparable Nestor, MoE, and dense models to four new domains: books, mathematics, code, and
                German. All three models improve on the new domain by about the same amount, but they differ in how much
                of their pretraining data they forget: Nestor forgets 3% less than the MoE on average, whereas the dense
                model forgets 21% more. Below, we show the gain and the forgetting on each domain.
              </p>
              <Figure
                src="domain.svg"
                number={3}
                width={441}
                height={202}
                size="column"
                alt="Adaptation gain and forgetting on four domains, relative to the MoE transformer."
                caption={
                  <>
                    Relative to the MoE. <strong>Top:</strong> gain on the new domain. <strong>Bottom:</strong> forgetting
                    of pretraining data. Faded marks: with 25% of the data replayed from pretraining.
                  </>
                }
              />
            </Section>

            <Section id="long-context" title="Long-context extension">
              <p>
                We also extend each model from a 4k context to as long as 64k. Since each column attends only among the
                tokens routed to it, Nestor's cost per token grows more slowly with the context length than the MoE's, so
                on the same compute budget it can train on more tokens. When each model receives the same number of
                tokens, Nestor preserves its advantage from pretraining; when each model uses the same compute, its
                advantage grows with the context length, as shown below.
              </p>
              <Figure
                src="cpt.svg"
                number={4}
                width={474}
                height={241}
                alt="Held-out loss at each context length from 4k to 64k, relative to the MoE transformer, at four scales."
                caption={
                  <>
                    Held-out loss from 4k to 64k context relative to the MoE (lower is better). <strong>Top:</strong> dense
                    transformer. <strong>Bottom:</strong> Nestor.
                  </>
                }
              />
            </Section>

            <Section id="theory" title="Why does nested sparsity help?">
              <p>
                To explain these observations, we compare stylized versions of Nestor and of the MoE transformer on a
                statistical model of data with hierarchical structure. In this analytically tractable model, we reproduce
                both the pretraining compute efficiency and the reduced forgetting that we observe empirically. We state
                the results informally here; the formal statements and proofs are in the paper.
              </p>

              <h3>A model of nested data</h3>
              <p>
                We assume that the next token is predicted in two stages. A coarse label <M>{String.raw`c \in [C]`}</M>{" "}
                determines which tokens of the context are relevant, how they are weighted, and which statistic of them
                is computed; a fine label <M>{String.raw`e \in [E]`}</M> determines how this statistic is used. In
                Nestor, <M>c</M> corresponds to the column router and <M>e</M> to the expert router of the column. We
                study the simplest case: one layer with one head, linear attention and maps, and the squared loss. The
                context holds tokens <M>{String.raw`\vx_i = (c_i, \vz_i)`}</M> with uniform labels and independent
                contents, the query <M>{String.raw`\vx_S = (c, \vz_S)`}</M> carries the label <M>c</M>, and the summary
                and target are
              </p>
              <Eq>{eq.linearModel}</Eq>
              <p>
                where <M>{String.raw`\vW_c^\star`}</M> is an invertible pattern,{" "}
                <M>{String.raw`\vA_c^\star \in \R^{r \times D}`}</M> and{" "}
                <M>{String.raw`\vB_{ce}^\star \in \R^{D \times r}`}</M> have rank <M>{String.raw`r < D`}</M>, and{" "}
                <M>{String.raw`\veps`}</M> is noise with covariance <M>{String.raw`\sigma^2 \vI`}</M>. The regression
                function is <M>{String.raw`\vf^\star(c, e, \vX) = \vG_{ce}^\star \vm`}</M>, where
              </p>
              <Eq>{eq.nestedStructure}</Eq>
              <p>
                The last property is the nested structure: for a fixed coarse label, the <M>E</M> maps share the pattern{" "}
                <M>{String.raw`\vW_c^\star`}</M> and the row space of <M>{String.raw`\vA_c^\star`}</M>, while both may
                differ from one coarse label to another.
              </p>
              <Figure
                src="theory.svg"
                number={5}
                width={583}
                height={205}
                alt="The linear model with coarse and fine labels, and the flat and nested predictors."
                caption={
                  <>
                    The model and the two predictors, for <M>C = 2</M> coarse labels (blue, green) and <M>E = 3</M> fine
                    labels. <em>(a)</em> One sample. <em>(b)</em> The flat predictor (MoE): the shared attention scores
                    every token, and each expert has its own row space. <em>(c)</em> The nested predictor (Nestor): a
                    column scores only the tokens with its label, and its row space is shared by its experts. Free
                    parameters are counted below each panel.
                  </>
                }
              />

              <h3>Sample and compute efficiency</h3>
              <p>
                The <em>nested predictor</em> is one layer of Nestor with one active column: column <M>c</M> attends over
                the tokens with label <M>c</M> through its own pattern <M>{String.raw`\vW_c`}</M> and value map{" "}
                <M>{String.raw`\vA_c`}</M>, and expert <M>(c, e)</M> applies a linear map{" "}
                <M>{String.raw`\vB_{ce}`}</M>. The <em>flat predictor</em> is one layer of an MoE transformer: a shared
                attention over all tokens, with one pattern block per pair of labels and a full-width value map{" "}
                <M>{String.raw`\vV`}</M>, followed by linear experts with <M>r</M> hidden units:
              </p>
              <Eq>{eq.predictors}</Eq>
              <p>
                We fit each predictor by least squares over its function class <M>{String.raw`\cM`}</M> on <M>T</M>{" "}
                samples, and measure its excess risk:
              </p>
              <Eq>{eq.leastSquares}</Eq>

              <Callout kind="Theorem" title="informal">
                <p>Let <M>{String.raw`C \geq 2`}</M>.</p>
                <p>
                  <em>(a)</em> Both predictors can represent <M>{String.raw`\vf^\star`}</M>, unlike a flat predictor with
                  a bilinear score on raw tokens.
                </p>
                <p>
                  <em>(b)</em> As <M>{String.raw`T \to \infty`}</M>, the excess risk of either predictor is
                  asymptotically <M>{String.raw`\sigma^2 \chi^2_d / T`}</M>, where <M>d</M> is the number of directions
                  in weight space that change the prediction. For <M>{String.raw`r \ll D`}</M> and{" "}
                  <M>{String.raw`E \gg 1`}</M>,
                </p>
                <Eq>{eq.dims}</Eq>
                <p>
                  <em>(c)</em> If a sample costs <M>{String.raw`F_0`}</M> FLOPs plus <M>{String.raw`F_1`}</M> FLOPs per
                  scored token, the flat predictor requires
                </p>
                <Eq>{eq.multiplier}</Eq>
                <p>
                  times as many FLOPs for the same expected excess risk, a factor that grows with <M>S</M> from{" "}
                  <M>{String.raw`\dF / \dN`}</M> to <M>{String.raw`C\, \dF / \dN`}</M>.
                </p>
              </Callout>
              <p>
                The count <M>d</M> has three parts: the patterns, the row spaces, and the maps from the row spaces to the
                output. Both predictors have <M>CE</M> maps to fit. The flat predictor also fits one row space per
                expert, from that expert's samples alone, and <M>{String.raw`C^2`}</M> pattern blocks, of which the{" "}
                <M>{String.raw`C(C - 1)`}</M> blocks between different labels are zero at the target but must still be
                fit from data. The nested predictor fits one row space and one pattern per column, from the samples of
                all of the column's experts, and its routing fixes the other blocks to zero. The ratio is near <M>C</M>{" "}
                when the patterns dominate the parameter count, and near 2 when the experts do. Since the nested
                predictor scores only <M>{String.raw`1/C`}</M> of the context, its compute advantage grows with the
                context length, as in our long-context experiments.
              </p>

              <h3>Forgetting</h3>
              <p>
                Finally, each predictor starts from weights that represent <M>{String.raw`\vf^\star`}</M> and is trained
                by gradient flow on a new domain, whose samples all carry the coarse label <M>{String.raw`c_Q`}</M> and
                new maps. The forgetting on the old labels <M>(c, e)</M> at time <M>t</M> is the increase of their risk
                since time 0.
              </p>
              <Callout kind="Proposition" title="informal" tone="blue">
                <p>
                  In the nested predictor, the columns <M>{String.raw`c \neq c_Q`}</M> never change, so there is no
                  forgetting on any other coarse label. In the flat predictor, the shared value map{" "}
                  <M>{String.raw`\vV(t)`}</M> moves, and the forgetting on <M>(c, e)</M> with{" "}
                  <M>{String.raw`c \neq c_Q`}</M> is
                </p>
                <Eq>{eq.forgetting}</Eq>
                <p>
                  where <M>{String.raw`\vXi \neq \vzero`}</M> whenever the value map starts moving. For orthonormal
                  rows, the coefficient lies between multiples of <M>{String.raw`\cos^2 \theta_r`}</M> and{" "}
                  <M>{String.raw`\cos^2 \theta_1`}</M>, where <M>{String.raw`\theta_1 \leq \dots \leq \theta_r`}</M> are
                  the principal angles between the row spaces of <M>{String.raw`\vA_c^\star`}</M> and{" "}
                  <M>{String.raw`\vA_{c_Q}^\star`}</M>.
                </p>
              </Callout>
              <p>
                In the flat predictor, the shared value map is the only source of forgetting on other labels, and the
                forgetting grows with the overlap between row spaces, which is the generic case. Overall, nesting improves
                the sample and compute efficiency when the relevant tokens, patterns, and row spaces differ between
                coarse labels, and it removes this forgetting, consistent with our domain adaptation experiments.
              </p>
            </Section>

            <Section id="code" title="Code">
              <p>
                The repository contains the model, the training stack, and the configurations of the twelve pretraining
                runs above.
              </p>
              <CodeBlock code={quickstart} label="shell" />
              <p>
                <a className="btn" href={paper.links.code} rel="noopener">
                  github.com/Ma-Lab-Berkeley/Nestor
                </a>
              </p>
            </Section>

            <Section id="bibtex" title="Citation">
              <BibTeX />
            </Section>
          </article>
        </div>
      </main>

      <footer className="footer">
        <p>© 2026 {paper.authors.map((a) => a.name).join(", ")}. Figures are those of the paper.</p>
      </footer>
    </>
  );
}
