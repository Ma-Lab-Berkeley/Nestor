import { paper, quickstart, toc } from "./content/paper";
import { eq } from "./content/equations";
import { Hero } from "./components/Hero";
import { Stats } from "./components/Stats";
import { Section } from "./components/Section";
import { Figure } from "./components/Figure";
import { ResultsTable } from "./components/ResultsTable";
import { CodeBlock } from "./components/CodeBlock";
import { BibTeX } from "./components/BibTeX";
import { Toc } from "./components/Toc";
import { Callout } from "./components/Callout";
import { Eq, M } from "./components/Math";
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
          <a href="#theory">Theory</a>
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
                Nested sparsity in Nestor. <em>(a)</em> In each block the column router sends the token{" "}
                <M>{String.raw`\vx_j`}</M> to <M>{String.raw`\Kc = 2`}</M> of the <M>C = 4</M> columns drawn; each column has{" "}
                <M>M</M> layers, each with its own attention and its own <M>E = 4</M> experts, of which a per-layer expert
                router selects <M>{String.raw`\Ke = 2`}</M>. The token uses only the coloured parameters. <em>(b)</em> The
                causal attention map computed by each column for <M>S = 8</M> tokens: column <M>c</M> attends only among
                the tokens routed to it.
              </>
            }
          />
          <Stats />
        </div>

        <div className="layout">
          <Toc entries={toc} />

          <article className="article">
            <Section id="abstract" title="Abstract">
              {paper.abstract.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </Section>

            <Section id="architecture" title="The Nestor architecture">
              <p>
                <strong>Conditional computation in MoE transformers.</strong> Each layer of a transformer updates the
                token features <M>{String.raw`\vx_1, \dots, \vx_S \in \R^D`}</M> with an attention operator, which mixes
                information across tokens, followed by an MLP, which acts on each token separately. The MoE transformer
                replaces the MLP by <M>E</M> expert MLPs, of which each token uses only <M>{String.raw`\Ke \ll E`}</M>. A
                learned router <M>{String.raw`\vr`}</M> scores the experts, and the operator computes (omitting
                normalization here and below)
              </p>
              <Eq tag="1">{eq.moe}</Eq>
              <p>
                with gating weights <M>{String.raw`\alpha_{je}`}</M> given by the softmax of the selected scores. The
                parameters grow with <M>E</M>; the compute per token grows only with <M>{String.raw`\Ke`}</M>. This
                conditional computation is confined to the MLP.
              </p>
              <p>
                <strong>Column-routed attention.</strong> Nestor applies the same principle to attention. The attention
                operator is replaced by <M>C</M> operators <M>{String.raw`\Attn_1, \dots, \Attn_C`}</M> with separate
                parameters, the <em>columns</em>. A column router with scores <M>{String.raw`\vs(\vx_j) \in \R^C`}</M>{" "}
                selects for each token a set <M>{String.raw`\cC_j`}</M> of <M>{String.raw`\Kc`}</M> columns with gating
                weights <M>{String.raw`w_{jc}`}</M>. Writing <M>{String.raw`\cU_c \doteq \{\, i \in [S] : c \in \cC_i \,\}`}</M>{" "}
                for the tokens routed to column <M>c</M>, the routed attention computes
              </p>
              <Eq tag="2">{eq.columnAttention}</Eq>
              <p>
                where <M>{String.raw`\Attn_c(\vx, \cdot)`}</M> is multi-head attention with the query from{" "}
                <M>{String.raw`\vx`}</M> and keys and values from the given context. Each column is a
                sequence-to-sequence operator on a subsequence chosen by the router, so one routing decision sparsifies
                both the parameters a token uses (<M>{String.raw`\Kc`}</M> of the <M>C</M> attention operators) and the
                attention pattern (Figure 1b).
              </p>
              <p>
                <strong>Nested sparsity.</strong> The two routing mechanisms compose hierarchically. Each column{" "}
                <M>c</M> has its own experts <M>{String.raw`\MLP_{c,1}, \dots, \MLP_{c,E}`}</M> and its own expert router{" "}
                <M>{String.raw`\vr_c`}</M>, which selects <M>{String.raw`\Ke`}</M> of them in every active column:
              </p>
              <Eq tag="3">{eq.nested}</Eq>
              <p>
                An expert is indexed by the pair <M>(c, e)</M>. One layer stores <M>C</M> attention operators and{" "}
                <M>CE</M> experts, while a token uses <M>{String.raw`\Kc`}</M> and <M>{String.raw`\Kc \Ke`}</M> of them;
                a small number of shared experts is tied across columns. Setting <M>C = 1</M> recovers the MoE
                transformer exactly, and further setting <M>E = 1</M> recovers the dense transformer.
              </p>
              <p>
                <strong>Blocks and the depth read.</strong> The layers are grouped into <em>blocks</em> of <M>M</M>{" "}
                layers, and the column routing is computed once per block from its input, so a token stays in the same
                columns for all <M>M</M> layers: within a block, each column is a small MoE transformer of depth{" "}
                <M>M</M> on its own subsequence. The number of active columns grows with depth. All experiments use{" "}
                <M>C = 8</M> and <M>{String.raw`\Kc = 3, 4, 5, 6, 6, 7, 8`}</M> over seven blocks, so the last block
                computes full attention over the whole sequence; the baselines use full attention in every layer and
                therefore spend more attention compute. The walkthrough below follows one token through the 200M model.
              </p>

              <div className="breakout">
                <ArchFlow />
              </div>
              <p className="arch__caption">
                <strong>Walkthrough.</strong> One token's path through the 200M configuration (seven blocks, <M>C = 8</M>,{" "}
                <M>{String.raw`\Kc = 3, 4, 5, 6, 6, 7, 8`}</M>). Routing, expert choices and depth-read weights are
                synthetic but balanced; the shapes are the paper's. Click a token to follow it, or a block to jump to it.
              </p>

              <p>
                <strong>Residual connections.</strong> Equations (2) and (3) show plain residuals; in Nestor each active
                column accumulates its outputs in its own residual stream, and the streams are combined within and
                across blocks by an attention residual. Given sources <M>{String.raw`\vz_1, \dots, \vz_N \in \R^D`}</M>,
                a learned query <M>{String.raw`\vq`}</M> and gains <M>{String.raw`g_i`}</M>, it returns the mixture
              </p>
              <Eq tag="4">{eq.attnRes}</Eq>
              <p>
                The input of each layer is such a mixture of the token embedding, the outputs of the earlier blocks and
                the mean of the streams of the current block; the input of the output head also includes the deviation
                of each stream from its block mean. The baselines use the same attention residual.
              </p>
              <p>
                <strong>Routing.</strong> Both routers select by top-<M>K</M> per token. The columns are balanced
                without an auxiliary loss: a bias <M>{String.raw`\vb \in \R^C`}</M> enters only the selection, receives
                no gradient, and the gates are the softmax of the selected scores:
              </p>
              <Eq tag="5">{eq.routerGate}</Eq>
              <p>
                After every optimizer step the bias moves towards the columns with below-average load{" "}
                <M>{String.raw`n_c`}</M> (the number of tokens in the batch that selected column <M>c</M>), with no
                auxiliary balancing loss:
              </p>
              <Eq tag="6">{eq.routerBias}</Eq>
            </Section>

            <Section id="pretraining" title="Compute-efficient pretraining">
              <p>
                We compare three architectures at four scales, named 100M, 200M, 500M and 1B after their <em>active</em>{" "}
                parameters, all trained on DCLM-baseline at context length <M>S = 4096</M> for 50 tokens per active
                parameter. The dense transformer has width <M>{String.raw`D \in \{480, 640, 960, 1280\}`}</M>, depth{" "}
                <M>{String.raw`\{12, 16, 24, 32\}`}</M>, 16 heads and MLP width <M>4D</M>. The MoE transformer has the same
                shape with each MLP replaced by <M>E = 24</M> experts of which <M>{String.raw`\Ke = 3`}</M> are active.
                Nestor uses no more training FLOPs than either baseline and stores about as many parameters as the MoE.
                Each architecture's loss–compute curve is fitted as{" "}
                <M>{String.raw`L(F) = A\,F^{-\alpha} + L_\infty`}</M>; the horizontal ticks mark the compute each baseline
                needs to match Nestor.
              </p>
              <Figure
                src="scaling.svg"
                number={2}
                width={519}
                height={194}
                alt="Held-out loss against training compute for Nestor, MoE and dense models, with fitted scaling laws."
                caption={
                  <>
                    Held-out loss scaling laws. Nestor reaches the lowest held-out loss at every scale; the horizontal
                    ticks mark the projected compute the MoE and dense baselines need to match it.
                  </>
                }
              />
              <p>
                Nestor reaches the lowest held-out loss at all four scales. At 1B it reaches 2.571; matching it is
                estimated to need 1.22× the compute for the MoE transformer and 4.1× for the dense transformer, and
                neither multiplier shrinks as the scale grows. The fitted exponents of Nestor and the MoE agree to three
                digits; Nestor has a smaller coefficient and a smaller irreducible loss.
              </p>
              <div className="breakout">
                <ResultsTable />
              </div>
              <p className="table-caption">
                <strong>Table 1:</strong> The twelve runs behind Figure 2. The last column is the compute each baseline is
                estimated to need to match Nestor's held-out loss at the same scale, from the fitted laws.
              </p>
            </Section>

            <Section id="continual" title="Continual pretraining">
              <p>
                A pretrained model is customized by continual training on new domains and longer contexts, and should
                keep its capabilities on the old ones. Nestor adapts to new domains as well as the MoE transformer while
                forgetting less, and it extends to long contexts more efficiently.
              </p>

              <Section id="domain-adaptation" level={3} title="Domain adaptation">
                <p>
                  The 500M Nestor, the 500M MoE and the 1B dense transformer, chosen for their similar pretraining
                  losses, are first extended to a 32k context and then trained on 0.5B tokens of one of four target
                  domains: books (PG-19), mathematics (OpenWebMath), code (AlgebraicStack) and German (FineWeb-2).
                  Routers are frozen, every other parameter is trained. Adaptation is the reduction in held-out loss on
                  the target corpus and forgetting the increase on the pretraining corpus, both relative to the model
                  before adaptation.
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
                      Domain adaptation and forgetting relative to the MoE transformer. <strong>Top:</strong> reduction in
                      held-out loss on the target corpus. <strong>Bottom:</strong> increase in held-out loss on the
                      pretraining corpus. Faded marks: 25% of the training tokens replayed from the pretraining corpus.
                    </>
                  }
                />
                <p>
                  All three models gain about the same on every domain: averaged over the four, Nestor and the 1B dense
                  model are within 1% of the MoE. They differ in what they forget. Nestor forgets 3% less than the MoE
                  on average, the dense model 21% more. Replaying 25% pretraining data cuts forgetting by about 70% for
                  all three at a cost of about 6% of the gain, and keeps the ordering.
                </p>
              </Section>

              <Section id="long-context" level={3} title="Long-context extension">
                <p>
                  Starting from the 4k checkpoints, each model is extended to <M>{String.raw`S \in \{8\text{k}, 16\text{k}, 32\text{k}, 64\text{k}\}`}</M>{" "}
                  by continued pretraining on documents of at least <M>S</M> tokens. The baselines train on 0.5B tokens.
                  Because Nestor's FLOPs per token fall relative to the baselines as <M>S</M> grows, it is trained both
                  at <em>equal tokens</em> and at <em>equal FLOPs</em>, where the saved compute buys more tokens of
                  repeated or new documents.
                </p>
                <Figure
                  src="cpt.svg"
                  number={4}
                  width={474}
                  height={241}
                  alt="Held-out loss at each context length from 4k to 64k, relative to the MoE transformer, at four scales."
                  caption={
                    <>
                      Long-context extension by continued pretraining: held-out loss at each length relative to the MoE
                      transformer of the same scale. <strong>Top:</strong> the dense transformer. <strong>Bottom:</strong>{" "}
                      Nestor at equal FLOPs with new documents, at equal FLOPs with repeated documents, and at equal
                      tokens. Open markers at 4k are the pretrained checkpoints.
                    </>
                  }
                />
                <p>
                  At equal tokens Nestor keeps its advantage from pretraining. At equal FLOPs the advantage grows with the
                  context length: the saved compute spent on repeated data already lowers the loss relative to the MoE,
                  and spent on fresh data it lowers it most. For equal shapes and balanced routing, Nestor trains on{" "}
                  <M>R(S)</M> times as many tokens as the MoE at equal FLOPs,
                </p>
                <Eq tag="7">{eq.tokensMultiplier}</Eq>
                <p>
                  where <M>{String.raw`\rho`}</M> is the average fraction <M>{String.raw`\Kc / C`}</M> of active columns and{" "}
                  <M>{String.raw`S_0`}</M> the context length at which the attention FLOPs of the MoE transformer equal its
                  other FLOPs.
                </p>
                <Figure
                  src="flops.svg"
                  width={491}
                  height={280}
                  size="column"
                  alt="Forward FLOPs per token against context length for the three architectures at four scales."
                  caption={
                    <>
                      Forward FLOPs per token of the three architectures at each scale as the sequence length grows;
                      Nestor's relative cost falls towards the fraction of active columns.
                    </>
                  }
                />
              </Section>
            </Section>

            <Section id="theory" title="Theory: why nesting helps">
              <p>
                We compare stylized versions of Nestor and of the MoE transformer on a statistical model with
                hierarchical latent structure. The next token is predicted in two stages: a coarse label{" "}
                <M>{String.raw`c \in [C]`}</M> determines which tokens of the context are relevant, how they are weighted and
                which statistic of them is computed, and a fine label <M>{String.raw`e \in [E]`}</M> determines how that
                statistic is used. In Nestor, <M>c</M> corresponds to the column router and <M>e</M> to the expert router
                of the column. We study the simplest case: one layer, one head, linear attention and maps, squared loss.
                The context holds tokens <M>{String.raw`\vx_i = (c_i, \vz_i)`}</M> with uniform labels and independent
                contents, the query carries the label <M>c</M>, and the summary and target are
              </p>
              <Eq tag="8">{eq.linearModel}</Eq>
              <p>
                with patterns <M>{String.raw`\vW_c^\star`}</M>, value maps <M>{String.raw`\vA_c^\star \in \R^{r \times D}`}</M>{" "}
                and expert maps <M>{String.raw`\vB_{ce}^\star \in \R^{D \times r}`}</M>, <M>{String.raw`r < D`}</M>. The
                regression function is <M>{String.raw`\vG_{ce}^\star \vm`}</M>, where
              </p>
              <Eq tag="9">{eq.nestedStructure}</Eq>
              <p>
                The last property is the nested structure: for a fixed coarse label the <M>E</M> maps share the pattern
                and the row space, while both may differ between coarse labels.
              </p>
              <Figure
                src="theory.svg"
                number={5}
                width={583}
                height={205}
                alt="The linear model with coarse and fine labels, and the flat and nested predictors."
                caption={
                  <>
                    The linear model and the two predictors, for <M>C = 2</M> coarse labels (blue, green) and{" "}
                    <M>E = 3</M> fine labels. <em>(a)</em> One sample. <em>(b)</em> The flat predictor (MoE transformer):
                    the shared attention scores every token, and each expert has its own row space, fit from its own
                    samples. <em>(c)</em> The nested predictor (Nestor): the attention of a column scores only the tokens
                    with its label, and its row space is fit once from the samples of all of its experts. Free parameters
                    are counted below each panel.
                  </>
                }
              />
              <p>
                <strong>The two predictors.</strong> The <em>nested</em> predictor is one layer of Nestor with{" "}
                <M>{String.raw`\Kc = 1`}</M>: column <M>c</M> attends over the tokens with label <M>c</M> through its own
                pattern <M>{String.raw`\vW_c`}</M> and value map <M>{String.raw`\vA_c`}</M>, and expert <M>(c, e)</M>{" "}
                applies <M>{String.raw`\vB_{ce}`}</M>. The <em>flat</em> predictor is one layer of an MoE transformer: a
                shared attention over all tokens with a full-width value map <M>{String.raw`\vV`}</M>, and linear experts
                with <M>r</M> hidden units,
              </p>
              <Eq tag="10">{eq.predictors}</Eq>
              <p>Each is fit by least squares, and we measure its excess risk over the regression function.</p>

              <Callout kind="Theorem" title="informal">
                <p>
                  Let <M>{String.raw`C \geq 2`}</M>. <em>(a)</em> Both predictors can represent the regression function,
                  unlike a flat predictor with a bilinear score on raw tokens. <em>(b)</em> As the number of samples{" "}
                  <M>{String.raw`T \to \infty`}</M>, the excess risk of either predictor is asymptotically{" "}
                  <M>{String.raw`\sigma^2 \chi^2_d / T`}</M>, where <M>d</M> counts the directions in weight space that change
                  the prediction. For <M>{String.raw`r \ll D`}</M> and <M>{String.raw`E \gg 1`}</M>,
                </p>
                <Eq tag="11">{eq.dims}</Eq>
                <p>
                  <em>(c)</em> If a sample costs <M>{String.raw`F_0`}</M> FLOPs plus <M>{String.raw`F_1`}</M> per scored
                  token, the flat predictor needs
                </p>
                <Eq tag="12">{eq.multiplier}</Eq>
                <p>
                  times as many FLOPs for the same excess risk, a factor that grows with <M>S</M> from{" "}
                  <M>{String.raw`\dF / \dN`}</M> to <M>{String.raw`C\, \dF / \dN`}</M>.
                </p>
              </Callout>

              <p>
                The count <M>d</M> has three parts: the patterns, the row spaces, and the maps from the row spaces to
                the output. The flat predictor fits one row space per expert and <M>{String.raw`C^2`}</M> pattern blocks,
                of which the <M>C(C-1)</M> blocks between different labels are zero at the target but still have to be
                fit from data; the nested predictor fits one row space and one pattern per column, from the samples of
                all of the column's experts, and its routing fixes the other blocks to zero. The ratio is near <M>C</M>{" "}
                when the patterns dominate and near 2 when the experts do. Part (c) is the multiplier <M>R(S)</M> of
                equation (7) with <M>{String.raw`\rho = 1 / C`}</M>.
              </p>

              <Callout kind="Proposition" title="informal" tone="blue">
                <p>
                  Train each predictor, starting from weights that represent the regression function, by gradient flow
                  on a new domain whose samples all carry the coarse label <M>{String.raw`c_Q`}</M> and new maps. In the
                  nested predictor the columns <M>{String.raw`c \neq c_Q`}</M> never change, so nothing is forgotten on
                  any other coarse label. In the flat predictor the shared value map <M>{String.raw`\vV(t)`}</M> moves,
                  and the forgetting on <M>(c, e)</M> at time <M>t</M> is
                </p>
                <Eq tag="13">{eq.forgetting}</Eq>
                <p>
                  where <M>{String.raw`\vXi \neq 0`}</M> whenever the value map starts moving. For orthonormal rows the
                  coefficient lies between multiples of <M>{String.raw`\cos^2 \theta_r`}</M> and{" "}
                  <M>{String.raw`\cos^2 \theta_1`}</M>, the principal angles between the row spaces of{" "}
                  <M>{String.raw`\vA_c^\star`}</M> and <M>{String.raw`\vA_{c_Q}^\star`}</M>.
                </p>
              </Callout>

              <p>
                In the flat predictor the shared value map is the only source of forgetting on other labels, and it
                grows with the overlap between row spaces, the generic case. Nesting improves the sample and compute
                efficiency when the relevant tokens, patterns and row spaces differ between coarse labels, and removes
                this forgetting, consistent with Figure 3.
              </p>
            </Section>

            <Section id="code" title="Code">
              <p>
                The repository holds the model, the training stack and the registry of the twelve pretraining runs of
                Figure 2, <code>{"{dense,moe,nestor}_{100m,200m,500m,1b}"}</code>, with the data preparation for
                DCLM-baseline and the held-out evaluation.
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
