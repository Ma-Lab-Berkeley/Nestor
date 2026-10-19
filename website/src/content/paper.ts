/** Static content of the project page: everything quoted from the paper lives here. */

export type Affiliation = { key: string; name: string };

export const affiliations: Affiliation[] = [
  { key: "1", name: "The University of Hong Kong" },
  { key: "2", name: "University of California, Berkeley" },
];

export const paper = {
  title: "Nested Sparsity for Scalable Pretraining and Continual Learning",
  shortName: "Nestor",
  venue: "Preprint",
  date: "2026",
  authors: [
    { name: "Tianzhe Chu", equal: true, affiliations: ["1"] },
    { name: "Druv Pai", equal: true, affiliations: ["2"] },
    { name: "Yi Ma", equal: false, affiliations: ["1", "2"] },
  ],
  equalNote: "Equal contribution; ordered alphabetically.",
  links: {
    /** Set when the PDF is public; an empty string renders the button as "coming soon". */
    paper: "",
    code: "https://github.com/Ma-Lab-Berkeley/Nestor",
  },
  abstract: [
    "Mixture-of-experts (MoE) transformers, the standard backbone for modern large-scale deep learning, employ conditional computation: only a sparse subset of model weights is used in each forward pass. In MoE transformers, this sparsity arises in the MoE operator, which generalizes the multi-layer perceptron in the classical transformer. In this work, we explore an alternative mechanism for conditional computation that analogously generalizes the attention operator. The sparsity induced by this mechanism naturally nests (i.e., hierarchically composes) with the sparsity induced by the MoE operator.",
    "We show that a transformer employing this nested sparsity mechanism, which we call Nestor, performs better than a conventional MoE transformer at equal compute and equal sparsity in language model pretraining. We also show that Nestor is well-suited for continual pretraining: it can be adapted to longer contexts more effectively than conventional MoEs, as well as adapt more efficiently to new domains while forgetting less on old domains. Furthermore, we demonstrate that these benefits persist as model size, dataset size, and context length grow. Our results demonstrate the potential of alternative conditional computation mechanisms for deep neural network architectures.",
  ],
  bibtex: `@article{chu2026nestor,
  title   = {Nested Sparsity for Scalable Pretraining and Continual Learning},
  author  = {Chu, Tianzhe and Pai, Druv and Ma, Yi},
  year    = {2026},
  note    = {Preprint}
}`,
} as const;

/** Entries of the table of contents, in page order. */
export const toc = [
  { id: "abstract", label: "Abstract" },
  { id: "architecture", label: "Architecture" },
  { id: "pretraining", label: "Pretraining" },
  { id: "continual", label: "Continual pretraining" },
  { id: "domain-adaptation", label: "Domain adaptation", depth: 2 as const },
  { id: "long-context", label: "Long-context extension", depth: 2 as const },
  { id: "theory", label: "Theory" },
  { id: "code", label: "Code" },
  { id: "bibtex", label: "Citation" },
];

export type Rung = {
  size: string;
  tokens: string;
  compute: string;
  rows: { model: "Dense transformer" | "MoE" | "Nestor"; params: string; loss: string; multiplier: string }[];
};

/** Table 2 of the paper: the twelve pretraining runs behind the scaling figure. */
export const rungs: Rung[] = [
  {
    size: "100M",
    tokens: "5B",
    compute: "0.02",
    rows: [
      { model: "Dense transformer", params: "0.14B", loss: "3.578", multiplier: "3.1×" },
      { model: "MoE", params: "0.37B", loss: "3.386", multiplier: "1.17×" },
      { model: "Nestor", params: "0.38B", loss: "3.356", multiplier: "1.00×" },
    ],
  },
  {
    size: "200M",
    tokens: "10B",
    compute: "0.10",
    rows: [
      { model: "Dense transformer", params: "0.23B", loss: "3.293", multiplier: "3.2×" },
      { model: "MoE", params: "0.79B", loss: "3.108", multiplier: "1.13×" },
      { model: "Nestor", params: "0.76B", loss: "3.092", multiplier: "1.00×" },
    ],
  },
  {
    size: "500M",
    tokens: "25B",
    compute: "0.78",
    rows: [
      { model: "Dense transformer", params: "0.55B", loss: "2.953", multiplier: "4.1×" },
      { model: "MoE", params: "2.41B", loss: "2.793", multiplier: "1.29×" },
      { model: "Nestor", params: "2.44B", loss: "2.758", multiplier: "1.00×" },
    ],
  },
  {
    size: "1B",
    tokens: "50B",
    compute: "3.49",
    rows: [
      { model: "Dense transformer", params: "1.10B", loss: "2.743", multiplier: "4.1×" },
      { model: "MoE", params: "5.49B", loss: "2.592", multiplier: "1.22×" },
      { model: "Nestor", params: "5.53B", loss: "2.571", multiplier: "1.00×" },
    ],
  },
];

/** The twelve experiments shipped in the repository's registry. */
export const quickstart = `uv sync
uv run nestor_cli --list-experiments   # {dense,moe,nestor}_{100m,200m,500m,1b}
uv run nestor_cli --experiment nestor_200m`;
