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
  bibtex: `@article{chu2026nestor,
  title   = {Nested Sparsity for Scalable Pretraining and Continual Learning},
  author  = {Chu, Tianzhe and Pai, Druv and Ma, Yi},
  year    = {2026},
  note    = {Preprint}
}`,
} as const;

/** Entries of the table of contents, in page order. */
export const toc = [
  { id: "summary", label: "Summary" },
  { id: "architecture", label: "What is Nestor?" },
  { id: "pretraining", label: "Pretraining" },
  { id: "domain-adaptation", label: "Domain adaptation" },
  { id: "long-context", label: "Long-context extension" },
  { id: "theory", label: "Theory" },
  { id: "code", label: "Code" },
  { id: "bibtex", label: "Citation" },
];

/** The twelve experiments shipped in the repository's registry. */
export const quickstart = `uv sync
uv run nestor_cli --list-experiments   # {dense,moe,nestor}_{100m,200m,500m,1b}
uv run nestor_cli --experiment nestor_200m`;
