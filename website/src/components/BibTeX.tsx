import { paper } from "../content/paper";
import { CodeBlock } from "./CodeBlock";

export function BibTeX() {
  return <CodeBlock code={paper.bibtex} label="BibTeX" />;
}
