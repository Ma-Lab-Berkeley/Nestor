import type { ReactNode } from "react";

type Props = {
  /** "Theorem", "Proposition", "Model", ... */
  kind: string;
  /** Short title after the kind, e.g. "informal". */
  title?: string;
  tone?: "purple" | "blue" | "green";
  children: ReactNode;
};

/** A theorem-style statement set off from the running text by a thin rule. */
export function Callout({ kind, title, tone = "purple", children }: Props) {
  return (
    <aside className={`callout callout--${tone}`}>
      <p className="callout__head">
        <strong>{kind}</strong>
        {title && <span> ({title})</span>}
      </p>
      {children}
    </aside>
  );
}
