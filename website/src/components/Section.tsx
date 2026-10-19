import type { ReactNode } from "react";

type Props = {
  id: string;
  title: string;
  /** Section depth: 2 for a top-level section, 3 for a subsection. */
  level?: 2 | 3;
  children: ReactNode;
};

/** A titled section of the article with an anchor id for the table of contents. */
export function Section({ id, title, level = 2, children }: Props) {
  const Tag = level === 2 ? "h2" : "h3";
  return (
    <section id={id} className={level === 2 ? "section" : "subsection"}>
      <Tag>{title}</Tag>
      {children}
    </section>
  );
}
