import type { ReactNode } from "react";

type Props = {
  id: string;
  title: string;
  children: ReactNode;
};

/** A titled section of the article with an anchor id for the table of contents. */
export function Section({ id, title, children }: Props) {
  return (
    <section id={id} className="section">
      <h2>{title}</h2>
      {children}
    </section>
  );
}
