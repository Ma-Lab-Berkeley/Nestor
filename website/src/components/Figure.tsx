import type { ReactNode } from "react";

type Props = {
  /** File name under public/figures, e.g. "scaling.svg". */
  src: string;
  alt: string;
  /** Figure number shown in the caption; omit for an unnumbered figure. */
  number?: number;
  caption: ReactNode;
  /** Intrinsic size of the SVG (its viewBox in CSS px), so the layout reserves space before the file loads. */
  width: number;
  height: number;
  /** Rendered width: the text column or the wider breakout. */
  size?: "column" | "breakout";
  /** Cap the rendered width in px for small figures. */
  maxWidth?: number;
  /** Load immediately instead of lazily (for the first figure on the page). */
  eager?: boolean;
};

/** A paper figure. The SVGs have dark ink and no background, which is why the page is light-only. */
export function Figure({ src, alt, number, caption, width, height, size = "breakout", maxWidth, eager = false }: Props) {
  return (
    <figure className={`figure figure--${size}`}>
      <div className="figure__surface">
        <img
          src={`${import.meta.env.BASE_URL}figures/${src}`}
          alt={alt}
          width={width}
          height={height}
          style={maxWidth ? { maxWidth } : undefined}
          loading={eager ? "eager" : "lazy"}
        />
      </div>
      <figcaption>
        {number !== undefined && <strong>Figure {number}: </strong>}
        {caption}
      </figcaption>
    </figure>
  );
}
