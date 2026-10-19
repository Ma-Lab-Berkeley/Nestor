import { useEffect, useState } from "react";

export type TocEntry = { id: string; label: string; depth?: 1 | 2 };

/**
 * The sticky table of contents on wide screens. The active entry follows the
 * section nearest the top of the viewport.
 */
export function Toc({ entries }: { entries: TocEntry[] }) {
  const [active, setActive] = useState(entries[0]?.id ?? "");

  useEffect(() => {
    const sections = entries.map((e) => document.getElementById(e.id)).filter((el): el is HTMLElement => el !== null);
    if (sections.length === 0) return;
    const update = () => {
      const line = window.innerHeight * 0.3;
      let current = sections[0].id;
      for (const s of sections) {
        if (s.getBoundingClientRect().top <= line) current = s.id;
      }
      setActive(current);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, [entries]);

  return (
    <nav className="toc" aria-label="Contents">
      <ol>
        {entries.map((e) => (
          <li key={e.id} className={`toc__item${e.depth === 2 ? " toc__item--sub" : ""}${e.id === active ? " is-active" : ""}`}>
            <a href={`#${e.id}`}>{e.label}</a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
