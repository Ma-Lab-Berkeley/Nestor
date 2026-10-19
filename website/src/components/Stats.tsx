import { stats } from "../content/paper";

/** The four headline numbers in one quiet row. */
export function Stats() {
  return (
    <ul className="stats" aria-label="Headline results">
      {stats.map((s) => (
        <li key={s.value} className="stat">
          <span className="stat__value">{s.value}</span>
          <span className="stat__label">{s.label}</span>
        </li>
      ))}
    </ul>
  );
}
