import { affiliations, paper } from "../content/paper";

/** Title, authors with affiliation marks, and the action links. */
export function Hero() {
  const hasPaper = paper.links.paper.length > 0;
  return (
    <header className="hero">
      <h1 className="hero__title">{paper.title}</h1>
      <p className="hero__tagline">{paper.tagline}</p>
      <p className="hero__authors">
        {paper.authors.map((a, i) => (
          <span key={a.name} className="author">
            {a.name}
            <sup>
              {a.affiliations.join(",")}
              {a.equal && "*"}
            </sup>
            {i < paper.authors.length - 1 && <span className="sep">·</span>}
          </span>
        ))}
      </p>
      <p className="hero__affil">
        {affiliations.map((af) => (
          <span key={af.key}>
            <sup>{af.key}</sup>
            {af.name}
          </span>
        ))}
      </p>
      <p className="hero__equal">
        <sup>*</sup>
        {paper.equalNote}
      </p>
      <p className="hero__meta">
        {paper.venue}, {paper.date}
      </p>
      <nav className="hero__links" aria-label="Paper resources">
        {hasPaper ? (
          <a className="btn btn--primary" href={paper.links.paper}>
            Paper
          </a>
        ) : (
          <span className="btn btn--primary is-disabled" aria-disabled="true">
            Paper · coming soon
          </span>
        )}
        <a className="btn" href={paper.links.code} rel="noopener">
          Code
        </a>
        <a className="btn" href="#bibtex">
          BibTeX
        </a>
      </nav>
    </header>
  );
}
