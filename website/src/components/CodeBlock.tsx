import { useState } from "react";

type Props = { code: string; label: string };

/** A code block with a copy button. */
export function CodeBlock({ code, label }: Props) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable: the text is selectable */
    }
  };
  return (
    <div className="code">
      <div className="code__bar">
        <span>{label}</span>
        <button type="button" className="btn btn--ghost btn--sm" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}
