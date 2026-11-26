import { useState, type ReactNode } from "react";
import { api, errorText } from "./api";
import { linkCitations } from "./lib";
import type { SourceText } from "./types";

export function Panel({ title, source, children, actions }: {
  title: string; source?: string; children: ReactNode; actions?: ReactNode;
}) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{title}</h2>
        <div className="panel-actions">{actions}</div>
      </header>
      {children}
      {source && <SourceTag source={source} />}
    </section>
  );
}

export function SourceTag({ source }: { source: string }) {
  return <p className="source">source: <code>{source}</code></p>;
}

export function Loading({ what = "" }: { what?: string }) {
  return <div className="muted pulse">loading {what}…</div>;
}

export function ErrorBox({ error, retry }: { error: unknown; retry?: () => void }) {
  return (
    <div className="error">
      <strong>Request failed.</strong> {errorText(error)}
      {retry && <button className="btn small" onClick={retry}>retry</button>}
      <div className="muted">Is the API running? <code>python scripts/serve_phase6.py</code></div>
    </div>
  );
}

export function Pill({ on, children, title }: { on: boolean; children: ReactNode; title?: string }) {
  return <span className={`pill ${on ? "on" : "off"}`} title={title}>{children}</span>;
}

/** A citation id that opens the cited text when clicked. */
export function Cite({ id }: { id: string }) {
  const [open, setOpen] = useState(false);
  const [src, setSrc] = useState<SourceText | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const toggle = () => {
    setOpen(!open);
    if (!src && !err) api.source(id).then(setSrc).catch((e) => setErr(errorText(e)));
  };
  return (
    <span className="cite-wrap">
      <button className="cite" onClick={toggle} title="show the cited source">[{id}]</button>
      {open && (
        <span className="cite-pop" role="dialog">
          {err ? <span className="error-text">{err}</span> : !src ? <span className="muted">loading…</span> : (
            <>
              <span className="cite-origin">{src.origin}{src.section ? ` › ${src.section}` : ""}</span>
              <span className="cite-text">{src.text}</span>
            </>
          )}
        </span>
      )}
    </span>
  );
}

/** Text with any known citation ids turned into Cite links. */
export function CitedText({ text, ids }: { text: string; ids: string[] }) {
  return (
    <>
      {linkCitations(text, ids).map((p, i) => ("cite" in p ? <Cite key={i} id={p.cite} /> : <span key={i}>{p.text}</span>))}
    </>
  );
}

export function Citations({ ids }: { ids: string[] }) {
  if (!ids.length) return <span className="muted">no citation</span>;
  return <span className="cites">{ids.map((c) => <Cite key={c} id={c} />)}</span>;
}
