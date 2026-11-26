import { useState } from "react";
import { api, asRejection, errorText } from "../api";
import { Citations, CitedText, Panel, Pill } from "../components";
import type { AnswerOutput, Generated, Rejection } from "../types";

const SUGGEST = [
  "Why does XGBoost flag benign lab traffic?",
  "How well do the models do on a network they were not trained on?",
  "What did the GRU attention test show?",
  "Which held-out attack class was missed?",
];

type Turn = { q: string; a?: Generated<AnswerOutput>; rej?: Rejection; err?: string };

export default function Ask() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);

  const send = (question: string) => {
    const text = question.trim();
    if (text.length < 3 || busy) return;
    setBusy(true); setQ("");
    api.ask(text)
      .then((a) => setTurns((t) => [{ q: text, a }, ...t]))
      .catch((e) => { const r = asRejection(e); setTurns((t) => [{ q: text, ...(r ? { rej: r } : { err: errorText(e) }) }, ...t]); })
      .finally(() => setBusy(false));
  };

  return (
    <>
      <h1>Ask</h1>
      <p className="lead">Questions are answered only from the project's documents, reports and explanation objects, with citations. Each answer is single-turn.</p>
      <Panel title="Question">
        <form className="ask" onSubmit={(e) => { e.preventDefault(); send(q); }}>
          <input value={q} onChange={(e) => setQ(e.target.value)} maxLength={500} placeholder="Ask about the models, the lab capture or an alert…" />
          <button className="btn primary" disabled={busy || q.trim().length < 3}>{busy ? "thinking…" : "Ask"}</button>
        </form>
        <div className="chips">{SUGGEST.map((s) => <button key={s} className="chip" onClick={() => send(s)} disabled={busy}>{s}</button>)}</div>
      </Panel>
      {turns.map((t, i) => (
        <Panel key={turns.length - i} title={t.q}>
          {t.err && <div className="error">{t.err}</div>}
          {t.rej && <div className="error"><b>Answer rejected: {t.rej.reason}.</b> {[...(t.rej.unsupported_numbers ?? []), ...(t.rej.unsupported_ips ?? []), ...(t.rej.unknown_citations ?? [])].join(", ")}</div>}
          {t.a && (
            <>
              <div className="report-meta">
                <Pill on={t.a.output.in_evidence}>{t.a.output.in_evidence ? "answered from evidence" : "not in the evidence"}</Pill>
                {" "}provider <b>{t.a.provider}</b>{t.a.fallback_used && " · fallback model"}
              </div>
              <div className="answer"><CitedText text={t.a.output.answer} ids={t.a.sources} /></div>
              <p className="muted">Cited: <Citations ids={t.a.output.citations} /></p>
            </>
          )}
        </Panel>
      ))}
    </>
  );
}
