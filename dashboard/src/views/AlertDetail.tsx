import { useState } from "react";
import { api, asRejection, errorText } from "../api";
import { Citations, CitedText, ErrorBox, Loading, Panel, Pill } from "../components";
import { useAsync } from "../hooks";
import { contributionBars, fmt, linear, timeOf } from "../lib";
import type { Explanation, Generated, Rejection, ReportOutput } from "../types";

const W = 360;

function ContributionChart({ e }: { e: Explanation }) {
  const c = e.contributions!;
  const bars = contributionBars(c.items, W);
  const rowH = 22;
  const attackWord = e.model.stage === 3 ? "more abnormal" : "towards attack";
  return (
    <>
      <svg className="chart" viewBox={`0 0 ${W + 290} ${bars.length * rowH + 24}`} role="img"
        aria-label="feature contributions">
        <line x1={170 + W / 2} x2={170 + W / 2} y1={0} y2={bars.length * rowH} className="axis" />
        {bars.map((b, i) => (
          <g key={b.feature} transform={`translate(0, ${i * rowH})`}>
            <text x={164} y={15} className="lbl" textAnchor="end">{b.feature}</text>
            <rect x={170 + b.x} y={4} width={Math.max(1, b.width)} height={rowH - 8} rx={2}
              className={b.positive ? "bar-pos" : "bar-neg"}>
              <title>{`${b.feature} = ${b.value}: ${b.contribution > 0 ? "+" : ""}${b.contribution}`}</title>
            </rect>
            <text x={170 + W + 8} y={15} className="val">{b.value}</text>
            <text x={W + 290 - 4} y={15} className="num-t" textAnchor="end">{b.contribution > 0 ? "+" : ""}{fmt(b.contribution)}</text>
          </g>
        ))}
        <text x={170 + W / 2 + 6} y={bars.length * rowH + 16} className="lbl">→ {attackWord}</text>
        {e.model.stage !== 3 && <text x={170 + W / 2 - 6} y={bars.length * rowH + 16} className="lbl" textAnchor="end">towards benign ←</text>}
      </svg>
      <p className="muted">
        Space: {c.space}. Base value {fmt(c.base_value)}, output {fmt(c.output)}; the remaining features add {fmt(c.rest)}.
        Additivity error {c.additivity_error.toExponential(1)} (the contributions sum to the model's real output).
      </p>
    </>
  );
}

function AttentionChart({ e }: { e: Explanation }) {
  const items = e.attention!;
  const max = Math.max(...items.map((a) => a.weight));
  return (
    <>
      <table className="tbl">
        <thead><tr><th>Earlier flow</th><th>Attention</th><th className="num">Removing it moves P(attack) by</th></tr></thead>
        <tbody>
          {items.map((a) => (
            <tr key={a.offset}>
              <td>{a.flow}</td>
              <td><span className="meter wide"><span style={{ width: `${(a.weight / max) * 100}%` }} /></span>{fmt(a.weight)}</td>
              <td className="num">{fmt(a.occlusion_delta)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">Attention shows where the GRU looked. The last column is the remove-one-flow test: what actually changed the decision.</p>
    </>
  );
}

function Gauge({ e }: { e: Explanation }) {
  const d = e.decision;
  const hi = Math.max(d.score, d.threshold) * 1.15 || 1;
  const s = linear(0, hi, 0, 300);
  return (
    <svg className="gauge" viewBox="0 0 320 46" role="img" aria-label="score against threshold">
      <rect x={10} y={16} width={300} height={10} rx={5} className="g-track" />
      <rect x={10} y={16} width={Math.min(300, s(d.score))} height={10} rx={5} className={d.flagged ? "g-hot" : "g-cool"} />
      <line x1={10 + s(d.threshold)} x2={10 + s(d.threshold)} y1={8} y2={34} className="g-thr" />
      <text x={10 + s(d.threshold)} y={44} className="lbl" textAnchor="middle">threshold {fmt(d.threshold)}</text>
      <text x={10} y={10} className="lbl">score {fmt(d.score, 4)} ({d.score_kind})</text>
    </svg>
  );
}

function ReportPanel({ id }: { id: string }) {
  const [state, setState] = useState<"idle" | "busy">("idle");
  const [rep, setRep] = useState<Generated<ReportOutput> | null>(null);
  const [rej, setRej] = useState<Rejection | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const run = () => {
    setState("busy"); setRep(null); setRej(null); setErr(null);
    api.report(id).then(setRep).catch((e) => { const r = asRejection(e); r ? setRej(r) : setErr(errorText(e)); })
      .finally(() => setState("idle"));
  };
  const o = rep?.output;
  return (
    <Panel title="Grounded incident report" actions={
      <button className="btn primary" onClick={run} disabled={state === "busy"}>
        {state === "busy" ? "generating…" : rep ? "regenerate" : "Generate report"}
      </button>}>
      {!rep && !rej && !err && <p className="muted">Builds a report from this explanation object and the project's findings through the Phase 5 pipeline. Every number, IP and citation must appear in the sources, or the report is rejected and not stored.</p>}
      {err && <div className="error">{err}</div>}
      {rej && (
        <div className="error">
          <strong>Report rejected: {rej.reason}.</strong>
          {!!rej.unsupported_numbers?.length && <div>numbers not in the sources: {rej.unsupported_numbers.join(", ")}</div>}
          {!!rej.unsupported_ips?.length && <div>IPs not in the sources: {rej.unsupported_ips.join(", ")}</div>}
          {!!rej.unknown_citations?.length && <div>unknown citations: {rej.unknown_citations.join(", ")}</div>}
        </div>
      )}
      {o && rep && (
        <div className="report">
          <div className="report-meta">
            <Pill on>grounded</Pill> provider <b>{rep.provider}</b> · model {rep.model}
            {rep.report_id && <> · stored as {rep.report_id} (in memory)</>}
            {rep.fallback_used && <> · <b>a fallback model produced this</b></>}
          </div>
          <h3>{o.title}</h3>
          <p><CitedText text={o.summary} ids={rep.sources} /></p>
          <h4>Evidence</h4>
          <ul>{o.evidence.map((ev, i) => <li key={i}><CitedText text={ev.statement} ids={rep.sources} /> <Citations ids={ev.citations} /></li>)}</ul>
          <h4>ATT&CK mapping</h4>
          <p><b>{o.attack_mapping.tactic}</b> — <CitedText text={o.attack_mapping.note} ids={rep.sources} /> <Citations ids={o.attack_mapping.citations} /></p>
          {!!o.caveats.length && <><h4>Caveats</h4><ul>{o.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul></>}
          {!!o.recommended_checks.length && <><h4>Recommended checks</h4><ul>{o.recommended_checks.map((c, i) => <li key={i}>{c}</li>)}</ul></>}
          <p className="muted">Sources given to the model: <Citations ids={rep.sources} /></p>
        </div>
      )}
    </Panel>
  );
}

export default function AlertDetail({ id }: { id: string }) {
  const a = useAsync(() => api.alert(id), [id]);
  if (a.error) return <ErrorBox error={a.error} retry={a.reload} />;
  if (!a.data) return <Loading what="alert" />;
  const e = a.data;
  const f = e.flow;
  return (
    <>
      <p><a href="#/alerts">← all alerts</a></p>
      <h1 className="mono">{e.alert_id}</h1>
      <div className="grid2">
        <Panel title="Decision" source={`explanation object (schema ${e.schema_version}), commit ${e.model.git_commit}`}>
          <div className="kv">
            <span>Verdict</span><span><Pill on={e.decision.flagged}>{e.decision.flagged ? "FLAGGED" : "not flagged"}</Pill></span>
            <span>Model</span><span>Stage {e.model.stage} · {e.model.name} ({e.model.feature_set} features)</span>
            <span>Flow</span><span className="mono">{String(f.protocol)} {String(f.src)}{f.src_port ? `:${f.src_port}` : ""} → {String(f.dst)}{f.dst_port ? `:${f.dst_port}` : ""}</span>
            <span>Dataset</span><span>{String(f.dataset ?? "—")}</span>
            <span>Start</span><span className="mono">{timeOf(f.start_ms as number)}</span>
            <span>Operating point</span><span>{e.decision.operating_point}</span>
            {e.tactic && <><span>Predicted tactic</span><span>{e.tactic.predicted} (confidence {fmt(e.tactic.confidence)})</span></>}
            {e.ground_truth && <><span>Ground truth</span><span>{e.ground_truth.attack} · {e.ground_truth.tactic}</span></>}
          </div>
          <Gauge e={e} />
        </Panel>
        <Panel title="Summary and caveats">
          <p>{e.summary}</p>
          <ul className="caveats">{e.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
        </Panel>
      </div>
      {e.contributions && (
        <Panel title={e.model.stage === 3 ? "Most abnormal features (reconstruction error)" : "What pushed the score (TreeSHAP)"}>
          <ContributionChart e={e} />
        </Panel>
      )}
      {e.attention && <Panel title="Earlier flows in the window (attention vs. removal test)"><AttentionChart e={e} /></Panel>}
      <ReportPanel id={e.alert_id} />
    </>
  );
}
