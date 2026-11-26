import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { ErrorBox, Loading, Panel, Pill } from "../components";
import { go, useAsync } from "../hooks";
import { fmt, shortModel, timeOf } from "../lib";
import type { AlertRow } from "../types";

export default function Alerts() {
  const all = useAsync(() => api.alerts(), []);
  const [model, setModel] = useState("");
  const [dataset, setDataset] = useState("");
  const [flagged, setFlagged] = useState("");
  const [text, setText] = useState("");
  const [replay, setReplay] = useState<number | null>(null);
  const timer = useRef<number>();

  const rows = all.data?.rows ?? [];
  const models = useMemo(() => [...new Set(rows.map((r) => r.model))].sort(), [rows]);
  const datasets = useMemo(() => [...new Set(rows.map((r) => r.dataset ?? ""))].sort(), [rows]);
  const shown = rows.filter((r) => (!model || r.model === model) && (!dataset || r.dataset === dataset)
    && (!flagged || String(r.flagged) === flagged)
    && (!text || `${r.alert_id} ${r.src} ${r.dst} ${r.true_tactic}`.toLowerCase().includes(text.toLowerCase())));

  // Replay: reveal the lab alerts one by one in capture order.
  const labOrdered = useMemo(() => rows.filter((r) => r.dataset === "lab")
    .sort((a, b) => (a.start_ms ?? 0) - (b.start_ms ?? 0)), [rows]);
  useEffect(() => {
    if (replay === null) return;
    if (replay >= labOrdered.length) { setReplay(null); return; }
    timer.current = window.setTimeout(() => setReplay(replay + 1), 120);
    return () => window.clearTimeout(timer.current);
  }, [replay, labOrdered.length]);
  const visible: AlertRow[] = replay === null ? shown : labOrdered.slice(0, replay).reverse();

  if (all.error) return <ErrorBox error={all.error} retry={all.reload} />;
  if (!all.data) return <Loading what="alerts" />;

  return (
    <>
      <h1>Alerts</h1>
      <p className="lead">One row per Phase 4 explanation object: the 54 lab conversations scored by three models, plus sampled benchmark examples.</p>
      <Panel title={replay === null ? `${shown.length} of ${rows.length} objects` : `Replaying lab capture v1: ${replay} of ${labOrdered.length}`}
        actions={
          replay === null
            ? <button className="btn" onClick={() => setReplay(0)} title="replay the lab alerts in capture order">▶ replay lab capture</button>
            : <button className="btn" onClick={() => setReplay(null)}>■ stop</button>
        }>
        {replay === null && (
          <div className="filters">
            <select value={model} onChange={(e) => setModel(e.target.value)} aria-label="model">
              <option value="">all models</option>
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)} aria-label="dataset">
              <option value="">all datasets</option>
              {datasets.map((d) => <option key={d} value={d}>{d || "(none)"}</option>)}
            </select>
            <select value={flagged} onChange={(e) => setFlagged(e.target.value)} aria-label="decision">
              <option value="">flagged or not</option>
              <option value="true">flagged</option>
              <option value="false">not flagged</option>
            </select>
            <input placeholder="search id, IP, tactic" value={text} onChange={(e) => setText(e.target.value)} />
            {(model || dataset || flagged || text) &&
              <button className="btn small" onClick={() => { setModel(""); setDataset(""); setFlagged(""); setText(""); }}>clear</button>}
          </div>
        )}
        <div className="scroll">
          <table className="tbl hover">
            <thead>
              <tr><th>Decision</th><th>Model</th><th>Dataset</th><th>Flow</th><th>Start (UTC)</th>
                <th className="num">Score</th><th className="num">Threshold</th><th>True tactic</th></tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.alert_id} onClick={() => go("alert", r.alert_id)} className={replay !== null ? "fade-in" : ""}>
                  <td><Pill on={r.flagged}>{r.flagged ? "FLAGGED" : "not flagged"}</Pill></td>
                  <td title={r.model}>{shortModel(r.model)}</td>
                  <td>{r.dataset}</td>
                  <td className="mono">{r.protocol?.match(/\((\w+)\)/)?.[1] ?? r.protocol}{r.src} → {r.dst}{r.dst_port ? `:${r.dst_port}` : ""}</td>
                  <td className="mono">{timeOf(r.start_ms)}</td>
                  <td className="num">{fmt(r.score)}</td>
                  <td className="num">{fmt(r.threshold)}</td>
                  <td>{r.true_tactic}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">Click a row for its explanation and a grounded report. Benchmark start times come from the source datasets.</p>
      </Panel>
    </>
  );
}
