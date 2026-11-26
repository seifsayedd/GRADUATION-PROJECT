import { useMemo, useState } from "react";
import { api } from "../api";
import { ErrorBox, Loading, Panel } from "../components";
import { go, useAsync } from "../hooks";
import { fmt, linear, shortModel, timeOf } from "../lib";
import type { TimelineRow } from "../types";

const W = 900, LANE = 46, LEFT = 110;

export default function TimelineView() {
  const t = useAsync(() => api.timeline(), []);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [hover, setHover] = useState<TimelineRow | null>(null);

  const rows = t.data?.rows ?? [];
  const times = rows.map((r) => r.start_ms ?? 0);
  const [t0, t1] = [Math.min(...times), Math.max(...times)];
  const x = useMemo(() => linear(t0, t1, LEFT, W - 20), [t0, t1]);

  if (t.error) return <ErrorBox error={t.error} retry={t.reload} />;
  if (!t.data) return <Loading what="timeline" />;
  const lanes = t.data.models.filter((m) => !hidden.has(m));
  const showGru = !hidden.has("gru");
  const nLanes = lanes.length + (showGru ? 1 : 0) + 1; // + truth lane
  const H = nLanes * LANE + 30;
  const toggle = (m: string) => { const s = new Set(hidden); s.has(m) ? s.delete(m) : s.add(m); setHidden(s); };
  const minutes = ((t1 - t0) / 60000).toFixed(1);

  return (
    <>
      <h1>Lab timeline</h1>
      <p className="lead">{rows.length} conversations from lab capture v1 over {minutes} minutes, in capture order. {t.data.note}</p>
      <Panel title="Verdicts over time" source={t.data.source} actions={
        <div className="chips">
          {[...t.data.models, "gru"].map((m) => (
            <button key={m} className={`chip ${hidden.has(m) ? "" : "on"}`} onClick={() => toggle(m)}>{m === "gru" ? "GRU score" : shortModel(m)}</button>
          ))}
          <button className={`chip ${onlyFlagged ? "on" : ""}`} onClick={() => setOnlyFlagged(!onlyFlagged)}>flagged only</button>
        </div>}>
        <svg className="chart timeline" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="lab timeline">
          {[{ name: "ground truth" }, ...lanes.map((m) => ({ name: shortModel(m) })), ...(showGru ? [{ name: "GRU P(attack)" }] : [])]
            .map((l, i) => (
              <g key={l.name}>
                <line x1={LEFT} x2={W - 20} y1={i * LANE + 24} y2={i * LANE + 24} className="lane" />
                <text x={LEFT - 10} y={i * LANE + 28} className="lbl" textAnchor="end">{l.name}</text>
              </g>
            ))}
          {rows.map((r) => {
            const cx = x(r.start_ms ?? t0);
            const attack = r.truth !== "Benign";
            return (
              <g key={r.row} onMouseEnter={() => setHover(r)} onMouseLeave={() => setHover(null)}>
                {attack && <rect x={cx - 6} y={4} width={12} height={H - 34} className="attack-band" />}
                <circle cx={cx} cy={24} r={attack ? 7 : 4} className={attack ? "dot-attack" : "dot-benign"} />
                {lanes.map((m, i) => {
                  const v = r.verdicts[m];
                  if (!v || (onlyFlagged && !v.flagged)) return null;
                  return <circle key={m} cx={cx} cy={(i + 1) * LANE + 24} r={v.flagged ? 6 : 3}
                    className={v.flagged ? "dot-flag" : "dot-quiet"} onClick={() => go("alert", v.alert_id)} style={{ cursor: "pointer" }} />;
                })}
                {showGru && r.gru_p_attack !== undefined && (
                  <rect x={cx - 2} y={(lanes.length + 1) * LANE + 24 - 18 * r.gru_p_attack} width={4}
                    height={Math.max(1, 18 * r.gru_p_attack)} className="gru-bar" />
                )}
              </g>
            );
          })}
          <text x={LEFT} y={H - 4} className="lbl">{timeOf(t0)}</text>
          <text x={W - 20} y={H - 4} className="lbl" textAnchor="end">{timeOf(t1)}</text>
        </svg>
        <div className="hover-card">
          {hover ? (
            <>
              <b>row {hover.row}</b> · {hover.session} · <span className="mono">{hover.protocol} {hover.src} → {hover.dst}{hover.dst_port ? `:${hover.dst_port}` : ""}</span>
              {" "}· truth <b>{hover.truth}</b> ({hover.true_tactic}) ·{" "}
              {Object.entries(hover.verdicts).map(([m, v]) => `${shortModel(m)} ${v.flagged ? "FLAG" : "ok"} ${fmt(v.score)}`).join(" · ")}
              {hover.gru_p_attack !== undefined && ` · GRU ${fmt(hover.gru_p_attack)}`}
            </>
          ) : <span className="muted">Hover a column for details; click a dot to open that explanation. The tall red band marks the attack conversation.</span>}
        </div>
      </Panel>
    </>
  );
}
