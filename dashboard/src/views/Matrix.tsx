import { useState } from "react";
import { api } from "../api";
import { ErrorBox, Loading, Panel } from "../components";
import { go, useAsync } from "../hooks";
import { buildMatrix, heat, shortModel } from "../lib";

export default function MatrixView() {
  const m = useAsync(() => api.matrix(), []);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState<"objects" | "flagged">("objects");
  if (m.error) return <ErrorBox error={m.error} retry={m.reload} />;
  if (!m.data) return <Loading what="matrix" />;
  const g = buildMatrix(m.data, hidden);
  const toggle = (x: string) => { const s = new Set(hidden); s.has(x) ? s.delete(x) : s.add(x); setHidden(s); };

  return (
    <>
      <h1>ATT&CK matrix</h1>
      <p className="lead">{m.data.note}</p>
      <Panel title="True tactic × model" source={m.data.source} actions={
        <div className="chips">
          <button className={`chip ${mode === "objects" ? "on" : ""}`} onClick={() => setMode("objects")}>objects</button>
          <button className={`chip ${mode === "flagged" ? "on" : ""}`} onClick={() => setMode("flagged")}>flagged / objects</button>
        </div>}>
        <div className="chips" style={{ marginBottom: 12 }}>
          {m.data.models.map((x) => (
            <button key={x} className={`chip ${hidden.has(x) ? "" : "on"}`} onClick={() => toggle(x)} title={x}>{shortModel(x)}</button>
          ))}
        </div>
        <div className="scroll">
          <table className="matrix">
            <thead><tr><th>Tactic</th>{g.models.map((x) => <th key={x} title={x}>{shortModel(x)}</th>)}</tr></thead>
            <tbody>
              {g.tactics.map((t) => (
                <tr key={t}>
                  <th>{t}</th>
                  {g.models.map((x) => {
                    const c = g.get(t, x);
                    if (!c) return <td key={x} className="empty">·</td>;
                    const v = mode === "objects" ? c.objects / (g.max || 1) : c.flagged / c.objects;
                    return (
                      <td key={x} style={{ background: heat(v) }} title={`${c.flagged} of ${c.objects} flagged`}
                        onClick={() => go("alerts")}>
                        {mode === "objects" ? c.objects : `${c.flagged}/${c.objects}`}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">Tactics follow docs/attack-taxonomy.md. Rows with no stored objects for the visible models are hidden.</p>
      </Panel>
    </>
  );
}
