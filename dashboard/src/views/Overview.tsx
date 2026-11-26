import { api } from "../api";
import { ErrorBox, Loading, Panel, Pill } from "../components";
import { useAsync } from "../hooks";
import { fmt, pct, shortModel } from "../lib";

export default function Overview() {
  const s = useAsync(() => api.summary(), []);
  if (s.error) return <ErrorBox error={s.error} retry={s.reload} />;
  if (!s.data) return <Loading what="summary" />;
  const { cards, lab_flags, holdout } = s.data;
  const maxFlag = Math.max(1, ...lab_flags.rows.map((r) => r.flagged));

  return (
    <>
      <h1>Overview</h1>
      <p className="lead">
        How the detectors do on the benchmark, on networks they were not trained on, and on the project's own lab capture.
      </p>
      <div className="cards">
        {cards.map((c) => (
          <div className="card" key={c.id}>
            <div className="card-label">{c.label}</div>
            <div className="card-value">
              {c.value !== null ? (Number.isInteger(c.value) ? c.value : fmt(c.value)) : c.range ? `${fmt(c.range[0], 2)}–${fmt(c.range[1], 2)}` : "—"}
            </div>
            {c.value !== null && c.range && <div className="card-range">95% CI {fmt(c.range[0])}–{fmt(c.range[1])}</div>}
            <div className="card-note">{c.note}</div>
            <div className="source" title={c.path}><code>{c.source}</code></div>
          </div>
        ))}
      </div>

      <div className="grid2">
        <Panel title="Lab capture v1: conversations each model flagged" source={lab_flags.source}>
          <div className="bars">
            {lab_flags.rows.map((r) => (
              <div className="bar-row" key={r.model}>
                <span className="bar-label">{shortModel(r.model)}</span>
                <span className="bar-track"><span className="bar-fill warn" style={{ width: `${(r.flagged / maxFlag) * 100}%` }} /></span>
                <span className="bar-num">{r.flagged}</span>
                <Pill on={r.attack_caught}>{r.attack_caught ? "attack caught" : "attack missed"}</Pill>
              </div>
            ))}
          </div>
          {lab_flags.rows.every((r) => !r.attack_caught) && (
            <p className="muted">Every flag above is a false alarm: none of the {lab_flags.rows.length} models flagged the attack conversation.</p>
          )}
        </Panel>

        <Panel title="Attack classes removed from training: share detected" source={holdout.source}>
          <table className="tbl">
            <thead><tr><th>Model</th><th>Held-out class</th><th className="num">Detected</th></tr></thead>
            <tbody>
              {holdout.rows.map((r) => (
                <tr key={r.model + r.class}>
                  <td>{shortModel(r.model)}</td><td>{r.class}</td>
                  <td className="num"><span className="meter"><span style={{ width: pct(r.detected, 0) }} /></span>{pct(r.detected)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>
    </>
  );
}
