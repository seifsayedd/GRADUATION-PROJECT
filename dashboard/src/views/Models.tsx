import { useState } from "react";
import { api } from "../api";
import { ErrorBox, Loading, Panel } from "../components";
import { useAsync } from "../hooks";
import { fmt, linear, shortModel } from "../lib";

const W = 260;

function F1Bar({ f1, lo, hi }: { f1: number; lo: number | null; hi: number | null }) {
  const s = linear(0, 1, 0, W);
  return (
    <svg className="ci" viewBox={`0 0 ${W} 16`} role="img" aria-label={`F1 ${f1}`}>
      <rect x={0} y={6} width={W} height={4} className="g-track" />
      {lo !== null && hi !== null && <rect x={s(lo)} y={3} width={s(hi) - s(lo)} height={10} className="ci-band" />}
      <circle cx={s(f1)} cy={8} r={4} className="ci-dot" />
    </svg>
  );
}

export default function Models() {
  const e = useAsync(() => api.evaluation(), []);
  const [fs, setFs] = useState<"all" | "full" | "lab">("all");
  if (e.error) return <ErrorBox error={e.error} retry={e.reload} />;
  if (!e.data) return <Loading what="evaluation" />;
  const rows = e.data.models.rows.filter((r) => fs === "all" || r.model.endsWith(`_${fs}`));
  const loso = e.data.loso.rows.filter((r) => fs === "all" || r.feature_set === fs);

  return (
    <>
      <h1>Models</h1>
      <p className="lead">Test results with the {e.data.operating_point}. Bars show F1 with its 95% block-bootstrap CI.</p>
      <div className="chips" style={{ marginBottom: 12 }}>
        {(["all", "full", "lab"] as const).map((x) => (
          <button key={x} className={`chip ${fs === x ? "on" : ""}`} onClick={() => setFs(x)}>{x === "all" ? "all feature sets" : `${x} features`}</button>
        ))}
      </div>
      <Panel title="Same networks, later flows (temporal) or random split (stratified)" source={e.data.models.source}>
        <div className="scroll">
          <table className="tbl">
            <thead><tr><th>Model</th><th>F1 (95% CI)</th><th className="num">F1</th><th className="num">Precision</th>
              <th className="num">Recall</th><th className="num">FPR</th><th className="num">Test rows</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.model}>
                  <td title={r.model}>{shortModel(r.model)}</td>
                  <td><F1Bar f1={r.f1} lo={r.f1_lo} hi={r.f1_hi} /></td>
                  <td className="num">{fmt(r.f1)}</td><td className="num">{fmt(r.precision)}</td>
                  <td className="num">{fmt(r.recall)}</td>
                  <td className={`num ${r.fpr > 0.02 ? "warn-text" : ""}`}>{fmt(r.fpr)}</td>
                  <td className="num">{r.n.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">FPR in amber is above 2%, although every threshold was set for 1% on validation flows.</p>
      </Panel>
      <Panel title="Network left out of training (leave-one-source-out, Stage 1 XGBoost)" source={e.data.loso.source}>
        <table className="tbl">
          <thead><tr><th>Features</th><th>Left-out source</th><th>F1</th><th className="num">F1</th><th className="num">Recall</th>
            <th className="num">FPR</th><th className="num">ROC-AUC</th></tr></thead>
          <tbody>
            {loso.map((r) => (
              <tr key={r.feature_set + r.test_source}>
                <td>{r.feature_set}</td><td>{r.test_source}</td><td><F1Bar f1={r.f1} lo={null} hi={null} /></td>
                <td className="num">{fmt(r.f1)}</td><td className="num">{fmt(r.recall)}</td>
                <td className={`num ${r.fpr > 0.02 ? "warn-text" : ""}`}>{fmt(r.fpr)}</td>
                <td className={`num ${r.roc_auc < 0.5 ? "warn-text" : ""}`}>{fmt(r.roc_auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted">ROC-AUC below 0.5 (amber) means the ranking is worse than chance on that network.</p>
      </Panel>
    </>
  );
}
