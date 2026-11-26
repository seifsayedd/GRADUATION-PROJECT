import { api } from "./api";
import { useAsync, useRoute } from "./hooks";
import AlertDetail from "./views/AlertDetail";
import Alerts from "./views/Alerts";
import Ask from "./views/Ask";
import MatrixView from "./views/Matrix";
import Models from "./views/Models";
import Overview from "./views/Overview";
import TimelineView from "./views/Timeline";

const NAV: [string, string][] = [
  ["overview", "Overview"], ["alerts", "Alerts"], ["timeline", "Lab timeline"], ["matrix", "ATT&CK matrix"],
  ["models", "Models"], ["ask", "Ask"],
];

export default function App() {
  const route = useRoute();
  const view = route[0] ?? "overview";
  const health = useAsync(() => api.health(), []);
  const active = view === "alert" ? "alerts" : view;

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <span className="brand-mark">◆</span>
          <div>
            <div className="brand-name">SENTINEL<b>AI</b></div>
            <div className="brand-sub">analyst console · phase 6</div>
          </div>
        </div>
        <nav>
          {NAV.map(([id, label]) => (
            <a key={id} href={`#/${id}`} className={active === id ? "active" : ""}>{label}</a>
          ))}
        </nav>
        <div className="status">
          {health.error ? <span className="dot bad" /> : <span className={`dot ${health.data ? "ok" : ""}`} />}
          {health.error ? "API offline" : health.data
            ? <>API online · provider <b>{health.data.provider}</b> · {health.data.store} store</>
            : "connecting…"}
        </div>
        <p className="side-note">
          Every number here is read from the repository's report files and Phase 4 explanation objects.
          Lab data is a replay of lab capture v1, not live traffic.
        </p>
      </aside>
      <main className="main">
        {view === "overview" && <Overview />}
        {view === "alerts" && <Alerts />}
        {view === "alert" && route[1] && <AlertDetail id={route[1]} />}
        {view === "timeline" && <TimelineView />}
        {view === "matrix" && <MatrixView />}
        {view === "models" && <Models />}
        {view === "ask" && <Ask />}
        {!["overview", "alerts", "alert", "timeline", "matrix", "models", "ask"].includes(view) && (
          <p className="muted">Unknown page. <a href="#/overview">Back to the overview</a>.</p>
        )}
      </main>
    </div>
  );
}
