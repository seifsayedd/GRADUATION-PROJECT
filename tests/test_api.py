"""Phase 6 API: offline tests with FastAPI's TestClient. No API key, no network, no database."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinelai.api import data
from sentinelai.api.app import create_app
from sentinelai.api.settings import ApiSettings
from sentinelai.genai.config import Settings
from sentinelai.genai.embeddings import HashEmbedder
from sentinelai.genai.providers import FakeProvider, TemplateProvider
from sentinelai.genai.store import Chunk, MemoryStore

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "http://localhost:5173"


def explanation(alert_id, flagged=True, tactic="Benign", dataset="lab", start=1.0, model="xgboost_temporal_lab"):
    return {"schema_version": "1.0", "alert_id": alert_id,
            "flow": {"src": "10.0.0.5", "dst": "224.0.0.251", "protocol": "17 (UDP)", "start_ms": start,
                     "dst_port": 5353.0, "dataset": dataset},
            "model": {"stage": 1, "name": model, "feature_set": "lab", "git_commit": "abc"},
            "ground_truth": {"label": 0.0, "attack": "Benign", "tactic": tactic},
            "decision": {"score": 0.912, "score_kind": "probability of attack", "threshold": 0.5,
                         "operating_point": "default", "flagged": flagged},
            "tactic": {"predicted": "Initial Access", "confidence": 0.58},
            "contributions": {"space": "log-odds", "base_value": 0.05, "output": 2.34, "rest": 0.1,
                              "additivity_error": 1e-6,
                              "items": [{"feature": "FLOW_DURATION_MILLISECONDS", "value": "0", "contribution": 3.624,
                                         "direction": "towards attack"}]},
            "caveats": ["Trained on BigFlow; on an unseen network F1 fell to 0.45-0.79."],
            "summary": "Stage 1 (xgboost_temporal_lab) flagged the UDP flow 10.0.0.5:5353 -> 224.0.0.251:5353 "
                       "(score 0.912, threshold 0.500, probability of attack)."}


@pytest.fixture
def store():
    st, emb = MemoryStore(384), HashEmbedder(384)
    docs = [Chunk("d#1", "docs/a.md", "Lab", "doc", "Lab false alarms are zero-duration one-way mDNS flows."),
            Chunk("d#2", "docs/b.md", "Stage 2", "doc", "The GRU attention picks the influential flow 6.6% of the time.")]
    st.add(docs, emb.embed([c.text for c in docs]))
    st.add_explanation(explanation("lab-xgboost_temporal_lab-1", start=20.0))
    st.add_explanation(explanation("lab-xgboost_temporal_lab-0", flagged=False, start=10.0))
    st.add_explanation(explanation("lab-ae-0", flagged=False, start=10.0, model="autoencoder_temporal_lab"))
    st.add_explanation(explanation("s1-xgboost_temporal_full-7", tactic="Impact", dataset="ToN-IoT",
                                   model="xgboost_temporal_full"))
    return st


def client(store, provider=None, api=None):
    app = create_app(ROOT, settings=Settings(embedder="hash"), api=api, store=store, embedder=HashEmbedder(384),
                     provider=provider or TemplateProvider())
    return TestClient(app)


def test_health_needs_no_key(store, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = client(store).get("/api/health")
    assert r.status_code == 200 and r.json()["provider"] == "template"


def test_alert_list_filters_and_order(store):
    c = client(store)
    rows = c.get("/api/alerts").json()["rows"]
    assert len(rows) == 4
    lab = c.get("/api/alerts", params={"dataset": "lab", "model": "xgboost_temporal_lab"}).json()["rows"]
    assert [r["alert_id"] for r in lab] == ["lab-xgboost_temporal_lab-0", "lab-xgboost_temporal_lab-1"]  # by time
    assert c.get("/api/alerts", params={"flagged": True}).json()["count"] == 2
    assert c.get("/api/alerts", params={"stage": 9}).status_code == 422


def test_alert_detail_and_unknown(store):
    c = client(store)
    assert c.get("/api/alerts/lab-ae-0").json()["alert_id"] == "lab-ae-0"
    assert c.get("/api/alerts/nope").status_code == 404
    assert c.get("/api/alerts/" + "x" * 500).status_code == 404
    assert c.post("/api/alerts/nope/report").status_code == 404


def test_report_is_grounded_and_stored(store):
    r = client(store).post("/api/alerts/lab-xgboost_temporal_lab-1/report")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "template" and body["report_id"] and len(store.reports) == 1
    assert "alert:lab-xgboost_temporal_lab-1" in body["sources"]


def test_ungrounded_report_returns_422_and_is_not_stored(store):
    bad = {"title": "t", "summary": "Score was 0.999 from 8.8.8.8.", "evidence": [], "caveats": [],
           "attack_mapping": {"tactic": "Impact", "note": "", "citations": ["made-up#9"]}, "recommended_checks": []}
    r = client(store, provider=FakeProvider([bad])).post("/api/alerts/lab-xgboost_temporal_lab-1/report")
    assert r.status_code == 422
    d = r.json()["detail"]
    assert d["rejected"] and "0.999" in d["unsupported_numbers"] and "8.8.8.8" in d["unsupported_ips"]
    assert "made-up#9" in d["unknown_citations"] and store.reports == []


def test_ask_validates_length(store):
    c = client(store, api=ApiSettings(max_question_chars=40))
    assert c.post("/api/ask", json={"question": "Why lab false alarms?"}).status_code == 200
    assert c.post("/api/ask", json={"question": "x" * 41}).status_code == 422
    assert c.post("/api/ask", json={"question": ""}).status_code == 422
    assert c.post("/api/ask", json={}).status_code == 422


def test_missing_api_key_gives_503_not_a_crash(store, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = create_app(ROOT, settings=Settings(provider="anthropic", embedder="hash"), store=store,
                     embedder=HashEmbedder(384))           # provider built lazily from settings, on first use
    r = TestClient(app).post("/api/ask", json={"question": "Why lab false alarms?"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_internal_errors_do_not_leak_details(store):
    def boom(**_):
        raise RuntimeError("secret path C:/x/y")
    c = TestClient(create_app(ROOT, settings=Settings(embedder="hash"), store=store, embedder=HashEmbedder(384),
                              provider=FakeProvider(boom)), raise_server_exceptions=False)
    r = c.post("/api/ask", json={"question": "Why lab false alarms?"})
    assert r.status_code == 500 and "secret" not in r.text


def test_cors_allows_only_the_dashboard_origin(store):
    c = client(store)
    ok = c.get("/api/health", headers={"Origin": ORIGIN})
    assert ok.headers.get("access-control-allow-origin") == ORIGIN
    bad = c.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_api_settings_from_env():
    a = ApiSettings.from_env({})
    assert a.host == "127.0.0.1" and ORIGIN in a.cors_origins
    a = ApiSettings.from_env({"SENTINEL_API_CORS_ORIGINS": "http://a:1, http://b:2", "SENTINEL_API_PORT": "9000"})
    assert a.cors_origins == ("http://a:1", "http://b:2") and a.port == 9000


def test_sources_endpoint(store):
    c = client(store)
    assert c.get("/api/sources/d%231").json()["origin"] == "docs/a.md"
    assert c.get("/api/sources/alert:lab-ae-0").json()["text"].startswith("Stage 1")
    assert c.get("/api/sources/zzz").status_code == 404


def test_timeline_groups_lab_rows_in_time_order(store):
    t = client(store).get("/api/timeline").json()
    assert [r["row"] for r in t["rows"]] == [0, 1]
    assert set(t["rows"][0]["verdicts"]) == {"xgboost_temporal_lab", "autoencoder_temporal_lab"}
    assert "not live traffic" in t["note"]


def test_attack_matrix_counts(store):
    m = client(store).get("/api/attack-matrix").json()
    assert m["tactics"][0] == "Benign" and "Impact" in m["tactics"]
    cell = next(x for x in m["cells"] if x["tactic"] == "Benign" and x["model"] == "xgboost_temporal_lab")
    assert cell == {"tactic": "Benign", "model": "xgboost_temporal_lab", "objects": 2, "flagged": 1}


def test_lab_index():
    assert data.lab_index("lab-ae-12") == ("autoencoder_temporal_lab", 12)
    assert data.lab_index("lab-xgboost_temporal_lab-3") == ("xgboost_temporal_lab", 3)
    assert data.lab_index("s1-xgboost_temporal_full-3") is None and data.lab_index("lab-ae-x") is None


def test_tactics_follow_the_taxonomy_doc():
    t = data.tactics(ROOT)
    assert t[0] == "Benign" and "Command and Control" in t and "Exfiltration" in t


# ------------------------------------------------------------------ numbers match the report files
def _json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def test_summary_numbers_match_the_report_files(store):
    s = client(store).get("/api/summary").json()
    cards = {c["id"]: c for c in s["cards"]}
    op = _json("reports/phase3/models.json")["models"]["stage1_xgboost_temporal_full"]["operating_points"]["val_fpr_0.01"]
    assert cards["seen"]["value"] == round(op["test"]["f1"], 3)
    assert cards["seen"]["range"] == [round(op["test_ci"]["f1"]["lo"], 3), round(op["test_ci"]["f1"]["hi"], 3)]
    runs = _json("reports/phase3/loso.json")["loso"]["runs"]
    for fs in ("full", "lab"):
        f1 = [r["val_fpr_0.01"]["f1"] for r in runs if r["feature_set"] == fs]
        assert cards[f"unseen_{fs}"]["range"] == [round(min(f1), 3), round(max(f1), 3)]
    lab = _json("reports/phase2/lab_scores_summary.json")["models"]
    got = {r["model"]: r for r in s["lab_flags"]["rows"]}
    for k, v in lab.items():
        assert got[k]["false_alarms"] == v["fp"] and got[k]["attack_caught"] == (v["tp"] > 0)
    assert all(c["source"].startswith("reports/") for c in s["cards"])


def test_evaluation_rows_match_models_json(store):
    e = client(store).get("/api/evaluation").json()
    models = _json("reports/phase3/models.json")["models"]
    assert {r["model"] for r in e["models"]["rows"]} == set(models)
    for r in e["models"]["rows"]:
        t = models[r["model"]]["operating_points"]["val_fpr_0.01"]["test"]
        assert r["fpr"] == round(t["fpr"], 3) and r["n"] == t["n"]


def test_missing_report_file_gives_503(tmp_path, store):
    app = create_app(tmp_path, settings=Settings(embedder="hash"), store=store, embedder=HashEmbedder(384))
    r = TestClient(app).get("/api/summary")
    assert r.status_code == 503 and "missing" in r.json()["detail"]
