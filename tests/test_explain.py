"""Phase 4 explanation code on small fixtures (CPU, seconds)."""
import json

import numpy as np
import pandas as pd
import pytest

from sentinelai.explain import build, checks
from sentinelai.explain.attention import attention_vs_occlusion, occlusion, windows_for
from sentinelai.explain.features import aggregate, group_columns, source_feature, value_text
from sentinelai.explain.schema import SCHEMA, SchemaError, finalize, numbers_in, validate
from sentinelai.explain.tree_shap import AdditivityError, check_additivity, explain_flow_classifier
from sentinelai.models.flow_classifier import FlowClassifier

INFO = {"stage": 1, "name": "test", "feature_set": "full", "git_commit": "abc"}


def split3(df):
    n = len(df)
    return df.iloc[: n * 6 // 10], df.iloc[n * 6 // 10: n * 8 // 10], df.iloc[n * 8 // 10:]


@pytest.fixture(scope="module")
def xgb_model():
    from conftest import make_labeled_fixture
    df = make_labeled_fixture(1200, 0)
    tr, va, te = split3(df)
    m = FlowClassifier("xgboost", params={"n_estimators": 40, "early_stopping_rounds": 5, "max_depth": 4}).fit(tr, va)
    return m, te


# ------------------------------------------------------------------ features
def test_columns_map_back_to_original_features():
    cols = ["IN_BYTES", "TCP_FLAGS__SYN", "TCP_FLAGS__ACK", "L4_DST_PORT__bucket1", "PROTOCOL__17"]
    assert [source_feature(c) for c in cols] == ["IN_BYTES", "TCP_FLAGS", "TCP_FLAGS", "L4_DST_PORT", "PROTOCOL"]
    assert group_columns(cols) == {"IN_BYTES": [0], "TCP_FLAGS": [1, 2], "L4_DST_PORT": [3], "PROTOCOL": [4]}
    c = np.random.default_rng(0).normal(size=(4, 5))
    agg = aggregate(c, cols)
    assert np.allclose(agg.sum(1), c.sum(1))                          # nothing lost or double-counted
    assert np.allclose(agg.TCP_FLAGS, c[:, 1] + c[:, 2])
    with pytest.raises(ValueError):
        aggregate(c[:, :3], cols)


def test_value_text_is_readable():
    assert value_text("TCP_FLAGS", 0x12) == "SYN|ACK (0x12)"
    assert value_text("L4_DST_PORT", 443) == "443 (well-known)"
    assert value_text("PROTOCOL", 17) == "17 (UDP)"
    assert value_text("IN_BYTES", 1500.0) == "1,500"


# ------------------------------------------------------------------ SHAP
def test_xgboost_shap_adds_up_to_model_output(xgb_model):
    m, te = xgb_model
    r = explain_flow_classifier(m, te)
    p = 1 / (1 + np.exp(-(r["binary"].values.sum(1) + r["binary"].base)))
    assert np.allclose(p, m.predict(te).p_attack, atol=1e-4)          # same score the classifier reports
    assert r["binary"].max_additivity_error < 1e-3 and r["tactic"].max_additivity_error < 1e-3


def test_random_forest_shap_adds_up_to_probability(labeled_df):
    tr, va, te = split3(labeled_df)
    m = FlowClassifier("random_forest", params={"n_estimators": 10, "max_depth": 5}).fit(tr, va)
    r = explain_flow_classifier(m, te.iloc[:60])
    assert r["binary"].space == "probability"
    assert np.allclose(r["binary"].values.sum(1) + r["binary"].base, m.predict(te.iloc[:60]).p_attack, atol=1e-6)


def test_additivity_mismatch_is_refused():
    with pytest.raises(AdditivityError):
        check_additivity(np.ones((2, 3)), np.zeros(2), np.array([3.0, 2.0]), tol=1e-6)


def test_deletion_test_finds_the_planted_feature(xgb_model, labeled_df):
    m, te = xgb_model
    flagged = te[m.predict(te).is_attack == 1]
    objs, fr = build.stage1(m, flagged, INFO, [])
    imp = checks.global_importance(fr["binary"])["overall"]
    assert next(iter(imp)) == "IN_BYTES"                              # IN_BYTES x50 is the planted signal
    ref = checks.benign_reference(m.transform(labeled_df[labeled_df.Label == 0]))
    d = checks.deletion_test(lambda X: m.binary_.predict_proba(X)[:, 1], fr["X"], fr["binary"], ref, ks=(1,), n_random=3)
    assert d["k=1"]["mean_drop_top"] > d["k=1"]["mean_drop_random"]
    assert d["k=1"]["share_top_beats_random"] > 0.8


# ------------------------------------------------------------------ schema
def test_stage1_objects_validate_roundtrip_and_summary_uses_only_object_numbers(xgb_model):
    m, te = xgb_model
    objs, _ = build.stage1(m, te.iloc[:20], INFO, ["Caveat with 0.43-0.81."])
    assert any(o["decision"]["flagged"] for o in objs) and any(not o["decision"]["flagged"] for o in objs)
    for o in objs:
        back = json.loads(json.dumps(o))
        validate(back)
        assert back == o
        body = json.dumps({k: v for k, v in o.items() if k != "summary"})
        for num in numbers_in(o["summary"]):
            n = num.lstrip("+-").rstrip(".")
            assert n in body or n.replace(",", "") in body or f"{float(n.replace(',', '')):.3f}" in o["summary"]
        if o["decision"]["flagged"]:
            assert "tactic" in o and "tactic_contributions" in o


def test_template_numbers_are_rounded_copies_of_object_values(xgb_model):
    m, te = xgb_model
    o = build.stage1(m, te.iloc[:1], INFO, [])[0][0]
    assert f"score {o['decision']['score']:.3f}" in o["summary"]
    top = o["contributions"]["items"][0]
    assert f"({top['contribution']:+.3f})" in o["summary"] or top["direction"] == "towards benign"


@pytest.mark.parametrize("mutate,msg", [
    (lambda o: o.pop("decision"), "missing"),
    (lambda o: o["decision"].update(score=float("nan")), "finite"),
    (lambda o: o["contributions"].update(space="dollars"), "not in"),
    (lambda o: o.update(extra=1), "unexpected"),
])
def test_validator_rejects_bad_objects(xgb_model, mutate, msg):
    m, te = xgb_model
    o = build.stage1(m, te.iloc[:1], INFO, [])[0][0]
    mutate(o)
    with pytest.raises(SchemaError, match=msg):
        validate(o)


def test_explanation_needs_contributions_or_attention():
    with pytest.raises(SchemaError):
        finalize({"alert_id": "x", "flow": {"src": "a", "dst": "b", "protocol": "p", "start_ms": 0}, "model": INFO,
                  "decision": {"score": 1, "score_kind": "k", "threshold": 0.5, "operating_point": "d", "flagged": True},
                  "caveats": []})
    assert SCHEMA["properties"]["schema_version"]["enum"] == ["1.0"]


# ------------------------------------------------------------------ Stage 2 / 3
def test_attention_occlusion_shapes_and_stage2_objects():
    from sentinelai.models.sequence import SequenceModel, block_split
    from test_models import _seq_df
    df = _seq_df(800)
    p = block_split(df, block_minutes=5)
    m = SequenceModel(window=8, hidden=8, epochs=1, batch_size=64, device="cpu").fit(p["train"], p["val"])
    x, mask, ordered, pos, _ = windows_for(m, p["test"], np.arange(30))
    base, att, delta = occlusion(m, x, mask)
    assert att.shape == mask.shape == delta.shape and np.allclose(att.sum(1), 1, atol=1e-5)
    assert np.isnan(delta[:, -1]).all()                               # the labelled flow is never removed
    assert np.isnan(delta[~mask]).all() and (delta[mask[:, :-1].nonzero()[0], mask[:, :-1].nonzero()[1]] >= 0).all()
    s = attention_vs_occlusion(att, delta, mask)
    assert set(s) >= {"windows", "top_attended_is_most_influential", "expected_random_rate"}
    objs, _ = build.stage2(m, p["test"], np.arange(10), {**INFO, "stage": 2}, [], 0.5, "test")
    assert len(objs) == 10 and all("attention" in o for o in objs)


def test_stage3_error_contributions_sum_to_score(labeled_df):
    from sentinelai.models.autoencoder import Autoencoder
    ben = labeled_df[labeled_df.Label == 0]
    ae = Autoencoder(hidden=[8, 4], epochs=2, batch_size=128, device="cpu").fit(ben.iloc[:400], ben.iloc[400:500])
    objs, agg = build.stage3(ae, labeled_df.iloc[:15], {**INFO, "stage": 3}, [])
    assert np.allclose(agg.sum(1).to_numpy(), ae.score(labeled_df.iloc[:15]), rtol=1e-5)
    assert all(o["contributions"]["additivity_error"] < 1e-5 for o in objs)
    assert all(i["direction"] == "abnormal" for o in objs for i in o["contributions"]["items"])
