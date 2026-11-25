"""Model tests on small fixtures (CPU, seconds). They check behaviour, not accuracy."""
import pickle

import numpy as np
import pandas as pd
import pytest

from sentinelai.data.mapping import available_features
from sentinelai.data.taxonomy import BENIGN
from sentinelai.models.autoencoder import Autoencoder, BenignOnlyError
from sentinelai.models.common import balanced_weights, load_model, save_model
from sentinelai.models.flow_classifier import FlowClassifier
from sentinelai.models.sequence import SequenceModel, WindowIndex, temporal_split


def split3(df):
    n = len(df)
    return df.iloc[: n * 6 // 10], df.iloc[n * 6 // 10: n * 8 // 10], df.iloc[n * 8 // 10:]


# ---------------------------------------------------------------- common
def test_balanced_weights_equalise_classes():
    y = np.array([0] * 90 + [1] * 10)
    w = balanced_weights(y)
    assert np.isclose(w[y == 0].sum(), w[y == 1].sum())


def test_registry_roundtrip_and_size_flag(tmp_path):
    meta = save_model({"a": 1}, "stageX", "tiny", {"k": "v"}, root=tmp_path)
    assert meta["commit_allowed"] and meta["size_mb"] < 1
    obj, m2 = load_model("stageX", "tiny", root=tmp_path)
    assert obj == {"a": 1} and m2["k"] == "v"


# --------------------------------------------------------------- stage 1
@pytest.mark.parametrize("algo,params", [
    ("random_forest", {"n_estimators": 20, "max_depth": 6}),
    ("xgboost", {"n_estimators": 30, "early_stopping_rounds": 5, "grid": {"max_depth": [3, 4]}}),
])
def test_flow_classifier_learns_planted_signal(labeled_df, algo, params):
    tr, va, te = split3(labeled_df)
    m = FlowClassifier(algo, params=params).fit(tr, va)
    p = m.predict(te)
    assert list(p.columns) == ["p_attack", "is_attack", "tactic_pred", "tactic_conf"]
    assert (p.loc[p.is_attack == 0, "tactic_pred"] == BENIGN).all()
    ev = m.evaluate(te)
    assert ev["binary"]["f1"] > 0.9                         # IN_BYTES x50 is easy to learn
    assert "tactic_given_attack" in ev
    if algo == "xgboost":
        assert len([h for h in m.history_ if h["task"] == "binary"]) == 2   # grid was searched


def test_flow_classifier_lab_feature_set(labeled_df, lab_full_path):
    from sentinelai.data.mapping import lab_to_bigflow
    tr, va, _ = split3(labeled_df)
    m = FlowClassifier("random_forest", available_features(), {"n_estimators": 10, "max_depth": 4}).fit(tr, va)
    p = m.predict(lab_to_bigflow(lab_full_path))
    assert len(p) == 54


def test_flow_classifier_pickles(labeled_df):
    tr, va, te = split3(labeled_df)
    m = FlowClassifier("random_forest", params={"n_estimators": 5, "max_depth": 3}).fit(tr, va)
    m2 = pickle.loads(pickle.dumps(m))
    pd.testing.assert_frame_equal(m.predict(te), m2.predict(te))


# --------------------------------------------------------------- stage 2
def _seq_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    from conftest import make_labeled_fixture
    df = make_labeled_fixture(n, seed)
    df["IPV4_SRC_ADDR"] = [f"h{i % 12}" for i in range(n)]
    df["FLOW_START_MILLISECONDS"] = np.sort(rng.integers(0, 10**7, n))
    df["slice"] = np.where(np.arange(n) < n // 2, "a", "b")
    return df


def test_windows_stay_inside_one_host():
    df = pd.DataFrame({"IPV4_SRC_ADDR": ["a", "b", "a", "a", "b"], "FLOW_START_MILLISECONDS": [5, 1, 1, 9, 7]})
    idx = WindowIndex(df, window=3)
    hosts = df.loc[idx.order, "IPV4_SRC_ADDR"].to_numpy()
    for i in range(len(idx)):
        assert set(hosts[idx.start[i]: i + 1]) == {hosts[i]}
    assert idx.log_gap[0] == 0                                 # first flow of a host has no gap


def test_temporal_split_per_slice_is_ordered():
    df = _seq_df()
    p = temporal_split(df)
    for sl in ("a", "b"):
        tr, va, te = (p[k][p[k].slice == sl].FLOW_START_MILLISECONDS for k in ("train", "val", "test"))
        assert tr.max() < va.min() and va.max() < te.min()


def test_sequence_model_fit_predict_attention_pickle():
    df = _seq_df()
    p = temporal_split(df)
    m = SequenceModel(window=8, hidden=8, epochs=1, batch_size=64, device="cpu").fit(p["train"], p["val"])
    pred, att = m.predict(p["test"], return_attention=True)
    assert set(pred.index) == set(p["test"].index)
    assert att.shape == (len(p["test"]), 8)
    assert np.allclose(att.sum(1), 1, atol=1e-5)               # attention is a distribution per window
    m2 = pickle.loads(pickle.dumps(m))
    m2.device = "cpu"
    pd.testing.assert_frame_equal(pred, m2.predict(p["test"]))


def test_sequence_cell_must_be_known():
    from sentinelai.models.sequence import build_net
    with pytest.raises(ValueError):
        build_net(3, 2, cell="transformer")


# --------------------------------------------------------------- stage 3
def test_autoencoder_refuses_attack_rows(labeled_df):
    with pytest.raises(BenignOnlyError):
        Autoencoder(epochs=1, device="cpu").fit(labeled_df, labeled_df)


def test_autoencoder_benign_only_threshold_and_scores(labeled_df):
    ben = labeled_df[labeled_df.Label == 0]
    tr, va = ben.iloc[: len(ben) * 3 // 4], ben.iloc[len(ben) * 3 // 4:]
    m = Autoencoder(hidden=[16, 4], epochs=5, batch_size=128, threshold_percentile=95, device="cpu").fit(tr, va)
    assert 0.02 <= m.predict(va).is_anomaly.mean() <= 0.08     # ~5% FPR on the data the threshold came from
    att = labeled_df[labeled_df.Label == 1]
    assert np.median(m.score(att)) > np.median(m.score(va))      # planted attacks reconstruct worse
    fe = m.feature_errors(att.iloc[:3])
    assert fe.shape[0] == 3 and (fe.to_numpy() >= 0).all()
    m2 = pickle.loads(pickle.dumps(m))
    m2.device = "cpu"
    assert np.allclose(m.score(va), m2.score(va))
