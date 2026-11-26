"""Phase 3 evaluation code on small hand-made inputs."""
import json

import numpy as np
import pandas as pd
import pytest

from sentinelai.eval import metrics as M
from sentinelai.eval.generalise import flag_per_source, loso_splits, per_source_thresholds
from sentinelai.eval.report import write_report


# ------------------------------------------------------------- confusion etc.
def test_confusion_counts_and_row_normalisation():
    c = M.confusion(["a", "a", "b", "b", "b"], ["a", "b", "b", "b", "a"], labels=["a", "b"])
    assert c["counts"] == [[1, 1], [1, 2]]
    assert c["row_normalised"] == [[0.5, 0.5], [0.3333, 0.6667]]


def test_binary_rates_match_hand_count():
    r = M.binary([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0])
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 3)
    assert r["recall"] == pytest.approx(2 / 3) and r["fpr"] == pytest.approx(1 / 4)
    assert r["precision"] == pytest.approx(2 / 3) and r["f1"] == pytest.approx(2 / 3)


def test_per_class_detection_and_per_source():
    df = pd.DataFrame({"Attack": ["Benign", "Benign", "DoS", "DoS", "Bot"], "Label": [0, 0, 1, 1, 1],
                       "source": ["X", "Y", "X", "X", "Y"]})
    flag = [1, 0, 1, 0, 1]
    pc = M.per_class_detection(df, flag)
    assert pc["DoS"]["flagged"] == 0.5 and pc["Bot"]["flagged"] == 1.0 and pc["Benign"]["flagged"] == 0.5
    ps = M.per_source(df, flag)
    assert ps["X"] == {"benign_n": 1, "benign_fpr": 1.0, "attack_n": 2, "attack_detected": 0.5}
    assert ps["Y"]["benign_fpr"] == 0.0


# ------------------------------------------------------------------ threshold
def test_threshold_at_fpr_uses_benign_val_scores_only_and_hits_target():
    rng = np.random.default_rng(0)
    benign_val = rng.normal(size=10_000)
    thr = M.threshold_at_fpr(benign_val, 0.01)
    assert (benign_val > thr).mean() == pytest.approx(0.01, abs=0.002)
    fresh = rng.normal(size=100_000)                       # same distribution, unseen
    assert (fresh > thr).mean() == pytest.approx(0.01, abs=0.003)


@pytest.mark.parametrize("bad", [0, 1, -0.1])
def test_threshold_rejects_bad_targets(bad):
    with pytest.raises(ValueError):
        M.threshold_at_fpr([0.1, 0.2], bad)


def test_threshold_rejects_empty():
    with pytest.raises(ValueError):
        M.threshold_at_fpr([], 0.01)


# ------------------------------------------------------------------ bootstrap
def _blocky(n_blocks=60, per_block=50, seed=0):
    """Each block is all-detected or all-missed: rows inside a block are near-copies."""
    rng = np.random.default_rng(seed)
    hit = rng.random(n_blocks) < 0.7
    y = np.ones(n_blocks * per_block, int)
    flag = np.repeat(hit, per_block).astype(int)
    groups = np.repeat(np.arange(n_blocks), per_block)
    return y, flag, groups


def test_block_bootstrap_ci_contains_point_estimate():
    y, f, g = _blocky()
    ci = M.block_bootstrap_ci(y, f, g, stats=("recall",), n_boot=500)
    r = ci["recall"]
    assert r["lo"] <= r["point"] <= r["hi"] and r["point"] == pytest.approx(f.mean())
    assert ci["n_groups"] == 60


def test_block_bootstrap_is_wider_than_row_bootstrap_on_correlated_rows():
    y, f, g = _blocky()
    blk = M.block_bootstrap_ci(y, f, g, stats=("recall",), n_boot=500)["recall"]
    row = M.block_bootstrap_ci(y, f, np.arange(len(y)), stats=("recall",), n_boot=500)["recall"]
    assert blk["hi"] - blk["lo"] > 3 * (row["hi"] - row["lo"])


def test_block_ids_split_by_source_and_time():
    df = pd.DataFrame({"source": ["A", "A", "A", "B"],
                       "FLOW_START_MILLISECONDS": [0, 299_999, 300_000, 0]})
    g = M.block_ids(df, block_minutes=5)
    assert g[0] == g[1] and g[1] != g[2] and g[0] != g[3]


# ------------------------------------------------------------ curves/calibration
def test_calibration_perfect_and_curves_shape():
    p = np.array([0.1] * 10 + [0.9] * 10)
    y = np.array([1] + [0] * 9 + [1] * 9 + [0])
    c = M.calibration(y, p)
    assert c["ece"] == pytest.approx(0.0, abs=1e-9)
    cv = M.curves(y, p)
    assert set(cv) == {"roc", "pr"} and len(cv["roc"]["fpr"]) == len(cv["roc"]["tpr"])
    assert M.curves([1, 1], [0.2, 0.3]) == {}


# ------------------------------------------------------------------ option A/B
def _flows():
    def part(n, seed):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"source": rng.choice(["S1", "S2", "S3"], n), "Label": rng.integers(0, 2, n),
                             "Attack": "x", "FLOW_START_MILLISECONDS": np.arange(n)})
    return {"train": part(300, 1), "val": part(100, 2), "test": part(100, 3)}


def test_loso_never_trains_on_test_source():
    f = _flows()
    for src in ("S1", "S2", "S3"):
        s = loso_splits(f, src)
        assert src not in set(s["train"].source) and src not in set(s["val"].source)
        assert set(s["test"].source) == {src}
        assert len(s["test"]) == sum((f[p].source == src).sum() for p in ("train", "val", "test"))


def test_per_source_thresholds_from_benign_val_only():
    val = pd.DataFrame({"source": ["A"] * 4 + ["B"] * 4, "Label": [0, 0, 0, 1, 0, 0, 0, 1]})
    score = np.array([1, 2, 3, 100, 10, 20, 30, 1000], float)
    thr = per_source_thresholds(val, score, 100)
    assert thr == {"A": 3.0, "B": 30.0}                  # attack scores (100, 1000) ignored
    flags = flag_per_source(val, score, thr)
    assert flags.tolist() == [False, False, False, True, False, False, False, True]
    with pytest.raises(ValueError):
        flag_per_source(pd.DataFrame({"source": ["C"]}), np.array([1.0]), thr)


# ---------------------------------------------------------------------- report
def test_report_marks_missing_runs_and_prints_only_recorded_numbers(tmp_path):
    p = write_report(tmp_path)
    text = p.read_text(encoding="utf-8")
    assert text.count("*Not run yet") == 5
    (tmp_path / "loso.json").write_text(json.dumps({"loso": {"target_fpr": 0.01, "runs": [{
        "feature_set": "full", "test_source": "S1", "test_rows": 10, "roc_auc": 0.5,
        "default_ci": {"f1": {"point": 0.123, "lo": 0.1, "hi": 0.2}, "fpr": {"point": 0.5, "lo": 0.4, "hi": 0.6}},
        "val_fpr_0.01": {"f1": 0.11, "fpr": 0.02}, "in_source_baseline": {"f1": 0.9, "fpr": 0.01},
        "tactics_unseen_in_training": ["Other"], "tactic_correct_on_known_attacks": None}]}}), encoding="utf-8")
    text = write_report(tmp_path).read_text(encoding="utf-8")
    assert text.count("*Not run yet") == 4 and "0.123 [0.100, 0.200]" in text and "| Other |" in text
