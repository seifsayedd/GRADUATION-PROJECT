"""Phase 2 training runs. Each run saves its model to models/ and its metrics to reports/phase2/.

Every number written here comes from the held-apart validation, test or
holdout splits produced by ``data.prepare``. Nothing is tuned on test.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.mapping import lab_to_bigflow
from ..data.prepare import ROOT, git_commit
from ..data.taxonomy import TAXONOMY, lab_tactic
from .autoencoder import Autoencoder
from .common import binary_metrics, feature_set, load_model, multiclass_metrics, save_model, write_json
from .flow_classifier import FlowClassifier
from .sequence import SequenceModel, block_split

REPORTS = ROOT / "reports" / "phase2"


def _flows(cfg: dict, split: str, interim: Path | None = None) -> dict[str, pd.DataFrame]:
    base = Path(interim or ROOT / cfg["data"]["interim_dir"]) / "flows"
    return {p: pd.read_parquet(base / f"{split}_{p}.parquet") for p in ("train", "val", "test", "holdout")}


def holdout_report(pred: pd.DataFrame, hold: pd.DataFrame, score_col="p_attack", flag_col="is_attack") -> dict:
    """Unseen-attack test: detection rate per held-out class, and whether its tactic was guessed right."""
    out = {}
    for cls, g in hold.groupby("Attack"):
        p = pred.loc[g.index]
        r = {"n": len(g), "detected_rate": float(p[flag_col].mean()), "expected_tactic": TAXONOMY[cls].tactic}
        if "tactic_pred" in p:
            r["tactic_correct_rate"] = float((p.tactic_pred == TAXONOMY[cls].tactic).mean())
            r["tactic_pred_counts"] = p.tactic_pred.value_counts().to_dict()
        out[cls] = r
    return out


def run_stage1(cfg: dict, split: str, fset: str, algo: str, interim: Path | None = None,
               models_root: Path | None = None, reports: Path = REPORTS) -> dict:
    d = _flows(cfg, split, interim)
    m = FlowClassifier(algo, feature_set(fset), cfg["stage1"][algo], seed=cfg["seed"]).fit(d["train"], d["val"])
    hold_pred = m.predict(d["holdout"])
    res = {"stage": 1, "algo": algo, "split": split, "feature_set": fset, "fit_seconds": m.fit_seconds_,
           "n_features": len(m.feature_names_), "search": m.history_,
           "val": m.evaluate(d["val"]), "test": m.evaluate(d["test"]),
           "holdout": holdout_report(hold_pred, d["holdout"]),
           "train_rows": len(d["train"]), "git_commit": git_commit()}
    name = f"{algo}_{split}_{fset}"
    res["model"] = save_model(m, "stage1", name, {k: res[k] for k in ("algo", "split", "feature_set", "n_features")})
    write_json(res, reports / f"stage1_{name}.json")
    return res


def run_stage3(cfg: dict, split: str, fset: str, interim: Path | None = None, reports: Path = REPORTS) -> dict:
    d = _flows(cfg, split, interim)
    c = cfg["stage3"]
    tr, va = d["train"][d["train"].Label == 0], d["val"][d["val"].Label == 0]
    m = Autoencoder(feature_set(fset), c["hidden"], c["epochs"], c["batch_size"], c["lr"],
                    c["threshold_percentile"], seed=cfg["seed"]).fit(tr, va)
    test_pred = m.predict(d["test"])
    per_source = {}
    for src, g in d["test"].groupby("source"):
        b = g.Label == 0
        per_source[src] = {"benign_n": int(b.sum()),
                           "benign_fpr": float(test_pred.loc[g.index[b], "is_anomaly"].mean()) if b.any() else None,
                           "attack_n": int((~b).sum()),
                           "attack_detected": float(test_pred.loc[g.index[~b], "is_anomaly"].mean()) if (~b).any() else None}
    hold_pred = m.predict(d["holdout"])
    res = {"stage": 3, "split": split, "feature_set": fset, "fit_seconds": m.fit_seconds_,
           "threshold": m.threshold_, "history": m.history_, "train_benign_rows": len(tr),
           "val_benign_fpr": float(m.predict(va).is_anomaly.mean()),
           "test": m.evaluate(d["test"]), "test_per_source": per_source,
           "test_per_tactic_detected": test_pred.join(d["test"].tactic).groupby("tactic").is_anomaly.mean().to_dict(),
           "holdout": holdout_report(hold_pred, d["holdout"], "anomaly_score", "is_anomaly"),
           "git_commit": git_commit()}
    name = f"autoencoder_{split}_{fset}"
    res["model"] = save_model(m, "stage3", name, {k: res[k] for k in ("split", "feature_set", "threshold")})
    write_json(res, reports / f"stage3_{name}.json")
    return res


def load_sequences(cfg: dict, interim: Path | None = None) -> pd.DataFrame:
    base = Path(interim or ROOT / cfg["data"]["interim_dir"]) / "sequences"
    frames = [pd.read_parquet(base / f"{s['name']}.parquet") for s in cfg["stage2"]["slices"]]
    df = pd.concat(frames, ignore_index=True)
    return df


def run_stage2(cfg: dict, fset: str, interim: Path | None = None, reports: Path = REPORTS,
               df: pd.DataFrame | None = None) -> dict:
    c = cfg["stage2"]
    df = load_sequences(cfg, interim) if df is None else df
    parts = block_split(df, cfg["data"]["val_size"], cfg["data"]["test_size"], c.get("block_minutes", 5), cfg["seed"])
    missing = {k: sorted(set(parts["train"].tactic) - set(v.tactic)) for k, v in parts.items() if k != "train"}
    unseen = sorted(set(parts["test"].tactic) - set(parts["train"].tactic))
    if unseen:
        raise ValueError(f"test has tactics absent from train: {unseen}; adjust block size or slices")
    m = SequenceModel(feature_set(fset), c["window"], c["hidden"], c["cell"], c["epochs"],
                      c["batch_size"], c["lr"], seed=cfg["seed"]).fit(parts["train"], parts["val"])
    test = parts["test"]
    pred = m.predict(test)
    per_slice = {}
    for sl, g in test.groupby("slice"):
        p = pred.loc[g.index]
        per_slice[sl] = {"binary": binary_metrics(g.Label, p.is_attack, p.p_attack),
                         "tactic": multiclass_metrics(g.tactic, p.tactic_pred)["per_class"]}
    res = {"stage": 2, "cell": c["cell"], "window": c["window"], "feature_set": fset,
           "fit_seconds": m.fit_seconds_, "history": m.history_,
           "rows": {k: len(v) for k, v in parts.items()}, "split": f"{c.get('block_minutes', 5)}-min time blocks per day",
           "train_tactics_missing_from": missing,
           "class_counts": {k: v.tactic.value_counts().to_dict() for k, v in parts.items()},
           "val": m.evaluate(parts["val"]), "test": m.evaluate(test), "test_per_slice": per_slice,
           "git_commit": git_commit()}
    name = f"{c['cell']}_w{c['window']}_{fset}"
    res["model"] = save_model(m, "stage2", name, {k: res[k] for k in ("cell", "window", "feature_set")})
    write_json(res, reports / f"stage2_{name}.json")
    return res


def score_lab(cfg: dict, split: str = "temporal", reports: Path = REPORTS, models_root: Path | None = None) -> pd.DataFrame:
    """Score the REAL lab conversations with every lab-compatible model.

    Lab v1 has 54 conversations and 1 attack, so this checks that the models
    run on real lab data; it is not evidence of accuracy.
    """
    lab = lab_to_bigflow(ROOT / cfg["data"]["lab_csv"])
    lab["tactic"] = lab_tactic(lab.Attack).values
    out = lab[["session_id", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "PROTOCOL", "IN_PKTS", "OUT_PKTS",
               "Label", "tactic"]].copy()
    summary = {"rows": len(lab), "attacks": int(lab.Label.sum()), "models": {}}
    for algo in cfg["stage1"]["models"]:
        m, _ = load_model("stage1", f"{algo}_{split}_lab", models_root)
        p = m.predict(lab)
        out[f"s1_{algo}_p_attack"], out[f"s1_{algo}_tactic"] = p.p_attack.round(4), p.tactic_pred
        summary["models"][f"stage1_{algo}"] = binary_metrics(lab.Label, p.is_attack, p.p_attack)
    c2 = cfg["stage2"]
    m, _ = load_model("stage2", f"{c2['cell']}_w{c2['window']}_lab", models_root)
    p = m.predict(lab)
    out.loc[p.index, "s2_p_attack"], out.loc[p.index, "s2_tactic"] = p.p_attack.round(4), p.tactic_pred
    summary["models"]["stage2"] = binary_metrics(lab.Label, p.is_attack.reindex(lab.index), p.p_attack.reindex(lab.index))
    m, _ = load_model("stage3", f"autoencoder_{split}_lab", models_root)
    p = m.predict(lab)
    out["s3_anomaly_score"], out["s3_is_anomaly"] = p.anomaly_score.round(4), p.is_anomaly
    summary["models"]["stage3"] = binary_metrics(lab.Label, p.is_anomaly, p.anomaly_score)
    reports.mkdir(parents=True, exist_ok=True)
    out.to_csv(reports / "lab_scores.csv", index=False)
    write_json(summary, reports / "lab_scores_summary.json")
    return out
