"""Per-model evaluation, the combined Stage 1 + Stage 3 detector, and the lab breakdown.

Every threshold and the combination rule are picked on the **validation**
part of the cached prediction tables; test and holdout are only scored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.mapping import lab_to_bigflow
from ..data.prepare import ROOT
from ..data.taxonomy import TACTICS, lab_tactic
from ..models.common import load_model
from . import metrics as M
from .predict import tactic_at

P2 = ROOT / "reports" / "phase2"


def _phase2(name: str) -> dict:
    p = P2 / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _kind(key: str) -> str:
    return key.split("_", 1)[0]


def score_col(key: str) -> str:
    return "anomaly_score" if _kind(key) == "stage3" else "p_attack"


def default_flag(key: str, t: pd.DataFrame) -> np.ndarray:
    k = _kind(key)
    if k == "stage1":
        return (t.p_attack >= 0.5).to_numpy()
    if k == "stage2":
        return (t.tactic_pred != "Benign").to_numpy()
    return (t.anomaly_score > _phase2(key)["threshold"]).to_numpy()


def operating_point(key: str, t: pd.DataFrame, flag: np.ndarray, groups: np.ndarray, cfg: dict) -> dict:
    te = (t.part == "test").to_numpy()
    tt = t[te]
    out = {"test": M.binary(tt.Label, flag[te]),
           "test_ci": M.block_bootstrap_ci(tt.Label, flag[te], groups[te], n_boot=cfg["phase3"]["n_boot"], seed=cfg["seed"]),
           "val": M.binary(t.Label[t.part == "val"], flag[(t.part == "val").to_numpy()])}
    if "source" in t:
        out["test_per_source"] = M.per_source(tt, flag[te])
        out["test_per_class"] = M.per_class_detection(tt, flag[te])
    else:
        out["test_per_slice"] = {sl: M.binary(g.Label, flag[te][(tt.slice == sl).to_numpy()]) for sl, g in tt.groupby("slice")}
        out["test_per_class"] = M.per_class_detection(tt.assign(source=tt.slice), flag[te])
    if _kind(key) == "stage1":
        out["test_tactic_confusion"] = M.confusion(tt.tactic, tactic_at(tt, flag[te]),
                                                   [x for x in TACTICS if x in set(tt.tactic) | set(tt.tactic_any)])
    ho = (t.part == "holdout").to_numpy()
    if ho.any():
        h, hf = t[ho], flag[ho]
        out["holdout"] = {}
        for cls, g in h.groupby("Attack"):
            m = (h.Attack == cls).to_numpy()
            r = {"n": int(m.sum()), "detected": float(hf[m].mean())}
            if _kind(key) == "stage1":
                r["tactic_counts"] = pd.Series(tactic_at(g, hf[m])).value_counts().to_dict()
            out["holdout"][cls] = r
    return out


def evaluate_model(key: str, t: pd.DataFrame, cfg: dict) -> dict:
    sc = score_col(key)
    groups = t.block.to_numpy() if "block" in t else M.block_ids(t, cfg["phase3"]["block_minutes"])
    val_benign = t.loc[(t.part == "val") & (t.Label == 0), sc]
    target = cfg["phase3"]["target_fpr"]
    thr = M.threshold_at_fpr(val_benign, target)
    flags = {"default": default_flag(key, t), f"val_fpr_{target:g}": (t[sc] > thr).to_numpy()}
    te = t[t.part == "test"]
    res = {"model": key, "score": sc, "threshold_val": thr, "target_fpr": target,
           "operating_points": {k: operating_point(key, t, f, groups, cfg) for k, f in flags.items()},
           "test_curves": M.curves(te.Label, te[sc])}
    if sc == "p_attack":
        res["test_calibration"] = M.calibration(te.Label, te.p_attack)
    if _kind(key) == "stage2":
        res["test_tactic_confusion"] = M.confusion(te.tactic, te.tactic_pred,
                                                   [x for x in TACTICS if x in set(te.tactic) | set(te.tactic_pred)])
    # reproduction check: the default operating point must match what Phase 2 recorded
    p2 = _phase2(key).get("test", {})
    p2f1 = (p2.get("binary") or p2).get("f1")
    res["phase2_test_f1"] = p2f1
    res["reproduces_phase2"] = None if p2f1 is None else bool(abs(res["operating_points"]["default"]["test"]["f1"] - p2f1) < 1e-6)
    return res


# ------------------------------------------------------- combined detector
def combined_detector(s1: pd.DataFrame, s3: pd.DataFrame, cfg: dict, name: str) -> dict:
    """Stage 1 alone vs "Stage 1 OR Stage 3", both held to the same validation FPR budget."""
    if not (len(s1) == len(s3) and (s1.Label.to_numpy() == s3.Label.to_numpy()).all()
            and (s1.FLOW_START_MILLISECONDS.to_numpy() == s3.FLOW_START_MILLISECONDS.to_numpy()).all()):
        raise ValueError("Stage 1 and Stage 3 tables are not row-aligned")
    c = cfg["phase3"]
    budget = c["target_fpr"]
    vb = ((s1.part == "val") & (s1.Label == 0)).to_numpy()
    va = (s1.part == "val").to_numpy()
    p, a = s1.p_attack.to_numpy(), s3.anomaly_score.to_numpy()
    grid = []
    for f1_ in c["combo_fpr_grid"]:
        for f3_ in [0.0] + c["combo_fpr_grid"]:
            t1 = M.threshold_at_fpr(p[vb], f1_)
            t3 = np.inf if f3_ == 0 else M.threshold_at_fpr(a[vb], f3_)
            flag = (p > t1) | (a > t3)
            v = M.binary(s1.Label[va], flag[va])
            grid.append({"s1_fpr": f1_, "s3_fpr": f3_, "t1": t1, "t3": None if f3_ == 0 else t3,
                         "val_fpr": v["fpr"], "val_recall": v["recall"]})
    ok = [g for g in grid if g["val_fpr"] <= budget]
    alone = max((g for g in ok if g["s3_fpr"] == 0), key=lambda g: g["val_recall"])
    combo = max((g for g in ok if g["s3_fpr"] > 0), key=lambda g: g["val_recall"], default=None)
    groups = M.block_ids(s1, c["block_minutes"])

    def run(g):
        flag = (p > g["t1"]) | (a > (np.inf if g["t3"] is None else g["t3"]))
        return {"chosen_on_val": g, **operating_point("stage1_combined", s1, flag, groups, cfg)}

    return {"name": name, "fpr_budget": budget, "val_grid": grid,
            "stage1_alone": run(alone), "stage1_or_stage3": run(combo) if combo else None}


# -------------------------------------------------------------------- lab
def lab_breakdown(cfg: dict, tables: dict) -> dict:
    """Which real lab flows the lab-compatible models flag, by protocol and destination port.

    One attack in 54 flows: this describes the false alarms; it is not a
    precision/recall estimate.
    """
    lab = lab_to_bigflow(ROOT / cfg["data"]["lab_csv"])
    lab["tactic"] = lab_tactic(lab.Attack).values
    target = cfg["phase3"]["target_fpr"]
    out = {"rows": len(lab), "attacks": int(lab.Label.sum()), "target_fpr": target, "models": {}}
    keys = [f"stage1_{a}_temporal_lab" for a in cfg["stage1"]["models"]] + ["stage3_autoencoder_temporal_lab"]
    for key in keys:
        stage, name = key.split("_", 1)
        m, _ = load_model(stage, name)
        t = tables[key]
        sc = score_col(key)
        thr = M.threshold_at_fpr(t.loc[(t.part == "val") & (t.Label == 0), sc], target)
        s = m.predict(lab).p_attack.to_numpy() if stage == "stage1" else m.score(lab)
        for op, flag in (("default", s >= 0.5 if stage == "stage1" else s > _phase2(key)["threshold"]),
                         (f"val_fpr_{target:g}", s > thr)):
            d = lab.assign(flag=flag.astype(int))
            out["models"].setdefault(key, {})[op] = {
                **M.binary(d.Label, d.flag),
                "flagged_by_protocol": d.groupby("PROTOCOL").flag.agg(["size", "sum"]).rename(
                    columns={"size": "n", "sum": "flagged"}).reset_index().to_dict("records"),
                "flagged_by_dst_port": d[d.flag == 1].L4_DST_PORT.value_counts().head(15).to_dict(),
                "attack_flagged": bool(d.loc[d.Label == 1, "flag"].all()) if d.Label.any() else None}
    return out
