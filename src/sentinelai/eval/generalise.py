"""Generalisation experiments that need retraining.

A. Leave-one-source-out (LOSO): train Stage 1 XGBoost on three source
   datasets and test on the fourth, which it has never seen. This is the
   closest BigFlow gets to "a new network", the failure seen on the lab data.

B. Stage 3 per-source variants, chosen on validation:
   * ``global``      the Phase 2 autoencoder and its single threshold
   * ``per_source_threshold``  same autoencoder, one threshold per source
     (99th percentile of that source's benign validation scores)
   * ``per_source_model``      one autoencoder per source, trained on that
     source's benign traffic only
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..models.autoencoder import Autoencoder
from ..models.common import feature_set
from ..models.flow_classifier import FlowClassifier
from . import metrics as M
from .predict import load_flows


def loso_splits(flows: dict, source: str) -> dict:
    """Train/val from every other source; test = every row of ``source`` (train, val and test parts).

    Held-out attack classes are not used at all here.
    """
    other = {p: flows[p][flows[p].source != source] for p in ("train", "val")}
    test = pd.concat([flows[p][flows[p].source == source] for p in ("train", "val", "test")])
    for p, d in other.items():
        if (d.source == source).any():
            raise AssertionError(f"{source} leaked into {p}")
    return {**other, "test": test}


def run_loso(cfg: dict, baseline_tables: dict, log=print) -> dict:
    flows = load_flows(cfg, "temporal", ("train", "val", "test"))
    target = cfg["phase3"]["target_fpr"]
    out = {"target_fpr": target, "runs": []}
    for fset in cfg["feature_sets"]:
        base = baseline_tables[f"stage1_xgboost_temporal_{fset}"]
        for src in sorted(flows["train"].source.unique()):
            log(f"LOSO {fset}: test on {src} ...")
            s = loso_splits(flows, src)
            t0 = time.time()
            m = FlowClassifier("xgboost", feature_set(fset), cfg["stage1"]["xgboost"], seed=cfg["seed"]).fit(s["train"], s["val"])
            vb = s["val"][s["val"].Label == 0]
            thr = M.threshold_at_fpr(m.predict(vb).p_attack, target)
            pred = m.predict(s["test"])
            te = s["test"]
            unseen = sorted(set(te.tactic) - set(m.tactic_enc_.classes_) - {"Benign"})
            atk = te.Label.to_numpy() == 1
            known = atk & te.tactic.isin(m.tactic_enc_.classes_).to_numpy()
            tac_correct = float((pred.tactic_pred.to_numpy()[known] == te.tactic.to_numpy()[known]).mean()) if known.any() else None
            bt = base[(base.part == "test") & (base.source == src)]
            groups = M.block_ids(te, cfg["phase3"]["block_minutes"])
            r = {"feature_set": fset, "test_source": src, "train_rows": len(s["train"]), "test_rows": len(te),
                 "fit_seconds": round(time.time() - t0, 1), "threshold_val": thr,
                 "default": M.binary(te.Label, pred.is_attack),
                 "default_ci": M.block_bootstrap_ci(te.Label, pred.is_attack, groups,
                                                    n_boot=cfg["phase3"]["n_boot"], seed=cfg["seed"]),
                 f"val_fpr_{target:g}": M.binary(te.Label, pred.p_attack > thr),
                 "roc_auc": _auc(te.Label, pred.p_attack),
                 "tactics_unseen_in_training": unseen,
                 "tactic_correct_on_known_attacks": tac_correct,
                 "per_class": M.per_class_detection(te, pred.is_attack),
                 "in_source_baseline": {"note": "Phase 2 temporal model (trained WITH this source), its test rows of this source",
                                        **M.binary(bt.Label, bt.p_attack >= 0.5)}}
            out["runs"].append(r)
            log(f"  F1={r['default']['f1']} FPR={r['default']['fpr']} (in-source F1={r['in_source_baseline']['f1']})")
    return out


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


# --------------------------------------------------------------- option B
def per_source_thresholds(val: pd.DataFrame, score: np.ndarray, percentile: float) -> dict:
    """One threshold per source from that source's **benign validation** scores."""
    s = pd.Series(score, index=val.index)
    return {src: float(np.percentile(s[g.index[g.Label == 0]], percentile))
            for src, g in val.groupby("source") if (g.Label == 0).any()}


def flag_per_source(df: pd.DataFrame, score: np.ndarray, thr: dict) -> np.ndarray:
    t = df.source.map(thr).to_numpy("float64")
    if np.isnan(t).any():
        raise ValueError(f"no threshold for sources {sorted(set(df.source[np.isnan(t)]))}")
    return np.asarray(score) > t


def run_stage3_variants(cfg: dict, s3_table: pd.DataFrame, fset: str, default_threshold: float, log=print) -> dict:
    c, pct = cfg["stage3"], cfg["stage3"]["threshold_percentile"]
    t = s3_table
    parts = {p: t[t.part == p] for p in ("val", "test", "holdout")}
    score = {p: parts[p].anomaly_score.to_numpy() for p in parts}
    variants = {"global": {p: score[p] > default_threshold for p in parts}}

    thr = per_source_thresholds(parts["val"], score["val"], pct)
    variants["per_source_threshold"] = {p: flag_per_source(parts[p], score[p], thr) for p in parts}

    # one autoencoder per source, benign train/val of that source only
    flows = load_flows(cfg, "temporal", ("train", *parts))
    for p in parts:
        if not (flows[p].FLOW_START_MILLISECONDS.to_numpy() == parts[p].FLOW_START_MILLISECONDS.to_numpy()).all():
            raise ValueError(f"cached Stage 3 table and flows/{p} are not row-aligned")
    pm_score = {p: np.full(len(parts[p]), np.nan) for p in parts}
    pm_thr, pm_info = {}, {}
    for src in sorted(flows["train"].source.unique()):
        tr = flows["train"][(flows["train"].source == src) & (flows["train"].Label == 0)]
        va = flows["val"][(flows["val"].source == src) & (flows["val"].Label == 0)]
        log(f"stage3 per-source AE {fset}: {src} ({len(tr):,} benign train rows) ...")
        m = Autoencoder(feature_set(fset), c["hidden"], c["epochs"], c["batch_size"], c["lr"], pct, seed=cfg["seed"]).fit(tr, va)
        pm_thr[src] = m.threshold_
        pm_info[src] = {"benign_train_rows": len(tr), "benign_val_rows": len(va), "threshold": m.threshold_,
                        "epochs_run": len(m.history_)}
        for p in parts:
            # flows[p] and the cached table hold the same rows in the same order
            mask = (parts[p].source == src).to_numpy()
            if mask.any():
                pm_score[p][mask] = m.score(flows[p][(flows[p].source == src).to_numpy()])
    if any(np.isnan(v).any() for v in pm_score.values()):
        raise ValueError("a row got no per-source autoencoder score")
    variants["per_source_model"] = {p: pm_score[p] > parts[p].source.map(pm_thr).to_numpy() for p in parts}

    out = {"feature_set": fset, "percentile": pct, "per_source_thresholds": thr, "per_source_models": pm_info,
           "variants": {}}
    for name, fl in variants.items():
        out["variants"][name] = {
            "val": M.binary(parts["val"].Label, fl["val"]),
            "test": M.binary(parts["test"].Label, fl["test"]),
            "test_ci": M.block_bootstrap_ci(parts["test"].Label, fl["test"], M.block_ids(parts["test"], cfg["phase3"]["block_minutes"]),
                                            n_boot=cfg["phase3"]["n_boot"], seed=cfg["seed"]),
            "test_per_source": M.per_source(parts["test"], fl["test"]),
            "holdout": {cls: {"n": int(len(g)), "detected": float(fl["holdout"][(parts["holdout"].Attack == cls).to_numpy()].mean())}
                        for cls, g in parts["holdout"].groupby("Attack")}}
    # selection on validation only: best recall among variants whose val benign FPR stays within budget
    budget = cfg["phase3"]["stage3_val_fpr_budget"]
    ok = {k: v for k, v in out["variants"].items() if v["val"]["fpr"] <= budget}
    out["selected_on_val"] = max(ok, key=lambda k: out["variants"][k]["val"]["recall"]) if ok else None
    out["selection_rule"] = f"highest validation recall with validation FPR <= {budget}"
    return out
