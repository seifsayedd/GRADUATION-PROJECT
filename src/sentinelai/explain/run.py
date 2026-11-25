"""Phase 4 analyses. Each function writes one JSON file to reports/phase4/ and returns it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.mapping import lab_to_bigflow
from ..data.prepare import ROOT, git_commit
from ..data.taxonomy import lab_tactic
from ..models.common import load_model, write_json
from ..models.sequence import block_split
from ..models.train import load_sequences
from . import build, checks
from .attention import attention_vs_occlusion

P3 = ROOT / "reports" / "phase3"


def _flows(cfg, part):
    return pd.read_parquet(ROOT / cfg["data"]["interim_dir"] / "flows" / f"temporal_{part}.parquet")


def _info(stage, name, fset):
    return {"stage": stage, "name": name, "feature_set": fset, "git_commit": git_commit()}


def _caveats(cfg, key):
    return list(cfg["phase4"]["caveats"].get(key, []))


def _stratified(df: pd.DataFrame, by: str, n: int, seed: int) -> pd.DataFrame:
    return df.sample(frac=1, random_state=seed).groupby(by, sort=True).head(n)


def _save_examples(out: Path, name: str, objs: list[dict]):
    p = out / "examples" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(objs, indent=2), encoding="utf-8")


def _one_per(objs: list[dict], key) -> list[dict]:
    seen, pick = set(), []
    for o in objs:
        k = key(o)
        if k not in seen:
            seen.add(k)
            pick.append(o)
    return pick


def _signed_top(agg: pd.DataFrame, top: int = 10) -> dict:
    m = agg.mean()
    return m.reindex(m.abs().sort_values(ascending=False).index[:top]).round(6).to_dict()


# ------------------------------------------------------------------ Stage 1
def run_stage1(cfg: dict, out: Path, log=print) -> dict:
    c4, seed = cfg["phase4"], cfg["seed"]
    test, train = _flows(cfg, "test"), None
    res = {}
    for fset in cfg["feature_sets"]:
        for algo in ("xgboost", "random_forest"):
            name = f"{algo}_temporal_{fset}"
            m, _ = load_model("stage1", name)
            p = m.predict(test)
            flagged = test[p.is_attack == 1]
            n = c4["per_tactic"] if algo == "xgboost" else c4["rf_per_tactic"]
            sample = _stratified(flagged, "tactic", n, seed)        # includes Benign = false alarms
            log(f"stage1 {name}: explaining {len(sample):,} flagged test flows ...")
            objs, fr = build.stage1(m, sample, _info(1, name, fset), _caveats(cfg, "stage1"), id_prefix=f"s1-{name}")
            r = {"rows": len(sample), "rows_per_true_tactic": sample.tactic.value_counts().to_dict(),
                 "additivity": fr["additivity"],
                 "importance_by_true_tactic": checks.global_importance(fr["binary"], sample.tactic),
                 "importance_by_source": checks.global_importance(fr["binary"], sample.source),
                 "false_alarm_drivers": _signed_top(fr["binary"][sample.Label.to_numpy() == 0])}
            if algo == "xgboost" or fset == "full":
                train = train if train is not None else _flows(cfg, "train")
                ben = train[train.Label == 0].sample(c4["reference_rows"], random_state=seed)
                ref = checks.benign_reference(m.transform(ben))
                log(f"  deletion test ...")
                r["deletion"] = checks.deletion_test(lambda X: m.binary_.predict_proba(X)[:, 1], fr["X"], fr["binary"], ref,
                                                     ks=tuple(c4["deletion_k"]), seed=seed)
            if algo == "xgboost":
                r["stability"] = checks.stability(fr["X"], fr["binary"], seed)
                _save_examples(out, f"stage1_{name}", _one_per(objs, lambda o: o["ground_truth"].get("tactic")))
            res[name] = r
    write_json({"git_commit": git_commit(), "stage1": res}, out / "stage1.json")
    return res


# ------------------------------------------------------------------ holdout
def run_holdout(cfg: dict, out: Path, log=print) -> dict:
    c4, seed = cfg["phase4"], cfg["seed"]
    hold = _flows(cfg, "holdout")
    res = {}
    for fset in cfg["feature_sets"]:
        name = f"xgboost_temporal_{fset}"
        m, _ = load_model("stage1", name)
        r = {}
        for cls, g in hold.groupby("Attack"):
            s = g.sample(min(len(g), c4["holdout_rows"]), random_state=seed)
            log(f"holdout {name}: {cls} ({len(s)}) ...")
            objs, fr = build.stage1(m, s, _info(1, name, fset), _caveats(cfg, "holdout"), id_prefix=f"ho-{name}")
            flag = fr["p_attack"].to_numpy() >= 0.5
            rr = {"rows": len(s), "detected": float(flag.mean()),
                  "binary_mean_shap_all": _signed_top(fr["binary"]),
                  "binary_mean_shap_missed": _signed_top(fr["binary"][~flag]) if (~flag).any() else None,
                  "tactic_given": pd.Series(fr["tactic_pred"][flag]).value_counts().to_dict()}
            if flag.any():
                rr["tactic_mean_shap_flagged"] = _signed_top(fr["tactic"][flag])
            r[cls] = rr
            _save_examples(out, f"holdout_{name}_{cls}", objs[:2])
        res[name] = r
    write_json({"git_commit": git_commit(), "holdout": res}, out / "holdout.json")
    return res


# ------------------------------------------------------------------ lab
def run_lab(cfg: dict, out: Path, log=print) -> dict:
    lab = lab_to_bigflow(ROOT / cfg["data"]["lab_csv"])
    lab["tactic"] = lab_tactic(lab.Attack).values
    lab["source"] = "lab"
    train = _flows(cfg, "train")
    ben = train[train.Label == 0]
    res = {"rows": len(lab), "attacks": int(lab.Label.sum())}
    for algo in ("xgboost", "random_forest"):
        name = f"{algo}_temporal_lab"
        m, _ = load_model("stage1", name)
        log(f"lab: explaining {name} ...")
        objs, fr = build.stage1(m, lab, _info(1, name, "lab"), _caveats(cfg, "stage1") + _caveats(cfg, f"lab_{algo}"),
                                id_prefix=f"lab-{name}")
        flag = fr["p_attack"].to_numpy() >= 0.5
        fp = flag & (lab.Label.to_numpy() == 0)
        top = _signed_top(fr["binary"][fp], 10) if fp.any() else {}
        cmp = {}
        for f in top:                                   # lab value vs BigFlow benign training distribution
            if f in lab and f in ben and pd.api.types.is_numeric_dtype(ben[f]):
                lv = lab.loc[fp, f].astype("float64")
                b = ben[f].astype("float64")
                cmp[f] = {"lab_false_alarm_median": float(lv.median()),
                          "bigflow_benign_median": float(b.median()),
                          "lab_median_percentile_in_bigflow_benign": float((b < lv.median()).mean() * 100),
                          "per_source_benign_median": ben.groupby("source")[f].median().astype(float).to_dict()}
        res[name] = {"flagged": int(flag.sum()), "false_alarms": int(fp.sum()),
                     "attack_flagged": bool(flag[lab.Label.to_numpy() == 1].all()),
                     "additivity": fr["additivity"], "false_alarm_drivers": top, "value_comparison": cmp,
                     "tactic_given_to_false_alarms": pd.Series(fr["tactic_pred"][fp]).value_counts().to_dict()}
        (out / "lab").mkdir(parents=True, exist_ok=True)
        (out / "lab" / f"{name}.json").write_text(json.dumps(objs, indent=2), encoding="utf-8")
    ae, _ = load_model("stage3", "autoencoder_temporal_lab")
    log("lab: explaining autoencoder ...")
    objs, agg = build.stage3(ae, lab, _info(3, "autoencoder_temporal_lab", "lab"), _caveats(cfg, "stage3"), id_prefix="lab-ae")
    res["autoencoder_temporal_lab"] = {"flagged": int(sum(o["decision"]["flagged"] for o in objs)),
                                       "mean_error_share_top": (agg.mean() / agg.mean().sum()).sort_values(ascending=False).head(10).round(4).to_dict()}
    (out / "lab" / "autoencoder_temporal_lab.json").write_text(json.dumps(objs, indent=2), encoding="utf-8")
    write_json({"git_commit": git_commit(), "lab": res}, out / "lab.json")
    return res


# ------------------------------------------------------------------ feature definitions
def run_units(cfg: dict, out: Path, log=print) -> dict:
    """Does BigFlow compute the rate features the way the lab mapping does? (BigFlow training flows only.)"""
    d = _flows(cfg, "train")
    d = d[d.FLOW_DURATION_MILLISECONDS > 0].sample(400_000, random_state=cfg["seed"])
    sec = d.FLOW_DURATION_MILLISECONDS / 1000

    def ratio(a, b):
        r = (a / b).replace([np.inf, -np.inf], np.nan).dropna()
        return {"median_ratio": float(r.median()), "share_within_5pct": float((abs(r - 1) < 0.05).mean())}

    res = {}
    for src, g in d.groupby("source"):
        s = sec.loc[g.index]
        log(f"units: {src}")
        res[src] = {"rows": len(g),
                    "SRC_TO_DST_SECOND_BYTES_zero_share": float((g.SRC_TO_DST_SECOND_BYTES == 0).mean()),
                    "SECOND_BYTES vs IN_BYTES / duration_s (lab mapping)": ratio(g.SRC_TO_DST_SECOND_BYTES, g.IN_BYTES / s),
                    "AVG_THROUGHPUT vs IN_BYTES*8 / FLOW_DURATION_s": ratio(g.SRC_TO_DST_AVG_THROUGHPUT, g.IN_BYTES * 8 / s),
                    "AVG_THROUGHPUT vs IN_BYTES*8 / DURATION_IN_s (lab mapping)": ratio(g.SRC_TO_DST_AVG_THROUGHPUT, g.IN_BYTES * 8 / (g.DURATION_IN / 1000))}
    write_json({"git_commit": git_commit(), "units": res}, out / "units.json")
    return res


# ------------------------------------------------------------------ Stage 3 (test)
def run_stage3(cfg: dict, out: Path, log=print) -> dict:
    c4, seed = cfg["phase4"], cfg["seed"]
    test = _flows(cfg, "test")
    res = {}
    for fset in cfg["feature_sets"]:
        name = f"autoencoder_temporal_{fset}"
        ae, _ = load_model("stage3", name)
        s = _stratified(test, "tactic", c4["stage3_per_tactic"], seed)
        log(f"stage3 {name}: explaining {len(s):,} test flows ...")
        objs, agg = build.stage3(ae, s, _info(3, name, fset), _caveats(cfg, "stage3"), id_prefix=f"s3-{name}")
        flag = np.array([o["decision"]["flagged"] for o in objs])
        share = agg.div(agg.sum(1), axis=0)
        res[name] = {"rows": len(s), "flagged": int(flag.sum()),
                     "error_share_by_true_tactic": {t: share[s.tactic.to_numpy() == t].mean().sort_values(ascending=False).head(8).round(4).to_dict()
                                                    for t in sorted(s.tactic.unique())},
                     "max_additivity_error": max(o["contributions"]["additivity_error"] for o in objs)}
        _save_examples(out, f"stage3_{name}", [o for o in objs if o["decision"]["flagged"]][:3])
    write_json({"git_commit": git_commit(), "stage3": res}, out / "stage3.json")
    return res


# ------------------------------------------------------------------ Stage 2
def run_stage2(cfg: dict, out: Path, log=print) -> dict:
    c4, c2, seed = cfg["phase4"], cfg["stage2"], cfg["seed"]
    seqs = load_sequences(cfg)
    test = block_split(seqs, cfg["data"]["val_size"], cfg["data"]["test_size"], c2.get("block_minutes", 5), seed)["test"]
    del seqs
    thr = json.loads((P3 / "models.json").read_text(encoding="utf-8"))["models"]
    res = {}
    for fset in cfg["feature_sets"]:
        name = f"{c2['cell']}_w{c2['window']}_{fset}"
        m, _ = load_model("stage2", name)
        t = thr[f"stage2_{name}"]["threshold_val"]
        pred = m.predict(test)                               # index in model order
        ordered_idx = pred.index
        flagged = pd.DataFrame({"pos": np.arange(len(pred)), "tactic": test.loc[ordered_idx, "tactic"].to_numpy()})[pred.p_attack.to_numpy() > t]
        pick = _stratified(flagged, "tactic", c4["stage2_per_tactic"], seed).pos.to_numpy()
        log(f"stage2 {name}: attention + occlusion on {len(pick):,} flagged flows ...")
        objs, raw = build.stage2(m, test, np.sort(pick), _info(2, name, fset), _caveats(cfg, "stage2"), t,
                                 "1% of benign validation flagged (Phase 3)", id_prefix=f"s2-{name}")
        agree = attention_vs_occlusion(raw["att"], raw["delta"], raw["mask"], seed)
        W = raw["mask"].shape[1]
        res[name] = {"rows": len(pick), "threshold": t, "attention_vs_occlusion": agree,
                     "mean_attention_on_labelled_flow": float(raw["att"][:, W - 1].mean()),
                     "mean_attention_by_offset": {int(j - (W - 1)): round(float(v), 5) for j, v in enumerate(raw["att"].mean(0))}}
        _save_examples(out, f"stage2_{name}", _one_per(objs, lambda o: o["ground_truth"].get("tactic")))
    write_json({"git_commit": git_commit(), "stage2": res}, out / "stage2.json")
    return res
