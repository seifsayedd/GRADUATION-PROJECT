#!/usr/bin/env python3
"""Phase 3: evaluate the saved Phase 2 models.

Needs the Phase 2 models in models/ and the prepared splits in data/interim/phase2/.
Writes JSON, figures and report.md to reports/phase3/.

Usage:
    python scripts/evaluate_phase3.py                 # everything (about 1 hour with A and B)
    python scripts/evaluate_phase3.py --only predict eval combined lab
    python scripts/evaluate_phase3.py --only loso     # option A: leave-one-source-out
    python scripts/evaluate_phase3.py --only stage3   # option B: per-source Stage 3
    python scripts/evaluate_phase3.py --only figures report
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sentinelai.data.prepare import git_commit, load_config  # noqa: E402
from sentinelai.eval import evaluate as E  # noqa: E402
from sentinelai.eval import figures as F  # noqa: E402
from sentinelai.eval import generalise as G  # noqa: E402
from sentinelai.eval.predict import score_all  # noqa: E402
from sentinelai.eval.report import write_report  # noqa: E402
from sentinelai.models.common import write_json  # noqa: E402

STEPS = ["predict", "eval", "combined", "lab", "loso", "stage3", "figures", "report"]


def config(path: Path) -> dict:
    p3 = load_config(path)
    cfg = load_config(ROOT / p3.pop("base_config"))
    cfg.update(p3)
    return cfg


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase3.yaml"))
    ap.add_argument("--only", nargs="+", choices=STEPS)
    ap.add_argument("--force", action="store_true", help="re-score models even if cached predictions exist")
    a = ap.parse_args()
    cfg = config(Path(a.config))
    steps = a.only or STEPS
    out = ROOT / cfg["phase3"]["reports_dir"]
    figs = out / "figures"

    paths = score_all(cfg, log, force=a.force and "predict" in steps) if set(steps) - {"report"} else {}
    tables = {k: pd.read_parquet(p) for k, p in paths.items()}

    def save(obj, name):
        obj = {"git_commit": git_commit(), **obj}
        write_json(obj, out / f"{name}.json")
        log(f"wrote {name}.json")

    def load(name):
        return json.loads((out / f"{name}.json").read_text(encoding="utf-8"))

    if "eval" in steps:
        res = {}
        for k, t in tables.items():
            log(f"evaluating {k} ...")
            res[k] = E.evaluate_model(k, t, cfg)
            r = res[k]
            log(f"  default F1={r['operating_points']['default']['test']['f1']:.4f}  reproduces Phase 2: {r['reproduces_phase2']}")
        save({"models": res}, "models")
    if "combined" in steps:
        res = {f: E.combined_detector(tables[f"stage1_xgboost_temporal_{f}"], tables[f"stage3_autoencoder_temporal_{f}"],
                                      cfg, f"xgboost + autoencoder ({f})") for f in cfg["feature_sets"]}
        save({"combined": res}, "combined")
    if "lab" in steps:
        save({"lab": E.lab_breakdown(cfg, tables)}, "lab")
    if "loso" in steps:
        save({"loso": G.run_loso(cfg, tables, log)}, "loso")
    if "stage3" in steps:
        res = {}
        for f in cfg["feature_sets"]:
            k = f"stage3_autoencoder_temporal_{f}"
            res[f] = G.run_stage3_variants(cfg, tables[k], f, E._phase2(k)["threshold"], log)
            log(f"  selected on val ({f}): {res[f]['selected_on_val']}")
        save({"stage3_variants": res}, "stage3_variants")
    if "figures" in steps:
        m = load("models")["models"]
        for k, r in m.items():
            op = r["operating_points"]
            if "test_tactic_confusion" in op["default"]:
                F.confusion_heatmap(op["default"]["test_tactic_confusion"], f"{k}: tactic (test, threshold 0.5)",
                                    figs / f"confusion_{k}.png")
            if "test_tactic_confusion" in r:
                F.confusion_heatmap(r["test_tactic_confusion"], f"{k}: tactic (test)", figs / f"confusion_{k}.png")
            F.per_class_bars(op["default"]["test_per_class"], f"{k}: detection per class (test, default threshold)",
                             figs / f"per_class_{k}.png")
        for split in ("temporal", "stratified"):
            sel = {k: r["test_curves"] for k, r in m.items() if split in k or (split == "temporal" and k.startswith("stage2"))}
            F.roc_pr(sel, f"{split} split + Stage 2", figs / f"roc_pr_{split}.png")
        F.calibration_plot({k: r["test_calibration"] for k, r in m.items() if "test_calibration" in r and "stratified" not in k},
                           figs / "calibration.png")
        for f in cfg["feature_sets"]:
            k = f"stage3_autoencoder_temporal_{f}"
            if k in tables:
                F.score_histograms(tables[k], E._phase2(k)["threshold"],
                                   f"Autoencoder scores per source ({f}); red = Phase 2 threshold", figs / f"ae_scores_{f}.png")
        if (out / "loso.json").exists():
            runs = load("loso")["loso"]["runs"]
            srcs = sorted({r["test_source"] for r in runs})
            for metric, ylab in (("f1", "F1"), ("fpr", "false-positive rate")):
                ser = {}
                for f in cfg["feature_sets"]:
                    rr = {r["test_source"]: r for r in runs if r["feature_set"] == f}
                    ser[f"{f}: unseen source"] = [rr[s]["default"][metric] if s in rr else None for s in srcs]
                    ser[f"{f}: seen (Phase 2)"] = [rr[s]["in_source_baseline"][metric] if s in rr else None for s in srcs]
                F.grouped_bars(srcs, ser, f"Leave-one-source-out, XGBoost: {ylab}", ylab, figs / f"loso_{metric}.png")
        if (out / "stage3_variants.json").exists():
            sv = load("stage3_variants")["stage3_variants"]
            for f, r in sv.items():
                srcs = sorted(next(iter(r["variants"].values()))["test_per_source"])
                F.grouped_bars(srcs, {v: [x["test_per_source"][s]["benign_fpr"] for s in srcs] for v, x in r["variants"].items()},
                               f"Stage 3 variants ({f}): benign FPR per source (test)", "benign false-positive rate",
                               figs / f"stage3_variants_fpr_{f}.png")
                F.grouped_bars(srcs, {v: [x["test_per_source"][s]["attack_detected"] for s in srcs] for v, x in r["variants"].items()},
                               f"Stage 3 variants ({f}): attacks detected per source (test)", "share detected",
                               figs / f"stage3_variants_recall_{f}.png")
        log(f"figures -> {figs}")
    if "report" in steps:
        log(f"report -> {write_report(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
