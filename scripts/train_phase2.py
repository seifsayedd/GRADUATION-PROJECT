#!/usr/bin/env python3
"""Train the Phase 2 detection models.

Run scripts/prepare_training_data.py first. Models go to models/ (git-ignored),
metrics to reports/phase2/*.json, and reports/phase2/summary.md is rebuilt
from whatever JSON files exist.

Usage:
    python scripts/train_phase2.py --stage 1                  # all Stage 1 runs from the config
    python scripts/train_phase2.py --stage 1 --split temporal --feature-set lab --algo xgboost
    python scripts/train_phase2.py --stage 2
    python scripts/train_phase2.py --stage 3
    python scripts/train_phase2.py --lab                      # score the real lab flows
    python scripts/train_phase2.py --summary                  # rebuild summary.md only
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelai.data.prepare import load_config  # noqa: E402
from sentinelai.models import train  # noqa: E402
from sentinelai.models.summary import write_summary  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3])
    ap.add_argument("--split", choices=["temporal", "stratified"])
    ap.add_argument("--feature-set", choices=["full", "lab"])
    ap.add_argument("--algo", choices=["random_forest", "xgboost"])
    ap.add_argument("--lab", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    splits = [a.split] if a.split else ["temporal", "stratified"]
    fsets = [a.feature_set] if a.feature_set else cfg["feature_sets"]

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    if a.stage == 1:
        for split in splits:
            for fset in fsets:
                for algo in ([a.algo] if a.algo else cfg["stage1"]["models"]):
                    log(f"stage1 {algo} {split} {fset} ...")
                    r = train.run_stage1(cfg, split, fset, algo)
                    log(f"  test F1={r['test']['binary']['f1']:.4f} FPR={r['test']['binary']['fpr']:.4f} "
                        f"tactic macroF1={r['test']['tactic_end_to_end']['macro_f1']:.4f} ({r['fit_seconds']}s, "
                        f"{r['model']['size_mb']} MB)")
    elif a.stage == 2:
        df = train.load_sequences(cfg)
        for fset in fsets:
            log(f"stage2 {cfg['stage2']['cell']} {fset} ...")
            r = train.run_stage2(cfg, fset, df=df)
            log(f"  test F1={r['test']['binary']['f1']:.4f} tactic macroF1={r['test']['tactic']['macro_f1']:.4f} "
                f"({r['fit_seconds']}s)")
    elif a.stage == 3:
        for split in ([a.split] if a.split else ["temporal"]):
            for fset in fsets:
                log(f"stage3 autoencoder {split} {fset} ...")
                r = train.run_stage3(cfg, split, fset)
                log(f"  test ROC-AUC={r['test'].get('roc_auc', float('nan')):.4f} recall={r['test']['recall']:.4f} "
                    f"FPR={r['test']['fpr']:.4f} ({r['fit_seconds']}s)")
    if a.lab:
        log("scoring real lab flows ...")
        train.score_lab(cfg)
    write_summary(train.REPORTS)
    log(f"summary -> {train.REPORTS / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
