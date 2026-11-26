#!/usr/bin/env python3
"""Phase 4: explain the saved Phase 2 models and check the explanations.

Needs the Phase 2 models in models/, the prepared splits in data/interim/phase2/
and the Phase 3 results in reports/phase3/ (Stage 2 thresholds).
Writes JSON, example explanation objects, figures and report.md to reports/phase4/.

Usage:
    python scripts/explain_phase4.py                       # everything
    python scripts/explain_phase4.py --only stage1 holdout lab
    python scripts/explain_phase4.py --only stage2 stage3
    python scripts/explain_phase4.py --only figures report
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinelai.data.prepare import load_config  # noqa: E402
from sentinelai.explain import run  # noqa: E402
from sentinelai.explain.report import write_figures, write_report  # noqa: E402
from sentinelai.explain.schema import write_schema  # noqa: E402

STEPS = ["stage1", "holdout", "lab", "units", "stage3", "stage2", "figures", "report"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase4.yaml"))
    ap.add_argument("--only", nargs="+", choices=STEPS)
    a = ap.parse_args()
    p4 = load_config(a.config)
    cfg = load_config(ROOT / p4.pop("base_config"))
    cfg.update(p4)
    out = ROOT / cfg["phase4"]["reports_dir"]
    write_schema(ROOT / "docs" / "schema" / "explanation.schema.json")
    for step in a.only or STEPS:
        if step in ("figures", "report"):
            continue
        log(f"== {step}")
        getattr(run, f"run_{step}")(cfg, out, log)
    if "figures" in (a.only or STEPS):
        log(f"figures -> {write_figures(out)}")
    if "report" in (a.only or STEPS):
        log(f"report -> {write_report(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
