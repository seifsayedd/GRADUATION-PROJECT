#!/usr/bin/env python3
"""Build Phase 2 training data from BigFlow into data/interim/phase2/ (git-ignored).

Usage:
    python scripts/prepare_training_data.py                 # flows + sequences
    python scripts/prepare_training_data.py --only flows
    python scripts/prepare_training_data.py --config configs/phase2.yaml --per-class 5000   # quick run
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelai.data.prepare import build_flow_dataset, build_sequence_dataset, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", choices=["flows", "sequences"])
    ap.add_argument("--per-class", type=int, help="override data.per_class_cap (quick runs)")
    ap.add_argument("--out", help="override data.interim_dir")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    if a.per_class:
        cfg["data"]["per_class_cap"] = a.per_class
        cfg["data"]["class_caps"] = {k: min(v, a.per_class * 10) for k, v in cfg["data"].get("class_caps", {}).items()}
    out = Path(a.out) if a.out else None
    if a.only in (None, "flows"):
        meta = build_flow_dataset(cfg, out)
        print("flows:", json.dumps({k: meta[k] for k in ("rows", "splits", "source_counts")}, default=str))
    if a.only in (None, "sequences"):
        meta = build_sequence_dataset(cfg, out)
        for name, m in meta["slices"].items():
            print(f"sequences/{name}: kept {m['rows_kept']:,} of {m['rows_raw_day']:,} rows, "
                  f"{m['hosts']:,} hosts, {m['class_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
