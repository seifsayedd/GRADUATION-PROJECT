#!/usr/bin/env python3
"""Sanity checks for a SentinelAI lab dataset CSV.

Checks: row/label counts, every real flow matched to an nfdump record,
derived fields consistent with packets/bytes/duration, and synthetic rows
flagged and traceable to a real source row.

Usage:
    python scripts/validate_dataset.py data/processed/combined_sessions_augmented.csv
"""
import sys

import pandas as pd


def main(path: str) -> int:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    flows = df[df.record_type == "flow"]
    synth = flows[flows.get("is_synthetic", "0").astype(str) == "1"] if "is_synthetic" in df else flows.iloc[0:0]
    real = flows.drop(synth.index)
    errors = []

    print(f"file: {path}")
    print(f"rows={len(df)}  flows={len(flows)}  real={len(real)}  synthetic={len(synth)}  "
          f"session_only={(df.record_type == 'session_only').sum()}")
    print("labels:", flows.label.value_counts().to_dict())
    print("sessions:", real.session_id.value_counts().sort_index().to_dict())

    if "nfdump_match" in real and (real.nfdump_match != "matched").any():
        errors.append(f"{(real.nfdump_match != 'matched').sum()} real flows without an nfdump match")

    f = flows.copy()
    for c in ("pkts", "bytes", "dur", "rate", "load"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    ok = f.dur > 0
    bad_rate = ((f.pkts / f.dur - f.rate).abs() > 1e-3 * f.rate.abs().clip(lower=1))[ok].sum()
    bad_load = ((f.bytes * 8 / f.dur - f.load).abs() > 1e-3 * f.load.abs().clip(lower=1))[ok].sum()
    if bad_rate or bad_load:
        errors.append(f"derived fields inconsistent: rate={bad_rate} load={bad_load}")

    if len(synth):
        known = set(df.row_id)
        if (~synth.source_row.isin(known)).any():
            errors.append("synthetic rows with unknown source_row")

    print("OK" if not errors else "FAIL:\n  " + "\n  ".join(errors))
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
