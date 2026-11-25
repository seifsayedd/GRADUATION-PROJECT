#!/usr/bin/env python3
"""Jitter-based synthetic augmentation for the lab flow dataset.

This is the method used to build data/processed/combined_sessions_augmented.csv.
Every synthetic row is flagged (`is_synthetic=1`) and points back to the real
row it was derived from (`source_row`), so it can always be filtered out.

Method
------
* Flows are grouped into *events*: same session, same IP pair (either
  direction) and same start second. One real ping, for example, appears as
  four rows (2 directions x 2 capture interfaces). All rows of an event get
  the same jitter, so the copies stay consistent with each other.
* Per synthetic copy:  time shift U(-120 s, +120 s), packet count x U(0.75, 1.30),
  duration x U(0.85, 1.15). Packet size is kept, so bytes scale with packets.
* Derived fields are recomputed: ltime, rate/srate/drate (pkts/s),
  load/sload/dload (bits/s) and mean packet sizes.
* Benign events get 2 copies, attack events (label 1) get 8, to reduce
  class imbalance.
* nfdump_* columns are cleared on synthetic rows because there is no
  matching collector record for them.

Rule: never use synthetic rows in a test/validation set.

Usage:
    python scripts/augment_flows.py data/processed/combined_sessions_full.csv out.csv [--seed 42]
"""
import argparse

import numpy as np
import pandas as pd

COPIES = {0: 2, 1: 8}


def event_key(r) -> str:
    pair = "|".join(sorted([r.saddr, r.daddr]))
    return f"{r.session_id}_{pair}_{int(float(r.stime))}"


def jitter(row: pd.Series, shift: float, pk: float, tm: float) -> pd.Series:
    r = row.copy()
    num = lambda c: float(r[c]) if str(r[c]) != "" else None
    stime, dur = float(r.stime) + shift, max(num("dur") * tm, 1e-6)
    spk = max(1, round(num("spkts") * pk)) if num("spkts") else 0
    dpk = max(1, round(num("dpkts") * pk)) if num("dpkts") else 0
    ssz = num("sbytes") / num("spkts") if num("spkts") else 0
    dsz = num("dbytes") / num("dpkts") if num("dpkts") else 0
    sb, db = round(spk * ssz), round(dpk * dsz)

    r["stime"], r["ltime"], r["dur"] = f"{stime:.6f}", f"{stime + dur:.6f}", f"{dur:.6f}"
    r["spkts"], r["dpkts"], r["pkts"] = spk, dpk, spk + dpk
    r["sbytes"], r["dbytes"], r["bytes"] = sb, db, sb + db
    r["rate"], r["srate"], r["drate"] = (f"{(spk + dpk) / dur:.6f}", f"{spk / dur:.6f}", f"{dpk / dur:.6f}")
    r["load"], r["sload"], r["dload"] = (f"{(sb + db) * 8 / dur:.6f}", f"{sb * 8 / dur:.6f}", f"{db * 8 / dur:.6f}")
    if spk:
        r["smeansz"] = f"{sb / spk:.1f}"
    if dpk:
        r["dmeansz"] = f"{db / dpk:.1f}"
    for c in [c for c in r.index if c.startswith("nfdump_")]:
        r[c] = ""
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    df = pd.read_csv(a.src, dtype=str, keep_default_na=False)
    df.insert(1, "row_id", [f"R{i:04d}" for i in range(len(df))])
    df["is_synthetic"], df["source_row"], df["synth_group"], df["augmentation_method"] = 0, "", "", "none"

    flows = df[df.record_type == "flow"].copy()
    flows["_event"] = flows.apply(event_key, axis=1)
    synth = []
    for ev, g in flows.groupby("_event", sort=False):
        label = int(float(g.label.iloc[0] or 0))
        for k in range(1, COPIES[label] + 1):
            shift, pk, tm = rng.uniform(-120, 120), rng.uniform(0.75, 1.30), rng.uniform(0.85, 1.15)
            tag = f"shift={shift:+.1f}s;pkts_x{pk:.2f};size_x1;time_x{tm:.2f}"
            for _, row in g.iterrows():
                s = jitter(row.drop("_event"), shift, pk, tm)
                s["is_synthetic"], s["source_row"] = 1, row.row_id
                s["synth_group"], s["augmentation_method"] = f"{ev}_syn{k}", tag
                synth.append(s)

    out = pd.concat([pd.DataFrame(synth), df], ignore_index=True)
    out["row_id"] = [f"S{i:04d}" if s == 1 else rid for i, (s, rid) in enumerate(zip(out.is_synthetic, out.row_id))]
    out.to_csv(a.out, index=False)
    print(f"real={len(df)} synthetic={len(synth)} total={len(out)}")


if __name__ == "__main__":
    main()
