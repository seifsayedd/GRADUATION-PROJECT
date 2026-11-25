#!/usr/bin/env python3
"""Merge flow records (Argus schema), nfdump exports (ens33 + ens37) and the session manifest
into one labeled dataset.

Each Argus flow row is matched to exactly one nfdump row captured on the same
interface, using (start time in ms, source IP, destination IP, packets, bytes)
and, if still ambiguous, destination port. Sessions that produced no network
traffic are kept as `session_only` metadata rows so nothing in the manifest is
lost.

Usage:
    python scripts/merge_sessions.py data/raw data/processed/combined_sessions_full.csv
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

NF_COLS = ["firstSeen", "duration", "proto", "srcPort", "dstPort", "flows"]


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def epoch_ms(ts: str) -> int:
    """nfdump firstSeen (UTC) -> epoch milliseconds."""
    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return round(dt.timestamp() * 1000)


def main(raw_dir: str, out_path: str) -> None:
    raw = Path(raw_dir)
    flows = read(raw / "flows_argus_schema.csv")
    manifest = read(raw / "session_manifest.csv")
    nf = {iface: read(raw / f"{iface}_nfdump.csv") for iface in ("ens33", "ens37")}

    for df in nf.values():
        df["_ms"] = df.firstSeen.map(epoch_ms)
        df["_used"] = False

    flows["capture_interface"] = flows.srcid.str.split("/").str[1]
    matched = {c: [] for c in NF_COLS}
    status = []

    for _, r in flows.iterrows():
        df = nf[r.capture_interface]
        ms = round(float(r.stime) * 1000)
        cand = df[(~df._used) & (df._ms == ms) & (df.srcAddr == r.saddr) & (df.dstAddr == r.daddr)
                  & (df.packets == r.pkts) & (df.bytes == r.bytes)]
        if len(cand) > 1 and r.dport:  # e.g. two broadcasts in the same ms on different ports
            cand = cand[cand.dstPort.str.replace(".0", "", regex=False) == r.dport]
        if len(cand):
            i = cand.index[0]
            df.at[i, "_used"] = True
            for c in NF_COLS:
                matched[c].append(df.at[i, c])
            status.append("matched")
        else:
            for c in NF_COLS:
                matched[c].append("")
            status.append("NO_MATCH")

    for c in NF_COLS:
        flows[f"nfdump_{c}"] = matched[c]
    flows["nfdump_match"] = status

    man = manifest.rename(columns={c: f"session_{c}" for c in manifest.columns if c != "session_id"})
    out = flows.merge(man, on="session_id", how="left")
    out.insert(0, "record_type", "flow")

    extra = man[~man.session_id.isin(set(flows.session_id))].copy()
    extra.insert(0, "record_type", "session_only")
    full = pd.concat([out, extra], ignore_index=True)[out.columns].fillna("")

    full["_order"] = pd.to_numeric(full.stime, errors="coerce")
    full = full.sort_values(["record_type", "_order", "capture_interface"], na_position="last")
    full.drop(columns="_order").to_csv(out_path, index=False)

    n_flow = (full.record_type == "flow").sum()
    print(f"rows={len(full)} flows={n_flow} session_only={len(full) - n_flow} "
          f"unmatched={status.count('NO_MATCH')} columns={full.shape[1] - 1}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
