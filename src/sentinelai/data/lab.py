"""Loaders for the SentinelAI lab CSVs (data/processed/*.csv).

The lab flows come from nfdump on two gateway interfaces, mapped to the Argus
field schema. Two properties of that data matter for modelling:

1. **One-directional records.** Each row carries one direction only
   (``spkts > 0``, ``dpkts == 0``). BigFlow's NetFlow-v3 records are
   two-directional (IN_* / OUT_*). ``pair_biflows`` joins a record with its
   reverse-direction partner so the two shapes line up.
2. **Two vantage points.** Traffic between the segments is recorded on both
   ``ens33`` and ``ens37``, so one event can appear twice. ``dedupe_vantage``
   keeps one copy per event while keeping flows that only one interface saw
   (e.g. the gateway's own mDNS on the target segment).

These files are read-only here; nothing in this module writes to data/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_LAB_COLS = [
    "stime", "ltime", "pkts", "spkts", "dpkts", "bytes", "sbytes", "dbytes", "dur",
    "rate", "srate", "drate", "load", "sload", "dload", "smeansz", "label",
]
PORTLESS_PROTOS = {"icmp", "igmp"}
PROTO_NUMBERS = {"icmp": 1, "igmp": 2, "tcp": 6, "udp": 17, "ipv6-icmp": 58}


class SyntheticLeakError(AssertionError):
    """Raised when synthetic rows show up where only real rows are allowed."""


def load_lab_flows(path: str | Path, include_synthetic: bool = False) -> pd.DataFrame:
    """Load real lab flow rows from a processed lab CSV.

    * Only ``record_type == "flow"`` rows are returned (``session_only``
      rows have no features).
    * Synthetic rows (``is_synthetic == 1``) are dropped unless
      ``include_synthetic`` is set. They are never allowed into evaluation;
      see ``split.assert_no_synthetic``.
    * Numeric columns are converted; empty strings become NaN, not 0.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "record_type" in df:
        df = df[df.record_type == "flow"]
    if "is_synthetic" in df:
        df["is_synthetic"] = pd.to_numeric(df.is_synthetic, errors="coerce").fillna(0).astype(int)
        if not include_synthetic:
            df = df[df.is_synthetic == 0]
    else:
        df["is_synthetic"] = 0
    df = df.mask(df == "")
    for c in NUMERIC_LAB_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label"] = df["label"].astype(int)
    if "capture_interface" not in df and "srcid" in df:
        df["capture_interface"] = df.srcid.str.split("/").str[1]
    return df.reset_index(drop=True)


def _ports(r) -> tuple:
    if r.proto in PORTLESS_PROTOS:
        return ("", "")
    return (str(r.sport), str(r.dport))


def dedupe_vantage(df: pd.DataFrame, prefer: str = "ens33", tol_s: float = 0.05) -> pd.DataFrame:
    """Drop a record when the same flow was also recorded on the preferred interface.

    Two records are the same flow when they have the same direction, protocol,
    addresses, ports, packets and bytes, and start within ``tol_s`` seconds.
    Records seen on only one interface are kept.
    """
    key = ["proto", "saddr", "daddr", "sport", "dport", "spkts", "sbytes", "is_synthetic"]
    d = df.copy()
    if "synth_group" in d:
        key.append("synth_group")
    # explicit "" for missing values: pandas 2 and 3 stringify NaN differently
    d["_k"] = d[key].apply(lambda col: col.map(lambda v: "" if pd.isna(v) else str(v))).agg("|".join, axis=1)
    pref, other = d[d.capture_interface == prefer], d[d.capture_interface != prefer]
    drop = []
    used = set()
    for i, r in other.iterrows():
        cand = pref[(pref._k == r._k) & ((pref.stime - r.stime).abs() <= tol_s)]
        cand = cand[~cand.index.isin(used)]
        if len(cand):
            used.add(cand.index[0])
            drop.append(i)
    return d.drop(index=drop).drop(columns="_k").reset_index(drop=True)


def _is_initiator(a, b) -> bool:
    """True if record ``a`` started the conversation with ``b``."""
    if a.proto == "icmp":
        ta, tb = str(a.dport).split(".")[0], str(b.dport).split(".")[0]
        if ta == "8" and tb == "0":      # echo request vs echo reply
            return True
        if ta == "0" and tb == "8":
            return False
    return (a.stime, a.name) <= (b.stime, b.name)


def pair_biflows(df: pd.DataFrame, tol_s: float = 1.0) -> pd.DataFrame:
    """Join each one-directional record with its reverse-direction partner.

    Partners share interface, protocol and session, have swapped addresses (and
    swapped ports for TCP/UDP), and start within ``tol_s`` seconds of each
    other. Matching is greedy, closest start time first. Unpaired records
    (broadcast, multicast, unanswered traffic) are kept with an empty reverse
    side, which is genuine: no packets went back.

    Returns one row per conversation with ``fwd_*`` (initiator) and ``rev_*``
    columns, plus ``paired`` and the label columns. The two halves of a pair
    must carry the same label; a mismatch raises ``ValueError``.
    """
    d = df.reset_index(drop=True)
    used: set[int] = set()
    rows = []
    for i, a in d.iterrows():
        if i in used:
            continue
        used.add(i)
        pa = _ports(a)
        cand = d[(~d.index.isin(used)) & (d.capture_interface == a.capture_interface)
                 & (d.proto == a.proto) & (d.session_id == a.session_id)
                 & (d.saddr == a.daddr) & (d.daddr == a.saddr)
                 & ((d.stime - a.stime).abs() <= tol_s)
                 & (d.is_synthetic == a.is_synthetic)]
        if "synth_group" in d:  # never pair a synthetic copy with a different copy or a real row
            cand = cand[cand.synth_group.fillna("") == ("" if pd.isna(a.synth_group) else a.synth_group)]
        if a.proto not in PORTLESS_PROTOS:
            cand = cand[(cand.sport.astype(str) == pa[1]) & (cand.dport.astype(str) == pa[0])]
        b = None
        if len(cand):
            j = (cand.stime - a.stime).abs().idxmin()
            used.add(j)
            b = d.loc[j]
            if int(a.label) != int(b.label) or a.get("attack_type") != b.get("attack_type"):
                raise ValueError(f"paired records disagree on label: rows {i} and {j}")
            if not _is_initiator(a, b):
                a, b = b, a
        rows.append(_biflow_row(a, b))
    return pd.DataFrame(rows)


# row_id / source_row (augmented file only) come from the initiator record, so a
# synthetic conversation still points at the real conversation it was copied from.
LABEL_PASSTHROUGH = ["session_id", "capture_interface", "label", "attack_cat", "attack_type",
                     "attack_name", "tool", "label_method", "is_synthetic", "row_id", "source_row"]


def _biflow_row(a, b) -> dict:
    row = {c: a.get(c) for c in LABEL_PASSTHROUGH if c in a.index}
    row.update(paired=b is not None, proto=a.proto, saddr=a.saddr, daddr=a.daddr,
               sport=a.sport, dport=a.dport,
               nfdump_dstPort=a.get("nfdump_dstPort"),
               stime=a.stime if b is None else min(a.stime, b.stime),
               ltime=a.ltime if b is None else max(a.ltime, b.ltime))
    for side, r in (("fwd", a), ("rev", b)):
        for c in ("pkts", "bytes", "dur", "stime", "ltime"):
            src = {"pkts": "spkts", "bytes": "sbytes"}.get(c, c)
            row[f"{side}_{c}"] = (r[src] if r is not None else (0 if c in ("pkts", "bytes") else np.nan))
    return row
