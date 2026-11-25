"""Map lab two-way flows (Argus/nfdump fields) onto BigFlow's NetFlow-v3 features.

Input: the output of ``lab.pair_biflows``. Output: a DataFrame with every
BigFlow feature column. Features the lab sensor cannot produce are **NaN**,
never 0, so a model cannot be scored on invented values.
``available_features`` lists the columns that have real values.

The full table, with reasons for each gap, is in docs/feature-mapping.md.
``FEATURE_MAP`` below is the source of truth; a test checks the doc and
this table cover the same 53 features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lab import PORTLESS_PROTOS, PROTO_NUMBERS
from .schema import BIGFLOW_FEATURES

# status: "mapped" (direct or exact derivation), "derived" (computed, see note),
#         "missing" (not observable with the current sensor)
FEATURE_MAP: dict[str, tuple[str, str]] = {
    "FLOW_START_MILLISECONDS": ("mapped", "min(fwd.stime, rev.stime) * 1000"),
    "FLOW_END_MILLISECONDS": ("mapped", "max(fwd.ltime, rev.ltime) * 1000"),
    "IPV4_SRC_ADDR": ("mapped", "initiator saddr"),
    "IPV4_DST_ADDR": ("mapped", "initiator daddr"),
    "L4_SRC_PORT": ("mapped", "initiator sport; 0 for ICMP/IGMP"),
    "L4_DST_PORT": ("mapped", "initiator dport; 0 for ICMP/IGMP (ICMP type goes to ICMP_TYPE)"),
    "PROTOCOL": ("mapped", "proto name -> IANA number (icmp=1, igmp=2, tcp=6, udp=17)"),
    "L7_PROTO": ("missing", "needs nDPI application detection; nfdump has none"),
    "IN_BYTES": ("mapped", "fwd.sbytes (IP-layer bytes, initiator -> responder)"),
    "IN_PKTS": ("mapped", "fwd.spkts"),
    "OUT_BYTES": ("mapped", "rev.sbytes; 0 when no reverse record exists"),
    "OUT_PKTS": ("mapped", "rev.spkts; 0 when no reverse record exists"),
    "TCP_FLAGS": ("missing", "Argus flgs/state empty; no TCP in lab v1 either"),
    "CLIENT_TCP_FLAGS": ("missing", "as TCP_FLAGS"),
    "SERVER_TCP_FLAGS": ("missing", "as TCP_FLAGS"),
    "FLOW_DURATION_MILLISECONDS": ("mapped", "(FLOW_END - FLOW_START)"),
    "DURATION_IN": ("mapped", "fwd.dur * 1000"),
    "DURATION_OUT": ("mapped", "rev.dur * 1000; 0 when no reverse record exists"),
    "MIN_TTL": ("missing", "sttl/dttl empty (Argus-only field)"),
    "MAX_TTL": ("missing", "sttl/dttl empty (Argus-only field)"),
    "LONGEST_FLOW_PKT": ("missing", "nfdump gives totals only; smaxsz in the lab CSV is the mean size"),
    "SHORTEST_FLOW_PKT": ("missing", "as LONGEST_FLOW_PKT"),
    "MIN_IP_PKT_LEN": ("missing", "as LONGEST_FLOW_PKT"),
    "MAX_IP_PKT_LEN": ("missing", "as LONGEST_FLOW_PKT"),
    # Phase 4 check (reports/phase4/units.json): BigFlow's SECOND_BYTES is 0 for 50-62% of flows in three
    # sources and matches bytes/s for 0% of flows, so a lab bytes/s value is not the same feature. Left NaN.
    "SRC_TO_DST_SECOND_BYTES": ("missing", "BigFlow's definition is unknown and does not match bytes/s (Phase 4 check); not filled"),
    "DST_TO_SRC_SECOND_BYTES": ("missing", "as SRC_TO_DST_SECOND_BYTES"),
    "RETRANSMITTED_IN_BYTES": ("missing", "retrans fields empty"),
    "RETRANSMITTED_IN_PKTS": ("missing", "retrans fields empty"),
    "RETRANSMITTED_OUT_BYTES": ("missing", "retrans fields empty"),
    "RETRANSMITTED_OUT_PKTS": ("missing", "retrans fields empty"),
    # Phase 4 check: BigFlow divides by the WHOLE-flow duration (ratio ~1.0 for 70-91% of flows), not one direction's.
    "SRC_TO_DST_AVG_THROUGHPUT": ("derived", "IN_BYTES * 8 / FLOW_DURATION seconds (bit/s, as BigFlow); 0 if duration == 0"),
    "DST_TO_SRC_AVG_THROUGHPUT": ("derived", "OUT_BYTES * 8 / FLOW_DURATION seconds (bit/s, as BigFlow); 0 if duration == 0"),
    "NUM_PKTS_UP_TO_128_BYTES": ("missing", "needs per-packet sizes"),
    "NUM_PKTS_128_TO_256_BYTES": ("missing", "needs per-packet sizes"),
    "NUM_PKTS_256_TO_512_BYTES": ("missing", "needs per-packet sizes"),
    "NUM_PKTS_512_TO_1024_BYTES": ("missing", "needs per-packet sizes"),
    "NUM_PKTS_1024_TO_1514_BYTES": ("missing", "needs per-packet sizes"),
    "TCP_WIN_MAX_IN": ("missing", "swin empty"),
    "TCP_WIN_MAX_OUT": ("missing", "dwin empty"),
    "ICMP_TYPE": ("mapped", "type*256 + code of the initiator, from nfdump dstPort 'type.code'; 0 for non-ICMP"),
    "ICMP_IPV4_TYPE": ("mapped", "ICMP type of the initiator; 0 for non-ICMP"),
    "DNS_QUERY_ID": ("missing", "needs DNS payload parsing (also dropped as a feature)"),
    "DNS_QUERY_TYPE": ("missing", "needs DNS payload parsing"),
    "DNS_TTL_ANSWER": ("missing", "needs DNS payload parsing"),
    "FTP_COMMAND_RET_CODE": ("missing", "needs FTP payload parsing"),
    "SRC_TO_DST_IAT_MIN": ("missing", "sintpkt* empty (Argus-only)"),
    "SRC_TO_DST_IAT_MAX": ("missing", "sintpkt* empty (Argus-only)"),
    "SRC_TO_DST_IAT_AVG": ("missing", "sintpkt* empty (Argus-only)"),
    "SRC_TO_DST_IAT_STDDEV": ("missing", "sintpkt* empty (Argus-only)"),
    "DST_TO_SRC_IAT_MIN": ("missing", "dintpkt* empty (Argus-only)"),
    "DST_TO_SRC_IAT_MAX": ("missing", "dintpkt* empty (Argus-only)"),
    "DST_TO_SRC_IAT_AVG": ("missing", "dintpkt* empty (Argus-only)"),
    "DST_TO_SRC_IAT_STDDEV": ("missing", "dintpkt* empty (Argus-only)"),
}

assert set(FEATURE_MAP) == set(BIGFLOW_FEATURES), set(FEATURE_MAP) ^ set(BIGFLOW_FEATURES)

LAB_META_COLS = ["session_id", "capture_interface", "label", "attack_cat", "attack_type",
                 "attack_name", "tool", "label_method", "is_synthetic", "paired",
                 "row_id", "source_row"]


def available_features() -> list[str]:
    """BigFlow features the lab data can actually fill (mapped or derived)."""
    return [f for f in BIGFLOW_FEATURES if FEATURE_MAP[f][0] != "missing"]


def missing_features() -> list[str]:
    return [f for f in BIGFLOW_FEATURES if FEATURE_MAP[f][0] == "missing"]


def _icmp_type_code(row) -> tuple[int, int]:
    """nfdump writes ICMP as dstPort 'type.code' (e.g. '8.0'); fall back to the Argus dport."""
    raw = row.get("nfdump_dstPort")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or raw == "":
        raw = row.get("dport")
    s = str(raw)
    if s in ("", "nan", "None"):
        return 0, 0
    t, _, c = s.partition(".")
    return int(float(t)), int(float(c or 0))


def _per_second(num: pd.Series, dur_s: pd.Series) -> pd.Series:
    return pd.Series(np.where(dur_s > 0, num / dur_s.where(dur_s > 0, 1), 0.0), index=num.index)


def map_lab_to_bigflow(biflows: pd.DataFrame) -> pd.DataFrame:
    """Turn ``pair_biflows`` output into BigFlow-shaped rows (+ lab label/meta columns)."""
    b = biflows
    out = pd.DataFrame(index=b.index, columns=BIGFLOW_FEATURES, dtype="float64")
    portless = b.proto.isin(PORTLESS_PROTOS)
    rev_dur = b.rev_dur.fillna(0.0)

    out["FLOW_START_MILLISECONDS"] = (b.stime * 1000).round()
    out["FLOW_END_MILLISECONDS"] = (b.ltime * 1000).round()
    out["FLOW_DURATION_MILLISECONDS"] = out.FLOW_END_MILLISECONDS - out.FLOW_START_MILLISECONDS
    out["IPV4_SRC_ADDR"] = b.saddr.values
    out["IPV4_DST_ADDR"] = b.daddr.values
    out["L4_SRC_PORT"] = np.where(portless, 0, pd.to_numeric(b.sport, errors="coerce"))
    out["L4_DST_PORT"] = np.where(portless, 0, pd.to_numeric(b.dport, errors="coerce"))
    unknown = set(b.proto) - set(PROTO_NUMBERS)
    if unknown:
        raise ValueError(f"no IANA number for protocol(s) {unknown}")
    out["PROTOCOL"] = b.proto.map(PROTO_NUMBERS)

    out["IN_BYTES"], out["IN_PKTS"] = b.fwd_bytes, b.fwd_pkts
    out["OUT_BYTES"], out["OUT_PKTS"] = b.rev_bytes.fillna(0), b.rev_pkts.fillna(0)
    out["DURATION_IN"] = b.fwd_dur * 1000
    out["DURATION_OUT"] = rev_dur * 1000
    flow_s = out.FLOW_DURATION_MILLISECONDS / 1000
    out["SRC_TO_DST_AVG_THROUGHPUT"] = _per_second(b.fwd_bytes * 8, flow_s)
    out["DST_TO_SRC_AVG_THROUGHPUT"] = _per_second(out.OUT_BYTES * 8, flow_s)

    tc = b.apply(lambda r: _icmp_type_code(r) if r.proto == "icmp" else (0, 0), axis=1)
    out["ICMP_IPV4_TYPE"] = [t for t, _ in tc]
    out["ICMP_TYPE"] = [t * 256 + c for t, c in tc]

    out = out.astype({c: "object" for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")})
    for c in LAB_META_COLS:
        if c in b:
            out[c] = b[c].values
    out["Label"] = b.label.astype(int).values
    out["Attack"] = b.attack_cat.values if "attack_cat" in b else np.nan
    return out


def lab_to_bigflow(path, include_synthetic: bool = False, prefer_interface: str = "ens33") -> pd.DataFrame:
    """Convenience: load a lab CSV, remove vantage duplicates, pair directions and map."""
    from .lab import dedupe_vantage, load_lab_flows, pair_biflows
    flows = load_lab_flows(path, include_synthetic=include_synthetic)
    return map_lab_to_bigflow(pair_biflows(dedupe_vantage(flows, prefer=prefer_interface)))
