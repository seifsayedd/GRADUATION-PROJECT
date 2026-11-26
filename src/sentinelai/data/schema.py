"""Column schema for BigFlow-NIDS (NetFlow-v3 feature set) and the SentinelAI lab data.

BigFlow-NIDS merges NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3 and
NF-CSE-CIC-IDS2018-v3. The names below are the NetFlow-v3 feature names used
by those sources: 53 flow attributes + ``Label`` + ``Attack`` = 55 columns,
which matches the 55 the dataset page reports.

ASSUMPTION (unverified): BigFlow keeps these names after its "feature
harmonization" step. The Mendeley page does not list the columns, so
``validate_bigflow_columns`` checks every file's real header at load time and
reports any difference instead of silently guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- identifiers: used for ordering, grouping and joining, never as model features
IDENTIFIER_COLS: list[str] = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
]

# --- categorical features (codes, not magnitudes)
CATEGORICAL_COLS: list[str] = [
    "PROTOCOL",
    "L7_PROTO",
    "ICMP_TYPE",        # type*256 + code
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE",
    "FTP_COMMAND_RET_CODE",
]

# --- ports: bucketed into service classes by the preprocessor
PORT_COLS: list[str] = ["L4_SRC_PORT", "L4_DST_PORT"]

# --- TCP flag bitmasks: expanded into one column per flag by the preprocessor
TCP_FLAG_COLS: list[str] = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]

# --- dropped: a random transaction id carries no behavioural signal and is a shortcut risk
DROP_COLS: list[str] = ["DNS_QUERY_ID"]

# --- numeric counters / statistics (heavy-tailed; log1p + standardise)
NUMERIC_COLS: list[str] = [
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "DNS_TTL_ANSWER",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]

BIGFLOW_FEATURES: list[str] = (
    IDENTIFIER_COLS + PORT_COLS + CATEGORICAL_COLS + TCP_FLAG_COLS + DROP_COLS + NUMERIC_COLS
)

LABEL_COL = "Label"     # 0 benign / 1 attack
ATTACK_COL = "Attack"   # attack category name ("Benign" for benign flows)
LABEL_COLS: list[str] = [LABEL_COL, ATTACK_COL]

# Columns that may exist in the harmonised release but are not NetFlow features.
OPTIONAL_META_COLS: list[str] = ["Dataset", "source", "SOURCE_DATASET"]

assert len(BIGFLOW_FEATURES) == 53, len(BIGFLOW_FEATURES)
assert len(set(BIGFLOW_FEATURES)) == 53


@dataclass
class SchemaReport:
    missing: list[str] = field(default_factory=list)   # expected but not in the file
    extra: list[str] = field(default_factory=list)     # in the file but not expected
    meta: list[str] = field(default_factory=list)      # known optional metadata columns

    @property
    def ok(self) -> bool:
        return not self.missing

    def __str__(self) -> str:
        return (f"missing={self.missing or '-'} extra={self.extra or '-'} "
                f"meta={self.meta or '-'}")


class SchemaError(ValueError):
    """Raised when a file does not have the columns the code expects."""


def validate_bigflow_columns(columns, *, require_labels: bool = True, strict: bool = True) -> SchemaReport:
    """Compare a file header with the expected BigFlow schema.

    ``strict`` raises ``SchemaError`` when expected columns are missing. Extra
    columns never raise; they are reported so the mapping can be updated.
    """
    cols = list(columns)
    expected = BIGFLOW_FEATURES + (LABEL_COLS if require_labels else [])
    rep = SchemaReport(
        missing=[c for c in expected if c not in cols],
        extra=[c for c in cols if c not in expected and c not in OPTIONAL_META_COLS],
        meta=[c for c in cols if c in OPTIONAL_META_COLS],
    )
    if strict and not rep.ok:
        raise SchemaError(f"BigFlow header does not match the expected NetFlow-v3 schema: {rep}")
    return rep
