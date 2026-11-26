"""Map model columns back to the original BigFlow features.

``FlowPreprocessor`` turns 53 BigFlow features into more columns: port
buckets (``L4_DST_PORT__bucket1``), TCP flag bits (``TCP_FLAGS__SYN``) and
one-hot codes (``PROTOCOL__17``). An analyst should see ``L4_DST_PORT`` or
``TCP_FLAGS``, so contributions are summed back per original feature. Because
SHAP values are additive, the per-feature sums still add up to the model output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.preprocess import TCP_FLAG_BITS, port_bucket
from ..data.schema import CATEGORICAL_COLS, PORT_COLS, TCP_FLAG_COLS

PORT_RANGE = {0: "none", 1: "well-known", 2: "registered", 3: "dynamic"}
PROTOCOLS = {1: "ICMP", 2: "IGMP", 6: "TCP", 17: "UDP", 58: "ICMPv6"}


def source_feature(column: str) -> str:
    """``TCP_FLAGS__SYN`` -> ``TCP_FLAGS``; plain numeric columns map to themselves."""
    return column.split("__", 1)[0]


def group_columns(columns) -> dict[str, list[int]]:
    """Original feature -> positions of its model columns, in first-seen order."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(columns):
        groups.setdefault(source_feature(c), []).append(i)
    return groups


def aggregate(contrib: np.ndarray, columns) -> pd.DataFrame:
    """Sum an (n x columns) contribution matrix into (n x original features)."""
    contrib = np.atleast_2d(np.asarray(contrib, dtype="float64"))
    if contrib.shape[1] != len(columns):
        raise ValueError(f"{contrib.shape[1]} contribution columns for {len(columns)} model columns")
    g = group_columns(columns)
    return pd.DataFrame({f: contrib[:, idx].sum(1) for f, idx in g.items()})


def flags_text(v) -> str:
    v = int(v)
    on = [n for n, bit in TCP_FLAG_BITS.items() if v & bit]
    return f"{'|'.join(on) or 'none'} (0x{v:02x})"


def value_text(feature: str, value) -> str:
    """Human-readable raw value of one original feature."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if feature in TCP_FLAG_COLS:
        return flags_text(value)
    if feature in PORT_COLS:
        b = int(port_bucket(pd.Series([value])).iloc[0])
        return f"{int(value)} ({PORT_RANGE[b]})"
    if feature == "PROTOCOL":
        return f"{int(value)} ({PROTOCOLS.get(int(value), 'other')})"
    if feature in CATEGORICAL_COLS:
        return str(int(value)) if float(value).is_integer() else str(value)
    v = float(value)
    return f"{int(v):,}" if v.is_integer() else f"{v:,.4g}"
