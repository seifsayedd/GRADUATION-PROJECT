"""Recover which benchmark dataset each BigFlow row came from.

BigFlow-NIDS has no source column, but each of its four sources was captured
in a distinct period, so ``FLOW_START_MILLISECONDS`` identifies the source.
The windows below were measured on BigFlow v2 (all 66,935,021 rows):

    UNSW-NB15        2015-01-22 .. 2015-02-18
    CSE-CIC-IDS2018  2018-02-14 .. 2018-03-02   (attack classes); benign rows Feb-Mar 2018
    BoT-IoT          2018-05 .. 2018-06
    ToN-IoT          2019-04-23 .. 2019-04-29

Each window is padded by a few days. A row outside every window gets
``"unknown"``, and ``infer_source(strict=True)`` raises instead, so a
schema or data change can't slip through silently.

Why time and not the class name: BigFlow's harmonisation merged same-named
classes across sources. "DDoS" spans 2018-06 .. 2019-04 because it holds both
BoT-IoT and ToN-IoT DDoS; "DoS", "Backdoor" and "Reconnaissance" are shared
the same way. The class name alone can't tell the sources apart; the
capture period can.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_MS = 86_400_000


def _ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").value // 1_000_000)


# (name, start inclusive, end exclusive) in epoch ms, padded by a few days
SOURCE_WINDOWS: list[tuple[str, int, int]] = [
    ("UNSW-NB15", _ms("2015-01-15"), _ms("2015-03-01")),
    ("CSE-CIC-IDS2018", _ms("2018-02-01"), _ms("2018-04-01")),
    ("BoT-IoT", _ms("2018-05-01"), _ms("2018-07-01")),
    ("ToN-IoT", _ms("2019-04-01"), _ms("2019-05-15")),
]
SOURCES = [s for s, _, _ in SOURCE_WINDOWS]
UNKNOWN = "unknown"


class SourceError(ValueError):
    pass


def infer_source(start_ms: pd.Series, strict: bool = False) -> pd.Series:
    """Map FLOW_START_MILLISECONDS to a source dataset name."""
    t = pd.to_numeric(start_ms, errors="coerce").to_numpy(dtype="float64")
    out = np.full(len(t), UNKNOWN, dtype=object)
    for name, lo, hi in SOURCE_WINDOWS:
        out[(t >= lo) & (t < hi)] = name
    res = pd.Series(out, index=start_ms.index, name="source")
    if strict and (res == UNKNOWN).any():
        bad = start_ms[res == UNKNOWN]
        raise SourceError(f"{len(bad)} rows outside every known source window, e.g. {bad.iloc[:3].tolist()}")
    return res
