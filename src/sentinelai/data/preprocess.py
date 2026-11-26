"""Cleaning, encoding and scaling for BigFlow-shaped flow tables.

Order of operations (enforced by the API):

1. ``clean(df)``             row-level fixes; safe before splitting (no statistics learned)
2. ``split.make_splits``     train / val / test (+ held-out attacks)
3. ``split.resample_train``  optional; train only
4. ``FlowPreprocessor.fit(train)`` then ``.transform(val/test/lab)``

``FlowPreprocessor`` learns encoders and scalers from the training split only,
so validation and test statistics never leak into training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .schema import (CATEGORICAL_COLS, DROP_COLS, IDENTIFIER_COLS, LABEL_COLS, NUMERIC_COLS,
                     PORT_COLS, TCP_FLAG_COLS)

TCP_FLAG_BITS = {"FIN": 1, "SYN": 2, "RST": 4, "PSH": 8, "ACK": 16, "URG": 32, "ECE": 64, "CWR": 128}


@dataclass
class CleanReport:
    rows_in: int = 0
    dropped_no_label: int = 0
    dropped_duplicates: int = 0
    inf_replaced: int = 0
    negatives_clipped: int = 0
    nan_filled: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"rows_in={self.rows_in} dropped_no_label={self.dropped_no_label} "
                f"dropped_duplicates={self.dropped_duplicates} inf={self.inf_replaced} "
                f"neg_clipped={self.negatives_clipped} nan_filled={self.nan_filled or '-'}")


def clean(df: pd.DataFrame, fill_numeric_nan: bool = True, drop_duplicates: bool = True,
          label_col: str = "Label") -> tuple[pd.DataFrame, CleanReport]:
    """Row-level cleaning with no learned statistics, so it is safe before the split.

    * rows without a label are dropped
    * +/-inf becomes NaN; negative counters are clipped to 0
    * with ``fill_numeric_nan`` NaN counters become 0 (in NetFlow-v3, an absent
      counter means "none observed"). Every fill is counted in the report.
      Do NOT use this on lab data, where NaN means "not measured": pass
      ``fill_numeric_nan=False`` and train on ``mapping.available_features()``.
    * exact duplicate rows are dropped
    """
    rep = CleanReport(rows_in=len(df))
    d = df.copy()
    if label_col in d:
        before = len(d)
        d = d[d[label_col].notna()]
        rep.dropped_no_label = before - len(d)
    num = [c for c in NUMERIC_COLS + PORT_COLS + TCP_FLAG_COLS + CATEGORICAL_COLS if c in d]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    vals = d[num].to_numpy(dtype="float64", na_value=np.nan)
    rep.inf_replaced = int(np.isinf(vals).sum())
    d[num] = d[num].replace([np.inf, -np.inf], np.nan)
    counters = [c for c in NUMERIC_COLS if c in d]
    rep.negatives_clipped = int((d[counters] < 0).sum().sum())
    d[counters] = d[counters].clip(lower=0)
    if fill_numeric_nan:
        n = d[num].isna().sum()
        rep.nan_filled = {c: int(v) for c, v in n.items() if v}
        d[num] = d[num].fillna(0)
    if drop_duplicates:
        before = len(d)
        d = d.drop_duplicates()
        rep.dropped_duplicates = before - len(d)
    return d.reset_index(drop=True), rep


def port_bucket(port: pd.Series) -> pd.Series:
    """IANA ranges: 0 = none/unknown, 1 = well-known (1-1023), 2 = registered (1024-49151), 3 = dynamic."""
    p = pd.to_numeric(port, errors="coerce").fillna(0)
    return pd.Series(np.select([p <= 0, p < 1024, p < 49152], [0, 1, 2], 3), index=port.index)


class NotFittedError(RuntimeError):
    pass


class FlowPreprocessor:
    """Turns a BigFlow-shaped table into a numeric model matrix.

    * identifiers (IPs, timestamps), labels and ``DROP_COLS`` never reach the output
    * ports become service-range buckets (one-hot), not raw numbers, to limit
      shortcut learning on dataset-specific port choices
    * TCP flag bitmasks are split into one 0/1 column per flag (readable in SHAP)
    * categorical codes are one-hot encoded; only the ``max_categories`` most
      frequent training values get a column, the rest go to ``__other``
    * numeric counters get ``log1p`` then ``StandardScaler``

    ``features`` limits the input columns, e.g. ``mapping.available_features()``
    for the lab-compatible model. NaN in a selected column at transform time is
    an error: missing measurements must be excluded, not imputed silently.
    """

    def __init__(self, features: list[str] | None = None, max_categories: int = 20):
        self.features = features
        self.max_categories = max_categories
        self.fitted_ = False

    # -- helpers
    def _select(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        allowed = set(self.features) if self.features is not None else None
        return [c for c in cols if c in df and (allowed is None or c in allowed)]

    def fit(self, train: pd.DataFrame) -> "FlowPreprocessor":
        self.numeric_ = self._select(train, NUMERIC_COLS)
        self.ports_ = self._select(train, PORT_COLS)
        self.flags_ = self._select(train, TCP_FLAG_COLS)
        self.categorical_ = self._select(train, CATEGORICAL_COLS)
        self._check_nan(train)
        self.categories_ = {}
        for c in self.categorical_:
            counts = pd.to_numeric(train[c], errors="coerce").value_counts()
            self.categories_[c] = list(counts.index[: self.max_categories])
        self.scaler_ = StandardScaler()
        if self.numeric_:
            self.scaler_.fit(self._log(train))
        self.fitted_ = True
        self.feature_names_ = list(self._build(train.iloc[:0]).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise NotFittedError("call fit() on the training split before transform()")
        missing = [c for c in self.numeric_ + self.ports_ + self.flags_ + self.categorical_ if c not in df]
        if missing:
            raise KeyError(f"columns seen in training are missing: {missing}")
        self._check_nan(df)
        out = self._build(df)
        return out[self.feature_names_]

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    def _check_nan(self, df: pd.DataFrame) -> None:
        cols = self.numeric_ + self.ports_ + self.flags_ + self.categorical_
        bad = [c for c in cols if pd.to_numeric(df[c], errors="coerce").isna().any()]
        if bad:
            raise ValueError(f"NaN in model inputs {bad}; drop these features or clean() the data first")

    def _log(self, df: pd.DataFrame) -> np.ndarray:
        return np.log1p(df[self.numeric_].to_numpy(dtype="float64").clip(min=0))

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.numeric_:
            x = self.scaler_.transform(self._log(df)) if len(df) else np.empty((0, len(self.numeric_)))
            parts.append(pd.DataFrame(x, columns=self.numeric_, index=df.index))
        for c in self.ports_:
            b = port_bucket(df[c])
            parts.append(pd.DataFrame({f"{c}__bucket{k}": (b == k).astype("int8") for k in range(4)},
                                      index=df.index))
        for c in self.flags_:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
            parts.append(pd.DataFrame({f"{c}__{n}": ((v & bit) > 0).astype("int8")
                                       for n, bit in TCP_FLAG_BITS.items()}, index=df.index))
        for c in self.categorical_:
            v = pd.to_numeric(df[c], errors="coerce")
            known = self.categories_[c]
            cols = {f"{c}__{int(k) if float(k).is_integer() else k}": (v == k).astype("int8") for k in known}
            cols[f"{c}__other"] = (~v.isin(known)).astype("int8")
            parts.append(pd.DataFrame(cols, index=df.index))
        if not parts:
            raise ValueError("no usable feature columns selected")
        return pd.concat(parts, axis=1)

    # -- persistence
    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FlowPreprocessor":
        return joblib.load(path)


NON_FEATURE_COLS = set(IDENTIFIER_COLS) | set(LABEL_COLS) | set(DROP_COLS)
