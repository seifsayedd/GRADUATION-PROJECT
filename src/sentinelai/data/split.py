"""Train / validation / test splitting, done before any resampling.

Rules this module enforces:

* **Split first, resample second.** ``resample_train`` only accepts the
  ``Splits`` object and only touches ``train``.
* **Synthetic rows never reach validation or test.** Splitting uses real rows
  only. A synthetic row joins ``train`` only if the real row it was derived
  from (``source_row``) is in ``train``; otherwise it is dropped, because a
  jittered copy of a test row in training is leakage.
* **Held-out attacks.** ``holdout_attacks`` removes those categories from
  train/val/test and returns them as ``holdout``. That is the generalisation
  test for unseen attack types (Phase 3).
* **Group splits** keep every row of a group (e.g. a lab session or a source
  dataset) on one side, which is what ai-team-data-requirements §7.4 asks for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .lab import SyntheticLeakError


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    holdout: pd.DataFrame
    resampled: bool = False

    def sizes(self) -> dict[str, int]:
        return {k: len(getattr(self, k)) for k in ("train", "val", "test", "holdout")}


def _synthetic_mask(df: pd.DataFrame) -> pd.Series:
    if "is_synthetic" not in df:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df.is_synthetic, errors="coerce").fillna(0).astype(int) == 1


def assert_no_synthetic(*frames: pd.DataFrame) -> None:
    """Raise ``SyntheticLeakError`` if any frame contains a synthetic row."""
    for i, f in enumerate(frames):
        n = int(_synthetic_mask(f).sum())
        if n:
            raise SyntheticLeakError(f"frame {i} contains {n} synthetic rows; synthetic data is train-only")


def _two_way(df, test_size, by, label_col, group_col, seed):
    if len(df) == 0 or test_size == 0:
        return df, df.iloc[0:0]
    if by == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        a, b = next(gss.split(df, groups=df[group_col]))
        return df.iloc[a], df.iloc[b]
    if by == "temporal":
        d = df.sort_values(group_col, kind="stable")
        cut = int(round(len(d) * (1 - test_size)))
        return d.iloc[:cut], d.iloc[cut:]
    strat = df[label_col] if df[label_col].value_counts().min() >= 2 else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)


def make_splits(df: pd.DataFrame, *, by: str = "stratified", label_col: str = "Attack",
                group_col: str | None = None, val_size: float = 0.15, test_size: float = 0.15,
                holdout_attacks: list[str] | None = None, attack_col: str = "Attack",
                id_col: str = "row_id", source_col: str = "source_row", seed: int = 42) -> Splits:
    """Split ``df`` into train/val/test (+ holdout) before any resampling.

    ``by``:
      * ``"stratified"`` random rows, stratified on ``label_col`` when every class has >= 2 rows
      * ``"group"``      no group (``group_col``, e.g. ``session_id``) spans two splits
      * ``"temporal"``   ordered by ``group_col`` (e.g. ``FLOW_START_MILLISECONDS``):
                         train is earliest, test is latest
    ``val_size`` and ``test_size`` are fractions of the non-holdout real rows.
    """
    if by not in ("stratified", "group", "temporal"):
        raise ValueError(f"unknown split mode {by!r}")
    if by in ("group", "temporal") and not group_col:
        raise ValueError(f"by={by!r} needs group_col")
    if not 0 <= val_size < 1 or not 0 <= test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be fractions that sum to < 1")

    syn = _synthetic_mask(df)
    real, synth = df[~syn], df[syn]

    hold_mask = real[attack_col].isin(holdout_attacks) if holdout_attacks else pd.Series(False, index=real.index)
    holdout, pool = real[hold_mask], real[~hold_mask]

    rest, test = _two_way(pool, test_size, by, label_col, group_col, seed)
    rel_val = val_size / (1 - test_size) if test_size < 1 else 0
    train, val = _two_way(rest, rel_val, by, label_col, group_col, seed + 1)

    if len(synth):
        if id_col not in df or source_col not in df:
            raise ValueError(f"synthetic rows need {id_col!r} and {source_col!r} to be split safely")
        ok = synth[source_col].isin(set(train[id_col]))
        if holdout_attacks:
            ok &= ~synth[attack_col].isin(holdout_attacks)
        train = pd.concat([train, synth[ok]])

    out = Splits(*(f.reset_index(drop=True) for f in (train, val, test, holdout)))
    assert_no_synthetic(out.val, out.test, out.holdout)
    return out


def resample_train(splits: Splits, *, label_col: str = "Attack", strategy: str = "undersample",
                   target: int | None = None, seed: int = 42) -> Splits:
    """Balance classes in ``splits.train`` only. val/test/holdout are returned unchanged.

    * ``undersample``: every class capped at ``target`` (default: smallest class size)
    * ``oversample``:  every class sampled with replacement up to ``target``
      (default: largest class size)
    """
    if not isinstance(splits, Splits):
        raise TypeError("resample_train takes the Splits from make_splits, so it can only touch train")
    tr = splits.train
    counts = tr[label_col].value_counts()
    if strategy == "undersample":
        n = target or int(counts.min())
        parts = [g.sample(n=min(len(g), n), random_state=seed) for _, g in tr.groupby(label_col)]
    elif strategy == "oversample":
        n = target or int(counts.max())
        parts = [g.sample(n=n, replace=len(g) < n, random_state=seed) for _, g in tr.groupby(label_col)]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    rng = np.random.default_rng(seed)
    new = pd.concat(parts)
    new = new.iloc[rng.permutation(len(new))].reset_index(drop=True)
    return Splits(new, splits.val, splits.test, splits.holdout, resampled=True)
