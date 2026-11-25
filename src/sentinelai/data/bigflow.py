"""Loaders for the BigFlow-NIDS dataset (CSV or Parquet, single file or folder).

The full release is ~66.9M rows, so nothing here reads the whole dataset into
memory unless asked to:

* ``iter_bigflow``   yields DataFrame chunks with only the requested columns.
* ``sample_bigflow`` builds a class-stratified uniform sample in one pass
                     (bottom-k sampling per class, seeded and reproducible).
* ``load_bigflow``   reads a small file or the first ``nrows`` rows.

The dataset is downloaded manually into ``data/external/`` (git-ignored).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .schema import ATTACK_COL, validate_bigflow_columns

CSV_SUFFIXES = {".csv", ".gz", ".zip"}
PARQUET_SUFFIXES = {".parquet", ".pq"}


def list_files(path: str | Path) -> list[Path]:
    """Return the data files at ``path`` (a file, or every CSV/Parquet file under a folder)."""
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"BigFlow path not found: {p}")
    files = sorted(f for f in p.rglob("*") if f.is_file()
                   and (f.suffix.lower() in PARQUET_SUFFIXES or f.suffix.lower() in CSV_SUFFIXES))
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet files under {p}")
    return files


def _is_parquet(f: Path) -> bool:
    return f.suffix.lower() in PARQUET_SUFFIXES


def read_header(f: Path) -> list[str]:
    if _is_parquet(f):
        return list(pq.ParquetFile(f).schema_arrow.names)
    return list(pd.read_csv(f, nrows=0).columns)


def iter_bigflow(path: str | Path, columns: Sequence[str] | None = None, chunksize: int = 500_000,
                 validate: bool = True) -> Iterator[pd.DataFrame]:
    """Yield chunks of at most ``chunksize`` rows from every file under ``path``.

    ``columns`` limits what is read (column projection is free for Parquet).
    With ``validate`` each file's header is checked against the expected schema
    before it is read.
    """
    for f in list_files(path):
        if validate:
            validate_bigflow_columns(read_header(f))
        cols = list(columns) if columns is not None else None
        if _is_parquet(f):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=chunksize, columns=cols):
                yield batch.to_pandas()
        else:
            yield from pd.read_csv(f, usecols=cols, chunksize=chunksize, low_memory=False)


def load_bigflow(path: str | Path, nrows: int | None = None, columns: Sequence[str] | None = None,
                 validate: bool = True) -> pd.DataFrame:
    """Read rows into one DataFrame. Use ``nrows`` (or a small file) for quick runs."""
    parts, n = [], 0
    for chunk in iter_bigflow(path, columns=columns, validate=validate,
                              chunksize=min(nrows, 500_000) if nrows else 500_000):
        if nrows is not None:
            chunk = chunk.iloc[: nrows - n]
        parts.append(chunk)
        n += len(chunk)
        if nrows is not None and n >= nrows:
            break
    if not parts:
        raise ValueError(f"No rows read from {path}")
    return pd.concat(parts, ignore_index=True)


def sample_bigflow(path: str | Path, per_class_cap: int, class_col: str = ATTACK_COL, seed: int = 42,
                   columns: Sequence[str] | None = None, chunksize: int = 500_000,
                   validate: bool = True) -> pd.DataFrame:
    """Class-stratified sample: up to ``per_class_cap`` rows per value of ``class_col``.

    One pass, bounded memory. Every row gets a random key and, per class, the
    ``per_class_cap`` rows with the smallest keys are kept. That is a uniform
    sample without replacement within each class, so rare classes are kept in
    full and large classes are capped. The result is unchanged for a given seed
    and chunk size.
    """
    if per_class_cap <= 0:
        raise ValueError("per_class_cap must be positive")
    rng = np.random.default_rng(seed)
    kept: dict[object, pd.DataFrame] = {}
    for chunk in iter_bigflow(path, columns=columns, chunksize=chunksize, validate=validate):
        if class_col not in chunk:
            raise KeyError(f"class column {class_col!r} not in data")
        chunk = chunk.assign(_key=rng.random(len(chunk)))
        for cls, g in chunk.groupby(class_col, sort=False):
            merged = g if cls not in kept else pd.concat([kept[cls], g])
            kept[cls] = merged.nsmallest(per_class_cap, "_key")
    out = pd.concat(kept.values()).sort_values("_key").drop(columns="_key")
    return out.reset_index(drop=True)
