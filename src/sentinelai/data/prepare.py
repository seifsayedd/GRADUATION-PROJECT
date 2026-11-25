"""Build the Phase 2 training datasets from BigFlow (read-only) into data/interim/.

Two products:

* **flow dataset** (Stages 1 and 3): a class-capped sample, cleaned, with
  ``source`` and ``tactic`` added, then split twice:
    - ``stratified``  random rows, stratified on ``Attack`` (comparable to published work)
    - ``temporal``    earliest 70% / next 15% / latest 15% inside every
                      (source, Attack) group, so near-identical neighbouring
                      flows can't sit on both sides of the split
  Held-out classes go to ``holdout`` in both.
* **sequence dataset** (Stage 2): contiguous days, all flows of the kept
  hosts, in time order (see ``configs/phase2.yaml`` ``stage2.slices``).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

from .bigflow import sample_bigflow
from .preprocess import clean
from .sources import infer_source
from .split import make_splits
from .taxonomy import to_tactic

ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path = ROOT / "configs" / "phase2.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - git missing
        return "unknown"


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``source`` (strict: unknown periods raise) and ``tactic``."""
    return df.assign(source=infer_source(df.FLOW_START_MILLISECONDS, strict=True).values,
                     tactic=to_tactic(df.Attack).values)


def build_flow_dataset(cfg: dict, out_dir: Path | None = None, bigflow_dir: Path | None = None) -> dict:
    d = cfg["data"]
    out = Path(out_dir or ROOT / d["interim_dir"]) / "flows"
    out.mkdir(parents=True, exist_ok=True)
    src = Path(bigflow_dir or ROOT / d["bigflow_dir"])

    raw = sample_bigflow(src, d["per_class_cap"], class_caps=d.get("class_caps"), seed=cfg["seed"])
    df, rep = clean(raw)
    df = annotate(df)
    df.insert(0, "flow_uid", np.arange(len(df), dtype="int64"))

    meta = {"git_commit": git_commit(), "seed": cfg["seed"], "rows": len(df), "clean": vars(rep),
            "class_counts": df.Attack.value_counts().to_dict(),
            "source_counts": df.source.value_counts().to_dict(),
            "holdout_attacks": d["holdout_attacks"], "splits": {}}
    common = dict(label_col="Attack", val_size=d["val_size"], test_size=d["test_size"],
                  holdout_attacks=d["holdout_attacks"], seed=cfg["seed"])
    for mode, kw in {"stratified": {"by": "stratified"},
                     "temporal": {"by": "temporal", "group_col": "FLOW_START_MILLISECONDS",
                                  "temporal_strata": ["source", "Attack"]}}.items():
        s = make_splits(df, **common, **kw)
        for part in ("train", "val", "test", "holdout"):
            getattr(s, part).to_parquet(out / f"{mode}_{part}.parquet", index=False)
        meta["splits"][mode] = s.sizes()
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def _ms(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").value // 1_000_000)


def select_hosts(df: pd.DataFrame, max_rows: int, seed: int, host_col: str = "IPV4_SRC_ADDR") -> pd.DataFrame:
    """Keep every host that sent an attack flow; add random benign-only hosts up to ``max_rows``.

    Whole hosts are kept or dropped so per-host sequences stay intact. This
    raises the share of attack hosts compared with the raw day, which is
    recorded in the dataset metadata.
    """
    attack_hosts = set(df.loc[df.Label == 1, host_col])
    keep = df[host_col].isin(attack_hosts)
    budget = max_rows - int(keep.sum())
    if budget > 0:
        sizes = df.loc[~keep].groupby(host_col).size()
        order = sizes.sample(frac=1.0, random_state=seed)
        chosen = order[order.cumsum() <= budget].index
        keep |= df[host_col].isin(set(chosen))
    return df[keep]


def build_sequence_dataset(cfg: dict, out_dir: Path | None = None, bigflow_dir: Path | None = None) -> dict:
    d, s2 = cfg["data"], cfg["stage2"]
    out = Path(out_dir or ROOT / d["interim_dir"]) / "sequences"
    out.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(Path(bigflow_dir or ROOT / d["bigflow_dir"]), format="parquet")
    meta = {"git_commit": git_commit(), "seed": cfg["seed"], "slices": {}}
    for sl in s2["slices"]:
        lo, hi = _ms(sl["start"]), _ms(sl["end"])
        t = dataset.to_table(filter=(ds.field("FLOW_START_MILLISECONDS") >= lo)
                             & (ds.field("FLOW_START_MILLISECONDS") < hi))
        df = t.to_pandas()
        held = df.Attack.isin(d["holdout_attacks"])
        if held.any():
            raise ValueError(f"slice {sl['name']} contains held-out classes "
                             f"{sorted(df.Attack[held].unique())}; pick another day")
        n_raw = len(df)
        df = select_hosts(df, s2["max_rows_per_slice"], cfg["seed"])
        df, rep = clean(df, drop_duplicates=False)   # duplicates are real repeated flows in a sequence
        df = annotate(df).assign(slice=sl["name"])
        df = df.sort_values(["IPV4_SRC_ADDR", "FLOW_START_MILLISECONDS"], kind="stable")
        df.to_parquet(out / f"{sl['name']}.parquet", index=False)
        meta["slices"][sl["name"]] = {"rows_raw_day": n_raw, "rows_kept": len(df),
                                      "hosts": int(df.IPV4_SRC_ADDR.nunique()),
                                      "class_counts": df.Attack.value_counts().to_dict(), "clean": vars(rep)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta
