"""Score the saved Phase 2 models once and cache the predictions.

Each cached table has one row per flow with the true labels, the source,
the time (for block bootstrap), the split part (val / test / holdout) and the
model's raw scores. Cached under data/interim/phase3/preds/ (git-ignored).

Stage 1 tables store the tactic the tactic layer would give **every** flow
(``tactic_any``), so a different alert threshold can be applied later
without re-running the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.prepare import ROOT
from ..models.common import load_model
from ..models.sequence import block_split
from ..models.train import load_sequences

META = ["Label", "Attack", "source", "tactic", "FLOW_START_MILLISECONDS"]
PARTS = ("val", "test", "holdout")


def preds_dir(cfg: dict) -> Path:
    return ROOT / cfg["phase3"]["interim_dir"] / "preds"


def load_flows(cfg: dict, split: str, parts=("train", "val", "test", "holdout"), columns=None) -> dict:
    base = ROOT / cfg["data"]["interim_dir"] / "flows"
    return {p: pd.read_parquet(base / f"{split}_{p}.parquet", columns=columns) for p in parts}


def stage1_table(model, flows: dict) -> pd.DataFrame:
    out = []
    for part in PARTS:
        df = flows[part]
        X = model.transform(df)
        proba = model.tactic_.predict_proba(X)
        t = df[META].copy()
        t["part"] = part
        t["p_attack"] = model.binary_.predict_proba(X)[:, 1]
        t["tactic_any"] = model.tactic_enc_.inverse_transform(proba.argmax(1))
        t["tactic_conf_any"] = proba.max(1)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def stage3_table(model, flows: dict) -> pd.DataFrame:
    out = []
    for part in PARTS:
        t = flows[part][META].copy()
        t["part"] = part
        t["anomaly_score"] = model.score(flows[part])
        out.append(t)
    return pd.concat(out, ignore_index=True)


def stage2_table(model, cfg: dict, seqs: pd.DataFrame) -> pd.DataFrame:
    """Re-create the Phase 2 block split (same seed) and score val + test."""
    c = cfg["stage2"]
    parts = block_split(seqs, cfg["data"]["val_size"], cfg["data"]["test_size"], c.get("block_minutes", 5), cfg["seed"])
    out = []
    for part in ("val", "test"):
        df = parts[part]
        p = model.predict(df)
        t = df.loc[p.index, ["Label", "Attack", "tactic", "slice", "seq_key", "FLOW_START_MILLISECONDS"]].copy()
        t["block"] = t.seq_key.str.rsplit("|", n=1).str[0]
        t["part"] = part
        t["p_attack"], t["tactic_pred"] = p.p_attack, p.tactic_pred
        out.append(t.drop(columns="seq_key"))
    return pd.concat(out, ignore_index=True)


def score_all(cfg: dict, log=print, force: bool = False) -> dict[str, Path]:
    """Score every saved model in the config. Existing caches are reused unless ``force``."""
    d = preds_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    written = {}
    for split in ("temporal", "stratified"):
        flows = None
        for fset in cfg["feature_sets"]:
            jobs = [("stage1", f"{a}_{split}_{fset}") for a in cfg["stage1"]["models"]]
            if split == "temporal":
                jobs.append(("stage3", f"autoencoder_{split}_{fset}"))
            for stage, name in jobs:
                path = d / f"{stage}_{name}.parquet"
                written[f"{stage}_{name}"] = path
                if path.exists() and not force:
                    continue
                flows = flows or load_flows(cfg, split, PARTS)
                log(f"scoring {stage} {name} ...")
                m, _ = load_model(stage, name)
                (stage1_table if stage == "stage1" else stage3_table)(m, flows).to_parquet(path)
    seqs = None
    for fset in cfg["feature_sets"]:
        name = f"{cfg['stage2']['cell']}_w{cfg['stage2']['window']}_{fset}"
        path = d / f"stage2_{name}.parquet"
        written[f"stage2_{name}"] = path
        if path.exists() and not force:
            continue
        seqs = load_sequences(cfg) if seqs is None else seqs
        log(f"scoring stage2 {name} ...")
        m, _ = load_model("stage2", name)
        stage2_table(m, cfg, seqs).to_parquet(path)
    return written


def tactic_at(t: pd.DataFrame, flag: np.ndarray) -> np.ndarray:
    """End-to-end tactic for a Stage 1 table at a given alert flag."""
    return np.where(np.asarray(flag).astype(bool), t.tactic_any.to_numpy(object), "Benign")
