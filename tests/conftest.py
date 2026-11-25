"""Shared fixtures.

The small BigFlow-shaped tables built here are UNIT-TEST FIXTURES ONLY: random
numbers in the right schema, used to exercise the code paths. They are never
used to train, evaluate or report anything.

The lab fixtures point at the real files in data/processed/, which tests only read.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinelai.data.schema import BIGFLOW_FEATURES, CATEGORICAL_COLS, NUMERIC_COLS, TCP_FLAG_COLS

ROOT = Path(__file__).resolve().parents[1]
LAB_FULL = ROOT / "data" / "processed" / "combined_sessions_full.csv"
LAB_AUG = ROOT / "data" / "processed" / "combined_sessions_augmented.csv"


@pytest.fixture(scope="session")
def lab_full_path() -> Path:
    assert LAB_FULL.exists(), LAB_FULL
    return LAB_FULL


@pytest.fixture(scope="session")
def lab_aug_path() -> Path:
    assert LAB_AUG.exists(), LAB_AUG
    return LAB_AUG


def make_bigflow_fixture(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Random rows in the BigFlow schema (test fixture, not data)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))
    start = 1_700_000_000_000 + np.sort(rng.integers(0, 10_000_000, n))
    df["FLOW_START_MILLISECONDS"] = start
    df["FLOW_END_MILLISECONDS"] = start + rng.integers(0, 5000, n)
    df["IPV4_SRC_ADDR"] = [f"10.0.{i % 7}.{i % 250}" for i in range(n)]
    df["IPV4_DST_ADDR"] = [f"10.1.{i % 5}.{i % 200}" for i in range(n)]
    df["L4_SRC_PORT"] = rng.integers(1024, 65535, n)
    df["L4_DST_PORT"] = rng.choice([22, 53, 80, 443, 3389, 8080, 50000], n)
    for c in CATEGORICAL_COLS:
        df[c] = rng.choice([0, 1, 6, 17], n)
    for c in TCP_FLAG_COLS:
        df[c] = rng.integers(0, 256, n)
    df["DNS_QUERY_ID"] = rng.integers(0, 65535, n)
    for c in NUMERIC_COLS:
        df[c] = rng.exponential(1000, n).round()
    df = df[BIGFLOW_FEATURES]
    df["Attack"] = rng.choice(["Benign"] * 6 + ["DDoS", "scanning", "Brute Force", "Exploits"], n)
    df["Label"] = (df.Attack != "Benign").astype(int)
    return df


@pytest.fixture
def bigflow_df() -> pd.DataFrame:
    return make_bigflow_fixture()


@pytest.fixture
def bigflow_files(tmp_path, bigflow_df):
    """The same fixture written as CSV and as two Parquet parts."""
    csv = tmp_path / "csv"
    pq_dir = tmp_path / "parquet"
    csv.mkdir()
    pq_dir.mkdir()
    bigflow_df.to_csv(csv / "bigflow.csv", index=False)
    half = len(bigflow_df) // 2
    bigflow_df.iloc[:half].to_parquet(pq_dir / "part-0.parquet", index=False)
    bigflow_df.iloc[half:].to_parquet(pq_dir / "part-1.parquet", index=False)
    return {"csv": csv / "bigflow.csv", "csv_dir": csv, "parquet_dir": pq_dir}


def make_labeled_fixture(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """BigFlow-shaped TEST FIXTURE with tactic/source columns and a planted signal
    (attack rows send more bytes), so tiny models have something to learn.
    Not data; never used for any reported number."""
    from sentinelai.data.taxonomy import TAXONOMY
    rng = np.random.default_rng(seed)
    df = make_bigflow_fixture(n, seed)
    attacks = ["Benign"] * 6 + ["DDoS", "scanning", "password", "Bot"]
    df["Attack"] = rng.choice(attacks, n)
    df["Label"] = (df.Attack != "Benign").astype(int)
    df.loc[df.Label == 1, "IN_BYTES"] *= 50
    df.loc[df.Attack == "scanning", "OUT_PKTS"] = 0
    df["tactic"] = df.Attack.map({k: v.tactic for k, v in TAXONOMY.items()})
    df["source"] = "ToN-IoT"
    return df


@pytest.fixture
def labeled_df() -> pd.DataFrame:
    return make_labeled_fixture()
