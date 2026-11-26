#!/usr/bin/env python3
"""Build a class-stratified BigFlow-NIDS sample for training.

Reads the full dataset in chunks (it never fits in memory), keeps up to
--per-class rows of every attack category, cleans the rows and writes one
Parquet file. Start with a small --per-class to test the pipeline.

Usage:
    python scripts/build_bigflow_sample.py data/external/bigflow data/interim/bigflow_sample.parquet --per-class 50000
    python scripts/build_bigflow_sample.py data/external/bigflow/part-0.csv out.parquet --per-class 1000 --check-only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelai.data.bigflow import list_files, read_header, sample_bigflow  # noqa: E402
from sentinelai.data.preprocess import clean  # noqa: E402
from sentinelai.data.schema import validate_bigflow_columns  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="BigFlow file or folder (CSV / Parquet)")
    ap.add_argument("out", help="output .parquet path (use data/interim/, which is git-ignored)")
    ap.add_argument("--per-class", type=int, default=50_000)
    ap.add_argument("--class-col", default="Attack")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--check-only", action="store_true", help="only validate file headers")
    a = ap.parse_args()

    files = list_files(a.src)
    print(f"{len(files)} file(s)")
    bad = 0
    for f in files:
        rep = validate_bigflow_columns(read_header(f), strict=False)
        print(f"  {f.name}: {'OK' if rep.ok else 'MISMATCH'}  {rep}")
        bad += not rep.ok
    if bad or a.check_only:
        return 1 if bad else 0

    df = sample_bigflow(a.src, a.per_class, class_col=a.class_col, seed=a.seed, chunksize=a.chunksize)
    df, rep = clean(df)
    print("clean:", rep)
    print("class counts:", df[a.class_col].value_counts().to_dict())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
