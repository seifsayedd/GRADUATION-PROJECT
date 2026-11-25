import pandas as pd
import pytest

from sentinelai.data.bigflow import iter_bigflow, list_files, load_bigflow, sample_bigflow
from sentinelai.data.schema import BIGFLOW_FEATURES, SchemaError, validate_bigflow_columns


def test_schema_has_55_columns_with_labels():
    assert len(BIGFLOW_FEATURES) == 53
    assert validate_bigflow_columns(BIGFLOW_FEATURES + ["Label", "Attack"]).ok


def test_schema_reports_missing_and_extra():
    cols = [c for c in BIGFLOW_FEATURES if c != "L7_PROTO"] + ["Label", "Attack", "NEW_COL", "Dataset"]
    with pytest.raises(SchemaError, match="L7_PROTO"):
        validate_bigflow_columns(cols)
    rep = validate_bigflow_columns(cols, strict=False)
    assert rep.missing == ["L7_PROTO"] and rep.extra == ["NEW_COL"] and rep.meta == ["Dataset"]


def test_csv_and_parquet_read_the_same_rows(bigflow_files, bigflow_df):
    a = load_bigflow(bigflow_files["csv"])
    b = load_bigflow(bigflow_files["parquet_dir"])
    assert len(a) == len(b) == len(bigflow_df)
    pd.testing.assert_series_equal(a.IN_BYTES, b.IN_BYTES, check_dtype=False)


def test_chunks_respect_size_and_columns(bigflow_files):
    chunks = list(iter_bigflow(bigflow_files["csv"], columns=["IN_BYTES", "Attack"], chunksize=100))
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) == 600
    assert list(chunks[0].columns) == ["IN_BYTES", "Attack"]


def test_nrows(bigflow_files):
    assert len(load_bigflow(bigflow_files["parquet_dir"], nrows=250)) == 250


def test_bad_header_fails_before_reading(tmp_path, bigflow_df):
    bigflow_df.drop(columns=["IN_BYTES"]).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(SchemaError):
        load_bigflow(tmp_path / "x.csv")


def test_list_files_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_files(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        list_files(tmp_path)


def test_stratified_sample_caps_each_class(bigflow_files, bigflow_df):
    s = sample_bigflow(bigflow_files["parquet_dir"], per_class_cap=40, chunksize=128, seed=1)
    counts = s.Attack.value_counts()
    full = bigflow_df.Attack.value_counts()
    for cls, n in counts.items():
        assert n == min(40, full[cls])
    assert set(counts.index) == set(full.index)


def test_sample_keeps_small_classes_whole(bigflow_files, bigflow_df):
    s = sample_bigflow(bigflow_files["csv"], per_class_cap=10_000)
    assert len(s) == len(bigflow_df)


def test_sample_is_reproducible(bigflow_files):
    a = sample_bigflow(bigflow_files["csv"], per_class_cap=30, seed=7, chunksize=200)
    b = sample_bigflow(bigflow_files["csv"], per_class_cap=30, seed=7, chunksize=200)
    pd.testing.assert_frame_equal(a, b)
