import numpy as np
import pandas as pd
import pytest

from sentinelai.data.mapping import available_features, lab_to_bigflow
from sentinelai.data.preprocess import FlowPreprocessor, NotFittedError, clean, port_bucket
from sentinelai.data.schema import IDENTIFIER_COLS


def test_clean_counts_every_fix(bigflow_df):
    d = bigflow_df.copy()
    d.loc[0, "IN_BYTES"] = np.inf
    d.loc[1, "OUT_BYTES"] = -5
    d.loc[2, "IN_PKTS"] = np.nan
    d.loc[3, "Label"] = np.nan
    d = pd.concat([d, d.iloc[[10]]])
    out, rep = clean(d)
    assert rep.dropped_no_label == 1 and rep.dropped_duplicates == 1
    assert rep.inf_replaced == 1 and rep.negatives_clipped == 1
    assert rep.nan_filled == {"IN_BYTES": 1, "IN_PKTS": 1}
    assert len(out) == len(bigflow_df) - 1
    assert out[["IN_BYTES", "OUT_BYTES", "IN_PKTS"]].min().min() >= 0


def test_clean_can_leave_nan(bigflow_df):
    d = bigflow_df.copy()
    d.loc[0, "IN_PKTS"] = np.nan
    out, rep = clean(d, fill_numeric_nan=False)
    assert out.IN_PKTS.isna().sum() == 1 and rep.nan_filled == {}


def test_port_buckets():
    b = port_bucket(pd.Series([0, 22, 1023, 1024, 49151, 49152, 65535]))
    assert b.tolist() == [0, 1, 1, 2, 2, 3, 3]


def test_transform_before_fit_raises(bigflow_df):
    with pytest.raises(NotFittedError):
        FlowPreprocessor().transform(bigflow_df)


def test_no_identifiers_or_labels_in_output(bigflow_df):
    X = FlowPreprocessor().fit_transform(bigflow_df)
    for c in IDENTIFIER_COLS + ["Label", "Attack", "DNS_QUERY_ID", "L4_DST_PORT"]:
        assert c not in X.columns
    assert "TCP_FLAGS__SYN" in X and "L4_DST_PORT__bucket1" in X
    assert np.isfinite(X.to_numpy(dtype=float)).all()


def test_scaler_uses_train_statistics_only(bigflow_df):
    train, test = bigflow_df.iloc[:400], bigflow_df.iloc[400:].copy()
    test["IN_BYTES"] *= 1000
    pp = FlowPreprocessor().fit(train)
    Xtr, Xte = pp.transform(train), pp.transform(test)
    assert abs(Xtr.IN_BYTES.mean()) < 1e-6                    # centred on train
    assert Xte.IN_BYTES.mean() > 3                            # test shift stays visible


def test_unseen_category_goes_to_other(bigflow_df):
    pp = FlowPreprocessor().fit(bigflow_df)
    t = bigflow_df.iloc[:5].copy()
    t["PROTOCOL"] = 99
    X = pp.transform(t)
    assert (X["PROTOCOL__other"] == 1).all()
    assert list(X.columns) == pp.feature_names_


def test_nan_inputs_are_rejected(bigflow_df):
    pp = FlowPreprocessor().fit(bigflow_df)
    t = bigflow_df.iloc[:5].copy()
    t.loc[t.index[0], "IN_BYTES"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pp.transform(t)


def test_lab_compatible_preprocessor(bigflow_df, lab_full_path):
    """Fit on (fixture) BigFlow with only the shared features, then transform real lab flows."""
    pp = FlowPreprocessor(features=available_features()).fit(bigflow_df)
    lab = lab_to_bigflow(lab_full_path)
    X = pp.transform(lab)
    assert len(X) == 54 and list(X.columns) == pp.feature_names_
    assert not any(c.startswith(("TCP_FLAGS", "L7_PROTO", "MIN_TTL")) for c in X.columns)


def test_full_preprocessor_refuses_lab_data(bigflow_df, lab_full_path):
    pp = FlowPreprocessor().fit(bigflow_df)
    with pytest.raises(ValueError, match="NaN"):
        pp.transform(lab_to_bigflow(lab_full_path))


def test_save_and_load(tmp_path, bigflow_df):
    pp = FlowPreprocessor().fit(bigflow_df)
    pp.save(tmp_path / "pp.joblib")
    pp2 = FlowPreprocessor.load(tmp_path / "pp.joblib")
    pd.testing.assert_frame_equal(pp.transform(bigflow_df), pp2.transform(bigflow_df))
