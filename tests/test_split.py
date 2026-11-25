import pandas as pd
import pytest

from sentinelai.data.lab import SyntheticLeakError, load_lab_flows
from sentinelai.data.split import Splits, assert_no_synthetic, make_splits, resample_train


def _ids(f):
    return set(f.index_id)


@pytest.fixture
def df(bigflow_df):
    return bigflow_df.assign(index_id=range(len(bigflow_df)), group=[i // 20 for i in range(len(bigflow_df))])


def test_splits_are_disjoint_and_complete(df):
    s = make_splits(df, seed=0)
    parts = [_ids(s.train), _ids(s.val), _ids(s.test)]
    assert sum(map(len, parts)) == len(df)
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def test_stratified_keeps_class_balance(df):
    s = make_splits(df, seed=0)
    for part in (s.val, s.test):
        assert set(part.Attack) == set(df.Attack)


def test_group_split_never_shares_groups(df):
    s = make_splits(df, by="group", group_col="group", seed=0)
    g = [set(f.group) for f in (s.train, s.val, s.test)]
    assert not (g[0] & g[1] or g[0] & g[2] or g[1] & g[2])


def test_temporal_split_orders_time(df):
    s = make_splits(df, by="temporal", group_col="FLOW_START_MILLISECONDS")
    assert s.train.FLOW_START_MILLISECONDS.max() <= s.val.FLOW_START_MILLISECONDS.min()
    assert s.val.FLOW_START_MILLISECONDS.max() <= s.test.FLOW_START_MILLISECONDS.min()


def test_holdout_attacks_never_in_train(df):
    s = make_splits(df, holdout_attacks=["scanning"])
    assert set(s.holdout.Attack) == {"scanning"}
    for f in (s.train, s.val, s.test):
        assert "scanning" not in set(f.Attack)


def test_bad_arguments(df):
    with pytest.raises(ValueError):
        make_splits(df, by="group")
    with pytest.raises(ValueError):
        make_splits(df, val_size=0.6, test_size=0.5)


def test_resample_touches_train_only(df):
    s = make_splits(df, seed=0)
    r = resample_train(s, strategy="undersample")
    assert r.resampled and r.train.Attack.value_counts().nunique() == 1
    pd.testing.assert_frame_equal(r.val, s.val)
    pd.testing.assert_frame_equal(r.test, s.test)
    o = resample_train(s, strategy="oversample")
    assert o.train.Attack.value_counts().nunique() == 1


def test_resample_requires_splits(df):
    with pytest.raises(TypeError):
        resample_train(df)


def test_assert_no_synthetic_fires():
    with pytest.raises(SyntheticLeakError):
        assert_no_synthetic(pd.DataFrame({"is_synthetic": [0, 1]}))
    assert_no_synthetic(pd.DataFrame({"is_synthetic": [0, 0]}), pd.DataFrame({"x": [1]}))


def test_real_augmented_file_keeps_synthetic_out_of_eval(lab_aug_path):
    """On the real augmented file: synthetic rows land in train only, and only
    when their source row is in train (no jittered copies of test rows)."""
    d = load_lab_flows(lab_aug_path, include_synthetic=True)
    s = make_splits(d, by="group", group_col="session_id", label_col="label",
                    attack_col="attack_cat", val_size=0.2, test_size=0.2, seed=3)
    assert_no_synthetic(s.val, s.test, s.holdout)
    syn = s.train[s.train.is_synthetic == 1]
    real_train = set(s.train.loc[s.train.is_synthetic == 0, "row_id"])
    assert set(syn.source_row) <= real_train
    eval_ids = set(s.val.row_id) | set(s.test.row_id)
    assert not (set(syn.source_row) & eval_ids)


def test_synthetic_without_lineage_is_refused():
    d = pd.DataFrame({"Attack": ["a", "b"] * 10, "is_synthetic": [0] * 18 + [1, 1]})
    with pytest.raises(ValueError, match="source_row"):
        make_splits(d)
