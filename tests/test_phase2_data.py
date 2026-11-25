import pandas as pd

from sentinelai.data.prepare import select_hosts
from sentinelai.data.split import make_splits


def test_temporal_split_inside_strata(labeled_df):
    s = make_splits(labeled_df, by="temporal", group_col="FLOW_START_MILLISECONDS",
                    temporal_strata=["source", "Attack"])
    for cls, g in labeled_df.groupby("Attack"):
        tr, te = s.train[s.train.Attack == cls], s.test[s.test.Attack == cls]
        assert len(tr) and len(te), cls                       # no class is test-only
        assert tr.FLOW_START_MILLISECONDS.max() <= te.FLOW_START_MILLISECONDS.min()


def test_select_hosts_keeps_attack_hosts_whole(labeled_df):
    df = labeled_df.assign(IPV4_SRC_ADDR=[f"h{i % 40}" for i in range(len(labeled_df))])
    df.loc[df.IPV4_SRC_ADDR == "h3", "Label"] = 1
    out = select_hosts(df, max_rows=400, seed=0)
    for h in set(df.loc[df.Label == 1, "IPV4_SRC_ADDR"]):
        assert (out.IPV4_SRC_ADDR == h).sum() == (df.IPV4_SRC_ADDR == h).sum()
    kept = set(out.IPV4_SRC_ADDR)
    for h in kept:                                          # whole hosts only
        assert (out.IPV4_SRC_ADDR == h).sum() == (df.IPV4_SRC_ADDR == h).sum()


def test_cap_time_window_is_contiguous_and_keeps_attacks():
    from sentinelai.data.prepare import cap_time_window
    df = pd.DataFrame({"FLOW_START_MILLISECONDS": range(1000), "Label": [0] * 600 + [1] * 50 + [0] * 350})
    out = cap_time_window(df, 200)
    t = out.FLOW_START_MILLISECONDS
    assert len(out) == 200 and t.max() - t.min() == 199          # one unbroken stretch
    assert out.Label.sum() == 50                                   # centred on the attacks
    assert cap_time_window(df, 5000) is df
