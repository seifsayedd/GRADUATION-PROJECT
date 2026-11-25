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
