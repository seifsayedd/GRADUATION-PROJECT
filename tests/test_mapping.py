import re
from pathlib import Path

import numpy as np

from sentinelai.data.mapping import (FEATURE_MAP, available_features, lab_to_bigflow,
                                     missing_features)
from sentinelai.data.schema import BIGFLOW_FEATURES

DOC = Path(__file__).resolve().parents[1] / "docs" / "feature-mapping.md"


def test_every_feature_is_accounted_for():
    assert set(FEATURE_MAP) == set(BIGFLOW_FEATURES)
    assert len(available_features()) + len(missing_features()) == 53


def test_doc_table_matches_code():
    rows = re.findall(r"^\| ([A-Z0-9_]+) \| (mapped|derived|missing) \|", DOC.read_text(encoding="utf-8"), re.M)
    doc = dict(rows)
    assert set(doc) == set(BIGFLOW_FEATURES)
    assert doc == {k: v[0] for k, v in FEATURE_MAP.items()}


def test_mapped_frame_shape_and_labels(lab_full_path):
    m = lab_to_bigflow(lab_full_path)
    assert len(m) == 54
    assert all(c in m for c in BIGFLOW_FEATURES + ["Label", "Attack"])
    assert m.Label.sum() == 1
    assert set(m.Attack) == {"Benign", "Discovery"}


def test_missing_features_are_nan_not_zero(lab_full_path):
    m = lab_to_bigflow(lab_full_path)
    for c in missing_features():
        assert m[c].isna().all(), c
    for c in available_features():
        assert m[c].notna().all(), c


def test_attack_ping_values(lab_full_path):
    m = lab_to_bigflow(lab_full_path)
    r = m[m.Label == 1].iloc[0]
    assert r.PROTOCOL == 1
    assert r.ICMP_IPV4_TYPE == 8 and r.ICMP_TYPE == 8 * 256
    assert (r.IN_PKTS, r.OUT_PKTS, r.IN_BYTES, r.OUT_BYTES) == (4, 4, 336, 336)
    assert r.L4_SRC_PORT == 0 and r.L4_DST_PORT == 0
    assert r.FLOW_DURATION_MILLISECONDS == 3029
    # throughput equals Argus sload for the forward direction: 336*8 / 3.029 s
    assert np.isclose(r.SRC_TO_DST_AVG_THROUGHPUT, 336 * 8 / 3.029)


def test_udp_ports_and_zero_duration(lab_full_path):
    m = lab_to_bigflow(lab_full_path)
    mdns = m[(m.PROTOCOL == 17) & (m.L4_DST_PORT == 5353) & (m.FLOW_DURATION_MILLISECONDS == 0)]
    assert len(mdns)
    assert (mdns.SRC_TO_DST_AVG_THROUGHPUT == 0).all()   # defined as 0, not inf
    assert (mdns.ICMP_TYPE == 0).all()


def test_rate_features_follow_bigflow_definitions(lab_full_path):
    """Phase 4 check: BigFlow throughput = bytes * 8 / whole-flow seconds; SECOND_BYTES cannot be reproduced."""
    m = lab_to_bigflow(lab_full_path)
    assert m.SRC_TO_DST_SECOND_BYTES.isna().all() and m.DST_TO_SRC_SECOND_BYTES.isna().all()
    assert "SRC_TO_DST_SECOND_BYTES" not in available_features()
    s = m.FLOW_DURATION_MILLISECONDS / 1000
    ok = s > 0
    assert ok.any()
    assert np.allclose(m.SRC_TO_DST_AVG_THROUGHPUT[ok], (m.IN_BYTES * 8 / s)[ok])
    assert np.allclose(m.DST_TO_SRC_AVG_THROUGHPUT[ok], (m.OUT_BYTES * 8 / s)[ok])
    assert (m.SRC_TO_DST_AVG_THROUGHPUT[~ok] == 0).all()
