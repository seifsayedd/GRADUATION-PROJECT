"""Tests on the REAL lab files (read-only)."""
import pandas as pd
import pytest

from sentinelai.data.lab import dedupe_vantage, load_lab_flows, pair_biflows


def test_full_file_has_68_real_flows(lab_full_path):
    d = load_lab_flows(lab_full_path)
    assert len(d) == 68
    assert d.label.value_counts().to_dict() == {0: 64, 1: 4}
    assert (d.is_synthetic == 0).all()
    assert d.capture_interface.isin(["ens33", "ens37"]).all()


def test_augmented_file_drops_synthetic_by_default(lab_aug_path):
    d = load_lab_flows(lab_aug_path)
    assert len(d) == 68 and d.is_synthetic.sum() == 0
    allrows = load_lab_flows(lab_aug_path, include_synthetic=True)
    assert allrows.is_synthetic.sum() == 160


def test_empty_fields_are_nan_not_zero(lab_full_path):
    d = load_lab_flows(lab_full_path)
    assert d.sttl.isna().all()          # Argus-only field, not measured


def test_dedupe_removes_cross_interface_copies(lab_full_path):
    d = load_lab_flows(lab_full_path)
    u = dedupe_vantage(d)
    assert len(u) == 60
    # gateway mDNS on the target segment is only on ens37 and must survive
    assert ((u.capture_interface == "ens37") & (u.saddr == "192.168.60.1")).any()
    # each S10 attack ping direction is kept exactly once
    s10 = u[u.session_id == "S10"]
    assert len(s10) == 2 and set(s10.capture_interface) == {"ens33"}


def test_pairing_builds_two_way_conversations(lab_full_path):
    b = pair_biflows(dedupe_vantage(load_lab_flows(lab_full_path)))
    assert len(b) == 54
    assert b.paired.sum() == 6
    assert b.label.value_counts().to_dict() == {0: 53, 1: 1}
    atk = b[b.label == 1].iloc[0]
    # initiator is the Kali echo request, response comes back from the target
    assert atk.saddr == "192.168.50.20" and atk.daddr == "192.168.60.10"
    assert (atk.fwd_pkts, atk.rev_pkts, atk.fwd_bytes, atk.rev_bytes) == (4, 4, 336, 336)


def test_unpaired_multicast_has_zero_reverse(lab_full_path):
    b = pair_biflows(dedupe_vantage(load_lab_flows(lab_full_path)))
    mc = b[b.daddr == "224.0.0.251"]
    assert len(mc) and (~mc.paired).all() and (mc.rev_pkts == 0).all()


def test_pairing_rejects_label_mismatch(lab_full_path):
    d = dedupe_vantage(load_lab_flows(lab_full_path))
    i = d.index[(d.session_id == "S10") & (d.saddr == "192.168.60.10")][0]
    d.loc[i, "label"] = 0
    with pytest.raises(ValueError, match="disagree"):
        pair_biflows(d)


def test_synthetic_copies_pair_only_within_their_group(lab_aug_path):
    d = load_lab_flows(lab_aug_path, include_synthetic=True)
    b = pair_biflows(dedupe_vantage(d))
    syn = b[b.is_synthetic == 1]
    # every synthetic conversation points at a real record
    assert syn.source_row.notna().all()
    assert set(syn.source_row) <= set(d.loc[d.is_synthetic == 0, "row_id"])
    # real conversations are unchanged by the presence of synthetic rows
    real = b[b.is_synthetic == 0]
    assert len(real) == 54


def test_loader_never_writes(lab_full_path):
    before = lab_full_path.stat().st_mtime_ns
    load_lab_flows(lab_full_path)
    assert lab_full_path.stat().st_mtime_ns == before
