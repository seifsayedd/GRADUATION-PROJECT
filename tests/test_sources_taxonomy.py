import re
from pathlib import Path

import pandas as pd
import pytest

from sentinelai.data.sources import SOURCES, SourceError, infer_source
from sentinelai.data.taxonomy import (TACTICS, TAXONOMY, UnknownAttackError, lab_tactic, markdown_table,
                                      to_tactic)

# The 32 class names found in BigFlow-NIDS v2 (66,935,021 rows, checked 2026-11-25).
BIGFLOW_V2_CLASSES = [
    "Benign", "DDoS", "DoS", "xss", "Reconnaissance", "password", "scanning", "DDOS_attack-HOIC",
    "FTP-BruteForce", "injection", "DDoS_attacks-LOIC-HTTP", "Backdoor", "Bot", "SSH-Bruteforce",
    "Infilteration", "DoS_attacks-SlowHTTPTest", "DoS_attacks-Hulk", "DoS_attacks-GoldenEye", "Exploits",
    "DoS_attacks-Slowloris", "Fuzzers", "Generic", "mitm", "ransomware", "DDOS_attack-LOIC-UDP", "Shellcode",
    "Brute_Force_-Web", "Theft", "Analysis", "Brute_Force_-XSS", "SQL_Injection", "Worms",
]
DOC = Path(__file__).resolve().parents[1] / "docs" / "attack-taxonomy.md"


def ms(s):
    return int(pd.Timestamp(s, tz="UTC").value // 1_000_000)


def test_source_windows_match_observed_ranges():
    t = pd.Series([ms("2015-01-22"), ms("2018-02-14"), ms("2018-06-04"), ms("2019-04-29 23:00")])
    assert infer_source(t).tolist() == ["UNSW-NB15", "CSE-CIC-IDS2018", "BoT-IoT", "ToN-IoT"]


def test_source_unknown_and_strict():
    t = pd.Series([ms("2026-11-12")])      # the lab capture date: not a BigFlow source
    assert infer_source(t).tolist() == ["unknown"]
    with pytest.raises(SourceError):
        infer_source(t, strict=True)
    assert len(SOURCES) == 4


def test_taxonomy_covers_all_bigflow_classes():
    assert set(TAXONOMY) == set(BIGFLOW_V2_CLASSES)
    assert {m.tactic for m in TAXONOMY.values()} <= set(TACTICS)
    assert all(m.confidence in {"high", "medium", "low"} and m.reason for m in TAXONOMY.values())


def test_to_tactic_and_unknown():
    s = to_tactic(pd.Series(["Bot", "SSH-Bruteforce", "Benign"]))
    assert s.tolist() == ["Command and Control", "Credential Access", "Benign"]
    with pytest.raises(UnknownAttackError):
        to_tactic(pd.Series(["NewAttack"]))


def test_lab_tactic_merges_discovery():
    assert lab_tactic(pd.Series(["Discovery", "Benign"])).tolist() == ["Reconnaissance/Discovery", "Benign"]
    with pytest.raises(UnknownAttackError):
        lab_tactic(pd.Series(["Lateral Movement"]))


def test_doc_table_is_generated_from_code():
    assert markdown_table() in DOC.read_text(encoding="utf-8")
