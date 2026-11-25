"""Map BigFlow's 32 attack categories to MITRE ATT&CK tactics.

The table is a judgement made by reading each source dataset's description
of the class; it is not provided by BigFlow. Every entry carries a
confidence and a reason, and docs/attack-taxonomy.md is generated from
this table (``python -m sentinelai.data.taxonomy``) so the two stay in sync.

Design decision (approved for Phase 2): ATT&CK "Reconnaissance" (TA0043,
pre-compromise) and "Discovery" (TA0007, post-compromise) are merged into
one class. On the wire a scan looks the same either way, and the lab's
labelled attack (a ping, T1016.001) is a Discovery technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BENIGN = "Benign"
RECON = "Reconnaissance/Discovery"
CRED = "Credential Access"
INITIAL = "Initial Access"
IMPACT = "Impact"
C2 = "Command and Control"
EXFIL = "Exfiltration"
OTHER = "Other"

TACTICS = [BENIGN, RECON, CRED, INITIAL, IMPACT, C2, EXFIL, OTHER]


@dataclass(frozen=True)
class Mapping:
    tactic: str
    technique: str      # most representative ATT&CK technique ID, "" if none fits
    confidence: str     # high / medium / low
    source: str         # dataset(s) the class comes from
    reason: str


TAXONOMY: dict[str, Mapping] = {
    "Benign": Mapping(BENIGN, "", "high", "all", "normal traffic"),
    # --- reconnaissance / discovery
    "Reconnaissance": Mapping(RECON, "T1046", "high", "UNSW-NB15, BoT-IoT", "service/OS scanning and probing"),
    "scanning": Mapping(RECON, "T1046", "high", "ToN-IoT", "network/port scanning (nmap, nessus)"),
    "Analysis": Mapping(RECON, "T1595", "medium", "UNSW-NB15",
                        "UNSW groups port scans, spam and HTML-file probing under 'Analysis'"),
    "Fuzzers": Mapping(RECON, "T1595.002", "low", "UNSW-NB15",
                       "random-input probing for crashes; closest to vulnerability scanning"),
    "Infilteration": Mapping(RECON, "T1046", "low", "CSE-CIC-IDS2018",
                             "after a malicious download the victim scans the internal network; the class "
                             "mixes delivery and scanning and is known to be noisily labelled (Liu et al., 2022)"),
    # --- credential access
    "password": Mapping(CRED, "T1110", "high", "ToN-IoT", "password guessing / cracking"),
    "FTP-BruteForce": Mapping(CRED, "T1110.001", "high", "CSE-CIC-IDS2018", "FTP brute force (Patator)"),
    "SSH-Bruteforce": Mapping(CRED, "T1110.001", "high", "CSE-CIC-IDS2018", "SSH brute force (Patator)"),
    "Brute_Force_-Web": Mapping(CRED, "T1110", "high", "CSE-CIC-IDS2018", "web login brute force"),
    "mitm": Mapping(CRED, "T1557", "medium", "ToN-IoT",
                    "ARP/DNS man-in-the-middle; ATT&CK lists T1557 under Credential Access and Collection"),
    # --- initial access / exploitation of public-facing services
    "xss": Mapping(INITIAL, "T1190", "medium", "ToN-IoT", "cross-site scripting against web apps"),
    "Brute_Force_-XSS": Mapping(INITIAL, "T1190", "medium", "CSE-CIC-IDS2018",
                                "automated XSS against a web app (name says brute force; behaviour is XSS)"),
    "injection": Mapping(INITIAL, "T1190", "high", "ToN-IoT", "SQL/code injection against web apps"),
    "SQL_Injection": Mapping(INITIAL, "T1190", "high", "CSE-CIC-IDS2018", "SQL injection"),
    "Exploits": Mapping(INITIAL, "T1190", "medium", "UNSW-NB15", "exploitation of known vulnerabilities"),
    "Shellcode": Mapping(INITIAL, "T1203", "low", "UNSW-NB15",
                         "shellcode payload delivery; could also be read as Execution"),
    "Worms": Mapping(INITIAL, "T1210", "low", "UNSW-NB15",
                     "self-replicating exploitation; T1210 is Lateral Movement, grouped here as exploitation"),
    # --- impact (denial of service, ransomware)
    "DoS": Mapping(IMPACT, "T1499", "high", "UNSW-NB15, BoT-IoT, ToN-IoT", "denial of service"),
    "DDoS": Mapping(IMPACT, "T1498", "high", "BoT-IoT, ToN-IoT", "distributed denial of service"),
    "DoS_attacks-GoldenEye": Mapping(IMPACT, "T1499.002", "high", "CSE-CIC-IDS2018", "HTTP keep-alive DoS"),
    "DoS_attacks-Slowloris": Mapping(IMPACT, "T1499.002", "high", "CSE-CIC-IDS2018", "slow HTTP header DoS"),
    "DoS_attacks-SlowHTTPTest": Mapping(IMPACT, "T1499.002", "high", "CSE-CIC-IDS2018", "slow HTTP DoS"),
    "DoS_attacks-Hulk": Mapping(IMPACT, "T1499.002", "high", "CSE-CIC-IDS2018", "HTTP flood DoS"),
    "DDoS_attacks-LOIC-HTTP": Mapping(IMPACT, "T1498", "high", "CSE-CIC-IDS2018", "LOIC HTTP flood"),
    "DDOS_attack-LOIC-UDP": Mapping(IMPACT, "T1498.001", "high", "CSE-CIC-IDS2018", "LOIC UDP flood"),
    "DDOS_attack-HOIC": Mapping(IMPACT, "T1498", "high", "CSE-CIC-IDS2018", "HOIC HTTP flood"),
    "ransomware": Mapping(IMPACT, "T1486", "medium", "ToN-IoT", "ransomware activity; network side only"),
    # --- command and control
    "Bot": Mapping(C2, "T1071.001", "high", "CSE-CIC-IDS2018", "Ares botnet: periodic HTTP C2 beaconing"),
    "Backdoor": Mapping(C2, "T1219", "medium", "UNSW-NB15, ToN-IoT", "remote access through a backdoor"),
    # --- exfiltration
    "Theft": Mapping(EXFIL, "T1041", "medium", "BoT-IoT", "keylogging and data theft sent out of the network"),
    # --- unmappable
    "Generic": Mapping(OTHER, "", "low", "UNSW-NB15",
                       "UNSW 'Generic' (attacks against block ciphers) has no clean ATT&CK tactic"),
}


class UnknownAttackError(KeyError):
    pass


def to_tactic(attack: pd.Series) -> pd.Series:
    """Map BigFlow ``Attack`` names to tactic names; unknown names raise."""
    unknown = set(attack.unique()) - set(TAXONOMY)
    if unknown:
        raise UnknownAttackError(f"no ATT&CK mapping for {sorted(unknown)}; add them to TAXONOMY")
    return attack.map({k: v.tactic for k, v in TAXONOMY.items()}).rename("tactic")


def lab_tactic(attack_cat: pd.Series) -> pd.Series:
    """Map lab ``attack_cat`` (ATT&CK tactic names) onto the model's tactic classes."""
    m = {"Benign": BENIGN, "Discovery": RECON, "Reconnaissance": RECON}
    out = attack_cat.map(lambda v: m.get(v, v))
    unknown = set(out.unique()) - set(TACTICS)
    if unknown:
        raise UnknownAttackError(f"lab tactics not in model classes: {sorted(unknown)}")
    return out.rename("tactic")


def markdown_table() -> str:
    lines = ["| BigFlow class | Tactic | Technique | Confidence | Source | Reason |",
             "|---|---|---|---|---|---|"]
    for name, m in TAXONOMY.items():
        lines.append(f"| {name} | {m.tactic} | {m.technique or '—'} | {m.confidence} | {m.source} | {m.reason} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown_table())
