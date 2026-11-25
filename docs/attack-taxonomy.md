# Attack taxonomy: BigFlow classes â†’ MITRE ATT&CK tactics

Stage 1's second step predicts an ATT&CK **tactic** for each attack flow. BigFlow labels flows with 32 dataset-specific category names, so each one is mapped to a tactic here. This mapping is **our own judgement**, based on each source dataset's description of the class. BigFlow doesn't provide it, so every row carries a confidence level and a reason.

Source of truth: [`src/sentinelai/data/taxonomy.py`](../src/sentinelai/data/taxonomy.py). The table below is generated from it with `python -m sentinelai.data.taxonomy`, and a test checks the two match.

## Decisions

- **Reconnaissance (TA0043) and Discovery (TA0007) are one class.** A scan looks the same on the network whether it runs before or after a compromise, and the lab's labelled attack (a ping, T1016.001) is a Discovery technique.
- **Low-confidence rows** (`Fuzzers`, `Infilteration`, `Shellcode`, `Worms`, `Generic`) are kept, but per-class results in Phase 3 report them separately. `Infilteration` is also one of the CSE-CIC-IDS2018 classes that Liu et al. (2022, *Error Prevalence in NIDS datasets*) found to be noisily labelled.
- **Same name, different sources.** BigFlow merged same-named classes across sources (`DoS`, `DDoS`, `Backdoor`, `Reconnaissance`). The source is recovered from the capture date instead ([`sources.py`](../src/sentinelai/data/sources.py)).
- **Held-out classes** for the unseen-attack test: `SSH-Bruteforce`, `DoS_attacks-Slowloris` and `Backdoor`. None of their rows are used in training, validation or test.

## Tactic classes

| Tactic | BigFlow classes |
|---|---|
| Benign | Benign |
| Reconnaissance/Discovery | Reconnaissance, scanning, Analysis, Fuzzers, Infilteration |
| Credential Access | password, FTP-BruteForce, SSH-Bruteforce, Brute_Force_-Web, mitm |
| Initial Access | xss, Brute_Force_-XSS, injection, SQL_Injection, Exploits, Shellcode, Worms |
| Impact | DoS, DDoS, DoS_attacks-GoldenEye, DoS_attacks-Slowloris, DoS_attacks-SlowHTTPTest, DoS_attacks-Hulk, DDoS_attacks-LOIC-HTTP, DDOS_attack-LOIC-UDP, DDOS_attack-HOIC, ransomware |
| Command and Control | Bot, Backdoor |
| Exfiltration | Theft |
| Other | Generic |

## Full table

| BigFlow class | Tactic | Technique | Confidence | Source | Reason |
|---|---|---|---|---|---|
| Benign | Benign | — | high | all | normal traffic |
| Reconnaissance | Reconnaissance/Discovery | T1046 | high | UNSW-NB15, BoT-IoT | service/OS scanning and probing |
| scanning | Reconnaissance/Discovery | T1046 | high | ToN-IoT | network/port scanning (nmap, nessus) |
| Analysis | Reconnaissance/Discovery | T1595 | medium | UNSW-NB15 | UNSW groups port scans, spam and HTML-file probing under 'Analysis' |
| Fuzzers | Reconnaissance/Discovery | T1595.002 | low | UNSW-NB15 | random-input probing for crashes; closest to vulnerability scanning |
| Infilteration | Reconnaissance/Discovery | T1046 | low | CSE-CIC-IDS2018 | after a malicious download the victim scans the internal network; the class mixes delivery and scanning and is known to be noisily labelled (Liu et al., 2022) |
| password | Credential Access | T1110 | high | ToN-IoT | password guessing / cracking |
| FTP-BruteForce | Credential Access | T1110.001 | high | CSE-CIC-IDS2018 | FTP brute force (Patator) |
| SSH-Bruteforce | Credential Access | T1110.001 | high | CSE-CIC-IDS2018 | SSH brute force (Patator) |
| Brute_Force_-Web | Credential Access | T1110 | high | CSE-CIC-IDS2018 | web login brute force |
| mitm | Credential Access | T1557 | medium | ToN-IoT | ARP/DNS man-in-the-middle; ATT&CK lists T1557 under Credential Access and Collection |
| xss | Initial Access | T1190 | medium | ToN-IoT | cross-site scripting against web apps |
| Brute_Force_-XSS | Initial Access | T1190 | medium | CSE-CIC-IDS2018 | automated XSS against a web app (name says brute force; behaviour is XSS) |
| injection | Initial Access | T1190 | high | ToN-IoT | SQL/code injection against web apps |
| SQL_Injection | Initial Access | T1190 | high | CSE-CIC-IDS2018 | SQL injection |
| Exploits | Initial Access | T1190 | medium | UNSW-NB15 | exploitation of known vulnerabilities |
| Shellcode | Initial Access | T1203 | low | UNSW-NB15 | shellcode payload delivery; could also be read as Execution |
| Worms | Initial Access | T1210 | low | UNSW-NB15 | self-replicating exploitation; T1210 is Lateral Movement, grouped here as exploitation |
| DoS | Impact | T1499 | high | UNSW-NB15, BoT-IoT, ToN-IoT | denial of service |
| DDoS | Impact | T1498 | high | BoT-IoT, ToN-IoT | distributed denial of service |
| DoS_attacks-GoldenEye | Impact | T1499.002 | high | CSE-CIC-IDS2018 | HTTP keep-alive DoS |
| DoS_attacks-Slowloris | Impact | T1499.002 | high | CSE-CIC-IDS2018 | slow HTTP header DoS |
| DoS_attacks-SlowHTTPTest | Impact | T1499.002 | high | CSE-CIC-IDS2018 | slow HTTP DoS |
| DoS_attacks-Hulk | Impact | T1499.002 | high | CSE-CIC-IDS2018 | HTTP flood DoS |
| DDoS_attacks-LOIC-HTTP | Impact | T1498 | high | CSE-CIC-IDS2018 | LOIC HTTP flood |
| DDOS_attack-LOIC-UDP | Impact | T1498.001 | high | CSE-CIC-IDS2018 | LOIC UDP flood |
| DDOS_attack-HOIC | Impact | T1498 | high | CSE-CIC-IDS2018 | HOIC HTTP flood |
| ransomware | Impact | T1486 | medium | ToN-IoT | ransomware activity; network side only |
| Bot | Command and Control | T1071.001 | high | CSE-CIC-IDS2018 | Ares botnet: periodic HTTP C2 beaconing |
| Backdoor | Command and Control | T1219 | medium | UNSW-NB15, ToN-IoT | remote access through a backdoor |
| Theft | Exfiltration | T1041 | medium | BoT-IoT | keylogging and data theft sent out of the network |
| Generic | Other | — | low | UNSW-NB15 | UNSW 'Generic' (attacks against block ciphers) has no clean ATT&CK tactic |

## Known gaps

- **No lateral-movement class.** `Worms` uses T1210 but is grouped as exploitation. Lateral movement has to come from future lab captures (SMB/WinRM/SSH between lab hosts).
- **Exfiltration has only 1,615 rows** (`Theft`). Expect weak per-class results there, and report them as such.
- **Lab labels use ATT&CK tactic names directly** (`attack_cat`). `taxonomy.lab_tactic()` maps `Discovery` onto the merged class.
