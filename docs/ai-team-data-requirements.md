> **Source:** Prepared by the SentinelAI AI team as the data-collection specification handed to the cybersecurity (data collection) team. Included here as the reference the lab capture work was built against.

# AI Team Requirements for Data Collection


## 1. Purpose

This document lists everything the AI team needs **before data collection starts**, so that the collected data can be used to train and evaluate our models without re-running attacks.

Anything not captured during collection is lost. Please review every section, answer the questions in **Section 12**, and confirm the checklist in **Section 13** before the first official session.

### How we will use the data
We are building a **hierarchical detection model** on network flows (Argus):

| Stage | Task | Output |
|---|---|---|
| 1 | Binary detection | `benign` / `attack` |
| 2 | Category classification | MITRE ATT&CK **tactic** (e.g. Discovery, Lateral Movement, Command and Control) |
| 3 | Type classification | MITRE ATT&CK **technique** (e.g. T1046, T1021.002, T1071.001) |

A future stage may add an **anomaly detector trained on benign traffic only**, so clean benign data is just as important as attack data.

Our goal is for the model to detect **real attacks, not only Caldera**. That is why we ask for attack diversity (Section 7) and clean identifiers for labeling (Sections 5–6).

---

## 2. Top 10 Requests (Summary)

1. **Keep raw PCAP** for every session, alongside Argus files.
2. **Enable extended Argus metrics** in `argus.conf` before the first session (Section 4).
3. **Sync all clocks via NTP; record everything in UTC.**
4. **Export Caldera operation reports and event logs** after every operation.
5. **Maintain a lab inventory** (VM → IP, MAC, OS, role).
6. **Fill in a session log** for every capture session.
7. **Run Caldera abilities sequentially with gaps**, not in parallel (clean labels).
8. **Record benign-only sessions** separately, and add benign traffic during attack sessions.
9. **Repeat each technique across multiple sessions and multiple tools** (Caldera + non-Caldera).
10. **Run a pilot session first** and hand it to the AI team for validation before full collection.

---

## 3. Lab Environment Requirements

| # | Requirement | Why |
|---|---|---|
| 3.1 | The Argus sensor must see **all** lab traffic, including VM-to-VM traffic on the same host (capture on the virtual switch / bridge or a mirror/SPAN port). | Lateral movement between internal VMs is invisible otherwise. |
| 3.2 | **NTP sync** on all VMs, the Caldera server, and the sensor, from the same source. All logs in **UTC**. | Labels are joined to flows by timestamp; a few seconds of drift mislabels short flows. |
| 3.3 | **Static IPs** (or DHCP reservations) for all VMs during collection. | IP → role mapping must stay stable for labeling. |
| 3.4 | Avoid NAT/proxies between attacker and victims inside the capture point, or document them. | NAT hides the real source and breaks labeling. |
| 3.5 | Use at least **2 victim OS types** (e.g. Windows + Linux) and, if possible, more than one attacker VM. | Prevents the model from learning one specific host. |
| 3.6 | Keep a **lab topology diagram** (segments, subnets, sensor position). | Needed for the report and for debugging. |

---

## 4. Capture Requirements

### 4.1 Artifacts per session
| Artifact | Required |
|---|---|
| Raw PCAP (`.pcap` / `.pcapng`) | ✅ |
| Argus binary (`.argus`) | ✅ |
| Argus CSV export (`.csv`, Section 5) | ✅ |
| Caldera operation report + event logs (`.json`) | ✅ (attack sessions) |
| Non-Caldera attack log (tool, command, start/end UTC, source/target) | ✅ (non-Caldera sessions) |
| Session log entry (Section 10.3) | ✅ |

### 4.2 `argus.conf` settings
```ini
ARGUS_MONITOR_ID=sensor01
ARGUS_FLOW_STATUS_INTERVAL=5
ARGUS_GENERATE_RESPONSE_TIME_DATA=yes
ARGUS_GENERATE_PACKET_SIZE=yes
ARGUS_GENERATE_JITTER_DATA=yes
ARGUS_GENERATE_MAC_DATA=yes
ARGUS_GENERATE_APPBYTE_METRIC=yes
ARGUS_GENERATE_TCP_PERF_METRIC=yes
ARGUS_GENERATE_BIDIRECTIONAL_TIMESTAMPS=yes
```
> `ARGUS_FLOW_STATUS_INTERVAL=5` gives a status record every 5 s (default is 60 s). This supports real-time detection and sequence models. We can merge records later with `racluster` if needed.

### 4.3 Do NOT
- ❌ Filter traffic at capture time (no BPF filters that drop protocols/hosts).
- ❌ Delete or overwrite PCAPs after exporting CSVs.
- ❌ Manually edit CSV files (open in Excel and re-save can corrupt timestamps/IPs).
- ❌ Anonymize IPs (or if required, use one consistent mapping and share it with us).

---

## 5. Fields to Export from Argus

### Tier A — Identifiers (used for labeling & splitting only, NOT as model features)
`srcid stime ltime saddr sport daddr dport smac dmac trans`

### Tier B — Core features (required)
`proto state flgs dir pkts spkts dpkts bytes sbytes dbytes dur rate srate drate load sload dload sttl dttl stos dtos loss sloss dloss`

### Tier C — Extended features (strongly requested)(VIP)
`smeansz dmeansz smaxsz dmaxsz sminsz dminsz sintpkt dintpkt sintpktact dintpktact sintpktidl dintpktidl sjit djit sjitact djitact sjitidl djitidl swin dwin stcpb dtcpb tcprtt synack ackdat retrans sretrans dretrans appbytes sappbytes dappbytes pcr idle runtime`

### Export command
```bash
ra -r SESSION.argus -n -u -c , -s \
  srcid stime ltime saddr sport daddr dport smac dmac trans \
  proto state flgs dir \
  pkts spkts dpkts bytes sbytes dbytes \
  dur rate srate drate load sload dload \
  sttl dttl stos dtos loss sloss dloss \
  smeansz dmeansz smaxsz dmaxsz sminsz dminsz \
  sintpkt dintpkt sintpktact dintpktact sintpktidl dintpktidl \
  sjit djit sjitact djitact sjitidl djitidl \
  swin dwin stcpb dtcpb tcprtt synack ackdat \
  retrans sretrans dretrans appbytes sappbytes dappbytes \
  pcr idle runtime \
  > SESSION_flows.csv
```
- `-n` numeric IPs/ports, `-u` Unix epoch timestamps, `-c ,` comma-separated.
- **Validation:** after the pilot, confirm no Tier B/C column is entirely empty. An empty column usually means an `argus.conf` flag is off.

---

## 6. Labeling Requirements

### 6.1 Label schema (final labeled CSV)
| Column | Type | Example | Notes |
|---|---|---|---|
| `session_id` | string | `S03` | Links to session log |
| `label` | int | `0` / `1` | 0 = benign, 1 = attack |
| `attack_cat` | string | `Discovery` | MITRE tactic; `Benign` for benign flows |
| `attack_type` | string | `T1046` | MITRE technique ID; `Benign` for benign flows |
| `attack_name` | string | `Network Service Discovery` | Human-readable |
| `tool` | string | `caldera-sandcat` / `nmap` / `hydra` / `manual` | Needed for leave-one-tool-out evaluation |
| `operation_id` | string | Caldera operation ID | Empty for non-Caldera |
| `ability_id` | string | Caldera ability ID | Empty for non-Caldera |
| `label_method` | string | `exact` / `time_window` / `inferred` | Confidence of the label |

### 6.2 Labeling rules (please confirm or propose changes)
1. A flow is labeled as an attack technique if it is **between the attacker/agent host and the target** and falls **inside the ability's execution window** (start → finish, plus a small tolerance, e.g. ±2 s).
2. Flows during an attack session that do **not** involve attacker/target pairs are **benign**.
3. **Caldera agent C2 beacons** (agent ↔ Caldera server, default HTTP port 8888) are labeled `Command and Control / T1071.001`, **not** benign.
4. **Agent deployment / payload download** (e.g. downloading the Sandcat agent) is labeled `Command and Control / T1105` (Ingress Tool Transfer).
5. If two abilities overlap in time and cannot be separated, mark `label_method = inferred` and record it in the session notes.
6. Failed abilities (status ≠ success) are still labeled with their technique, but recorded in the session log.

### 6.3 Execution practices that make labeling possible
- Run abilities **one at a time** (sequential / atomic planner), with a **30–60 s gap** between abilities.
- Note the **exact UTC start/end** of every non-Caldera attack.
- Do not run unrelated admin activity from the attacker VM during attack windows.

---

## 7. Attack Coverage & Diversity

### 7.1 Prioritize network-visible techniques
Many Caldera abilities run locally on the host and produce almost no network traffic, so our network model cannot detect them. Please prioritize techniques that **generate network traffic**:

| Tactic | Technique | ID | Example tools |
|---|---|---|---|
| Discovery | Network Service Discovery (port scan) | T1046 | Caldera, nmap, masscan |
| Discovery | Remote System Discovery | T1018 | Caldera, nmap ping sweep |
| Discovery | Network Share Discovery | T1135 | Caldera, smbclient, net view |
| Credential Access | Brute Force | T1110 | Hydra, Medusa, Caldera |
| Lateral Movement | Remote Services: SMB / Admin Shares | T1021.002 | Caldera, PsExec, Impacket |
| Lateral Movement | Remote Services: SSH | T1021.004 | Caldera, ssh |
| Lateral Movement | Remote Services: WinRM | T1021.006 | Caldera, evil-winrm |
| Lateral Movement | Lateral Tool Transfer | T1570 | Caldera, scp, SMB copy |
| Command and Control | Application Layer Protocol: Web | T1071.001 | Caldera (HTTP), Metasploit |
| Command and Control | Application Layer Protocol: DNS | T1071.004 | Caldera (DNS contact), dnscat2 |
| Command and Control | Ingress Tool Transfer | T1105 | curl/wget, Caldera |
| Exfiltration | Exfiltration Over C2 Channel | T1041 | Caldera |
| Exfiltration | Exfiltration Over Alternative Protocol | T1048 | Caldera, scp/ftp/curl |
| Initial Access | Exploit Public-Facing Application | T1190 | Metasploit *(if in scope)* |
| Impact | Network / Endpoint DoS | T1498 / T1499 | hping3 *(if in scope)* |

### 7.2 Diversity within Caldera
Please vary these across sessions:
- **Agents:** Sandcat, Manx, Ragdoll (where applicable)
- **C2 contact channels:** HTTP, TCP, UDP, DNS (where configured)
- **Sleep / jitter settings:** e.g. short (5–10 s), medium (30–60 s), long (120 s+)
- **Victim OS:** Windows and Linux
- **Adversary profiles / ability order**

### 7.3 Non-Caldera attacks (high priority)
Please repeat a subset of the same techniques **without Caldera** (nmap, Hydra, Metasploit, Impacket, Atomic Red Team, manual commands). We will train on Caldera data and **test on non-Caldera data** to show the model detects attack behavior, not the Caldera tool.

### 7.4 Repetitions
- Each technique should appear in **at least 5 separate sessions** (different days/configs if possible).
- Target **≥ 1,000 attack flows per technique** where the technique naturally produces that many (scans and brute force will produce far more; C2 and exfiltration fewer, which is fine).

> We split train/test **by session**, not by random rows, to avoid leakage. Techniques that appear in only one session cannot be evaluated fairly.

---

## 8. Benign Traffic Requirements

| # | Requirement |
|---|---|
| 8.1 | **Benign-only sessions** (no Caldera/attacks running, Caldera agents stopped) — kept as a separate, clean subset. |
| 8.2 | **Background benign traffic during attack sessions**, so attacks are mixed with normal activity. |
| 8.3 | Variety: HTTP/HTTPS browsing, DNS, SMB file sharing, SSH/RDP admin sessions, software updates, email, ICMP ping, NTP, file downloads. |
| 8.4 | Generated from **multiple hosts** and at **different times**. Automated scripts are fine, but please vary timing (not perfectly periodic). |
| 8.5 | Benign volume ≥ attack volume overall. |
| 8.6 | Document how benign traffic was generated (scripts/tools used). |

---

## 9. Data Volume & Session Planning (targets)

| Item | Target |
|---|---|
| Benign-only sessions | ≥ 5 sessions, ≥ 1 hour each |
| Caldera attack sessions | ≥ 5 per technique group |
| Non-Caldera attack sessions | ≥ 2 per technique (for the techniques in 7.1 that are in scope) |
| Pilot session | 1 short session (20–30 min) with benign + 2–3 techniques **before** full collection |

---

## 10. Deliverables & Formats

### 10.1 Folder structure
```
dataset/
├── inventory/
│   ├── lab_inventory.csv
│   └── topology.png
├── sessions/
│   └── S03_2026-10-12_lateral-movement/
│       ├── S03.pcap
│       ├── S03.argus
│       ├── S03_flows.csv
│       ├── S03_caldera_report.json
│       ├── S03_caldera_events.json
│       ├── S03_manual_attacks.csv      (non-Caldera only)
│       └── notes.md
├── session_log.csv
└── config/
    ├── argus.conf
    └── ra_fields.txt
```

### 10.2 `lab_inventory.csv`
| hostname | ip | mac | os | role | notes |
|---|---|---|---|---|---|
| kali-01 | 10.0.0.5 | 08:00:27:aa:bb:cc | Kali 2024.x | attacker | |
| caldera-srv | 10.0.0.2 | ... | Ubuntu 22.04 | c2_server | port 8888 |
| win10-01 | 10.0.0.20 | ... | Windows 10 | victim | Sandcat agent |
| ubuntu-01 | 10.0.0.21 | ... | Ubuntu 22.04 | victim / benign_client | |

Roles: `attacker`, `c2_server`, `victim`, `benign_client`, `infra` (DNS, DC, file server…).

### 10.3 `session_log.csv`
| session_id | date | start_utc | end_utc | type | scenario | tools | caldera_operation_id | agents | c2_channel | sleep_jitter | victims | benign_generator | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S01 | 2026-10-11 | 10:00:00 | 11:00:00 | benign | baseline office traffic | — | — | — | — | — | — | script v1 | |
| S03 | 2026-10-12 | 13:00:00 | 13:45:00 | attack | discovery + lateral movement | caldera-sandcat | abc123 | sandcat | HTTP | 10s/20% | win10-01 | script v1 | ability X failed |

### 10.4 `SXX_manual_attacks.csv` (non-Caldera sessions)
| start_utc | end_utc | tool | command | src_ip | dst_ip | dst_port | tactic | technique_id | success | notes |
|---|---|---|---|---|---|---|---|---|---|---|

### 10.5 Sharing
- One shared location for all sessions (to be agreed: Google Drive / NAS / OneDrive).
- Hand over **raw files**. Labeling scripts will be version-controlled in the project repo.

---

## 11. Quality Checks Before Handoff (per session)

- [ ] PCAP, `.argus`, and CSV all present and non-empty
- [ ] CSV has all Tier A/B/C columns, header row present, no fully-empty columns
- [ ] Timestamps are Unix epoch and match the session start/end in the session log (UTC)
- [ ] Caldera report/event JSON exported (attack sessions)
- [ ] Session log row filled in
- [ ] Any anomalies (crashes, failed abilities, network issues) written in `notes.md`

---

## 12. Questions for the Data Collection Team

Please answer in the **Answer** column (or reply in a separate doc referencing the IDs).

### Lab & infrastructure
| ID | Question | Why we need it | Answer |
|---|---|---|---|
| Q1 | What hypervisor/platform is used (VirtualBox, VMware, Proxmox, cloud)? How many VMs, which OSes and roles? | Inventory, topology, diversity | |
| Q2 | Where exactly is the Argus sensor placed? Does it capture VM-to-VM traffic on the same host? | Lateral movement visibility | |
| Q3 | Which versions of `argus` and `argus-clients` are installed? | Field availability differs by version | |
| Q4 | Can PCAPs be retained for all sessions? How much storage is available? | PCAP is our recovery option | |
| Q5 | How is time synchronized (NTP source)? Are all hosts in UTC? | Label accuracy | |
| Q6 | Is there any NAT, proxy, or VPN between hosts inside the lab? | Labeling by IP | |
| Q7 | Does any traffic go to the internet (e.g. external C2, updates)? | Benign variety; external C2 realism | |
| Q8 | Is any traffic IPv6? | Parsing and features | |

### Caldera & attacks
| ID | Question | Why we need it | Answer |
|---|---|---|---|
| Q9 | Which adversary profiles and abilities are planned? Please share the list with **tactic + technique IDs**. | Defines our classes | |
| Q10 | Which agents (Sandcat/Manx/Ragdoll) and C2 contact channels (HTTP/TCP/UDP/DNS) will be used? | Diversity, C2 labeling | |
| Q11 | What sleep/jitter settings will agents use? Can they be varied across sessions? | Avoid timing shortcuts | |
| Q12 | Which planner will be used (atomic/sequential vs batch)? Can abilities run one at a time with 30–60 s gaps? | Clean labels | |
| Q13 | How are agents deployed on victims (initial access method)? | Label deployment traffic (T1105) | |
| Q14 | Which non-Caldera tools can the red team use (nmap, Hydra, Metasploit, Impacket, Atomic Red Team, manual)? For which techniques? | Cross-tool generalization test | |
| Q15 | Are DoS (T1498/T1499) or exploitation (T1190) attacks in scope? | Class list | |
| Q16 | Will C2 traffic be encrypted (HTTPS) in any session? | Feature expectations | |

### Benign traffic
| ID | Question | Why we need it | Answer |
|---|---|---|---|
| Q17 | How will benign traffic be generated (scripts, real user activity, tools)? | Realism, documentation | |
| Q18 | Can benign traffic run in the background during all attack sessions? | Realistic mix | |
| Q19 | How many benign-only hours can be recorded? | Anomaly detector training data | |


## 13. Pre-Collection Sign-Off Checklist

| # | Item | Owner | Done |
|---|---|---|---|
| 1 | NTP synced on all hosts; UTC everywhere | Data team | ☐ |
| 2 | Argus sensor sees all lab traffic (incl. VM-to-VM) | Data team | ☐ |
| 3 | `argus.conf` updated per Section 4.2 | Data team | ☐ |
| 4 | PCAP capture running alongside Argus | Data team | ☐ |
| 5 | `lab_inventory.csv` + topology diagram complete | Data team | ☐ |
| 6 | Ability list with tactic/technique IDs shared (Q9) | Red team | ☐ |
| 7 | Non-Caldera tool plan agreed (Q14) | Red team | ☐ |
| 8 | Benign traffic generator ready (Q17) | Data team | ☐ |
| 9 | Labeling rules agreed (Section 6.2) | AI + Data team | ☐ |
| 10 | Folder structure + shared storage set up (Section 10) | Data team | ☐ |
| 11 | **Pilot session delivered** | Data team | ☐ |
| 12 | **Pilot validated by AI team** (all fields present, labels joinable) | AI team | ☐ |
| 13 | Full collection approved | All | ☐ |
