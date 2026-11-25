# Dataset Card — SentinelAI Lab Capture v1

Labeled network-flow data captured in the isolated SentinelAI cyber range on **2026-11-12 (UTC)**. See [`../docs/lab-architecture.md`](../docs/lab-architecture.md) for the topology.

## Files

| Path | Rows | Description |
|---|---|---|
| `raw/flows_argus_schema.csv` | 68 | Captured flows in the Argus field schema (77 cols) with labels |
| `raw/ens33_nfdump.csv` | — | Raw nfdump export, Red Team–side interface |
| `raw/ens37_nfdump.csv` | — | Raw nfdump export, target-side interface |
| `raw/session_manifest.csv` | 10 | One row per capture session: time window, tool, ATT&CK ID, status, notes |
| `processed/combined_sessions_full.csv` | 75 | **Real data only.** 68 flows matched 1:1 to nfdump + 7 `session_only` rows (96 cols) |
| `processed/combined_sessions_augmented.csv` | 235 | The 75 real rows + **160 flagged synthetic** flows (101 cols) |

Rebuild and check:

```bash
python scripts/merge_sessions.py data/raw data/processed/combined_sessions_full.csv
python scripts/validate_dataset.py data/processed/combined_sessions_full.csv
```

## Sessions

| ID | Activity | ATT&CK | Flows | Label |
|---|---|---|---|---|
| S07 | Extended benign baseline (mDNS, IGMP, SSDP, broadcast, connectivity pings) | — | 56 | benign |
| S08 | Caldera: Hostname Discovery | T1082 | 0 (local only) | — |
| S09 | Caldera: ping to external address | T1016.001 | 0 (left via NAT, not monitored) | — |
| S10 | Kali → Windows ping across the gateway | T1016.001 | 4 | **attack** |
| S11 | Caldera: List OS Information | T1082 | 0 (local only) | — |
| S12 | Caldera: Process Discovery | T1057 | 0 (local only) | — |
| S13 | Manual `ip addr show` | — | 0 (local only) | — |
| S14 | Scripted connectivity-check burst | — | 8 | benign |
| EX1 | Caldera geolocation lookup | — | 0 | **excluded**: out of approved scope |
| EX2 | Caldera network interface config | — | 0 | **excluded**: timed out |

Sessions with no flows are kept as `session_only` rows (feature fields empty) so the record is complete. **Filter `record_type == "flow"` before training.**

## Labels

| Column | Values |
|---|---|
| `label` | `0` benign, `1` attack |
| `attack_cat` | `Benign`, `Discovery` |
| `attack_type` | `Benign`, `T1016.001` |
| `label_method` | `exact_window`: flow timestamp falls inside the session window in the manifest |

Real data: **64 benign / 4 attack**. Augmented: **192 benign / 36 attack**.

## Synthetic augmentation

Synthetic rows were added only to ease class imbalance and to test the pipeline. They are **not** real captures.

- Every synthetic row has `is_synthetic = 1`, a `source_row` pointing to the real row it came from, a `synth_group` and the exact `augmentation_method` (for example `shift=-97.5s;pkts_x1.21;size_x1;time_x0.92`).
- Method: time shift ±120 s, packet count ×0.75–1.30, duration ×0.85–1.15, with rate/load/mean size recomputed. Benign events ×2, attack events ×8. See `scripts/augment_flows.py`.
- **Never put synthetic rows in a validation or test split.**

## Known limitations

- **Small and narrow.** 68 real flows, and one attack technique (T1016.001) produced network traffic. This is a pipeline proof, not a training corpus.
- **Argus-only fields are empty** (jitter, TCP RTT/window, TTL/TOS, inter-packet times, flags/state, MACs, app bytes) because flows came from nfdump after Argus failed.
- Protocols are ICMP, UDP and IGMP only. There's no TCP application traffic yet.
- The manifest notes mention an earlier combined file (`jojo_data_expanded.csv`). That file belongs to another team member and is **not** part of this repository.

## Scaling plan

- **Training volume:** the public [BigFlow-NIDS](https://data.mendeley.com/datasets/nv729tbdgz/1) dataset (~66.9M labeled NetFlow records, 55 features, 34 attack categories, 4 benchmark sources). Not included here; download it into `data/external/` (git-ignored).
- **Validation:** real lab captures only, through a planned 5-hour session (≈3 h benign → 5–10 min authorized attack window → benign), labeled by time window.
- **Feature alignment:** map Argus/nfdump fields to BigFlow's 55 NetFlow features so models trained on BigFlow can score lab flows.
