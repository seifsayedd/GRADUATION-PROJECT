# Feature mapping: lab flows → BigFlow-NIDS (NetFlow-v3)

Models are trained on [BigFlow-NIDS](https://data.mendeley.com/datasets/nv729tbdgz/1) and validated on our real lab captures. To do that, every lab flow has to be expressed in BigFlow's feature space. This page documents how, and is honest about what can't be mapped.

Code: [`src/sentinelai/data/mapping.py`](../src/sentinelai/data/mapping.py). `FEATURE_MAP` there is the source of truth, and `tests/test_mapping.py` checks that this table and the code list the same 53 features.

## Assumption to verify

BigFlow merges NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3 and NF-CSE-CIC-IDS2018-v3. The feature names below are the NetFlow-v3 names those sources use: 53 features + `Label` + `Attack` = the 55 attributes the dataset page reports. The page doesn't list its columns, so **this is not yet confirmed against a downloaded file**. The BigFlow loader checks every file header against this list and reports missing/extra columns ([`schema.py`](../src/sentinelai/data/schema.py)). If BigFlow renamed anything, update `schema.py` and this page together.

## Step 1: remove vantage-point duplicates

The sensor is dual-homed, so traffic between the two segments is recorded on both `ens33` and `ens37`. `dedupe_vantage()` drops a record when the same flow (same direction, protocol, addresses, ports, packets, bytes, start within 50 ms) was also recorded on `ens33`. Flows that only one interface saw are kept. Example: the gateway's own mDNS on `192.168.60.0/24` only appears on `ens37`.

## Step 2: pair the two directions

nfdump exports **one direction per record** (`spkts > 0`, `dpkts = 0` on every row). NetFlow-v3 records in BigFlow are **two-way** (`IN_*` = client→server, `OUT_*` = server→client). Mapping one-way records directly would set every `OUT_*` to 0, a shortcut a model would learn instantly.

`pair_biflows()` joins each record with its reverse partner. Partners have the same interface, protocol and session, swapped addresses (and swapped ports for TCP/UDP), and start within 1 s of each other. The initiator is the earlier record. For ICMP, the echo request (type 8) is always the initiator over the reply (type 0). Unpaired records (multicast, broadcast, unanswered traffic) keep `OUT_* = 0`. That's genuine: nothing came back. The two halves of a pair must carry the same label, or pairing fails loudly.

Result on `combined_sessions_full.csv`: **68 records → 60 after de-duplication → 54 conversations** (6 paired, 48 one-way). Labels: **53 benign, 1 attack** (the S10 Kali → Windows ping, T1016.001).

## Step 3: feature table

Status: **mapped** = direct copy or exact derivation · **derived** = computed from other fields (check the unit) · **missing** = not observable with the current sensor, left as NaN (never 0).

| BigFlow feature | Status | Lab source / reason |
|---|---|---|
| FLOW_START_MILLISECONDS | mapped | min(fwd.stime, rev.stime) × 1000 *(identifier, not a feature)* |
| FLOW_END_MILLISECONDS | mapped | max(fwd.ltime, rev.ltime) × 1000 *(identifier)* |
| IPV4_SRC_ADDR | mapped | initiator `saddr` *(identifier)* |
| IPV4_DST_ADDR | mapped | initiator `daddr` *(identifier)* |
| L4_SRC_PORT | mapped | initiator `sport`; 0 for ICMP/IGMP |
| L4_DST_PORT | mapped | initiator `dport`; 0 for ICMP/IGMP |
| PROTOCOL | mapped | `proto` name → IANA number (icmp 1, igmp 2, tcp 6, udp 17) |
| L7_PROTO | missing | needs nDPI application detection; nfdump has none |
| IN_BYTES | mapped | fwd `sbytes` (IP-layer bytes) |
| IN_PKTS | mapped | fwd `spkts` |
| OUT_BYTES | mapped | rev `sbytes`; 0 if no reverse record |
| OUT_PKTS | mapped | rev `spkts`; 0 if no reverse record |
| TCP_FLAGS | missing | Argus `flgs`/`state` empty; lab v1 has no TCP |
| CLIENT_TCP_FLAGS | missing | as above |
| SERVER_TCP_FLAGS | missing | as above |
| FLOW_DURATION_MILLISECONDS | mapped | FLOW_END − FLOW_START |
| DURATION_IN | mapped | fwd `dur` × 1000 |
| DURATION_OUT | mapped | rev `dur` × 1000; 0 if no reverse record |
| MIN_TTL | missing | `sttl`/`dttl` empty (Argus-only) |
| MAX_TTL | missing | as above |
| LONGEST_FLOW_PKT | missing | nfdump gives totals only. `smaxsz`/`sminsz` in the lab CSV hold the **mean** size (e.g. 80.18), not min/max |
| SHORTEST_FLOW_PKT | missing | as above |
| MIN_IP_PKT_LEN | missing | as above |
| MAX_IP_PKT_LEN | missing | as above |
| SRC_TO_DST_SECOND_BYTES | missing | **Not filled.** BigFlow's value is 0 for 50–62% of flows in three sources and matches bytes/s for 0% of flows, so its definition is unknown (Phase 4 check, `reports/phase4/units.json`). Lab v1 filled it with bytes/s, which caused false alarms |
| DST_TO_SRC_SECOND_BYTES | missing | as above |
| RETRANSMITTED_IN_BYTES | missing | retrans fields empty |
| RETRANSMITTED_IN_PKTS | missing | as above |
| RETRANSMITTED_OUT_BYTES | missing | as above |
| RETRANSMITTED_OUT_PKTS | missing | as above |
| SRC_TO_DST_AVG_THROUGHPUT | derived | IN_BYTES × 8 / **whole-flow** duration (bit/s), as BigFlow computes it (checked in Phase 4); 0 if duration = 0 |
| DST_TO_SRC_AVG_THROUGHPUT | derived | OUT_BYTES × 8 / whole-flow duration (bit/s), as BigFlow |
| NUM_PKTS_UP_TO_128_BYTES | missing | needs per-packet sizes |
| NUM_PKTS_128_TO_256_BYTES | missing | as above |
| NUM_PKTS_256_TO_512_BYTES | missing | as above |
| NUM_PKTS_512_TO_1024_BYTES | missing | as above |
| NUM_PKTS_1024_TO_1514_BYTES | missing | as above |
| TCP_WIN_MAX_IN | missing | `swin` empty |
| TCP_WIN_MAX_OUT | missing | `dwin` empty |
| ICMP_TYPE | mapped | type × 256 + code of the initiator, from nfdump `dstPort` `"type.code"`; 0 for non-ICMP |
| ICMP_IPV4_TYPE | mapped | ICMP type of the initiator; 0 for non-ICMP |
| DNS_QUERY_ID | missing | needs DNS parsing (also dropped as a model feature: a random ID) |
| DNS_QUERY_TYPE | missing | needs DNS parsing |
| DNS_TTL_ANSWER | missing | needs DNS parsing |
| FTP_COMMAND_RET_CODE | missing | needs FTP parsing |
| SRC_TO_DST_IAT_MIN | missing | `sintpkt*` empty (Argus-only) |
| SRC_TO_DST_IAT_MAX | missing | as above |
| SRC_TO_DST_IAT_AVG | missing | as above |
| SRC_TO_DST_IAT_STDDEV | missing | as above |
| DST_TO_SRC_IAT_MIN | missing | `dintpkt*` empty (Argus-only) |
| DST_TO_SRC_IAT_MAX | missing | as above |
| DST_TO_SRC_IAT_AVG | missing | as above |
| DST_TO_SRC_IAT_STDDEV | missing | as above |

**Totals: 18 mapped/derived (4 of them identifiers, so 14 usable model inputs), 35 missing.** Until Phase 4 the two `*_SECOND_BYTES` features were filled with bytes/s, giving 16 inputs; the Phase 2 and 3 results for lab-feature models before that fix used those 16.

## Labels

| BigFlow | Lab | Note |
|---|---|---|
| `Label` (0/1) | `label` | same meaning |
| `Attack` (attack category name) | `attack_cat` (ATT&CK tactic) | **different vocabularies.** BigFlow uses dataset category names; the lab uses ATT&CK tactics. A category → tactic table is needed for Stage 1's multi-class step (Phase 2). |

Lab-only columns (`session_id`, `attack_type`, `tool`, `label_method`, `capture_interface`, `paired`, `is_synthetic`) are carried through for splitting and reporting.

## What this means for the models

1. **A "lab-compatible" model is needed.** To score lab flows, train on the 14 features both datasets share. A model trained on all of BigFlow's features can't be fairly scored on lab data: imputing 33 features would mean scoring invented values. Phase 2 will train both and report them separately.
2. **Closing the gap is a sensor problem, not a code problem.** Most gaps (TTL, TCP flags, inter-arrival times, packet-size bins, L7) disappear if the planned 5-hour capture uses a collector that emits NetFlow-v3 fields directly (e.g. nProbe, the tool the NF-v3 datasets were built with), or a working Argus with the extended metrics in `docs/ai-team-data-requirements.md` §4.2. Raw PCAP from the session would also let us regenerate any of them.
3. **Lab v1 is too small to judge accuracy.** 54 conversations with 1 attack can show the pipeline runs end to end. It cannot support a recall or precision claim. Metrics on it will be reported with that caveat.
