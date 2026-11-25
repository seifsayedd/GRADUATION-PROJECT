# Phase 2 findings

All numbers come from [`reports/phase2/summary.md`](../reports/phase2/summary.md) and the JSON files next to it. These are training-time results; the full evaluation is Phase 3. Validation was used for model selection; test and holdout never were.

## Data facts that shape the results

- **BigFlow v2:** 66,935,021 flows, 32 classes, no source column. The source is recovered from the capture date ([`sources.py`](../src/sentinelai/data/sources.py)).
- **Training sample:** 2,647,605 rows (≤ 100k per class, ≤ 1M benign). 167,302 exact duplicate rows had to be dropped, even though BigFlow says it removed duplicates. 118,292 infinite values in the bytes-per-second columns were set to 0, and every change was counted.
- **BoT-IoT has almost no benign traffic** (208 benign rows in the test split, against 39,261 attack rows).

## Stage 1: Random Forest / XGBoost

- The **temporal** split (train on earlier flows, test on later ones) is the honest setting. It gives binary F1 ≈ 0.91 and a **false-positive rate ≈ 6–7%**. The random **stratified** split gives F1 ≈ 0.95 and a false-positive rate ≈ 2%, because near-identical neighbouring flows end up on both sides of the split.
- The **lab feature set** (16 features) costs little on BigFlow. For XGBoost on the temporal split, binary F1 is 0.913 vs 0.910, and tactic macro-F1 is 0.815 vs 0.832.
- **Unseen attack classes:**
  - `SSH-Bruteforce`: detected (100% with all features), but the tactic is almost never right. The model calls it something other than Credential Access.
  - `DoS_attacks-Slowloris`: 88–100% detected, and the tactic is right about half the time.
  - `Backdoor` (C2): **mostly missed** (4–34% detected).
- XGBoost files are 3–9 MB. Random Forest files are 53–74 MB, above the 50 MB commit limit; they stay in git-ignored `models/`.

## Stage 2: GRU + attention

Test = random 5-minute blocks covering 15% of each day's rows, with every tactic present in train and test.

| Features | F1 | FPR | Tactic macro-F1 |
|---|---|---|---|
| full | 0.922 | 6.6% | 0.939 |
| lab | 0.897 | 11.5% | 0.922 |

- `Bot` beaconing and slow DoS are detected almost perfectly (recall 1.00, false-positive rate ≈ 0). On the two ToN-IoT days the benign false-positive rate is **23–44%**. Those days have only 9–11 source hosts, so there's little variety in benign behaviour to learn from.
- Two earlier split designs were rejected, because both left whole attack classes out of train or out of test: one time cut per day, then 15-minute random blocks. See the docstring in [`sequence.py`](../src/sentinelai/models/sequence.py).
- Caveat: neighbouring 5-minute blocks can land in different splits, so a burst may continue across a boundary. That makes these numbers somewhat optimistic.

## Stage 3: benign-only autoencoder

- **It doesn't work well as a general detector here:** test ROC-AUC 0.58, and only 8% of attacks flagged at the 1% validation false-positive threshold.
- It does flag 64% of UNSW-NB15 attacks, and with all features, 82% of the unseen `Backdoor` class, which Stage 1 mostly missed. With lab features it flags almost nothing.
- Its benign false-positive rate on **BoT-IoT is 57%**, because BoT-IoT has too little benign traffic for the model to learn what normal looks like there.
- Likely next steps (Phase 3, decided on validation only): one autoencoder per source, or a score normalised per source.

## Real lab flows (lab-compatible models)

54 real conversations, 1 attack (S10 ping):

| Model | Flagged | Caught the attack | False alarms |
|---|---|---|---|
| Stage 1 RF | 17 | yes | 16 |
| Stage 1 XGBoost | 48 | yes | 47 |
| Stage 2 GRU | 0 | no | 0 |
| Stage 3 autoencoder | 54 | yes | 53 |

**The BigFlow-trained models don't transfer to the lab traffic.** The lab's benign traffic (mDNS, IGMP, SSDP, connectivity pings) looks nothing like BigFlow's benign traffic, so most of it is flagged. The one attack is an ICMP ping that's almost identical to the benign pings in the same capture, so catching it says little. Closing this gap needs more and more varied lab captures (see [feature-mapping.md](feature-mapping.md)), and probably fine-tuning or recalibrating on real lab benign traffic. Synthetic rows stay out of evaluation.
