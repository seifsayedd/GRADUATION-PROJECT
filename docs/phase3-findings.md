# Phase 3 findings

All numbers come from [`reports/phase3/report.md`](../reports/phase3/report.md) and the JSON files next to it. Every threshold and choice was made on **validation** data; test and holdout were only scored. Intervals are 95% bootstrap intervals that resample whole (source, 5-minute) blocks, so they are wider, and more honest, than row-level intervals.

**Update after Phase 4:** Phase 4 found a lab feature-mapping bug (see [phase4-findings.md](phase4-findings.md)). The lab-feature models were retrained on 14 corrected inputs, and every lab-feature number below is from that rerun. All-feature results are unchanged.

## Check first: the evaluation reproduces Phase 2

Re-scoring the 12 saved models from disk gives exactly the test F1 each Phase 2 run recorded (`reproduces_phase2 = true` for every model in `models.json`).

## 1. Seen networks: the Phase 2 numbers hold, with wide intervals

| Model (time-ordered split, all features) | F1 [95% CI] | False-positive rate [95% CI] |
|---|---|---|
| Stage 1 XGBoost | 0.910 [0.825, 0.957] | 0.067 [0.039, 0.105] |
| Stage 1 Random Forest | 0.916 [0.830, 0.962] | 0.062 [0.036, 0.098] |
| Stage 2 GRU | 0.922 [0.875, 0.988] | 0.066 [0.005, 0.132] |
| Stage 3 autoencoder | 0.149 [0.092, 0.230] | 0.024 [0.013, 0.043] |

- **Thresholds drift over time.** A Stage 1 threshold set to flag 1% of benign *validation* flows flags 6.9% of benign flows later in time (XGBoost). On the random split the same rule gives 0.9%, so the drift comes from time, not from the method. Stage 2 behaves better: its validation threshold gives 1.5% on test, with F1 0.942.
- **Where Stage 1 fails:** on later flows it catches almost none of ToN-IoT `scanning` (0.4%) and only 11% of CSE-CIC-IDS2018 `Infilteration`. Known labelling problems in CSE-CIC-IDS2018 (Liu et al. 2022) may explain part of the Infilteration result. Benign false alarms are highest on ToN-IoT (12–13%) and BoT-IoT (26–64%, but from only 208 benign flows).
- **Calibration:** Stage 1 probabilities are reasonably calibrated on the random split (ECE 0.016–0.027). They are worse on later flows (0.05–0.07).
- **Tactic confusion:** the main error is Reconnaissance/Discovery flows called Benign (60% of them, XGBoost, later flows), followed by Impact called Credential Access (16%).

## 2. Unseen attack types

| Class (never trained on) | Stage 1 XGBoost | Stage 3, one global model | Stage 3, one model per source (selected) |
|---|---|---|---|
| Backdoor (C2) | 5% | 82% | 94% |
| DoS_attacks-Slowloris | 100% | 51% | 15% |
| SSH-Bruteforce | 100% (tactic usually wrong: Impact) | 0% | 0% |

Stage 1 and Stage 3 cover different unseen attacks. **But a combined detector chosen on validation did not help.** Held to the same 1% validation false-alarm budget, "Stage 1 OR Stage 3" scored no better than Stage 1 alone (test recall 0.864 vs 0.872), and caught *fewer* Backdoor flows (3%). Validation contains no unseen attacks, so validation cannot reward catching them. Showing that the combination works needs a validation set with novel attacks, or a separate budget for Stage 3.

## 3. Option A: unseen networks (leave-one-source-out)

XGBoost trained on three source datasets and tested on every flow of the fourth:

| Unseen source | F1 [95% CI], all features | False-positive rate | ROC-AUC | F1 when the source was seen |
|---|---|---|---|---|
| BoT-IoT | 0.491 [0.341, 0.632] | 0.416 | 0.456 | 0.965 |
| CSE-CIC-IDS2018 | 0.627 [0.540, 0.700] | 0.644 | 0.581 | 0.940 |
| ToN-IoT | 0.715 [0.664, 0.766] | 0.167 | 0.752 | 0.810 |
| UNSW-NB15 | 0.577 [0.553, 0.599] | 0.356 | 0.472 | 1.000 |

With lab features, F1 is 0.45–0.79.

**This is the central result of Phase 3.** On a network it has not seen, Stage 1 loses most of its skill. On BoT-IoT and UNSW-NB15 its ranking of flows is worse than random (ROC-AUC below 0.5), and false alarms reach 17–64%. A threshold from validation does not fix this, because validation comes from the training networks.

This matches the lab result: even after the mapping fix, the lab-feature models flag 14–17 of 53 benign lab conversations and miss the one attack. The models learn what each benchmark network looks like, not what an attack looks like in general. Some tactics also exist in only one source (Exfiltration only in BoT-IoT, Command and Control only in CSE-CIC-IDS2018, Other only in UNSW-NB15), so leaving out that source removes the tactic from training entirely.

## 4. Option B: per-source Stage 3

| Variant | Validation recall | Test recall [CI] | Test false-positive rate [CI] |
|---|---|---|---|
| one global model (Phase 2) | 0.083 | 0.082 [0.049, 0.132] | 0.024 [0.013, 0.043] |
| one threshold per source | 0.073 | 0.084 [0.057, 0.123] | 0.069 [0.039, 0.116] |
| **one model per source (selected on validation)** | 0.165 | 0.172 [0.122, 0.239] | 0.074 [0.043, 0.123] |

- Per-source models cut BoT-IoT benign false alarms from 57% to 13%, and catch 94% of unseen Backdoor flows and all UNSW-NB15 attacks.
- **Trade-off:** ToN-IoT benign false alarms rise from 4% to 16%. Every variant hits about 1% on validation, so the rise is drift over time, not a selection error.
- With lab features, the per-source model is also best (test recall 0.032, false-positive rate 0.019, Backdoor 92%). Stage 3 is still a weak general detector.
- It needs to know which network a flow comes from. In deployment that means one model per site, trained on that site's benign traffic. For the lab, that means a real benign lab capture.

## 5. Real lab flows

After the mapping fix, the validation-chosen threshold changes nothing on the lab: XGBoost flags 14 of 54 conversations, Random Forest 15 and the autoencoder 17. None of them flags the attack. Stage 1's false alarms are all UDP (14 or 15 of 33 UDP conversations; no ICMP or IGMP), while the autoencoder also flags 3 of the 6 ICMP conversations. Before the fix, XGBoost flagged 48 and the autoencoder all 54. With one attack, this is a description, not a precision or recall estimate.

## What this means for the project

1. Benchmark scores (F1 ≈ 0.91) do not carry over to new networks. The thesis should report both numbers side by side.
2. The fix most clearly supported by these results is **local data**: a real benign capture from the lab, to calibrate thresholds and train Stage 3 per site, and to fine-tune Stage 1. The planned 5-hour capture is the next step that matters most.
3. Combining Stage 1 and Stage 3 only pays off if it is evaluated on attacks unseen in validation. That needs a deliberate design, not tuning on validation.
