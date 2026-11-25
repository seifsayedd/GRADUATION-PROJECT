# Phase 4 findings

All numbers come from [`reports/phase4/report.md`](../reports/phase4/report.md) and the JSON files next to it. Explanations come from the models themselves: TreeSHAP for Stage 1, attention weights for Stage 2, reconstruction error for Stage 3. The summary inside each explanation object is filled from a fixed template, not written by an LLM.

## 1. Stage 1 SHAP explanations are faithful

- **They add up.** For every explained flow, the base value plus the contributions equals the model's real output. The maximum error is 7e-6 for XGBoost (log-odds) and 1e-13 for Random Forest (probability). The code refuses an explanation that does not add up.
- **They point at what the model uses.** Replacing a flagged flow's top 3 SHAP features with typical benign values drops XGBoost's attack probability by 0.68 on average and turns 91% of the alerts benign. Replacing 3 random features drops it by 0.04 and turns 2% benign. The top-k change beat the random change on 93–100% of flows, across the models, feature sets and k values tested.
- **Stability** is a weak test here. Near-identical flows get identical explanations (median Spearman 1.0, against 0.15 for random pairs). But nearest neighbours are usually near-copies, since 35% of test flows duplicate another on every model input. So this shows the explanations are deterministic, not that they are robust.

## 2. What the full-feature model relies on: likely shortcuts

The most important features of the full-feature XGBoost model are `L7_PROTO`, `TCP_WIN_MAX_IN`, `FLOW_DURATION_MILLISECONDS`, `MAX_TTL`, `MIN_TTL` and `L4_DST_PORT`. They rank at or near the top for almost every tactic.

TTL and TCP window size mostly describe *which operating systems and testbed produced the traffic*, not what the traffic does. A model leaning on them learns each benchmark network's fingerprint. This is a likely explanation for the Phase 3 result that F1 falls to 0.43–0.81 on a network left out of training. This is an interpretation of the SHAP ranking, not a separate experiment; retraining without these features would test it.

Benign test flows that get flagged are pushed there mostly by `L7_PROTO` (+1.30 log-odds on average), `TCP_WIN_MAX_IN` (+1.07) and `MAX_TTL` (+0.48).

## 3. Attacks never seen in training

- **Backdoor** (C2) is missed mainly because of `LONGEST_FLOW_PKT`, `TCP_WIN_MAX_IN` and `MAX_TTL`. On missed flows these push towards benign by 1.15, 0.86 and 0.52 log-odds on average. Its packet sizes and TCP fingerprint look like benign traffic to the model.
- **SSH-Bruteforce** is always detected by the full-feature model, but all 400 sampled flows are called **Impact**. The lab-feature model calls most of them **Credential Access** (243 of the 262 it flags), which is the right tactic.

## 4. Why the real lab gets false alarms: a feature-definition bug

The lab false alarms have two concrete causes.

1. **`SRC_TO_DST_SECOND_BYTES` and `DST_TO_SRC_SECOND_BYTES` do not mean the same thing in the lab and in BigFlow.**
   - The lab mapping computes true bytes per second. In BigFlow the column is 0 for 50–62% of flows in three of the four sources. Its median ratio to bytes per second is 0 to 0.0012 depending on the source, and it is within 5% of bytes per second for 0% of flows in every source. No formula we tried reproduces it.
   - The lab's false-alarm median is 92, which sits at the 92nd percentile of BigFlow benign traffic.
   - It is the **largest driver of XGBoost's lab false alarms** (+1.66 log-odds on average), and it produces **46% of the autoencoder's reconstruction error** on lab flows. That is why the autoencoder flags all 54 conversations.
   - [`docs/feature-mapping.md`](feature-mapping.md) already said this unit needed verifying. This check shows the lab and BigFlow values don't match.
2. **The throughput features use a different duration.** BigFlow's `*_AVG_THROUGHPUT` is bits ÷ *whole-flow* duration (ratio 0.995–1.000, 80–91% of flows within 5%). The lab mapping divides by the *forward-direction* duration, which matches less well (22–73% of flows within 5%).

Other drivers come from real traffic differences, not bugs:

- Lab flows last much longer: median 1,202 ms against 1 ms for BigFlow benign traffic.
- Many lab flows are one-way multicast (mDNS on port 5353, SSDP), so `OUT_BYTES` and `DST_TO_SRC_AVG_THROUGHPUT` are 0. Both push towards attack.

**Suggested fix (needs your approval, because it changes the Phase 1 mapping and means retraining the lab models):**
- Drop the two `*_SECOND_BYTES` features from the lab feature set.
- Compute throughput over the whole-flow duration.
- Retrain the lab-feature models and score the lab again.

This fixes the mapping bug. It will not fix the real traffic differences, which still need a benign lab capture.

## 5. Stage 2 attention is a weak explanation

When one earlier flow is removed from the window at a time:

| Model | Top-attended earlier flow is the most influential one | Expected by chance | Median rank correlation, attention vs influence |
|---|---|---|---|
| GRU, all features | 6.6% | 3.4% | −0.005 |
| GRU, lab features | 23.0% | 3.4% | 0.113 |

For the full-feature GRU, attention weights are **barely better than chance** at finding the flow that changed the decision. Attention is spread over the window (about 0.03 per position), with no dominant step. Each Stage 2 explanation object therefore records both the attention weight and the measured `occlusion_delta`, and its caveat says attention is not proof. For alerts, the occlusion values are the better evidence.

## 6. Stage 3

On lab-feature test flows, the ICMP type fields carry the largest share of reconstruction error for most tactics. With all features, `L7_PROTO` and the port and TCP-flag fields lead. Every Stage 3 explanation adds up exactly: per-feature error shares sum to the score (max error 8e-8).

## What this means for the project

1. SHAP explanations for Stage 1 can be trusted as descriptions of *the model*. They expose a likely shortcut (TTL, TCP window) that matches the poor results on unseen networks.
2. Part of the lab failure is **our own feature-mapping bug**, which can be fixed now. The rest needs real benign lab traffic.
3. For Stage 2, use occlusion values rather than raw attention when explaining an alert.
4. The explanation objects (`reports/phase4/examples/`, `reports/phase4/lab/`) and their schema are ready for Phase 5 (GenAI reports) and Phase 6 (dashboard).
