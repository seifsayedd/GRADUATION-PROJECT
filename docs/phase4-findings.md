# Phase 4 findings

All numbers come from [`reports/phase4/report.md`](../reports/phase4/report.md) and the JSON files next to it. Explanations come from the models themselves: TreeSHAP for Stage 1, attention weights for Stage 2, reconstruction error for Stage 3. The summary inside each explanation object is filled from a fixed template, not written by an LLM.

Phase 4 found a bug in the lab feature mapping (section 4). It was fixed, the lab-feature models were retrained, and Phases 2–4 were rerun for the lab feature set. Lab-feature numbers below are **after the fix** unless marked "before". All-feature numbers did not change.

## 1. Stage 1 SHAP explanations are faithful

- **They add up.** For every explained flow, the base value plus the contributions equals the model's real output. The maximum error is 7e-6 for XGBoost (log-odds) and 1e-13 for Random Forest (probability). The code refuses an explanation that does not add up.
- **They point at what the model uses.** Replacing a flagged flow's top 3 SHAP features with typical benign values:

  | Model | Mean drop in attack probability | Alerts turned benign | Top-k change beats random |
  |---|---|---|---|
  | XGBoost, all features, top 3 | 0.68 | 91% | 100% of flows |
  | XGBoost, all features, 3 random | 0.04 | 2% | – |
  | XGBoost, lab features, top 3 | 0.51 | 52% | 96% of flows |
  | XGBoost, lab features, 3 random | 0.15 | 5% | – |

  Across every model, feature set and k tested, the top-k change beat the random change on 87–100% of flows. The lab-feature model has only 14 inputs, so 3 random features are already a large share of them.
- **Stability** is a weak test here. Near-identical flows get identical explanations (median Spearman 1.0, against 0.15 for random pairs). But nearest neighbours are usually near-copies, since 35% of test flows duplicate another on every model input. So this shows the explanations are deterministic, not that they are robust.

## 2. What the full-feature model relies on: likely shortcuts

The most important features of the full-feature XGBoost model are `L7_PROTO`, `TCP_WIN_MAX_IN`, `FLOW_DURATION_MILLISECONDS`, `MAX_TTL`, `MIN_TTL` and `L4_DST_PORT`. They rank at or near the top for almost every tactic.

TTL and TCP window size mostly describe *which operating systems and testbed produced the traffic*, not what the traffic does. A model leaning on them learns each benchmark network's fingerprint. This is a likely explanation for the Phase 3 result that F1 falls to 0.49–0.72 (all features) on a network left out of training. This is an interpretation of the SHAP ranking, not a separate experiment; retraining without these features would test it.

Benign test flows that get flagged are pushed there mostly by `L7_PROTO` (+1.30 log-odds on average), `TCP_WIN_MAX_IN` (+1.07) and `MAX_TTL` (+0.48).

## 3. Attacks never seen in training

- **Backdoor** (C2): the all-feature model misses it mainly because of `LONGEST_FLOW_PKT`, `TCP_WIN_MAX_IN` and `MAX_TTL`. On missed flows these push towards benign by 1.15, 0.86 and 0.52 log-odds on average. Its packet sizes and TCP fingerprint look like benign traffic to the model. The lab-feature model, which cannot see those fields, catches 92% of the sampled Backdoor flows (90% in the full Phase 3 holdout). It mostly calls them Reconnaissance/Discovery, which is the wrong tactic.
- **SSH-Bruteforce** is always detected by the full-feature model, but all 400 sampled flows are called **Impact**. The lab-feature model also detects all of them and calls 399 of 400 **Credential Access**, which is the right tactic.

## 4. The lab false alarms: a feature-definition bug, found and fixed

**What SHAP showed (before the fix).**

1. **`SRC_TO_DST_SECOND_BYTES` and `DST_TO_SRC_SECOND_BYTES` did not mean the same thing in the lab and in BigFlow.**
   - The lab mapping computed true bytes per second. In BigFlow the column is 0 for 50–62% of flows in three of the four sources. Its median ratio to bytes per second is 0 to 0.0012 depending on the source, and it matches bytes per second (within 5%) for 0% of flows in every source. No formula we tried reproduces it (`reports/phase4/units.json`).
   - It was the largest driver of XGBoost's lab false alarms (+1.66 log-odds on average), and 46% of the autoencoder's reconstruction error on lab flows.
2. **The throughput features used a different duration.** BigFlow's `*_AVG_THROUGHPUT` is bits ÷ *whole-flow* duration (ratio 0.97–1.00; 70–91% of flows within 5%, both directions). The lab mapping divided by one direction's duration.

**The fix (approved and applied).**
- The two `*_SECOND_BYTES` features are no longer filled for lab flows, so the lab feature set has 14 model inputs instead of 16.
- Throughput now divides by whole-flow duration, as BigFlow does.
- [`mapping.py`](../src/sentinelai/data/mapping.py) and [`feature-mapping.md`](feature-mapping.md) were updated, with a test.
- All six lab-feature models were retrained, and Phases 2–4 were rerun for the lab feature set.

**Result on the 54 real lab conversations:**

| Model | Flagged before | Flagged after | Caught the attack, before → after |
|---|---|---|---|
| Stage 1 XGBoost | 48 | **14** | yes → **no** |
| Stage 1 Random Forest | 17 | 15 | yes → **no** |
| Stage 2 GRU | 0 | 0 | no → no |
| Stage 3 autoencoder | 54 | **17** | yes → **no** |

- **False alarms fell sharply, but no model catches the lab attack any more.** Before the fix, the attack was "caught" because the broken feature made nearly every lab flow look like an attack. With one attack, neither result supports a precision or recall claim.
- **BigFlow results barely moved:** XGBoost F1 on later flows 0.913 → 0.910 with lab features.

**What still drives the remaining lab false alarms.** These are real traffic differences, not bugs:
- They are all UDP, and mostly **zero-duration single-packet** flows (median duration 0 ms, against 1 ms for BigFlow benign) to mDNS port 5353. `FLOW_DURATION_MILLISECONDS` pushes them towards attack by +3.6 log-odds on average (XGBoost).
- They are one-way, so `OUT_BYTES` is 0, which also pushes towards attack.
- Fixing these needs real benign lab traffic to learn from, which the planned capture would provide.

## 5. Stage 2 attention is a weak explanation

When one earlier flow is removed from the window at a time:

| Model | Top-attended earlier flow is the most influential one | Expected by chance | Median rank correlation, attention vs influence |
|---|---|---|---|
| GRU, all features | 6.6% | 3.4% | −0.005 |
| GRU, lab features | 9.2% | 3.3% | 0.010 |

Attention weights are **barely better than chance** at finding the flow that changed the decision. Attention is spread over the window (about 0.03 per position), with no dominant step. Each Stage 2 explanation object therefore records both the attention weight and the measured `occlusion_delta`, and its caveat says attention is not proof. For alerts, the occlusion values are the better evidence. (Before the fix, the lab-feature GRU scored 23.0% here; the retrained model is closer to the all-feature result.)

## 6. Stage 3

On lab-feature test flows, the ICMP type fields and flow duration carry the largest shares of reconstruction error for most tactics. With all features, `L7_PROTO` and the port and TCP-flag fields lead. Every Stage 3 explanation adds up exactly: per-feature error shares sum to the score (max error 8e-8).

## What this means for the project

1. SHAP explanations for Stage 1 can be trusted as descriptions of *the model*. They expose a likely shortcut (TTL, TCP window) that matches the poor results on unseen networks.
2. Part of the lab failure was **our own feature-mapping bug**. It is fixed: flagged lab conversations fell from 48 to 14 (XGBoost) and from 54 to 17 (autoencoder). The rest needs real benign lab traffic, and catching the lab attack needs more and more varied attack captures.
3. For Stage 2, use occlusion values rather than raw attention when explaining an alert.
4. The explanation objects (`reports/phase4/examples/`, `reports/phase4/lab/`) and their schema are ready for Phase 5 (GenAI reports) and Phase 6 (dashboard).
