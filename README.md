# SentinelAI — AI-Driven Cybersecurity Threat Defense and Alert System

**Graduation Project · Faculty of Computers & Information Science, Egyptian Chinese University (ECU)**

An AI-driven cybersecurity system that combines Red and Blue Teams. The Red Team runs controlled, ATT&CK-mapped attacks in an isolated lab. The Blue Team uses ML/DL to detect and classify the threats in the captured network traffic. Generative AI then explains the findings and writes security reports, so the system can be re-evaluated as attacks change.

```
AI Red Team Attack ─► Network Traffic Collection ─► Data Processing & Labeling ─► AI Detection
        ▲                                                                              │
        └──── GenAI Security Report ◄── Alert & Response ◄── Explainability (XAI) ◄────┘
```

| Layer | Technology |
|---|---|
| Red Team | MITRE Caldera + LLM-based planner, MITRE ATT&CK |
| Data collection | Argus / nfdump flow sensors, automatic labeling against Caldera logs |
| Detection | Stage 1 Random Forest / XGBoost (per flow) · Stage 2 LSTM / GRU (temporal) · Stage 3 Autoencoder (anomaly / zero-day) |
| Explainability | SHAP, attention weights |
| GenAI reporting | RAG over PostgreSQL + pgvector, LLM report engine, analyst chatbot |
| Dashboard | React |

Full design: [`docs/project-overview.md`](docs/project-overview.md)

---

## Repository contents

```
├── data/
│   ├── README.md                      dataset card: sessions, labels, limitations
│   ├── raw/                           flow records, nfdump exports (ens33, ens37), session manifest
│   └── processed/                     merged real dataset + flagged augmented dataset
├── docs/
│   ├── project-overview.md            full SentinelAI architecture and evaluation plan
│   ├── lab-architecture.md            cyber-range topology, tooling, lessons learned
│   ├── ai-team-data-requirements.md   data-collection spec from the AI team
│   ├── feature-mapping.md             lab (Argus/nfdump) → BigFlow NetFlow-v3 features, with gaps
│   └── attack-taxonomy.md             BigFlow's 32 attack classes → ATT&CK tactics
├── scripts/
│   ├── merge_sessions.py              flows + nfdump + manifest → one labeled CSV
│   ├── augment_flows.py               flagged jitter-based synthetic augmentation
│   ├── validate_dataset.py            consistency and labeling checks
│   ├── build_bigflow_sample.py        class-stratified BigFlow sample → data/interim/
│   ├── prepare_training_data.py       Phase 2 training sets (flows + sequences)
│   └── train_phase2.py                train Stages 1–3, score lab flows, write reports/phase2
├── configs/phase2.yaml                Phase 2 settings (sampling, splits, model parameters)
├── reports/phase2/                    training metrics (JSON), summary.md, lab_scores.csv
├── src/sentinelai/data/               data layer (loaders, mapping, preprocessing, splits, sources, ATT&CK taxonomy)
├── src/sentinelai/models/             Stage 1–3 detection models and training runs
├── tests/                             pytest suite
├── pyproject.toml
└── requirements.txt
```

## Current status

| Workstream | Status |
|---|---|
| Isolated cyber range (Kali · Ubuntu sensor · Windows 7) | ✅ Built and verified |
| Caldera + Sandcat adversary emulation | ✅ Running (agent on Kali) |
| Flow capture on both gateway interfaces | ✅ fprobe + nfcapd/nfdump |
| First labeled dataset (v1) | ✅ 68 real flows, 96 features, ATT&CK-labeled |
| Large-scale training data | 🔄 BigFlow-NIDS selected (~66.9M flows) |
| 5-hour benign/attack capture session | 🔄 Planned |
| Phase 1 data layer (loaders, feature mapping, preprocessing, splits) | ✅ `src/sentinelai/data` |
| Phase 2 detection models (RF/XGBoost, GRU + attention, autoencoder) | ✅ trained on BigFlow; results in [`reports/phase2/summary.md`](reports/phase2/summary.md) |
| Evaluation, XAI, GenAI reports, dashboard | ⏳ Phases 3–6 |

## Quick start

```bash
pip install -r requirements.txt

# Rebuild the merged dataset from raw files (output matches data/processed exactly)
python scripts/merge_sessions.py data/raw data/processed/combined_sessions_full.csv

# Validate
python scripts/validate_dataset.py data/processed/combined_sessions_full.csv
python scripts/validate_dataset.py data/processed/combined_sessions_augmented.csv

# Regenerate the augmented set (seeded)
python scripts/augment_flows.py data/processed/combined_sessions_full.csv augmented.csv --seed 42
```

> **Data integrity:** every flow in `combined_sessions_full.csv` is genuinely captured traffic matched 1:1 to a collector record. Synthetic rows exist only in the augmented file and are always flagged with `is_synthetic = 1`. Exclude them from evaluation.

## Phase 1: data layer

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt && pip install -e .
python -m pytest                                    # lab tests read data/processed (read-only)
```

**BigFlow-NIDS.** Download it from [Mendeley](https://data.mendeley.com/datasets/nv729tbdgz/1) into `data/external/bigflow/` (git-ignored). Check the headers first, then build a training sample:

```bash
python scripts/build_bigflow_sample.py data/external/bigflow data/interim/x.parquet --check-only
python scripts/build_bigflow_sample.py data/external/bigflow data/interim/bigflow_sample.parquet --per-class 50000
```

**Typical use:**

```python
from sentinelai.data import (load_bigflow, clean, make_splits, resample_train,
                             FlowPreprocessor, lab_to_bigflow, available_features)

df, report = clean(load_bigflow("data/interim/bigflow_sample.parquet"))
splits = make_splits(df, label_col="Attack", holdout_attacks=["<category>"])  # split FIRST
splits = resample_train(splits, strategy="undersample")                     # train only
pp = FlowPreprocessor(features=available_features()).fit(splits.train)      # lab-compatible feature set
X_train, X_test = pp.transform(splits.train), pp.transform(splits.test)

lab = lab_to_bigflow("data/processed/combined_sessions_full.csv")           # 54 real conversations, eval only
X_lab = pp.transform(lab)
```

Rules the code enforces: synthetic rows never reach val/test/holdout (`assert_no_synthetic`). Resampling only touches train. Scalers and encoders are fitted on train only. Lab features the sensor didn't measure stay NaN and can't be fed to a model. Details and the full gap list: [`docs/feature-mapping.md`](docs/feature-mapping.md).

## Phase 2: detection models

Needs Python ≥ 3.11 and BigFlow v2 Parquet in `data/external/bigflow/`. Install the CUDA build of PyTorch first if you have an NVIDIA GPU (see `requirements.txt`).

```bash
python scripts/prepare_training_data.py        # ~15 min: 2.6M-row class-capped sample + 4 contiguous days for Stage 2
python scripts/train_phase2.py --stage 1       # RF + XGBoost × {temporal, stratified} split × {full, lab} features
python scripts/train_phase2.py --stage 3       # benign-only autoencoder
python scripts/train_phase2.py --stage 2 --lab # GRU + attention, then score the real lab flows
```

| Stage | Model | What it does |
|---|---|---|
| 1 | Random Forest, XGBoost | Benign vs attack per flow, then ATT&CK tactic for flagged flows ([taxonomy](docs/attack-taxonomy.md)) |
| 2 | GRU (or LSTM) + attention | Labels each flow from that host's previous 31 flows plus the time gaps between them |
| 3 | Autoencoder | Trained on benign flows only. Flags flows it reconstructs badly; threshold set for a 1% false-positive rate on benign validation traffic |

Each model is trained twice: on the **full** feature set, and on the **lab** set (the 16 features our lab sensor can fill). Only the lab set can score lab flows. Three attack classes (`SSH-Bruteforce`, `DoS_attacks-Slowloris`, `Backdoor`) are held out of training completely, to test detection of unseen attacks.

Models are saved in `models/` (git-ignored). Metrics go to `reports/phase2/`, and **[`summary.md`](reports/phase2/summary.md)** lists every number, including the weak ones.

## Phase 3: evaluation

Needs the Phase 2 models in `models/` and the prepared splits in `data/interim/phase2/`. No model is tuned on test: every threshold and choice below is made on validation data.

```bash
python scripts/evaluate_phase3.py --only predict eval combined lab figures report   # ~5 min: score and evaluate every saved model
python scripts/evaluate_phase3.py --only loso      # option A: train on 3 source datasets, test on the 4th (~30 min on GPU)
python scripts/evaluate_phase3.py --only stage3    # option B: per-source Stage 3 variants (~10 min)
python scripts/evaluate_phase3.py --only figures report                             # rebuild figures and report.md
```

| What | How |
|---|---|
| Confidence intervals | 95% bootstrap that resamples whole (source, 5-minute) blocks, since neighbouring flows are near-copies |
| Operating points | The Phase 2 default, and a threshold that flags 1% of benign **validation** flows |
| Per-class / per-source | Detection rate for each of the 32 attack classes and false alarms per source dataset |
| Unseen attacks | Held-out classes: detection rate and which tactic each is mistaken for |
| Combined detector | Stage 1 alone vs Stage 1 OR Stage 3, both held to the same validation false-alarm budget |
| Option A | Leave-one-source-out XGBoost: how it does on a network it has never seen |
| Option B | Stage 3 with one threshold per source, or one autoencoder per source, chosen on validation |

Predictions are cached in `data/interim/phase3/` (git-ignored). The results, figures and **[`report.md`](reports/phase3/report.md)** go to `reports/phase3/`. The written findings are in [`docs/phase3-findings.md`](docs/phase3-findings.md).

An interactive map of the whole project (loop, data, models, roadmap) is in [`docs/visual/sentinelai-map.html`](docs/visual/sentinelai-map.html). Open it in any browser.

---

## My contributions — Seif Eldeen Elsayed (Cybersecurity)

- **Cyber range:** built the isolated VMware lab with Kali Linux (Red Team / Caldera), a dual-homed Ubuntu gateway/sensor and a Windows 7 target on two host-only segments. Fixed the Netplan/NetworkManager conflicts and verified routed connectivity across all VMs.
- **Flow capture:** compiled and configured Argus from source. When it kept crashing, deployed fprobe + nfcapd/nfdump as the working collector on both monitored interfaces (`ens33`, `ens37`).
- **Adversary emulation:** deployed MITRE Caldera and the Sandcat agent. Diagnosed the Sandcat/Windows 7 incompatibility, ran the agent from Kali, and ran controlled, manually approved discovery abilities (e.g. T1016.001) alongside benign baseline sessions.
- **Dataset engineering:** exported the raw ens33/ens37 data and merged it with the flow records and session manifest into one labeled dataset (68 verified flows, 96 features), with a session manifest that records zero-flow and excluded runs transparently.
- **Data strategy:** added clearly flagged synthetic samples to ease class imbalance, and found the public BigFlow-NIDS dataset for large-scale training, keeping lab captures for validation.
- **Research:** market-gap analysis of NDR / XDR / EDR / IPS / NGFW approaches, used to position SentinelAI.

## Team

A nine-member cross-disciplinary team (Cybersecurity, AI and Data Science), FCIS, Egyptian Chinese University.

## Ethics & scope

All offensive activity ran **only** inside an isolated, self-owned virtual lab with no bridged networking, using low-impact, ATT&CK-mapped discovery techniques for research and education. Nothing in this repository targets real systems.
