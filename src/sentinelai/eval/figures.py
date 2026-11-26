"""Figures for reports/phase3/figures/, drawn only from the result dicts (headless matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN = "#2563eb", "#ea580c", "#9ca3af", "#dc2626", "#16a34a"
SRC_COLORS = {"CSE-CIC-IDS2018": "#2563eb", "ToN-IoT": "#16a34a", "BoT-IoT": "#ea580c", "UNSW-NB15": "#9333ea"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def confusion_heatmap(conf: dict, title: str, path: Path) -> Path:
    labels, m = conf["labels"], np.array(conf["row_normalised"])
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(labels), 0.9 + 0.65 * len(labels)))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    counts = np.array(conf["counts"])
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j]:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] > 0.55 else "black")
    ax.set_xlabel("predicted"), ax.set_ylabel("true (row-normalised)"), ax.set_title(title, fontsize=10)
    return _save(fig, path)


def roc_pr(curves: dict[str, dict], title: str, path: Path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, c in curves.items():
        if not c:
            continue
        a.plot(c["roc"]["fpr"], c["roc"]["tpr"], label=name, lw=1.4)
        b.plot(c["pr"]["recall"], c["pr"]["precision"], label=name, lw=1.4)
    a.plot([0, 1], [0, 1], ls=":", c=GREY)
    a.set_xscale("symlog", linthresh=1e-3), a.set_xlim(0, 1)
    a.set_xlabel("false-positive rate (symlog)"), a.set_ylabel("true-positive rate"), a.set_title("ROC (test)")
    b.set_xlabel("recall"), b.set_ylabel("precision"), b.set_title("Precision-recall (test)")
    a.legend(fontsize=7), b.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def per_class_bars(per_class: dict, title: str, path: Path) -> Path:
    items = sorted(((k, v) for k, v in per_class.items() if k != "Benign"), key=lambda kv: kv[1]["flagged"])
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(items) + 1.2))
    cols = [SRC_COLORS.get(v["source"], GREY) for _, v in items]
    ax.barh([f"{k} (n={v['n']:,})" for k, v in items], [v["flagged"] for _, v in items], color=cols)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=s) for s, c in SRC_COLORS.items() if any(v["source"] == s for _, v in items)]
    if any(v["source"] not in SRC_COLORS for _, v in items):
        handles.append(Patch(color=GREY, label="several sources / other"))
    if "Benign" in per_class:
        handles.append(ax.axvline(per_class["Benign"]["flagged"], c=RED, ls="--", lw=1,
                                  label=f"benign FPR {per_class['Benign']['flagged']:.3f}"))
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.set_xlim(0, 1), ax.set_xlabel("share detected (test)"), ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=10)
    return _save(fig, path)


def grouped_bars(groups: list[str], series: dict[str, list], title: str, ylabel: str, path: Path,
                 colors=None, ylim=(0, 1)) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(groups)), 4))
    w = 0.8 / max(len(series), 1)
    for i, (name, vals) in enumerate(series.items()):
        x = np.arange(len(groups)) + i * w - 0.4 + w / 2
        v = [np.nan if x_ is None else x_ for x_ in vals]
        ax.bar(x, v, w, label=name, color=(colors or {}).get(name))
        for xi, vi in zip(x, v):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(groups)), groups, fontsize=8)
    ax.set_ylim(*ylim), ax.set_ylabel(ylabel), ax.set_title(title, fontsize=10), ax.legend(fontsize=7)
    return _save(fig, path)


def calibration_plot(cals: dict[str, dict], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], ls=":", c=GREY)
    for name, c in cals.items():
        b = c["bins"]
        ax.plot([x["mean_predicted"] for x in b], [x["observed_attack_rate"] for x in b], marker="o", ms=3,
                label=f"{name} (ECE {c['ece']:.3f})")
    ax.set_xlabel("predicted attack probability"), ax.set_ylabel("observed attack rate (test)")
    ax.set_title("Calibration", fontsize=10), ax.legend(fontsize=7)
    return _save(fig, path)


def score_histograms(t, threshold: float, title: str, path: Path) -> Path:
    """Autoencoder score distributions per source, benign vs attack (test)."""
    te = t[t.part == "test"]
    srcs = sorted(te.source.unique())
    fig, axs = plt.subplots(1, len(srcs), figsize=(3.2 * len(srcs), 3), sharey=False)
    bins = np.logspace(np.log10(max(te.anomaly_score.min(), 1e-4)), np.log10(te.anomaly_score.max() + 1e-9), 60)
    for ax, s in zip(np.atleast_1d(axs), srcs):
        g = te[te.source == s]
        for lab, col, name in ((0, BLUE, "benign"), (1, ORANGE, "attack")):
            x = g.anomaly_score[g.Label == lab]
            if len(x):
                ax.hist(x, bins=bins, color=col, alpha=0.55, density=True, label=f"{name} ({len(x):,})")
        ax.axvline(threshold, c=RED, lw=1)
        ax.set_xscale("log"), ax.set_title(s, fontsize=9), ax.legend(fontsize=6), ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)
