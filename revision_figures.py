"""
Figures 2 and 3 of the revised manuscript
=========================================
fig2_convergence.png  — convergence over 60 epochs and final-accuracy
                        distribution by topology (results_fix, 15 seeds)
fig3_crosstask.png    — cross-task accuracy with and without entangling
                        gates (results_generality, corrected), with the
                        readout ceiling of circuits without entangling
                        gates (results_revision/readout_ceiling.csv)
Usage:
    python revision_figures.py            # both figures
    python revision_figures.py --only 2   # one figure
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "results_revision"
TOPO_COLORS = {"ring": "#3B6FB6", "all_to_all": "#C0504D", "none": "#4E9A58"}
TOPO_LABELS = {"ring": "Ring", "all_to_all": "All-to-all", "none": "No entangling gates"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.3)


def fig2():
    r = json.load(open("results_fix/h1_fix_results.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    for topo in ["ring", "all_to_all", "none"]:
        h = np.array([x["history"]["test_acc"] for x in r[topo]])
        ep = np.arange(1, h.shape[1] + 1)
        m, s = h.mean(0), h.std(0, ddof=1)
        a.plot(ep, m, color=TOPO_COLORS[topo], lw=2, label=TOPO_LABELS[topo])
        a.fill_between(ep, m - s, m + s, color=TOPO_COLORS[topo], alpha=0.15, lw=0)
    a.axvline(30, color="0.4", ls=":", lw=1)
    a.text(30.8, 0.965, "30-epoch protocol of the factorial sweep", fontsize=8, color="0.35", va="top")
    a.set(xlim=(1, 60), ylim=(0.45, 1.0), xlabel="Epoch", ylabel="Test accuracy (mean ± SD)",
          title="(a) Convergence over 60 epochs")
    a.legend(frameon=False, loc="lower right")
    style(a)

    data = [[x["final_acc"] for x in r[t]] for t in ["ring", "all_to_all", "none"]]
    bp = b.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=1.5))
    rng = np.random.default_rng(0)
    for i, (t, d) in enumerate(zip(["ring", "all_to_all", "none"], data), 1):
        bp["boxes"][i - 1].set(facecolor=TOPO_COLORS[t], alpha=0.45)
        b.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=14, color="0.2", alpha=0.7, zorder=3)
    b.set_xticks([1, 2, 3], [TOPO_LABELS[t] for t in ["ring", "all_to_all", "none"]])
    b.set(ylim=(0.45, 1.0), ylabel="Final test accuracy", title="(b) Final accuracy distribution")
    style(b)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_convergence.png"), dpi=300)
    print("saved fig2_convergence.png")


def fig3():
    g = pd.read_csv("results_generality/generality_raw.csv")
    g = g.groupby(["dataset", "encoding", "topology", "seed"]).final_acc.mean().reset_index()
    stat = g.groupby(["dataset", "encoding", "topology"]).final_acc.agg(["mean", "std"])
    rc = pd.read_csv(os.path.join(OUT, "readout_ceiling.csv")).groupby(["dataset", "encoding"])
    ceil, bound = rc.ceiling_test.mean(), rc.test_bound.mean()
    tasks = ["mnist_0v1", "mnist_3v5", "mnist_4v9"]
    bars = [("amplitude", "ring", "#2F5F9E", "Amplitude + ring"),
            ("amplitude", "none", "#A9C1E0", "Amplitude + no entangling gates"),
            ("angular", "ring", "#B23A36", "Angle + ring"),
            ("angular", "none", "#E8B4B1", "Angle + no entangling gates")]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.2
    for k, (enc, topo, col, lab) in enumerate(bars):
        xs = np.arange(len(tasks)) + (k - 1.5) * w
        m = [stat.loc[(t, enc, topo), "mean"] for t in tasks]
        s = [stat.loc[(t, enc, topo), "std"] for t in tasks]
        ax.bar(xs, m, w * 0.95, yerr=s, color=col, label=lab, capsize=3, error_kw=dict(lw=1))
        if topo == "none":
            ax.scatter(xs, [ceil.loc[(t, enc)] for t in tasks], marker="_", s=400, color="black", lw=2.2, zorder=4,
                       label="Bloch-vector reference (no entangling gates)" if enc == "amplitude" else None)
            ax.scatter(xs, [bound.loc[(t, enc)] for t in tasks], marker="o", s=30, facecolor="white",
                       edgecolor="black", lw=1.2, zorder=5,
                       label="Exact test-set bound (no entangling gates)" if enc == "amplitude" else None)
    ax.axhline(0.5, color="0.4", ls="--", lw=1, label="Chance level")
    ax.set_xticks(range(len(tasks)), ["0 vs. 1", "3 vs. 5", "4 vs. 9"])
    ax.set(ylim=(0, 1.05), ylabel="Final test accuracy (mean ± SD over seeds)")
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_crosstask.png"), dpi=300)
    print("saved fig3_crosstask.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["2", "3"], default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.only in (None, "2"):
        fig2()
    if args.only in (None, "3"):
        fig3()
