"""
Statistical analyses added in the manuscript revision
=====================================================
S1  Topology comparison (results_fix_v2, 30 seeds): paired differences by
    seed, Holm-corrected Mann-Whitney p-values, and a paired TOST
    equivalence test for ring vs all-to-all with margin DELTA.
S2  Cross-task ring vs all-to-all (results_generality): runs are averaged
    per seed over training sizes (the sizes share data partitions), then
    paired Wilcoxon signed-rank tests with Holm correction, 90% CI of the
    paired difference, and TOST with margin DELTA.
S3  Gradient-variance decay (results_controls/E1): slope b of log2 Var vs n
    with a t-based 95% CI (regression over n) and a bootstrap 95% CI
    (resampling the active parameters at each n).
S4  Classical baselines trained on the SAME features given to the VQCs
    (16 central pixels; 4 pixels used by the 2-qubit amplitude circuits),
    with the partitions, seeds, and training protocol of run_all.py.
S5  VQC selected WITHOUT the test set (highest mean training accuracy) and
    comparison with classical models using seed-level paired differences.

Outputs in ./results_revision/
"""

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

import run_all as RA

OUT = "results_revision"
os.makedirs(OUT, exist_ok=True)
DELTA = 0.03          # equivalence margin (3 percentage points)
SEEDS = list(range(5))
SIZES = [50, 100, 200, 400]   # 400 is the nominal label of the 260-sample level


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tost(diff, delta=DELTA):
    """Two one-sided t-tests on paired differences; returns p_TOST and 90% CI."""
    diff = np.asarray(diff, float)
    n, m, se = len(diff), diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
    t_lo, t_hi = (m + delta) / se, (m - delta) / se
    p = max(1 - stats.t.cdf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    h = stats.t.ppf(0.95, n - 1) * se
    return p, m - h, m + h


# ── S1 ────────────────────────────────────────────────────────────────────
def s1():
    d = pd.read_csv("results_fix_v2/audit_h1_raw.csv").pivot(index="seed", columns="entanglement", values="final_acc")
    rows = []
    for a, b in [("ring", "none"), ("all_to_all", "none"), ("ring", "all_to_all")]:
        diff = d[a] - d[b]
        p_mw = stats.mannwhitneyu(d[a], d[b], alternative="two-sided").pvalue
        p_t, lo, hi = paired_tost(diff)
        rows.append(dict(comparison=f"{a} - {b}", mean_diff=diff.mean(), ci90_lo=lo, ci90_hi=hi,
                         p_mannwhitney=p_mw, p_tost=p_t))
    df = pd.DataFrame(rows)
    df["p_mannwhitney_holm"] = holm(df.p_mannwhitney)
    df.to_csv(os.path.join(OUT, "S1_topology_tost.csv"), index=False)
    print("\nS1 topology comparison (30 seeds)\n", df.round(4).to_string(index=False))


# ── S2 ────────────────────────────────────────────────────────────────────
def s2():
    g = pd.read_csv("results_generality/generality_raw.csv")
    g = g.groupby(["dataset", "encoding", "topology", "seed"]).final_acc.mean().unstack("topology")
    rows = []
    for (ds, enc), s in g.groupby(level=[0, 1]):
        diff = s["ring"] - s["all_to_all"]
        p_w = stats.wilcoxon(s["ring"], s["all_to_all"]).pvalue if (diff != 0).any() else 1.0
        p_t, lo, hi = paired_tost(diff)
        rows.append(dict(dataset=ds, encoding=enc, ring=s["ring"].mean(), all_to_all=s["all_to_all"].mean(),
                         mean_diff=diff.mean(), ci90_lo=lo, ci90_hi=hi, p_wilcoxon=p_w, p_tost=p_t))
    df = pd.DataFrame(rows)
    df["p_wilcoxon_holm"] = holm(df.p_wilcoxon)
    df.to_csv(os.path.join(OUT, "S2_crosstask_ring_vs_a2a.csv"), index=False)
    print("\nS2 cross-task ring vs all-to-all (seed means, 10 seeds)\n", df.round(4).to_string(index=False))


# ── S3 ────────────────────────────────────────────────────────────────────
def s3(n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    e = json.load(open("results_controls/E1_grad_variance.json"))
    rows = []
    for enc in ["angular", "pairs"]:
        for topo in ["ring", "all_to_all", "none"]:
            recs = sorted([r for r in e if r["encoding"] == enc and r["topology"] == topo], key=lambda r: r["n_qubits"])
            ns = np.array([r["n_qubits"] for r in recs], float)
            act = [np.array(r["var_k"])[np.array(r["var_k"]) > 1e-20] for r in recs]
            y = np.log2([a.mean() for a in act])
            res = stats.linregress(ns, y)
            h = stats.t.ppf(0.975, len(ns) - 2) * res.stderr
            boot = []
            for _ in range(n_boot):
                yb = np.log2([rng.choice(a, len(a)).mean() for a in act])
                boot.append(np.polyfit(ns, yb, 1)[0])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(encoding=enc, topology=topo, slope_b=res.slope, t_ci_lo=res.slope - h,
                             t_ci_hi=res.slope + h, boot_ci_lo=lo, boot_ci_hi=hi,
                             factor_per_qubit=2 ** res.slope))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S3_gradvar_slope_ci.csv"), index=False)
    print("\nS3 gradient-variance slope\n", df.round(3).to_string(index=False))


# ── S4 ────────────────────────────────────────────────────────────────────
class CNN4(nn.Module):
    """CNN for the 4x4 central region (same layer widths as run_all.TinyCNN)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 2 * 2, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_cnn4(Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    m = CNN4()
    opt, bce = torch.optim.Adam(m.parameters(), lr=RA.LR), nn.BCELoss()
    Xtr, Xte = Xtr.view(-1, 1, 4, 4).float(), Xte.view(-1, 1, 4, 4).float()
    for _ in range(RA.EPOCHS):
        m.train(); opt.zero_grad()
        bce(m(Xtr), ytr.float()).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return float(((m(Xte) > 0.5).float() == yte.float()).float().mean())


def s4():
    rows = []
    for size in SIZES:
        for seed in SEEDS:
            Xtr16, Xte16, _, _, ytr, yte = RA.load_mnist_binary(train_size=size, seed=seed)
            for feat, Xtr, Xte in [("16px", Xtr16, Xte16), ("4px", Xtr16[:, :4], Xte16[:, :4])]:
                Xtr_f, Xte_f = Xtr.float(), Xte.float()
                rows.append(dict(features=feat, model="svm", train_size=size, seed=seed,
                                 final_acc=RA.train_svm(Xtr_f, ytr, Xte_f, yte, seed=seed)["final_acc"]))
                rows.append(dict(features=feat, model="mlp", train_size=size, seed=seed,
                                 final_acc=RA.train_mlp(Xtr_f, ytr, Xte_f, yte, seed=seed)["final_acc"]))
                if feat == "16px":
                    rows.append(dict(features=feat, model="cnn", train_size=size, seed=seed,
                                     final_acc=train_cnn4(Xtr, ytr, Xte, yte, seed)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S4_classical_same_features.csv"), index=False)
    print("\nS4 classical models on the VQC features\n",
          df.groupby(["features", "model", "train_size"]).final_acc.agg(["mean", "std"]).unstack("train_size").round(3).to_string())
    return df


# ── S5 ────────────────────────────────────────────────────────────────────
def s5(same_feat):
    r = json.load(open("results/results.json"))
    v = pd.DataFrame([{k: x[k] for k in ["encoding", "entanglement", "n_qubits", "n_layers", "train_size", "seed",
                                         "final_acc", "train_acc"]} for x in r["vqc"]])
    cfg = ["encoding", "entanglement", "n_qubits", "n_layers"]
    by_train = v.groupby(cfg).train_acc.mean().sort_values(ascending=False)
    sel = by_train.index[0]
    print("\nS5 VQC selected by mean TRAINING accuracy:", sel, f"(train acc {by_train.iloc[0]:.3f})")
    top = v.groupby(cfg).agg(train_acc=("train_acc", "mean"), test_acc=("final_acc", "mean")).sort_values("train_acc", ascending=False).head(5)
    print(top.round(3).to_string())
    vs = v[(v[cfg] == pd.Series(sel, index=cfg)).all(1)][["train_size", "seed", "final_acc"]].assign(model="vqc_selected")

    c = pd.DataFrame([{k: x[k] for k in ["model", "train_size", "seed", "final_acc"]} for x in r["classical"]])
    c["model"] = c.model + "_8x8"
    s = same_feat[same_feat.features == "16px"].assign(model=lambda d: d.model + "_16px")[["model", "train_size", "seed", "final_acc"]]
    allm = pd.concat([vs, c, s])
    per_seed = allm.groupby(["model", "seed"]).final_acc.mean().unstack("model")
    rows = []
    for m in [x for x in per_seed.columns if x != "vqc_selected"]:
        diff = per_seed[m] - per_seed["vqc_selected"]
        h = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        by_size = (allm[allm.model == m].set_index(["train_size", "seed"]).final_acc
                   - allm[allm.model == "vqc_selected"].set_index(["train_size", "seed"]).final_acc)
        rows.append(dict(model=m, mean_acc=allm[allm.model == m].final_acc.mean(), diff_vs_vqc=diff.mean(),
                         ci95_lo=diff.mean() - h, ci95_hi=diff.mean() + h,
                         seeds_better=int((diff > 0).sum()), seeds_worse=int((diff < 0).sum()),
                         runs_better=int((by_size > 0).sum()), runs_tied=int((by_size == 0).sum()),
                         runs_worse=int((by_size < 0).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S5_selected_vqc_vs_classical.csv"), index=False)
    print(f"\nselected VQC mean test acc = {vs.final_acc.mean():.3f}")
    print(df.round(3).to_string(index=False))
    per_size = allm.groupby(["model", "train_size"]).final_acc.mean().unstack("train_size")
    per_size.to_csv(os.path.join(OUT, "S5_mean_by_size.csv"))
    print(per_size.round(3).to_string())


if __name__ == "__main__":
    s1(); s2(); s3()
    sf = s4()
    s5(sf)
