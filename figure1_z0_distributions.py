"""
Figure 1 — distribution of <Z0> by class on the test set (digits 0 vs 1)
=======================================================================
Retrains six configurations of the factorial sweep (3 layers, n = 100,
seeds 0–4) with exactly the factorial protocol of run_all.py
(N(0, 0.1^2) init, Adam lr = 0.05, 30 full-batch epochs), records the
circuit output <Z0> for every test sample, and checks that each final
accuracy reproduces the value stored in results/results.json.
Test outputs are pooled over the 5 seeds (500 samples per panel).

Outputs in ./results_z0/:
    z0_data.csv            — one row per (config, seed, test sample)
    z0_distributions.png   — Figure 1
Usage:
    python figure1_z0_distributions.py
    python figure1_z0_distributions.py --plots-only
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import run_all as RA
from control_experiments import make_circuit, validate_against_original

OUT = "results_z0"
SEEDS = list(range(5))
TRAIN_SIZE, N_LAYERS = 100, 3

# (label, encoding, topology, n_qubits)
CONFIGS = [
    ("Amplitude + ring, 4 qubits",           "amplitude", "ring", 4),
    ("Amplitude + no entangling gates, 4 qubits", "amplitude", "none", 4),
    ("Angle + ring, 2 qubits",               "angular",   "ring", 2),
    ("Angle + ring, 4 qubits",               "angular",   "ring", 4),
    ("Angle + ring, 6 qubits",               "angular",   "ring", 6),
    ("Angle + no entangling gates, 4 qubits",    "angular",   "none", 4),
]


def train_and_record(enc, topo, n, seed):
    """Replica of run_all.VQCClassifier.fit that also returns the test outputs."""
    Xtr, Xte, _, _, ytr, yte = RA.load_mnist_binary(train_size=TRAIN_SIZE, seed=seed)
    circ = make_circuit(n, N_LAYERS, topo, enc)
    torch.manual_seed(seed)
    w = nn.Parameter(torch.randn(N_LAYERS, n, 2, dtype=torch.float64) * 0.1)
    opt = torch.optim.Adam([w], lr=RA.LR)
    bce = nn.BCELoss()
    for _ in range(RA.EPOCHS):
        opt.zero_grad()
        bce((circ(Xtr, w) + 1) / 2, ytr).backward()
        opt.step()
    with torch.no_grad():
        z0 = circ(Xte, w).numpy()
    y = yte.numpy().astype(int)
    pred = ((z0 + 1) / 2 > 0.5).astype(int)
    return z0, y, pred


def factorial_reference():
    ref = {}
    for r in json.load(open(os.path.join(RA.OUT_DIR, "results.json")))["vqc"]:
        if r["train_size"] == TRAIN_SIZE and r["n_layers"] == N_LAYERS:
            ref[(r["encoding"], r["entanglement"], r["n_qubits"], r["seed"])] = r["final_acc"]
    return ref


def run():
    validate_against_original()
    ref = factorial_reference()
    rows = []
    for label, enc, topo, n in CONFIGS:
        for seed in SEEDS:
            z0, y, pred = train_and_record(enc, topo, n, seed)
            acc = float((pred == y).mean())
            expected = ref[(enc, topo, n, seed)]
            status = "ok" if abs(acc - expected) < 1e-6 else "MISMATCH"
            print(f"{label:40s} seed={seed}  acc={acc:.3f}  factorial={expected:.3f}  {status}")
            rows += [dict(config=label, encoding=enc, topology=topo, n_qubits=n,
                          seed=seed, sample=i, Z0=float(z0[i]), y_true=int(y[i]),
                          y_pred=int(pred[i])) for i in range(len(y))]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "z0_data.csv"), index=False)
    return df


def plot(df):
    colors = {0: "#3B6FB6", 1: "#E07B39"}
    bins = np.linspace(-1, 1, 41)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for k, (ax, (label, *_)) in enumerate(zip(axes.flat, CONFIGS)):
        d = df[df.config == label]
        accs = d.assign(c=d.y_true == d.y_pred).groupby("seed").c.mean()
        for cls in (0, 1):
            ax.hist(d.Z0[d.y_true == cls], bins=bins, color=colors[cls], alpha=0.6,
                    edgecolor="white", linewidth=0.4, label=f"digit {cls}")
        ax.axvline(0, color="0.25", ls="--", lw=1)
        ax.set_title(f"({'abcdef'[k]}) {label}\naccuracy = {accs.mean():.3f} ± {accs.std(ddof=1):.3f}",
                     fontsize=10)
        ax.set_xlim(-1, 1)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.3)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if k >= 3:
            ax.set_xlabel(r"$\langle Z_0 \rangle$")
    axes.flat[0].legend(frameon=False, fontsize=9, loc="upper center")
    fig.tight_layout()
    path = os.path.join(OUT, "z0_distributions.png")
    fig.savefig(path, dpi=300)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(OUT, "z0_data.csv")) if args.plots_only else run()
    plot(df)
