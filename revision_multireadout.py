"""
Multi-observable readout experiment (revision)
==============================================
Tests how much of the encoding/topology/task dependence comes from reading
out a single qubit. Cross-task protocol of genalizacao.py (4 qubits,
3 layers, U[-pi, pi] init, Adam lr 0.05, 60 full-batch epochs, same
partitions), seeds 0-4, n in {50, 100, 200}, two readout arms:

  z0     p = (<Z0> + 1) / 2                        (original readout)
  multi  p = sigmoid(a . (<Z0>, ..., <Z3>) + c)     (4 observables + a
                                                     trainable linear head,
                                                     a = 0, c = 0 at init)

The z0 arm replicates genalizacao.py and is checked against
results_generality/generality_raw.csv.

Output: results_revision/multireadout_raw.csv (resumable)
"""

import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn

import genalizacao as G

OUT = "results_revision"
RAW = os.path.join(OUT, "multireadout_raw.csv")
SEEDS = list(range(5))
N, L = G.N_QUBITS, G.N_LAYERS


def make_circuit(topo, enc, readout):
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(X, w):
        if enc == "angular":
            d = X.shape[1]
            for q in range(N):
                qml.RY(X[:, (q * 4) % d], wires=q)
                qml.RZ(X[:, (q * 4 + 2) % d], wires=q)
        else:
            Xa = X[:, :2 ** N]
            Xa = Xa / (torch.norm(Xa, dim=1, keepdim=True) + 1e-8)
            qml.AmplitudeEmbedding(Xa, wires=range(N), normalize=False)
        for l in range(L):
            for q in range(N):
                qml.RY(w[l, q, 0], wires=q)
                qml.RZ(w[l, q, 1], wires=q)
            if topo == "ring":
                for q in range(N):
                    qml.CNOT(wires=[q, (q + 1) % N])
            elif topo == "all_to_all":
                for a, b in combinations(range(N), 2):
                    qml.CNOT(wires=[a, b])
        if readout == "z0":
            return qml.expval(qml.PauliZ(0))
        return [qml.expval(qml.PauliZ(q)) for q in range(N)]

    return circ


def run(task, enc, topo, readout, size, seed):
    a, b = task
    Xtr, Xte, ytr, yte = G.load_mnist_pair(a, b, size, seed)
    circ = make_circuit(topo, enc, readout)
    torch.manual_seed(seed)
    np.random.seed(seed)
    w = nn.Parameter(torch.rand(L, N, 2, dtype=torch.float64) * 2 * np.pi - np.pi)
    params = [w]
    if readout == "multi":
        head_a = nn.Parameter(torch.zeros(N, dtype=torch.float64))
        head_c = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        params += [head_a, head_c]
        prob = lambda X: torch.sigmoid(torch.stack(circ(X, w), 1) @ head_a + head_c)
    else:
        prob = lambda X: (circ(X, w) + 1) / 2
    opt, bce = torch.optim.Adam(params, lr=G.LR), nn.BCELoss()
    t0 = time.time()
    for _ in range(G.EPOCHS):
        opt.zero_grad()
        bce(prob(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        te = float(((prob(Xte) > 0.5).double() == yte).double().mean())
        tr = float(((prob(Xtr) > 0.5).double() == ytr).double().mean())
    return dict(dataset=f"mnist_{a}v{b}", encoding=enc, topology=topo, readout=readout,
                train_size=size, seed=seed, final_acc=te, train_acc=tr, train_time_s=time.time() - t0)


def main():
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(RAW):
        d = pd.read_csv(RAW)
        done = set(zip(d.dataset, d.encoding, d.topology, d.readout, d.train_size, d.seed))
    torch.set_num_threads(4)
    tasks = [(t, e, tp, r, n, s) for t in G.MNIST_PAIRS for e in G.ENCODINGS for tp in G.TOPOLOGIES
             for r in ["z0", "multi"] for n in G.TRAIN_SIZES for s in SEEDS]
    for i, (t, e, tp, r, n, s) in enumerate(tasks):
        key = (f"mnist_{t[0]}v{t[1]}", e, tp, r, n, s)
        if key in done:
            continue
        row = run(t, e, tp, r, n, s)
        pd.DataFrame([row]).to_csv(RAW, mode="a", header=not os.path.exists(RAW), index=False)
        print(f"[{i + 1}/{len(tasks)}] {key} acc={row['final_acc']:.3f} ({row['train_time_s']:.1f}s)", flush=True)
    summarize()


def summarize():
    d = pd.read_csv(RAW)
    s = d.groupby(["dataset", "encoding", "topology", "readout"]).final_acc.mean().unstack("readout")
    ref = pd.read_csv("results_generality/generality_raw.csv")
    ref = ref[ref.seed.isin(SEEDS)].groupby(["dataset", "encoding", "topology"]).final_acc.mean()
    s["z0_original"] = ref
    s.to_csv(os.path.join(OUT, "multireadout_summary.csv"))
    print(s.round(3).to_string())
    m = d[d.readout == "z0"].merge(pd.read_csv("results_generality/generality_raw.csv"),
                                   on=["dataset", "encoding", "topology", "train_size", "seed"], suffixes=("", "_orig"))
    print(f"\nz0 replication: {int((abs(m.final_acc - m.final_acc_orig) < 1e-6).sum())}/{len(m)} runs identical")


if __name__ == "__main__":
    main()
