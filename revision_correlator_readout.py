"""
Correlator readout control (second revision)
============================================
Does the gap between circuits without entangling gates and ring circuits
come from the absence of entangling gates or from restricting the readout
to single-qubit observables? Same protocol as revision_multireadout.py
(cross-task protocol, 4 qubits, 3 layers, U[-pi, pi] init, 60 epochs,
seeds 0-4, n in {50, 100, 200}), two additional readouts, each combined by
a trainable linear layer followed by a sigmoid (weights and bias start at 0):

  zz     <Z_i> (4) and <Z_i Z_j> for all pairs i < j (6): 10 features
  probs  the full probability distribution over the 16 computational-basis
         outcomes; a linear function of it is a linear function of all 15
         Z-string correlators, i.e. the most general readout obtainable from
         measurements in the computational basis after the circuit

Topologies: ring and no entangling gates.
Output: results_revision/correlator_readout_raw.csv (resumable)
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
RAW = os.path.join(OUT, "correlator_readout_raw.csv")
SEEDS = list(range(5))
N, L = G.N_QUBITS, G.N_LAYERS
PAIRS = list(combinations(range(N), 2))


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
        if readout == "probs":
            return qml.probs(wires=range(N))
        return ([qml.expval(qml.PauliZ(q)) for q in range(N)]
                + [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in PAIRS])

    return circ


def features(circ, X, w, readout):
    out = circ(X, w)
    return out if readout == "probs" else torch.stack(out, 1)


def run(task, enc, topo, readout, size, seed):
    a, b = task
    Xtr, Xte, ytr, yte = G.load_mnist_pair(a, b, size, seed)
    circ = make_circuit(topo, enc, readout)
    torch.manual_seed(seed)
    np.random.seed(seed)
    w = nn.Parameter(torch.rand(L, N, 2, dtype=torch.float64) * 2 * np.pi - np.pi)
    k = 2 ** N if readout == "probs" else N + len(PAIRS)
    head_a = nn.Parameter(torch.zeros(k, dtype=torch.float64))
    head_c = nn.Parameter(torch.zeros(1, dtype=torch.float64))
    prob = lambda X: torch.sigmoid(features(circ, X, w, readout) @ head_a + head_c)
    opt, bce = torch.optim.Adam([w, head_a, head_c], lr=G.LR), nn.BCELoss()
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
    tasks = [(t, e, tp, r, n, s) for t in G.MNIST_PAIRS for e in G.ENCODINGS for tp in ["none", "ring"]
             for r in ["zz", "probs"] for n in G.TRAIN_SIZES for s in SEEDS]
    for i, (t, e, tp, r, n, s) in enumerate(tasks):
        key = (f"mnist_{t[0]}v{t[1]}", e, tp, r, n, s)
        if key in done:
            continue
        row = run(t, e, tp, r, n, s)
        pd.DataFrame([row]).to_csv(RAW, mode="a", header=not os.path.exists(RAW), index=False)
        print(f"[{i + 1}/{len(tasks)}] {key} acc={row['final_acc']:.3f} ({row['train_time_s']:.1f}s)", flush=True)
    d = pd.concat([pd.read_csv(os.path.join(OUT, "multireadout_raw.csv")), pd.read_csv(RAW)])
    d = d[d.topology.isin(["none", "ring"])]
    s = d.groupby(["dataset", "encoding", "topology", "readout"]).final_acc.mean().unstack(["topology", "readout"])
    s.to_csv(os.path.join(OUT, "correlator_readout_summary.csv"))
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
