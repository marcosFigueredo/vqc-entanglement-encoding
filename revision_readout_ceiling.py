"""
Readout ceiling of circuits without entangling gates
====================================================
Without entangling gates, the measured qubit 0 evolves only under
single-qubit rotations, so <Z0> = n . r(x), where r(x) is the Bloch
vector of the reduced state of qubit 0 after encoding and n is a unit
vector set by the trainable rotations (any direction is reachable with
two or more RY.RZ layers). The best accuracy that ANY such circuit can
reach with the decision rule <Z0> > 0 -> class 1 is therefore

    max_n  acc( sign(n . r(x)) ).

Bloch vector of qubit 0 after encoding (4 qubits, 16 features x):
  angle     : RZ(x2) RY(x0)|0>  ->  (sin x0 cos x2, sin x0 sin x2, cos x0)
  amplitude : |psi> = sum_i x_i |i>, qubit 0 = most significant bit
              r = (2 sum_j x_j x_{j+8}, 0, sum_j x_j^2 - sum_j x_{j+8}^2)
              with x normalized; |r| < 1 when qubit 0 is entangled with
              the rest of the encoded state.

n is chosen on the TRAINING set (dense grid over the sphere) and the
accuracy is reported on the test set, with the partitions of the
cross-task protocol (genalizacao.py: 10 seeds, n in {50,100,200}).

Output: results_revision/readout_ceiling.csv
"""

import os

import numpy as np
import pandas as pd

import genalizacao as G

OUT = "results_revision"
os.makedirs(OUT, exist_ok=True)


def fibonacci_sphere(m=20000):
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


DIRS = fibonacci_sphere()


def bloch_q0(X, encoding):
    X = np.asarray(X)
    if encoding == "angular":
        a, b = X[:, 0], X[:, 2]
        return np.stack([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)], 1)
    v = X[:, :16] / (np.linalg.norm(X[:, :16], axis=1, keepdims=True) + 1e-8)
    top, bot = v[:, :8], v[:, 8:]
    return np.stack([2 * (top * bot).sum(1), np.zeros(len(v)), (top ** 2).sum(1) - (bot ** 2).sum(1)], 1)


def ceiling(r_tr, y_tr, r_te, y_te):
    acc_tr = (((r_tr @ DIRS.T) > 0) == y_tr[:, None]).mean(0)
    k = int(np.argmax(acc_tr))
    return acc_tr[k], float((((r_te @ DIRS[k]) > 0) == y_te).mean())


def main():
    rows = []
    for a, b in G.MNIST_PAIRS:
        for enc in G.ENCODINGS:
            for n in G.TRAIN_SIZES:
                for seed in range(G.N_SEEDS):
                    Xtr, Xte, ytr, yte = G.load_mnist_pair(a, b, n, seed)
                    ytr, yte = ytr.numpy().astype(bool), yte.numpy().astype(bool)
                    r_tr, r_te = bloch_q0(Xtr, enc), bloch_q0(Xte, enc)
                    tr, te = ceiling(r_tr, ytr, r_te, yte)
                    rows.append(dict(dataset=f"mnist_{a}v{b}", encoding=enc, train_size=n, seed=seed,
                                     ceiling_train=tr, ceiling_test=te,
                                     bloch_norm_mean=float(np.linalg.norm(r_te, axis=1).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "readout_ceiling.csv"), index=False)

    obs = pd.read_csv("results_generality/generality_raw.csv")
    obs = obs[obs.topology == "none"].groupby(["dataset", "encoding"]).final_acc.mean()
    s = df.groupby(["dataset", "encoding"]).agg(ceiling=("ceiling_test", "mean"),
                                               ceiling_sd=("ceiling_test", "std"),
                                               bloch_norm=("bloch_norm_mean", "mean"))
    s["observed_no_ent"] = obs
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
