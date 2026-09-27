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
    """Bloch-vector reference: direction chosen on the training set, accuracy on the test set."""
    acc_tr = (((r_tr @ DIRS.T) > 0) == y_tr[:, None]).mean(0)
    k = int(np.argmax(acc_tr))
    return acc_tr[k], float((((r_te @ DIRS[k]) > 0) == y_te).mean())


def _acc(pred, y):
    return float((pred == y).mean())


def best_2d(p, y):
    """Exact max over directions w in the plane of #correct(w . p > 0), p of shape (m, 2).

    The count is constant on the open arcs between consecutive critical angles
    (where w is orthogonal to some nonzero p_k), so evaluating one angle per
    arc is exact. Points with p_k = 0 always give output 0 (class 0)."""
    nz = np.linalg.norm(p, axis=1) > 1e-12
    if not nz.any():
        return int((~y).sum())
    phi = np.arctan2(p[nz, 1], p[nz, 0])
    crit = np.unique(np.round(np.mod(np.concatenate([phi + np.pi / 2, phi - np.pi / 2]), 2 * np.pi), 12))
    mids = crit + np.diff(np.append(crit, crit[0] + 2 * np.pi)) / 2
    W = np.stack([np.cos(mids), np.sin(mids)], 1)
    pred = (p @ W.T) > 0
    return int((pred == y[:, None]).sum(0).max())


def exact_max_2d(r, y):
    """Exact max over unit u of acc(u . r > 0) when r_y = 0 (amplitude encoding).

    u . r = u_x r_x + u_z r_z; directions with u_x = u_z = 0 give output 0 for
    every sample, the others reduce to the planar problem."""
    best = int((~y).sum())
    best = max(best, best_2d(r[:, [0, 2]], y))
    return best / len(y), 0


def exact_max_3d(r, y, tol=1e-9):
    """Exact max over unit u of acc(u . r > 0) for arbitrary (possibly degenerate) points.

    The accuracy is constant on the open regions of the arrangement of great
    circles {u : u . r_k = 0} on the sphere. Every region is adjacent to a vertex
    v = +-(d_i x d_j) of two non-parallel point directions, and the regions
    around v correspond to the open arcs of the planar problem formed by the
    points lying on the plane u . r = 0 through v: points off that plane keep
    the sign of v . r_k, points on it take the sign of w . r_k for a direction w
    in the plane. Solving that planar problem exactly at every vertex gives the
    exact maximum."""
    dirs = np.unique(np.round(r / np.linalg.norm(r, axis=1, keepdims=True), 12), axis=0)
    best, n_on = 0, 0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            v = np.cross(dirs[i], dirs[j])
            if np.linalg.norm(v) < 1e-12:
                continue
            v /= np.linalg.norm(v)
            e1 = dirs[i] - (dirs[i] @ v) * v
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(v, e1)
            proj = r @ v
            on = np.abs(proj) < tol
            n_on = max(n_on, int(on.sum()))
            p_on = np.stack([r[on] @ e1, r[on] @ e2], 1)
            inner = best_2d(p_on, y[on])
            for sv in (1, -1):                      # vertex u = +v or u = -v
                off = ~on
                correct_off = int(((sv * proj[off] > 0) == y[off]).sum())
                best = max(best, correct_off + inner)
    return best / len(y), n_on


def exact_test_bound(r_te, y_te, encoding):
    if encoding == "amplitude":
        return exact_max_2d(r_te, y_te)
    return exact_max_3d(r_te, y_te)


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
                    bound, other = exact_test_bound(r_te, yte, enc)
                    grid_test_max = float((((r_te @ DIRS.T) > 0) == yte[:, None]).mean(0).max())
                    assert grid_test_max <= bound + 1e-12, "grid exceeds exact bound"
                    rows.append(dict(dataset=f"mnist_{a}v{b}", encoding=enc, train_size=n, seed=seed,
                                     ceiling_train=tr, ceiling_test=te, test_bound=bound,
                                     grid_test_max=grid_test_max, max_points_on_vertex_plane=other,
                                     bloch_norm_mean=float(np.linalg.norm(r_te, axis=1).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "readout_ceiling.csv"), index=False)

    raw = pd.read_csv("results_generality/generality_raw.csv")
    raw = raw[raw.topology == "none"].merge(df, on=["dataset", "encoding", "train_size", "seed"])
    exceed = raw[raw.final_acc > raw.test_bound + 1e-5]   # raw accuracies are rounded to 6 decimals
    print(f"trained runs without entangling gates above the exact test bound: {len(exceed)}/{len(raw)}")
    print(f"max points on a single vertex plane (handled exactly): {int(df.max_points_on_vertex_plane.max())}")
    obs = raw.groupby(["dataset", "encoding"]).final_acc.mean()
    s = df.groupby(["dataset", "encoding"]).agg(reference=("ceiling_test", "mean"),
                                               test_bound=("test_bound", "mean"),
                                               grid_test_max=("grid_test_max", "mean"),
                                               bloch_norm=("bloch_norm_mean", "mean"))
    s["observed_no_ent"] = obs
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
