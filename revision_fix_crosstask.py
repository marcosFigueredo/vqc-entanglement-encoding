"""
Correction of cross-task runs evaluated on the wrong test split (revision)
==========================================================================
Some runs in the original cross-task data were produced by an earlier
version of the script that held out a fixed test set of 100 samples,
whereas the cross-task protocol of genalizacao.py uses a 25% stratified
split (90, 92, and 91 test samples for 0 vs 1, 3 vs 5, and 4 vs 9). Such
runs are detected because their test accuracy is not a multiple of
1 / (test-set size), or because they belong to the 0 vs 1 ring
configurations, all of which were produced by the earlier version.

Every affected run is repeated with the current protocol, using the batched
replica in revision_multireadout.py (verified to reproduce genalizacao.py
exactly), and the corrected file is written. The original file is kept as
generality_raw_original.csv.
"""

import os
import shutil

import numpy as np
import pandas as pd
import torch

import genalizacao as G
import revision_multireadout as M

DIR = "results_generality"
RAW, ORIG = os.path.join(DIR, "generality_raw.csv"), os.path.join(DIR, "generality_raw_original.csv")


def test_sizes():
    return {f"mnist_{a}v{b}": len(G.load_mnist_pair(a, b, 50, 0)[3]) for a, b in G.MNIST_PAIRS}


def affected(g, sizes):
    k = g.dataset.map(sizes)
    frac = (g.final_acc * k) % 1
    off = ~(np.isclose(frac, 0, atol=1e-3) | np.isclose(frac, 1, atol=1e-3))
    return off | ((g.dataset == "mnist_0v1") & (g.topology == "ring"))


def main():
    if not os.path.exists(ORIG):
        shutil.copy(RAW, ORIG)
    g = pd.read_csv(ORIG)
    sizes = test_sizes()
    mask = affected(g, sizes)
    print("test-set sizes:", sizes, "| runs to repeat:", int(mask.sum()))
    print(g[mask].groupby(["dataset", "encoding", "topology"]).size().to_string())
    torch.set_num_threads(4)
    new = []
    for _, row in g[mask].iterrows():
        a, b = (int(c) for c in row.dataset.split("_")[1].split("v"))
        r = M.run((a, b), row.encoding, row.topology, "z0", int(row.train_size), int(row.seed))
        new.append(dict(dataset=row.dataset, topology=row.topology, encoding=row.encoding,
                        train_size=int(row.train_size), seed=int(row.seed),
                        final_acc=round(r["final_acc"], 6), train_acc=round(r["train_acc"], 6),
                        train_time_s=round(r["train_time_s"], 1), timestamp="rerun-2026-09-27"))
    out = pd.concat([g[~mask], pd.DataFrame(new)], ignore_index=True)
    assert not affected(out[out.timestamp != "rerun-2026-09-27"], sizes).any()
    out.to_csv(RAW, index=False)
    print(out.groupby(["dataset", "encoding", "topology"]).final_acc.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
