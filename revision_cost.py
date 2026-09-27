"""
Compiled circuit cost (revision)
================================
Decomposes the factorial circuits of run_all.py into {CNOT, RY, RZ, RX,
PhaseShift} with PennyLane and counts two-qubit gates and depth,
separately for state preparation (encoding) and for the variational
layers. Amplitude encoding is decomposed with PennyLane's default
StatePrep decomposition (Mottonen et al.) for a generic dense real input.

Output: results_revision/S6_compiled_cost.csv
"""

import os
from functools import partial

import numpy as np
import pandas as pd
import pennylane as qml
import torch

import run_all as RA

OUT = "results_revision"
os.makedirs(OUT, exist_ok=True)
GATES = {"CNOT", "RY", "RZ", "RX", "PhaseShift", "GlobalPhase"}


def count(n, L, topo, enc):
    circuit, _ = RA.build_circuit(n, L, topo, enc)
    compiled = partial(qml.transforms.decompose, gate_set=GATES)(circuit)
    rng = np.random.default_rng(0)
    x = torch.tensor(np.pi * rng.random(16))      # generic dense input
    w = torch.tensor(rng.normal(size=(max(L, 1), n, 2)))
    res = qml.specs(compiled)(x, w)["resources"]
    return res.gate_types.get("CNOT", 0), res.depth, res.num_gates


def main():
    rows = []
    for enc in ["angular", "amplitude"]:
        for n in [2, 4, 6]:
            enc_cnot, enc_depth, _ = count(n, 0, "none", enc)
            for topo in ["ring", "all_to_all", "none"]:
                for L in [1, 3]:
                    cnot, depth, gates = count(n, L, topo, enc)
                    rows.append(dict(encoding=enc, n_qubits=n, topology=topo, n_layers=L,
                                     cnot_encoding=enc_cnot, cnot_variational=cnot - enc_cnot,
                                     cnot_total=cnot, depth_total=depth, gates_total=gates))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S6_compiled_cost.csv"), index=False)
    print(df[df.n_layers == 3].to_string(index=False))


if __name__ == "__main__":
    main()
