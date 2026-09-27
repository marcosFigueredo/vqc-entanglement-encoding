"""
Experimentos de controle — revisão do manuscrito (set/2026)
===========================================================

E1  Variância real do gradiente, Var_θ[∂θ_k L] sobre inicializações U[-π, π]
    (definição de McClean et al. 2018), n ∈ {2,4,6,8,10}, 3 topologias, L=3.
    Duas codificações angulares:
      - "angular"      : regra original do paper (16 features, índices 4q e 4q+2 mod 16)
      - "angle_pairs"  : 2 features DISTINTAS por qubit, tiradas da imagem 8×8
E2  Controle do Achado 4 (amplitude + all-to-all + 6q):
      - "orig"     : 16 features + zero padding, leitura Z0   (replicação)
      - "full64"   : 64 features (8×8), sem padding, leitura Z0
      - "readout5" : 16 features + zero padding, leitura Z5
E3  Achado 2 limpo (angular + anel + L=3, n ∈ {2,4,6}):
      - "orig"   : regra original (replicação da Tabela V)
      - "grow"   : 2n features distintas (informação cresce com n)
      - "fixed4" : as mesmas 4 features em todos os n (só a largura cresce)

Protocolo de treino idêntico a run_all.py (Adam lr=0.05, 30 épocas, batch completo,
BCE, pesos ~ 0.1·N(0,1) com torch.manual_seed(seed), sementes 0–4, partição por semente).

Uso:
    python control_experiments.py            # tudo
    python control_experiments.py --only E2  # um experimento
Saídas em ./results_controls/
"""

import argparse
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

import run_all as RA  # reutiliza load_mnist_binary e build_circuit originais

OUT = "results_controls"
os.makedirs(OUT, exist_ok=True)

SEEDS = list(range(5))
TRAIN_SIZES = [50, 100, 200, 400]
EPOCHS, LR = RA.EPOCHS, RA.LR


# ─────────────────────────────────────────────────────────────────────────────
# Dados
# ─────────────────────────────────────────────────────────────────────────────

def load(train_size, seed):
    """Retorna X16 (4×4 central, [0,π]), X64 (8×8, [0,π]) e rótulos, na partição original."""
    Xtr16, Xte16, Xtr_cls, Xte_cls, ytr, yte = RA.load_mnist_binary(
        train_size=train_size, seed=seed)
    # X_cls já passou por MinMaxScaler → [0,1] (ajustado no treino); ×π ≡ MinMax para [0,π]
    Xtr64 = Xtr_cls.double() * np.pi
    Xte64 = Xte_cls.double() * np.pi
    # ranking não supervisionado das 64 features pela variância no TREINO (sem vazamento)
    rank = torch.argsort(Xtr64.var(dim=0), descending=True)
    return dict(Xtr16=Xtr16, Xte16=Xte16, Xtr64=Xtr64, Xte64=Xte64,
                ytr=ytr, yte=yte, rank=rank)


# ─────────────────────────────────────────────────────────────────────────────
# Circuito (broadcast sobre o batch) — mesma lógica de run_all.build_circuit
# ─────────────────────────────────────────────────────────────────────────────

def make_circuit(n, L, topo, enc, readout=0):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circ(X, w):
        d = X.shape[1]
        if enc == "angular":                      # regra original do paper
            for q in range(n):
                qml.RY(X[:, (q * 4) % d], wires=q)
                qml.RZ(X[:, (q * 4 + 2) % d], wires=q)
        elif enc == "pairs":                      # 2 features distintas por qubit
            for q in range(n):
                if 2 * q + 1 < d:                 # qubits sem feature ficam em |0>
                    qml.RY(X[:, 2 * q], wires=q)
                    qml.RZ(X[:, 2 * q + 1], wires=q)
        elif enc == "amplitude":                  # idêntico a run_all (normaliza, pad, renormaliza)
            Xn = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
            dim = 2 ** n
            if d < dim:
                Xn = torch.cat([Xn, torch.zeros(X.shape[0], dim - d, dtype=X.dtype)], dim=1)
            else:
                Xn = Xn[:, :dim]
            Xn = Xn / (torch.norm(Xn, dim=1, keepdim=True) + 1e-8)
            qml.StatePrep(Xn, wires=range(n))
        for l in range(L):
            for q in range(n):
                qml.RY(w[l, q, 0], wires=q)
                qml.RZ(w[l, q, 1], wires=q)
            if topo == "ring":
                for q in range(n):
                    qml.CNOT(wires=[q, (q + 1) % n])
            elif topo == "all_to_all":
                for a, b in combinations(range(n), 2):
                    qml.CNOT(wires=[a, b])
        return qml.expval(qml.PauliZ(readout))

    return circ


def validate_against_original():
    """Confere que o circuito broadcast reproduz run_all.build_circuit."""
    torch.manual_seed(123)
    worst = 0.0
    for enc in ["angular", "amplitude"]:
        for topo in ["ring", "all_to_all", "none"]:
            for n in [2, 4, 6]:
                for L in [1, 3]:
                    orig, _ = RA.build_circuit(n, L, topo, enc)
                    mine = make_circuit(n, L, topo, enc)
                    X = torch.rand(4, 16, dtype=torch.float64) * np.pi
                    w = torch.randn(L, n, 2, dtype=torch.float64)
                    a = torch.stack([orig(x, w) for x in X])
                    b = mine(X, w)
                    worst = max(worst, float((a - b).abs().max()))
    print(f"[validação] max |orig - broadcast| = {worst:.2e}")
    assert worst < 1e-8, "circuito broadcast difere do original"
    return worst


# ─────────────────────────────────────────────────────────────────────────────
# Treino — réplica de run_all.VQCClassifier.fit
# ─────────────────────────────────────────────────────────────────────────────

def train(circ, n, L, Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    w = nn.Parameter(torch.randn(L, n, 2, dtype=torch.float64) * 0.1)
    opt = torch.optim.Adam([w], lr=LR)
    bce = nn.BCELoss()
    prob = lambda X: (circ(X, w) + 1) / 2
    with torch.no_grad():
        out_std_init = float(circ(Xte, w).std())
    grad_std0, te_hist = None, []
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = bce(prob(Xtr), ytr)
        loss.backward()
        if ep == 0:
            grad_std0 = float(w.grad.detach().flatten().std())
        opt.step()
        with torch.no_grad():
            te_hist.append(float(((prob(Xte) > 0.5).double() == yte).double().mean()))
    with torch.no_grad():
        out_std_final = float(circ(Xte, w).std())
        tr_acc = float(((prob(Xtr) > 0.5).double() == ytr).double().mean())
    return dict(final_acc=te_hist[-1], best_acc=max(te_hist), train_acc=tr_acc,
                grad_std_ep0=grad_std0, out_std_init=out_std_init,
                out_std_final=out_std_final, test_hist=te_hist)


def run_grid(name, arms, path):
    """arms: lista de dicts com chaves arm, n, L, topo, enc, readout, feats(fn)."""
    done = []
    if os.path.exists(path):
        done = json.load(open(path))
    key = lambda r: (r["arm"], r["n_qubits"], r["n_layers"], r["train_size"], r["seed"])
    seen = {key(r) for r in done}
    total = len(arms) * len(TRAIN_SIZES) * len(SEEDS)
    i = 0
    t0 = time.time()
    for ts in TRAIN_SIZES:
        for sd in SEEDS:
            D = load(ts, sd)
            for a in arms:
                i += 1
                k = (a["arm"], a["n"], a["L"], ts, sd)
                if k in seen:
                    continue
                Xtr, Xte = a["feats"](D)
                circ = make_circuit(a["n"], a["L"], a["topo"], a["enc"], a.get("readout", 0))
                r = train(circ, a["n"], a["L"], Xtr, D["ytr"], Xte, D["yte"], sd)
                r.update(arm=a["arm"], n_qubits=a["n"], n_layers=a["L"], topology=a["topo"],
                         encoding=a["enc"], readout=a.get("readout", 0),
                         n_features=int(Xtr.shape[1]), train_size=ts, seed=sd)
                done.append(r)
                json.dump(done, open(path, "w"), indent=1)
                print(f"[{name} {i}/{total}] {a['arm']:9s} n={a['n']} L={a['L']} ts={ts:3d} "
                      f"sd={sd} acc={r['final_acc']:.3f} out_std={r['out_std_final']:.1e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return done


# ─────────────────────────────────────────────────────────────────────────────
# E2 — controle do Achado 4
# ─────────────────────────────────────────────────────────────────────────────

def run_E2():
    f16 = lambda D: (D["Xtr16"], D["Xte16"])
    f64 = lambda D: (D["Xtr64"], D["Xte64"])
    arms = []
    for L in [1, 2, 3]:
        arms += [
            dict(arm="orig", n=6, L=L, topo="all_to_all", enc="amplitude", readout=0, feats=f16),
            dict(arm="full64", n=6, L=L, topo="all_to_all", enc="amplitude", readout=0, feats=f64),
            dict(arm="readout5", n=6, L=L, topo="all_to_all", enc="amplitude", readout=5, feats=f16),
        ]
    return run_grid("E2", arms, os.path.join(OUT, "E2_achado4_control.json"))


# ─────────────────────────────────────────────────────────────────────────────
# E3 — Achado 2 limpo
# ─────────────────────────────────────────────────────────────────────────────

def run_E3():
    f16 = lambda D: (D["Xtr16"], D["Xte16"])

    def ranked(k):
        return lambda D: (D["Xtr64"][:, D["rank"][:k]], D["Xte64"][:, D["rank"][:k]])

    arms = []
    for n in [2, 4, 6]:
        arms += [
            dict(arm="orig", n=n, L=3, topo="ring", enc="angular", feats=f16),
            dict(arm="grow", n=n, L=3, topo="ring", enc="pairs", feats=ranked(2 * n)),
            dict(arm="fixed4", n=n, L=3, topo="ring", enc="pairs", feats=ranked(4)),
        ]
    return run_grid("E3", arms, os.path.join(OUT, "E3_achado2_clean.json"))


# ─────────────────────────────────────────────────────────────────────────────
# E1 — variância do gradiente sobre inicializações
# ─────────────────────────────────────────────────────────────────────────────

def run_E1(n_inits=200, L=3, train_size=100, data_seed=0):
    path = os.path.join(OUT, "E1_grad_variance.json")
    res = json.load(open(path)) if os.path.exists(path) else []
    seen = {(r["encoding"], r["topology"], r["n_qubits"]) for r in res}
    D = load(train_size, data_seed)
    bce = nn.BCELoss()
    t0 = time.time()
    for enc in ["angular", "pairs"]:
        for topo in ["ring", "all_to_all", "none"]:
            for n in [2, 4, 6, 8, 10]:
                if (enc, topo, n) in seen:
                    continue
                if enc == "angular":
                    X = D["Xtr16"]
                else:
                    X = D["Xtr64"][:, D["rank"][:2 * n]]
                circ = make_circuit(n, L, topo, enc)
                G = torch.zeros(n_inits, L * n * 2, dtype=torch.float64)
                gen = torch.Generator().manual_seed(1000 + n)
                for i in range(n_inits):
                    w = (torch.rand(L, n, 2, dtype=torch.float64, generator=gen) * 2 * np.pi
                         - np.pi).requires_grad_()
                    loss = bce((circ(X, w) + 1) / 2, D["ytr"])
                    loss.backward()
                    G[i] = w.grad.detach().flatten()
                var_k = G.var(dim=0)                        # Var_θ[∂θ_k L] por parâmetro
                active = var_k > 1e-20
                r = dict(
                    encoding=enc, topology=topo, n_qubits=n, n_layers=L,
                    n_inits=n_inits, n_params=int(var_k.numel()),
                    mean_var_all=float(var_k.mean()),
                    mean_var_active=float(var_k[active].mean()),
                    frac_active=float(active.double().mean()),
                    var_theta000=float(var_k[0]),            # RY da 1ª camada, qubit 0
                    var_last_q0_RY=float(var_k[(L - 1) * n * 2]),  # RY da última camada, qubit 0
                    old_proxy=float(G.var(dim=1).mean()),   # métrica da Tabela VII (var entre componentes)
                    var_k=var_k.tolist(),
                )
                res.append(r)
                json.dump(res, open(path, "w"), indent=1)
                print(f"[E1] {enc:7s} {topo:10s} n={n:2d} meanVar={r['mean_var_all']:.2e} "
                      f"active={r['frac_active']:.2f} VarActive={r['mean_var_active']:.2e} "
                      f"Var(theta000)={r['var_theta000']:.2e} ({time.time()-t0:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["E1", "E2", "E3"], default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    validate_against_original()
    for name, fn in [("E2", run_E2), ("E3", run_E3), ("E1", run_E1)]:
        if args.only in (None, name):
            print(f"\n===== {name} =====", flush=True)
            fn()
    print("\nOK — todos os experimentos concluídos.")
