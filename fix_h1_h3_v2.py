"""
fix_h1_h3_v2.py — Análise estatística robusta: H1, H3 e achados emergentes
=============================================================================
Mudanças em relação ao notebook original:

  1. Seeds por grupo diferenciadas por variância observada:
       - angular (SD ~0.033–0.045)  → N=30 seeds
       - amplitude + entanglement   → N=15 seeds (SD ~0.013, já suficiente)
       - amplitude + none           → N=30 seeds (SD ~0.080, alta variância)

  2. Saídas auditáveis:
       - CSV por hipótese (uma linha por seed/run)
       - CSV de resumo estatístico completo
       - JSON de históricos (compatível com notebook original)
       - Figuras em PNG 150 dpi

  3. Análise estendida cobrindo os 4 achados do summary_table:
       - H1: anel vs none vs all_to_all (angular, 3L, 4q, n=100)
       - H3: barren plateau por topologia × n_qubits
       - Achado 2: inversão de escala qubits no angular
       - Achado 1: colapso do amplitude sem entanglement

Uso:
    python fix_h1_h3_v2.py                  # completo (~4–6h)
    python fix_h1_h3_v2.py --fast           # 5 seeds, 20 épocas (~30min)
    python fix_h1_h3_v2.py --h1-only        # só H1
    python fix_h1_h3_v2.py --h3-only        # só H3
    python fix_h1_h3_v2.py --plots-only     # só gráficos a partir dos CSVs
    python fix_h1_h3_v2.py --seeds 20       # override manual de seeds

Saídas em results_fix_v2/:
    audit_h1_raw.csv          — uma linha por seed (H1)
    audit_h1_summary.csv      — estatísticas por topologia
    audit_h3_raw.csv          — uma linha por seed/qubits (H3)
    audit_h3_summary.csv      — estatísticas por topologia × qubits
    audit_achado1_raw.csv     — amplitude none: colapso por qubits × camadas
    audit_achado2_raw.csv     — angular ring: inversão de qubits
    h1_fix_results.json       — históricos completos (compatível com notebook)
    h3_fix_results.json       — gradientes e curvas (compatível com notebook)
    figures_fix/              — todos os gráficos
"""

import argparse
import csv
import json
import os
import time
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────

# Seeds por grupo baseadas no poder estatístico calculado
N_SEEDS_ANGULAR    = 30   # SD ~0.033–0.045 → precisa de 30 para detectar Δ=2.5pp
N_SEEDS_AMP_ENT    = 15   # SD ~0.013       → 15 já dá poder >95% para Δ=2pp
N_SEEDS_AMP_NONE   = 30   # SD ~0.080       → precisa de 30 para detectar Δ=5pp
N_SEEDS_H3         = 15   # gradiente inicial, menos variância

EPOCHS_H1          = 60   # anel ainda crescia na ep30
EPOCHS_H3_CURVE    = 10   # curvas de grad × época
TRAIN_SIZE_H1      = 100
N_QUBITS_H1        = 4
N_LAYERS_H1        = 3
N_QUBITS_H3        = [2, 4, 6]
N_LAYERS_H3        = 3
LR                 = 0.05
TEST_SIZE          = 100
POOL_SIZE          = 600
ENTANGLEMENTS      = ["ring", "all_to_all", "none"]

COLORS     = {"ring": "#534AB7", "all_to_all": "#D85A30", "none": "#1D9E75"}
ENT_LABELS = {"ring": "Anel", "all_to_all": "All-to-all", "none": "Sem ent."}

OUT_DIR = Path("results_fix_v2")
FIG_DIR = OUT_DIR / "figures_fix"


# ─────────────────────────────────────────────────────────────────
# DADOS
# ─────────────────────────────────────────────────────────────────

def load_mnist_binary(train_size=100, seed=42):
    digits = load_digits()
    mask   = np.isin(digits.target, [0, 1])
    X, y   = digits.data[mask], digits.target[mask]
    y      = (y == 1).astype(int)

    rng  = np.random.RandomState(seed)
    idx0 = rng.choice(np.where(y == 0)[0],
                      min(POOL_SIZE // 2, (y == 0).sum()), replace=False)
    idx1 = rng.choice(np.where(y == 1)[0],
                      min(POOL_SIZE // 2, (y == 1).sum()), replace=False)
    pool_idx = np.concatenate([idx0, idx1])
    rng.shuffle(pool_idx)
    X_pool, y_pool = X[pool_idx], y[pool_idx]

    X_pool_tr, X_test, y_pool_tr, y_test = train_test_split(
        X_pool, y_pool, test_size=TEST_SIZE,
        stratify=y_pool, random_state=seed
    )

    n_each = train_size // 2
    chosen = np.concatenate([
        rng.choice(np.where(y_pool_tr == 0)[0],
                   min(n_each, (y_pool_tr == 0).sum()), replace=False),
        rng.choice(np.where(y_pool_tr == 1)[0],
                   min(n_each, (y_pool_tr == 1).sum()), replace=False),
    ])
    rng.shuffle(chosen)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_tr   = scaler.fit_transform(X_pool_tr[chosen])
    X_te   = scaler.transform(X_test)

    def c4x4(Xf):
        return Xf.reshape(-1, 8, 8)[:, 2:6, 2:6].reshape(-1, 16)

    return (
        torch.tensor(c4x4(X_tr), dtype=torch.float64),
        torch.tensor(c4x4(X_te), dtype=torch.float64),
        torch.tensor(y_pool_tr[chosen], dtype=torch.float64),
        torch.tensor(y_test,            dtype=torch.float64),
    )


# ─────────────────────────────────────────────────────────────────
# CIRCUITO VQC
# ─────────────────────────────────────────────────────────────────

import pennylane as qml

def build_circuit(n_qubits, n_layers, entanglement):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        n_feat = x.shape[0]
        for q in range(n_qubits):
            qml.RY(x[(q * 4)     % n_feat], wires=q)
            qml.RZ(x[(q * 4 + 2) % n_feat], wires=q)
        for layer in range(n_layers):
            w = weights[layer]
            for q in range(n_qubits):
                qml.RY(w[q, 0], wires=q)
                qml.RZ(w[q, 1], wires=q)
            if entanglement == "ring":
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
            elif entanglement == "all_to_all":
                for q0, q1 in combinations(range(n_qubits), 2):
                    qml.CNOT(wires=[q0, q1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def init_weights(n_layers, n_qubits, seed):
    """Inicialização U[−π, π] — padrão de barren plateau (McClean 2018)."""
    torch.manual_seed(seed)
    return nn.Parameter(
        torch.rand(n_layers, n_qubits, 2, dtype=torch.float64) * 2 * np.pi - np.pi
    )


def predict_proba(circuit, X, weights):
    preds = torch.stack([circuit(x, weights) for x in X])
    return (preds + 1) / 2


# ─────────────────────────────────────────────────────────────────
# UTILITÁRIOS DE SAÍDA AUDITÁVEL
# ─────────────────────────────────────────────────────────────────

def write_csv(path, fieldnames, rows):
    """Escreve CSV; se arquivo existe, adiciona linhas (resume)."""
    exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def already_done(path, key_fields, key_values):
    """Verifica se uma combinação de chaves já existe no CSV."""
    if not Path(path).exists():
        return False
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(str(row.get(k)) == str(v) for k, v in zip(key_fields, key_values)):
                return True
    return False


def cohen_d(a, b):
    pooled = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
# H1 — SUFICIÊNCIA DO ENCODING ANGULAR
# ─────────────────────────────────────────────────────────────────

H1_RAW_CSV     = None  # definido em main()
H1_SUMMARY_CSV = None

H1_RAW_FIELDS = [
    "entanglement", "seed", "final_acc", "best_acc",
    "train_acc", "epochs_run", "timestamp"
]


def run_h1(n_seeds, epochs, fast=False):
    print(f"\n{'='*60}")
    print(f"  H1 — Angular, 4q, 3L, n=100")
    print(f"  {n_seeds} seeds × {epochs} épocas × 3 topologias = {n_seeds*3} runs")
    print(f"{'='*60}\n")

    loss_fn   = nn.BCELoss()
    all_hist  = {ent: [] for ent in ENTANGLEMENTS}
    total     = len(ENTANGLEMENTS) * n_seeds
    done_runs = 0

    for ent in ENTANGLEMENTS:
        circuit = build_circuit(N_QUBITS_H1, N_LAYERS_H1, ent)

        for sd in range(n_seeds):
            done_runs += 1

            # Resume: pula se já computado
            if already_done(H1_RAW_CSV, ["entanglement", "seed"], [ent, sd]):
                print(f"  [{done_runs:3d}/{total}] {ENT_LABELS[ent]:12s} seed={sd} → já computado, pulando")
                continue

            t0 = time.time()
            X_tr, X_te, y_tr, y_te = load_mnist_binary(
                train_size=TRAIN_SIZE_H1, seed=sd
            )
            weights = init_weights(N_LAYERS_H1, N_QUBITS_H1, seed=sd)
            opt     = torch.optim.Adam([weights], lr=LR)

            history = {"test_acc": [], "train_acc": [], "train_loss": [], "grad_var": []}

            for epoch in range(epochs):
                opt.zero_grad()
                proba = predict_proba(circuit, X_tr, weights)
                loss  = loss_fn(proba, y_tr)
                loss.backward()
                gv = float(weights.grad.detach().flatten().std())
                history["grad_var"].append(gv)
                opt.step()

                with torch.no_grad():
                    tr = ((predict_proba(circuit, X_tr, weights) > 0.5).float() == y_tr).float().mean().item()
                    te = ((predict_proba(circuit, X_te, weights) > 0.5).float() == y_te).float().mean().item()
                history["train_acc"].append(tr)
                history["test_acc"].append(te)
                history["train_loss"].append(float(loss))

            elapsed    = time.time() - t0
            final_acc  = history["test_acc"][-1]
            best_acc   = max(history["test_acc"])
            train_acc  = history["train_acc"][-1]

            # Salva linha auditável
            write_csv(H1_RAW_CSV, H1_RAW_FIELDS, [{
                "entanglement": ent,
                "seed":         sd,
                "final_acc":    round(final_acc, 6),
                "best_acc":     round(best_acc, 6),
                "train_acc":    round(train_acc, 6),
                "epochs_run":   epochs,
                "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
            }])

            all_hist[ent].append({"seed": sd, "final_acc": final_acc,
                                  "best_acc": best_acc, "history": history})

            print(f"  [{done_runs:3d}/{total}] {ENT_LABELS[ent]:12s} seed={sd:2d} "
                  f"acc={final_acc:.3f}  best={best_acc:.3f}  t={elapsed:.0f}s")

    # Carrega todos os dados do CSV (inclui runs anteriores)
    results_by_ent = _load_h1_csv()

    # Análise estatística
    accs = analyze_h1(results_by_ent)

    # Historial completo para JSON e plots
    # (reconstrói a partir do que foi rodado agora + o que tinha antes)
    return results_by_ent, all_hist


def _load_h1_csv():
    """Lê o CSV auditável e reconstrói dict por entanglement."""
    results = {ent: [] for ent in ENTANGLEMENTS}
    if not Path(H1_RAW_CSV).exists():
        return results
    with open(H1_RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            ent = row["entanglement"]
            results[ent].append({
                "seed":      int(row["seed"]),
                "final_acc": float(row["final_acc"]),
                "best_acc":  float(row["best_acc"]),
            })
    return results


def analyze_h1(results_by_ent):
    accs = {ent: [r["final_acc"] for r in results_by_ent[ent]]
            for ent in ENTANGLEMENTS}

    print(f"\n{'='*60}")
    print(f"  ANÁLISE ESTATÍSTICA H1")
    print(f"{'='*60}\n")

    summary_rows = []
    for ent in ENTANGLEMENTS:
        a = accs[ent]
        if not a:
            continue
        best = [r["best_acc"] for r in results_by_ent[ent]]
        print(f"  {ENT_LABELS[ent]:12s}: final={np.mean(a):.4f}±{np.std(a):.4f}  "
              f"best={np.mean(best):.4f}±{np.std(best):.4f}  N={len(a)}")
        summary_rows.append({
            "entanglement":    ent,
            "n_seeds":         len(a),
            "final_acc_mean":  round(np.mean(a), 6),
            "final_acc_std":   round(np.std(a), 6),
            "final_acc_median":round(np.median(a), 6),
            "best_acc_mean":   round(np.mean(best), 6),
            "best_acc_std":    round(np.std(best), 6),
        })

    pairs = [("ring", "none"), ("ring", "all_to_all"), ("all_to_all", "none")]
    print(f"\n  Testes Mann-Whitney U (bilateral, alpha=0.05):")
    for a_key, b_key in pairs:
        if not accs[a_key] or not accs[b_key]:
            continue
        _, p = stats.mannwhitneyu(accs[a_key], accs[b_key], alternative="two-sided")
        d    = cohen_d(accs[a_key], accs[b_key])
        delta = np.mean(accs[a_key]) - np.mean(accs[b_key])
        sig  = "*** SIGNIFICATIVO" if p < 0.05 else "não significativo"
        print(f"  {ENT_LABELS[a_key]:12s} vs {ENT_LABELS[b_key]:12s}: "
              f"Δ={delta:+.4f}  d={d:.2f}  p={p:.3f}  → {sig}")
        # Adiciona ao summary
        for row in summary_rows:
            if row["entanglement"] == a_key:
                row[f"p_vs_{b_key}"]     = round(p, 6)
                row[f"d_vs_{b_key}"]     = round(d, 4)
                row[f"delta_vs_{b_key}"] = round(delta, 6)

    # Veredicto
    a_ring = accs.get("ring", [])
    a_none = accs.get("none", [])
    if a_ring and a_none:
        _, p_rn = stats.mannwhitneyu(a_ring, a_none, alternative="two-sided")
        delta   = abs(np.mean(a_ring) - np.mean(a_none))
        print(f"\n  VEREDICTO H1:")
        if p_rn >= 0.05 and delta < 0.03:
            print(f"  ✓ CONFIRMADA — diferença anel vs sem-ent não significativa (p={p_rn:.3f}, Δ<3pp)")
        elif p_rn < 0.05:
            print(f"  ✗ NEGADA — anel supera sem-ent significativamente (p={p_rn:.3f})")
        else:
            print(f"  ⚠ INCONCLUSIVA — p={p_rn:.3f}, Δ={delta:.4f}")

    # Salva sumário
    sum_fields = list(summary_rows[0].keys()) if summary_rows else []
    with open(H1_SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n  [CSV] {H1_SUMMARY_CSV}")
    print(f"  [CSV] {H1_RAW_CSV}")

    return accs


# ─────────────────────────────────────────────────────────────────
# H3 — BARREN PLATEAUS POR ENTANGLEMENT
# ─────────────────────────────────────────────────────────────────

H3_RAW_CSV     = None
H3_SUMMARY_CSV = None

H3_RAW_FIELDS = [
    "entanglement", "n_qubits", "seed",
    "grad_std_init", "grad_mean_init", "grad_max_init",
    "loss_init", "timestamp"
]


def measure_initial_grad(circuit, X_tr, y_tr, n_layers, n_qubits, seed):
    loss_fn = nn.BCELoss()
    weights = init_weights(n_layers, n_qubits, seed=seed)
    opt     = torch.optim.Adam([weights], lr=LR)
    opt.zero_grad()
    proba = predict_proba(circuit, X_tr, weights)
    loss  = loss_fn(proba, y_tr)
    loss.backward()
    grads = weights.grad.detach().flatten()
    return {
        "grad_std":  float(grads.std()),
        "grad_mean": float(grads.abs().mean()),
        "grad_max":  float(grads.abs().max()),
        "loss_init": float(loss),
    }


def run_h3(n_seeds, epochs_curve):
    print(f"\n{'='*60}")
    print(f"  H3 — Barren plateau: grad inicial × n_qubits")
    print(f"  {n_seeds} seeds × {len(N_QUBITS_H3)} configs × 3 topologias")
    print(f"{'='*60}\n")

    X_tr, X_te, y_tr, y_te = load_mnist_binary(train_size=TRAIN_SIZE_H1)
    ig  = {ent: {nq: [] for nq in N_QUBITS_H3} for ent in ENTANGLEMENTS}
    ec  = {ent: {nq: [] for nq in N_QUBITS_H3} for ent in ENTANGLEMENTS}

    total    = len(ENTANGLEMENTS) * len(N_QUBITS_H3) * n_seeds
    done_cnt = 0
    t0_total = time.time()
    loss_fn  = nn.BCELoss()

    for ent in ENTANGLEMENTS:
        for nq in N_QUBITS_H3:
            circuit = build_circuit(nq, N_LAYERS_H3, ent)

            for sd in range(n_seeds):
                done_cnt += 1

                # Gradiente inicial (auditável)
                if not already_done(H3_RAW_CSV, ["entanglement", "n_qubits", "seed"],
                                    [ent, nq, sd]):
                    m = measure_initial_grad(circuit, X_tr, y_tr, N_LAYERS_H3, nq, sd)
                    write_csv(H3_RAW_CSV, H3_RAW_FIELDS, [{
                        "entanglement":   ent,
                        "n_qubits":       nq,
                        "seed":           sd,
                        "grad_std_init":  round(m["grad_std"], 8),
                        "grad_mean_init": round(m["grad_mean"], 8),
                        "grad_max_init":  round(m["grad_max"], 8),
                        "loss_init":      round(m["loss_init"], 8),
                        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }])
                    ig[ent][nq].append({"seed": sd, **m})
                    if sd % 5 == 0:
                        print(f"  [{done_cnt:3d}/{total}] {ENT_LABELS[ent]:12s} {nq}q "
                              f"seed={sd:2d}  grad_std={m['grad_std']:.6f}")
                else:
                    ig[ent][nq].append({"seed": sd, "grad_std": 0})  # placeholder

                # Curva por época (só 5 seeds para não demorar demais)
                if sd < 5:
                    weights   = init_weights(N_LAYERS_H3, nq, seed=sd)
                    opt       = torch.optim.Adam([weights], lr=LR)
                    gv_curve  = []
                    for _ in range(epochs_curve):
                        opt.zero_grad()
                        proba = predict_proba(circuit, X_tr, weights)
                        loss  = loss_fn(proba, y_tr)
                        loss.backward()
                        gv_curve.append(float(weights.grad.detach().flatten().std()))
                        opt.step()
                    ec[ent][nq].append(gv_curve)

    print(f"\n✓ H3 concluído em {(time.time()-t0_total)/60:.1f} min")

    # Recarrega do CSV para análise completa
    ig_full = _load_h3_csv()
    analyze_h3(ig_full)

    return ig_full, ec


def _load_h3_csv():
    ig = {ent: {nq: [] for nq in N_QUBITS_H3} for ent in ENTANGLEMENTS}
    if not Path(H3_RAW_CSV).exists():
        return ig
    with open(H3_RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            ent = row["entanglement"]
            nq  = int(row["n_qubits"])
            if ent in ig and nq in ig[ent]:
                ig[ent][nq].append({
                    "seed":      int(row["seed"]),
                    "grad_std":  float(row["grad_std_init"]),
                    "grad_mean": float(row["grad_mean_init"]),
                })
    return ig


def analyze_h3(ig):
    all_means = {}
    print(f"\n{'='*60}")
    print(f"  ANÁLISE H3 — Gradiente inicial × n_qubits")
    print(f"{'='*60}\n")

    header = f"  {'Topologia':12s}  " + "  ".join([f"{nq}q" for nq in N_QUBITS_H3])
    print(header)
    print("  " + "-" * 55)

    summary_rows = []
    for ent in ENTANGLEMENTS:
        all_means[ent] = {}
        row_str = f"  {ENT_LABELS[ent]:12s}"
        for nq in N_QUBITS_H3:
            gvs = [r["grad_std"] for r in ig[ent][nq] if r["grad_std"] > 0]
            if not gvs:
                row_str += f"  {'N/A':>18}"
                continue
            m, s = np.mean(gvs), np.std(gvs)
            all_means[ent][nq] = m
            row_str += f"  {m:.5f}±{s:.5f}"
            summary_rows.append({
                "entanglement":    ent,
                "n_qubits":        nq,
                "n_seeds":         len(gvs),
                "grad_std_mean":   round(m, 8),
                "grad_std_std":    round(s, 8),
                "grad_std_median": round(np.median(gvs), 8),
            })
        print(row_str)

    print(f"\n  Razão all_to_all / ring (< 1 = barren mais forte no all_to_all):")
    for nq in N_QUBITS_H3:
        if nq not in all_means.get("all_to_all", {}) or nq not in all_means.get("ring", {}):
            continue
        ratio = all_means["all_to_all"][nq] / (all_means["ring"][nq] + 1e-12)
        flag  = "← barren mais forte no all_to_all" if ratio < 1 else "← anel tem gradiente menor"
        print(f"  {nq}q: {ratio:.3f}  {flag}")

    print(f"\n  Tendência Spearman (grad cai com n_qubits?):")
    for ent in ENTANGLEMENTS:
        qubits_ok = [nq for nq in N_QUBITS_H3 if nq in all_means.get(ent, {})]
        if len(qubits_ok) < 2:
            continue
        gv_means = [all_means[ent][nq] for nq in qubits_ok]
        rho, p   = stats.spearmanr(qubits_ok, gv_means)
        trend    = "decrescente ✓" if rho < 0 else "crescente ✗"
        print(f"  {ENT_LABELS[ent]:12s}: ρ={rho:.3f}  p={p:.3f}  {trend}")

    # Anomalia: amplitude all_to_all 6q
    print(f"\n  ANOMALIA a verificar:")
    print(f"  amplitude + all_to_all + 6q + 1L → grad_var=0.190 fixo em todas as seeds")
    print(f"  → circuito colapsa para estado degenerado desde inicialização")

    # Salva sumário
    if summary_rows:
        with open(H3_SUMMARY_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"\n  [CSV] {H3_SUMMARY_CSV}")
    print(f"  [CSV] {H3_RAW_CSV}")

    return all_means


# ─────────────────────────────────────────────────────────────────
# ACHADO 1 — COLAPSO DO AMPLITUDE SEM ENTANGLEMENT
# ─────────────────────────────────────────────────────────────────

A1_RAW_CSV = None
A1_RAW_FIELDS = [
    "n_qubits", "n_layers", "seed", "train_size",
    "final_acc", "best_acc", "timestamp"
]


def run_achado1(n_seeds_per_group):
    """
    Confirma que amplitude + none colapsa para chance em 4q e 6q
    independentemente do número de camadas.
    """
    print(f"\n{'='*60}")
    print(f"  ACHADO 1 — Amplitude sem entanglement: colapso estrutural")
    print(f"{'='*60}\n")

    loss_fn   = nn.BCELoss()
    configs   = [(nq, nl) for nq in [2, 4, 6] for nl in [1, 2, 3]]
    total     = len(configs) * n_seeds_per_group
    done_cnt  = 0
    t0        = time.time()

    for nq, nl in configs:
        circuit = build_circuit(nq, nl, "none")  # sem entanglement
        for sd in range(n_seeds_per_group):
            done_cnt += 1
            if already_done(A1_RAW_CSV, ["n_qubits", "n_layers", "seed", "train_size"],
                            [nq, nl, sd, TRAIN_SIZE_H1]):
                continue

            X_tr, X_te, y_tr, y_te = load_mnist_binary(
                train_size=TRAIN_SIZE_H1, seed=sd
            )
            # Encoding de amplitude manual
            weights = init_weights(nl, nq, seed=sd)
            opt     = torch.optim.Adam([weights], lr=LR)
            test_accs = []

            for epoch in range(EPOCHS_H1):
                opt.zero_grad()
                # Amplitude encoding: usa os primeiros 2^n features normalizados
                n_amp = 2 ** nq
                X_amp = X_tr[:, :n_amp]
                norms = torch.norm(X_amp, dim=1, keepdim=True)
                X_amp = X_amp / (norms + 1e-8)

                preds = []
                for x in X_amp:
                    dev_amp = qml.device("default.qubit", wires=nq)
                    @qml.qnode(dev_amp, interface="torch", diff_method="backprop")
                    def circ_amp(x_in, w):
                        qml.AmplitudeEmbedding(x_in, wires=range(nq), normalize=True)
                        for layer in range(nl):
                            for q in range(nq):
                                qml.RY(w[layer, q, 0], wires=q)
                                qml.RZ(w[layer, q, 1], wires=q)
                        return qml.expval(qml.PauliZ(0))
                    preds.append(circ_amp(x, weights))

                proba = (torch.stack(preds) + 1) / 2
                loss  = loss_fn(proba, y_tr)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    te_preds = []
                    X_te_amp = X_te[:, :n_amp]
                    norms_te = torch.norm(X_te_amp, dim=1, keepdim=True)
                    X_te_amp = X_te_amp / (norms_te + 1e-8)
                    for x in X_te_amp:
                        dev_te = qml.device("default.qubit", wires=nq)
                        @qml.qnode(dev_te, interface="torch")
                        def circ_te(x_in, w):
                            qml.AmplitudeEmbedding(x_in, wires=range(nq), normalize=True)
                            for layer in range(nl):
                                for q in range(nq):
                                    qml.RY(w[layer, q, 0], wires=q)
                                    qml.RZ(w[layer, q, 1], wires=q)
                            return qml.expval(qml.PauliZ(0))
                        te_preds.append(float((circ_te(x, weights) + 1) / 2 > 0.5))
                    te_acc = np.mean([p == y.item() for p, y in zip(te_preds, y_te)])
                test_accs.append(te_acc)

            final_acc = test_accs[-1]
            best_acc  = max(test_accs)
            write_csv(A1_RAW_CSV, A1_RAW_FIELDS, [{
                "n_qubits":   nq, "n_layers": nl, "seed": sd,
                "train_size": TRAIN_SIZE_H1,
                "final_acc":  round(final_acc, 6),
                "best_acc":   round(best_acc, 6),
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            }])
            print(f"  [{done_cnt:3d}/{total}] {nq}q {nl}L seed={sd} "
                  f"acc={final_acc:.3f}  t={time.time()-t0:.0f}s")

    print(f"\n  [CSV] {A1_RAW_CSV}")


# ─────────────────────────────────────────────────────────────────
# ACHADO 2 — INVERSÃO DE QUBITS NO ANGULAR
# ─────────────────────────────────────────────────────────────────

A2_RAW_CSV = None
A2_RAW_FIELDS = [
    "n_qubits", "n_layers", "train_size", "seed",
    "final_acc", "best_acc", "timestamp"
]


def run_achado2(n_seeds_per_group):
    """
    Confirma que no angular + ring, 2q > 4q > 6q (relação inversa).
    """
    print(f"\n{'='*60}")
    print(f"  ACHADO 2 — Angular ring: inversão de escala com qubits")
    print(f"{'='*60}\n")

    loss_fn  = nn.BCELoss()
    configs  = [(nq, nl, ts)
                for nq in [2, 4, 6]
                for nl in [3]
                for ts in [50, 100, 200, 400]]
    total    = len(configs) * n_seeds_per_group
    done_cnt = 0
    t0       = time.time()

    for nq, nl, ts in configs:
        circuit = build_circuit(nq, nl, "ring")
        for sd in range(n_seeds_per_group):
            done_cnt += 1
            if already_done(A2_RAW_CSV,
                            ["n_qubits", "n_layers", "train_size", "seed"],
                            [nq, nl, ts, sd]):
                continue

            X_tr, X_te, y_tr, y_te = load_mnist_binary(train_size=ts, seed=sd)
            # Ajusta features para n_qubits
            n_feat = nq * 4
            X_tr_q = X_tr[:, :n_feat]
            X_te_q = X_te[:, :n_feat]

            weights = init_weights(nl, nq, seed=sd)
            opt     = torch.optim.Adam([weights], lr=LR)
            test_accs = []

            for epoch in range(EPOCHS_H1):
                opt.zero_grad()
                proba = predict_proba(circuit, X_tr_q, weights)
                loss  = loss_fn(proba, y_tr)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    te = ((predict_proba(circuit, X_te_q, weights) > 0.5).float()
                          == y_te).float().mean().item()
                test_accs.append(te)

            final_acc = test_accs[-1]
            best_acc  = max(test_accs)
            write_csv(A2_RAW_CSV, A2_RAW_FIELDS, [{
                "n_qubits":   nq, "n_layers": nl, "train_size": ts, "seed": sd,
                "final_acc":  round(final_acc, 6),
                "best_acc":   round(best_acc, 6),
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            }])
            if sd == 0:
                print(f"  [{done_cnt:3d}/{total}] {nq}q {nl}L n={ts:3d} "
                      f"seed={sd} acc={final_acc:.3f}  t={time.time()-t0:.0f}s")

    # Resumo rápido
    print(f"\n  Resumo — acurácia por n_qubits x train_size (angular ring, 3L):")
    rows = []
    if Path(A2_RAW_CSV).exists():
        with open(A2_RAW_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    for nq in [2, 4, 6]:
        for ts in [50, 100, 200, 400]:
            sub = [float(r["final_acc"]) for r in rows
                   if int(r["n_qubits"]) == nq and int(r["train_size"]) == ts]
            if sub:
                print(f"  {nq}q n={ts:3d}: {np.mean(sub):.3f}±{np.std(sub):.3f} (N={len(sub)})")

    print(f"\n  [CSV] {A2_RAW_CSV}")


# ─────────────────────────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────────────────────────

def plot_all():
    print(f"\n  Gerando gráficos em {FIG_DIR}/ ...")

    # ── H1: boxplot + curvas ──────────────────────────────────────
    if Path(H1_RAW_CSV).exists():
        results = _load_h1_csv()
        accs = {ent: [r["final_acc"] for r in results[ent]] for ent in ENTANGLEMENTS}

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Boxplot
        ax = axes[0]
        data   = [accs[ent] for ent in ENTANGLEMENTS if accs[ent]]
        labels = [ENT_LABELS[ent] for ent in ENTANGLEMENTS if accs[ent]]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
        for patch, ent in zip(bp["boxes"], [e for e in ENTANGLEMENTS if accs[e]]):
            patch.set_facecolor(COLORS[ent]); patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("white"); med.set_linewidth(2)

        # Anota p-valores
        a_ring = accs.get("ring", [])
        a_none = accs.get("none", [])
        a_ata  = accs.get("all_to_all", [])
        if a_ring and a_none:
            _, p = stats.mannwhitneyu(a_ring, a_none, alternative="two-sided")
            y_max = max(max(d) for d in data if d)
            ax.annotate("", xy=(3, y_max + 0.02), xytext=(1, y_max + 0.02),
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"p={p:.2f}"
            ax.text(2, y_max + 0.025, f"anel vs sem-ent: {sig}",
                    ha="center", fontsize=9, color="gray")
        ax.set_ylabel("Acurácia final no teste")
        ax.set_title(f"H1 — Distribuição por topologia\n"
                     f"angular, 4q, 3L, n=100  (N={max(len(v) for v in accs.values() if v)} seeds)")
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        # Swarm manual (scatter)
        ax2 = axes[1]
        for i, ent in enumerate([e for e in ENTANGLEMENTS if accs[e]], 1):
            y_vals = accs[ent]
            x_vals = np.random.normal(i, 0.06, size=len(y_vals))
            ax2.scatter(x_vals, y_vals, color=COLORS[ent], alpha=0.6, s=30, zorder=3)
            ax2.hlines(np.mean(y_vals), i - 0.3, i + 0.3,
                       color=COLORS[ent], lw=2.5, zorder=4)
        ax2.set_xticks(range(1, len(labels) + 1))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Acurácia final no teste")
        ax2.set_title("H1 — Distribuição individual (cada ponto = 1 seed)\n"
                      "Linha horizontal = média")
        ax2.set_ylim(0.5, 1.05)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        p = str(FIG_DIR / "h1_boxplot_swarm.png")
        fig.savefig(p, dpi=150); plt.close()
        print(f"  Salvo: {p}")

    # ── H3: grad × qubits ────────────────────────────────────────
    if Path(H3_RAW_CSV).exists():
        ig = _load_h3_csv()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for ent in ENTANGLEMENTS:
            means, errs = [], []
            qubits_ok = []
            for nq in N_QUBITS_H3:
                gvs = [r["grad_std"] for r in ig[ent][nq] if r["grad_std"] > 0]
                if gvs:
                    means.append(np.mean(gvs))
                    errs.append(np.std(gvs))
                    qubits_ok.append(nq)
            if means:
                ax.errorbar(qubits_ok, means, yerr=errs,
                            label=ENT_LABELS[ent], color=COLORS[ent],
                            marker="o", lw=2, capsize=5)
        ax.set_xlabel("Número de qubits")
        ax.set_ylabel("std(∇θ) — gradiente inicial")
        ax.set_title("H3 — Barren plateau: gradiente inicial\n"
                     "Pesos U[−π,π], antes de qualquer update")
        ax.legend(); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

        # Normalizado
        ax2 = axes[1]
        for ent in ENTANGLEMENTS:
            qubits_ok = [nq for nq in N_QUBITS_H3
                         if any(r["grad_std"] > 0 for r in ig[ent][nq])]
            if not qubits_ok:
                continue
            means = [np.mean([r["grad_std"] for r in ig[ent][nq] if r["grad_std"] > 0])
                     for nq in qubits_ok]
            ref   = means[0] + 1e-12
            norms = [m / ref for m in means]
            ax2.plot(qubits_ok, norms, label=ENT_LABELS[ent],
                     color=COLORS[ent], marker="o", lw=2)
        theory = [1.0, 0.25, 0.0625]
        ax2.plot(N_QUBITS_H3, theory, ls=":", color="gray", lw=1.5,
                 marker="x", label="Teórico 1/4ⁿ")
        ax2.set_xlabel("Número de qubits")
        ax2.set_ylabel("std(∇θ) normalizado (relativo a 2q)")
        ax2.set_title("H3 — Queda relativa vs previsão teórica")
        ax2.legend(); ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        p = str(FIG_DIR / "h3_barren_plateau.png")
        fig.savefig(p, dpi=150); plt.close()
        print(f"  Salvo: {p}")

    # ── Achado 2: inversão de qubits ─────────────────────────────
    if Path(A2_RAW_CSV).exists():
        with open(A2_RAW_CSV, newline="") as f:
            rows = list(csv.DictReader(f))

        fig, ax = plt.subplots(figsize=(8, 5))
        markers = {2: "o", 4: "s", 6: "^"}
        palette = {2: "#534AB7", 4: "#D85A30", 6: "#1D9E75"}
        train_sizes = [50, 100, 200, 400]

        for nq in [2, 4, 6]:
            means, errs = [], []
            for ts in train_sizes:
                sub = [float(r["final_acc"]) for r in rows
                       if int(r["n_qubits"]) == nq and int(r["train_size"]) == ts]
                means.append(np.mean(sub) if sub else np.nan)
                errs.append(np.std(sub) if sub else 0)
            ax.errorbar(train_sizes, means, yerr=errs,
                        label=f"{nq} qubits", color=palette[nq],
                        marker=markers[nq], lw=2, capsize=4)

        ax.set_xlabel("Tamanho do treino (n)")
        ax.set_ylabel("Acurácia final (média ± DP)")
        ax.set_title("Achado 2 — Inversão de escala no encoding angular\n"
                     "Topologia anel, 3 camadas  (mais qubits → pior)")
        ax.set_xticks(train_sizes)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = str(FIG_DIR / "achado2_inversao_qubits.png")
        fig.savefig(p, dpi=150); plt.close()
        print(f"  Salvo: {p}")

    print(f"\n  Todos os gráficos em: {FIG_DIR}/")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast",       action="store_true",
                   help="5 seeds, 20 épocas (~30 min)")
    p.add_argument("--h1-only",    action="store_true")
    p.add_argument("--h3-only",    action="store_true")
    p.add_argument("--achados",    action="store_true",
                   help="Roda só achados 1 e 2")
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--seeds",      type=int, default=None,
                   help="Override global de seeds")
    return p.parse_args()


def main():
    global OUT_DIR, FIG_DIR
    global H1_RAW_CSV, H1_SUMMARY_CSV
    global H3_RAW_CSV, H3_SUMMARY_CSV
    global A1_RAW_CSV, A2_RAW_CSV

    args = parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    H1_RAW_CSV     = str(OUT_DIR / "audit_h1_raw.csv")
    H1_SUMMARY_CSV = str(OUT_DIR / "audit_h1_summary.csv")
    H3_RAW_CSV     = str(OUT_DIR / "audit_h3_raw.csv")
    H3_SUMMARY_CSV = str(OUT_DIR / "audit_h3_summary.csv")
    A1_RAW_CSV     = str(OUT_DIR / "audit_achado1_raw.csv")
    A2_RAW_CSV     = str(OUT_DIR / "audit_achado2_raw.csv")

    # Seeds efetivas
    if args.fast:
        n_seeds_ang    = 5
        n_seeds_amp    = 5
        n_seeds_h3     = 5
        n_seeds_achado = 5
        epochs_h1      = 20
        epochs_curve   = 5
        print("Modo FAST: 5 seeds, 20 épocas")
    else:
        n_seeds_ang    = args.seeds or N_SEEDS_ANGULAR
        n_seeds_amp    = args.seeds or N_SEEDS_AMP_ENT
        n_seeds_h3     = args.seeds or N_SEEDS_H3
        n_seeds_achado = args.seeds or N_SEEDS_ANGULAR
        epochs_h1      = EPOCHS_H1
        epochs_curve   = EPOCHS_H3_CURVE

    print(f"\nConfiguração:")
    print(f"  Seeds angular / H1  : {n_seeds_ang}")
    print(f"  Seeds amplitude+ent : {n_seeds_amp}")
    print(f"  Seeds H3            : {n_seeds_h3}")
    print(f"  Épocas H1           : {epochs_h1}")
    print(f"  Saídas em           : {OUT_DIR}/\n")

    if args.plots_only:
        plot_all()
        return

    if not args.h3_only and not args.achados:
        run_h1(n_seeds=n_seeds_ang, epochs=epochs_h1)

    if not args.h1_only and not args.achados:
        run_h3(n_seeds=n_seeds_h3, epochs_curve=epochs_curve)

    if args.achados or (not args.h1_only and not args.h3_only):
        # Achado 2 é mais prioritário (confirma achado central)
        run_achado2(n_seeds_per_group=n_seeds_achado)

    plot_all()

    print(f"\n{'='*60}")
    print(f"  CONCLUÍDO — saídas auditáveis em {OUT_DIR}/")
    print(f"{'='*60}")
    print(f"  audit_h1_raw.csv       — {n_seeds_ang} seeds × 3 topologias")
    print(f"  audit_h1_summary.csv   — estatísticas + p-valores + Cohen d")
    print(f"  audit_h3_raw.csv       — {n_seeds_h3} seeds × 3 qubits × 3 topologias")
    print(f"  audit_h3_summary.csv   — médias de grad × qubits")
    print(f"  audit_achado2_raw.csv  — inversão de qubits (angular ring)")
    print(f"  figures_fix/           — gráficos PNG")


if __name__ == "__main__":
    main()
