"""
fix_h1_h3.py — Melhoria cirúrgica de H1 e H3
=============================================
Roda SOMENTE o que é necessário para resolver os dois problemas
diagnosticados, sem repetir os 40h do sweep completo.

Problemas e soluções:
  H1 → N=5 seeds (poder ~25%) + 30 épocas (anel ainda crescendo)
       Fix: N=15 seeds + 60 épocas, só n=100, 4q, 3L, encoding angular
       Teste: Mann-Whitney U (mais robusto que t-test para N pequeno)

  H3 → grad_var medida na ep30 (convergência ≠ barren plateau)
       Fix: medir grad ANTES de qualquer update (pesos aleatórios)
       Análise: grad_var(ep1) × n_qubits para as 3 topologias

Tempo estimado: ~3–5h CPU  (vs 40h do sweep completo)

Uso:
    python fix_h1_h3.py              # roda tudo
    python fix_h1_h3.py --h1-only   # só H1
    python fix_h1_h3.py --h3-only   # só H3
    python fix_h1_h3.py --fast      # 5 seeds, 30 épocas — valida em ~30 min

Saídas em ./results_fix/:
    h1_fix_results.json
    h3_fix_results.json
    figures_fix/h1_*.png
    figures_fix/h3_*.png
"""

import argparse
import json
import os
import time
import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pennylane as qml

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — apenas o necessário para H1 e H3
# ─────────────────────────────────────────────────────────────────────────────

SEED          = 42
TRAIN_SIZE_H1 = 100          # H1: n fixo em 100 (pior caso para poder estatístico)
N_QUBITS_H1   = 4
N_LAYERS_H1   = 3
ENCODING_H1   = "angular"    # encoding que estava com resultado inconclusivo
ENTANGLEMENTS = ["ring", "all_to_all", "none"]

# H1: mais seeds + mais épocas
N_SEEDS_FIX   = 15           # de 5 → 15  (poder ~25% → ~70%)
EPOCHS_FIX    = 60           # de 30 → 60 (anel ainda crescia na ep30)
LR            = 0.05

# H3: sweep de qubits com medição na época 1
N_QUBITS_H3   = [2, 4, 6]   # mesmo do sweep original
N_LAYERS_H3   = 3
N_SEEDS_H3    = 10           # mais seeds para H3 (grad na ep1 é muito ruidoso)
EPOCHS_H3     = 1            # só precisa de 1 época para medir o gradiente inicial
TEST_SIZE     = 100
POOL_SIZE     = 600

OUT_DIR = "results_fix"
FIG_DIR = os.path.join(OUT_DIR, "figures_fix")
os.makedirs(FIG_DIR, exist_ok=True)

# Paleta consistente com o pipeline original
COLORS = {
    "ring":       "#534AB7",
    "all_to_all": "#D85A30",
    "none":       "#1D9E75",
}
ENT_LABELS = {"ring": "Anel", "all_to_all": "All-to-all", "none": "Sem ent."}


# ─────────────────────────────────────────────────────────────────────────────
# DADOS — copiado diretamente de run_all.py para não criar dependência
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_binary(train_size=100, seed=SEED):
    digits = load_digits()
    mask = np.isin(digits.target, [0, 1])
    X, y = digits.data[mask], digits.target[mask]
    y = (y == 1).astype(int)

    rng = np.random.RandomState(seed)
    idx0 = rng.choice(np.where(y == 0)[0], min(POOL_SIZE // 2, (y==0).sum()), replace=False)
    idx1 = rng.choice(np.where(y == 1)[0], min(POOL_SIZE // 2, (y==1).sum()), replace=False)
    pool_idx = np.concatenate([idx0, idx1])
    rng.shuffle(pool_idx)
    X_pool, y_pool = X[pool_idx], y[pool_idx]

    X_pool_tr, X_test, y_pool_tr, y_test = train_test_split(
        X_pool, y_pool, test_size=TEST_SIZE, stratify=y_pool, random_state=seed
    )

    n_each_tr = train_size // 2
    idx0_tr = np.where(y_pool_tr == 0)[0]
    idx1_tr = np.where(y_pool_tr == 1)[0]
    chosen = np.concatenate([
        rng.choice(idx0_tr, min(n_each_tr, len(idx0_tr)), replace=False),
        rng.choice(idx1_tr, min(n_each_tr, len(idx1_tr)), replace=False),
    ])
    rng.shuffle(chosen)
    X_train_raw = X_pool_tr[chosen]
    y_train     = y_pool_tr[chosen]

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_raw = scaler.fit_transform(X_train_raw)
    X_test      = scaler.transform(X_test)

    def central_4x4(Xf):
        return Xf.reshape(-1, 8, 8)[:, 2:6, 2:6].reshape(-1, 16)

    X_train_vqc = torch.tensor(central_4x4(X_train_raw), dtype=torch.float64)
    X_test_vqc  = torch.tensor(central_4x4(X_test),      dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    y_te = torch.tensor(y_test,  dtype=torch.float64)
    return X_train_vqc, X_test_vqc, y_tr, y_te


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUITO — idêntico ao run_all.py
# ─────────────────────────────────────────────────────────────────────────────

def build_circuit(n_qubits, n_layers, entanglement):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        # Encoding angular (único encoding investigado aqui)
        n_features = x.shape[0]
        for q in range(n_qubits):
            qml.RY(x[(q * 4)     % n_features], wires=q)
            qml.RZ(x[(q * 4 + 2) % n_features], wires=q)

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
            # none: sem CNOT

        return qml.expval(qml.PauliZ(0))

    return circuit


def init_weights(n_layers, n_qubits, seed):
    torch.manual_seed(seed)
    # Inicialização U[-π, π] — padrão para análise de barren plateau
    # (pesos pequenos podem mascarar o efeito; inicialização uniforme é o padrão teórico)
    return nn.Parameter(
        torch.rand(n_layers, n_qubits, 2, dtype=torch.float64) * 2 * np.pi - np.pi
    )


def predict_proba(circuit, X, weights):
    preds = torch.stack([circuit(x, weights) for x in X])
    return (preds + 1) / 2


# ─────────────────────────────────────────────────────────────────────────────
# H1 — EXPERIMENTO CORRIGIDO
# N=15 seeds, 60 épocas, n=100, 4q, 3L, angular
# ─────────────────────────────────────────────────────────────────────────────

def run_h1_fix(fast=False):
    n_seeds = 5 if fast else N_SEEDS_FIX
    epochs  = 30 if fast else EPOCHS_FIX

    print(f"\n{'='*55}")
    print(f"  H1 FIX — {n_seeds} seeds × {epochs} épocas × 3 topologias")
    print(f"  Config: n={TRAIN_SIZE_H1}, {N_QUBITS_H1}q, {N_LAYERS_H1}L, {ENCODING_H1}")
    print(f"  Total runs: {n_seeds * 3}")
    print(f"{'='*55}\n")

    results = {ent: [] for ent in ENTANGLEMENTS}
    total_t0 = time.time()
    run_n = 0

    for ent in ENTANGLEMENTS:
        circuit = build_circuit(N_QUBITS_H1, N_LAYERS_H1, ent)
        loss_fn = nn.BCELoss()

        for sd in range(n_seeds):
            run_n += 1
            print(f"  [{run_n}/{n_seeds*3}] ent={ent} seed={sd}", end=" ", flush=True)
            t0 = time.time()

            X_tr, X_te, y_tr, y_te = load_mnist_binary(
                train_size=TRAIN_SIZE_H1, seed=sd
            )
            weights = init_weights(N_LAYERS_H1, N_QUBITS_H1, seed=sd)
            opt = torch.optim.Adam([weights], lr=LR)

            history = {
                "test_acc":  [],
                "train_acc": [],
                "train_loss":[],
                "grad_var":  [],
            }

            for epoch in range(epochs):
                opt.zero_grad()
                proba = predict_proba(circuit, X_tr, weights)
                loss  = loss_fn(proba, y_tr)
                loss.backward()

                # grad_var registrado ANTES do step (mesma lógica do original)
                gv = float(weights.grad.detach().flatten().std())
                history["grad_var"].append(gv)
                opt.step()

                with torch.no_grad():
                    tr_acc = ((predict_proba(circuit, X_tr, weights) > 0.5).float()
                              == y_tr).float().mean().item()
                    te_acc = ((predict_proba(circuit, X_te, weights) > 0.5).float()
                              == y_te).float().mean().item()

                history["train_acc"].append(tr_acc)
                history["test_acc"].append(te_acc)
                history["train_loss"].append(float(loss))

            elapsed = time.time() - t0
            print(f"→ acc={history['test_acc'][-1]:.3f} | {elapsed:.0f}s")

            results[ent].append({
                "seed":       sd,
                "final_acc":  history["test_acc"][-1],
                "best_acc":   max(history["test_acc"]),
                "train_acc":  history["train_acc"][-1],
                "history":    history,
            })

    elapsed_total = time.time() - total_t0
    print(f"\n  H1 concluído em {elapsed_total/60:.1f} min")

    # Salva
    path = os.path.join(OUT_DIR, "h1_fix_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Salvo: {path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# H3 — EXPERIMENTO CORRIGIDO
# Mede grad_var na ÉPOCA 1 (antes de qualquer update) vs n_qubits
# Inicialização U[-π, π] — padrão da literatura de barren plateau
# ─────────────────────────────────────────────────────────────────────────────

def measure_initial_grad(circuit, X_tr, y_tr, n_layers, n_qubits, seed):
    """
    Mede std(∇θ) com pesos aleatórios ANTES de qualquer update de gradiente.
    Esta é a definição correta de barren plateau (McClean et al. 2018):
    o gradiente deve ser medido na paisagem de otimização inexplorada.
    """
    loss_fn = nn.BCELoss()
    weights = init_weights(n_layers, n_qubits, seed=seed)
    opt = torch.optim.Adam([weights], lr=LR)

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


def run_h3_fix(fast=False):
    n_seeds = 5 if fast else N_SEEDS_H3

    print(f"\n{'='*55}")
    print(f"  H3 FIX — grad inicial (ep1) × n_qubits × topologia")
    print(f"  Seeds por config: {n_seeds}")
    print(f"  Configs: {len(N_QUBITS_H3)} qubits × 3 topologias = "
          f"{len(N_QUBITS_H3)*3} configs × {n_seeds} seeds = "
          f"{len(N_QUBITS_H3)*3*n_seeds} medições")
    print(f"{'='*55}\n")

    results = {ent: {nq: [] for nq in N_QUBITS_H3} for ent in ENTANGLEMENTS}
    total_t0 = time.time()
    run_n = 0
    total_runs = len(ENTANGLEMENTS) * len(N_QUBITS_H3) * n_seeds

    X_tr, X_te, y_tr, y_te = load_mnist_binary(train_size=TRAIN_SIZE_H1)

    for ent in ENTANGLEMENTS:
        for nq in N_QUBITS_H3:
            circuit = build_circuit(nq, N_LAYERS_H3, ent)

            for sd in range(n_seeds):
                run_n += 1
                print(f"  [{run_n}/{total_runs}] ent={ent} {nq}q seed={sd}",
                      end=" ", flush=True)
                t0 = time.time()

                # Usa n_qubits correto nos dados (mesmo esquema do original)
                measurement = measure_initial_grad(
                    circuit, X_tr, y_tr, N_LAYERS_H3, nq, seed=sd
                )
                elapsed = time.time() - t0
                print(f"→ grad_std={measurement['grad_std']:.6f} | {elapsed:.1f}s")

                results[ent][nq].append({
                    "seed":     sd,
                    "n_qubits": nq,
                    **measurement,
                })

    # Também roda 5 épocas completas para confirmar comportamento ao longo do treino
    # (complemento, não substitui — agora com escala de qubits)
    print("\n  Rodando curvas de grad_var × época por n_qubits (5 épocas, confirmação)...")
    epoch_results = {ent: {nq: [] for nq in N_QUBITS_H3} for ent in ENTANGLEMENTS}
    EPOCHS_CONFIRM = 5 if fast else 10

    X_tr, X_te, y_tr, y_te = load_mnist_binary(train_size=TRAIN_SIZE_H1)
    for ent in ENTANGLEMENTS:
        for nq in N_QUBITS_H3:
            circuit = build_circuit(nq, N_LAYERS_H3, ent)
            loss_fn = nn.BCELoss()

            for sd in range(min(n_seeds, 5)):   # 5 seeds suficientes aqui
                weights = init_weights(N_LAYERS_H3, nq, seed=sd)
                opt = torch.optim.Adam([weights], lr=LR)
                gv_curve = []

                for _ in range(EPOCHS_CONFIRM):
                    opt.zero_grad()
                    proba = predict_proba(circuit, X_tr, weights)
                    loss  = loss_fn(proba, y_tr)
                    loss.backward()
                    gv_curve.append(float(weights.grad.detach().flatten().std()))
                    opt.step()

                epoch_results[ent][nq].append(gv_curve)

    elapsed_total = time.time() - total_t0
    print(f"\n  H3 concluído em {elapsed_total/60:.1f} min")

    # Salva
    out = {"initial_grad": results, "epoch_curves": epoch_results}
    path = os.path.join(OUT_DIR, "h3_fix_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Salvo: {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISE ESTATÍSTICA H1
# ─────────────────────────────────────────────────────────────────────────────

def analyze_h1(results):
    """
    Compara topologias com Mann-Whitney U (não-paramétrico, robusto para N pequeno).
    Reporta: média ± std, mediana, Cohen's d, p-valor, conclusão.
    """
    print(f"\n{'='*55}")
    print("  ANÁLISE ESTATÍSTICA H1")
    print(f"{'='*55}")

    accs = {}
    for ent in ENTANGLEMENTS:
        accs[ent] = [r["final_acc"] for r in results[ent]]
        mean, std = np.mean(accs[ent]), np.std(accs[ent])
        best = [r["best_acc"] for r in results[ent]]
        print(f"\n  {ENT_LABELS[ent]:12s}: "
              f"final={mean:.4f}±{std:.4f}  "
              f"best={np.mean(best):.4f}±{np.std(best):.4f}  "
              f"N={len(accs[ent])}")

    print("\n  Testes Mann-Whitney U (unilateral: anel > alternativa):")
    pairs = [("ring", "none"), ("ring", "all_to_all"), ("all_to_all", "none")]
    for a, b in pairs:
        u, p_two = stats.mannwhitneyu(accs[a], accs[b], alternative="two-sided")
        _, p_one = stats.mannwhitneyu(accs[a], accs[b], alternative="greater")
        pool_std = np.sqrt((np.std(accs[a])**2 + np.std(accs[b])**2) / 2)
        d = (np.mean(accs[a]) - np.mean(accs[b])) / pool_std if pool_std > 0 else 0
        conclusion = "SIGNIFICATIVO" if p_two < 0.05 else "não significativo"
        print(f"  {ENT_LABELS[a]:12s} vs {ENT_LABELS[b]:12s}: "
              f"Δ={np.mean(accs[a])-np.mean(accs[b]):+.4f}  "
              f"d={d:.2f}  p(2-sided)={p_two:.3f}  p(1-sided)={p_one:.3f}  "
              f"→ {conclusion}")

    # Veredicto H1
    d_ring_none = (np.mean(accs["ring"]) - np.mean(accs["none"])) / \
                  np.sqrt((np.std(accs["ring"])**2 + np.std(accs["none"])**2) / 2)
    _, p_rn = stats.mannwhitneyu(accs["ring"], accs["none"], alternative="two-sided")
    print(f"\n  VEREDICTO H1:")
    if p_rn >= 0.05 and abs(np.mean(accs["ring"]) - np.mean(accs["none"])) < 0.03:
        print(f"  ✓ CONFIRMADA — diferença anel vs sem-ent não significativa (p={p_rn:.3f}, Δ<3pp)")
        print(f"    Encoding angular provê não-linearidade suficiente.")
    elif p_rn < 0.05:
        print(f"  ✗ NEGADA — anel supera sem-ent significativamente (p={p_rn:.3f})")
        print(f"    Entanglement contribui para o poder discriminativo.")
    else:
        print(f"  ⚠ INCONCLUSIVA — p={p_rn:.3f}, d={d_ring_none:.2f}.")
        print(f"    Efeito presente mas poder insuficiente. Aumentar N ou épocas.")

    return accs


def analyze_h3(results):
    """
    Analisa grad_var inicial × n_qubits por topologia.
    Evidência de barren plateau: queda de grad com n_qubits deve ser
    mais acentuada em all_to_all do que em ring.
    """
    ig = results["initial_grad"]

    print(f"\n{'='*55}")
    print("  ANÁLISE ESTATÍSTICA H3 — Gradiente inicial × n_qubits")
    print(f"{'='*55}")
    print(f"\n  {'Topologia':12s}  " +
          "  ".join([f"{nq}q grad_std" for nq in N_QUBITS_H3]))
    print("  " + "-"*50)

    all_means = {}
    for ent in ENTANGLEMENTS:
        row = f"  {ENT_LABELS[ent]:12s}"
        all_means[ent] = {}
        for nq in N_QUBITS_H3:
            gvs = [r["grad_std"] for r in ig[ent][nq]]
            m, s = np.mean(gvs), np.std(gvs)
            all_means[ent][nq] = m
            row += f"  {m:.6f}±{s:.6f}"
        print(row)

    print("\n  Razão all_to_all / ring por n_qubits (< 1 = barren mais forte no all):")
    for nq in N_QUBITS_H3:
        ratio = all_means["all_to_all"][nq] / all_means["ring"][nq]
        direction = "← barren mais forte no all_to_all" if ratio < 1 else "← anel tem menos gradiente"
        print(f"  {nq}q: {ratio:.3f}  {direction}")

    print("\n  Teste de tendência — grad cai com n_qubits? (Spearman)")
    for ent in ENTANGLEMENTS:
        gv_means = [all_means[ent][nq] for nq in N_QUBITS_H3]
        rho, p = stats.spearmanr(N_QUBITS_H3, gv_means)
        direction = "decrescente" if rho < 0 else "crescente"
        print(f"  {ENT_LABELS[ent]:12s}: ρ={rho:.3f}  p={p:.3f}  tendência={direction}")

    print(f"\n  VEREDICTO H3:")
    ratio_4q = all_means["all_to_all"][4] / all_means["ring"][4]
    ratio_6q = all_means["all_to_all"][6] / all_means["ring"][6]
    rho_all, p_all = stats.spearmanr(N_QUBITS_H3,
                                      [all_means["all_to_all"][nq] for nq in N_QUBITS_H3])
    if ratio_4q < 1 and ratio_6q < 1:
        print(f"  ✓ CONFIRMADA — all_to_all tem grad inicial menor que anel em 4q e 6q")
        print(f"    (razão 4q={ratio_4q:.2f}, 6q={ratio_6q:.2f})")
        if rho_all < 0 and p_all < 0.1:
            print(f"    Tendência decrescente no all_to_all (ρ={rho_all:.2f})")
    else:
        print(f"  ⚠ PARCIAL/INCONCLUSIVA — padrão não consistente entre n_qubits")
        print(f"    (razão 4q={ratio_4q:.2f}, 6q={ratio_6q:.2f})")
        print(f"    Possivelmente 4 qubits insuficiente para manifestar barren plateau claro.")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS H1
# ─────────────────────────────────────────────────────────────────────────────

def plot_h1_curves(results, epochs_fix):
    """Curvas de acurácia × época para as 3 topologias — 60 épocas, 15 seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curvas de acurácia
    ax = axes[0]
    for ent in ENTANGLEMENTS:
        curves = np.array([r["history"]["test_acc"] for r in results[ent]])
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        ep = range(1, len(mean) + 1)
        ax.plot(ep, mean, label=ENT_LABELS[ent], color=COLORS[ent], lw=2)
        ax.fill_between(ep, mean - std, mean + std, color=COLORS[ent], alpha=0.15)

    ax.axvline(30, color="gray", ls=":", lw=1, alpha=0.7, label="ep30 (original)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no teste")
    ax.set_title(f"H1 — Curvas de convergência (N={len(results['ring'])} seeds, {epochs_fix} épocas)\n"
                 f"n=100, 4q, 3L, encoding angular")
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)

    # Boxplot acurácia final
    ax2 = axes[1]
    data = [np.array([r["final_acc"] for r in results[ent]]) for ent in ENTANGLEMENTS]
    bp = ax2.boxplot(data, labels=[ENT_LABELS[e] for e in ENTANGLEMENTS],
                     patch_artist=True, notch=False)
    for patch, ent in zip(bp["boxes"], ENTANGLEMENTS):
        patch.set_facecolor(COLORS[ent])
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)

    # Adiciona p-valores entre os pares
    pairs = [("ring", "none"), ("ring", "all_to_all")]
    y_max = max(max(d) for d in data)
    for i, (a, b) in enumerate(pairs):
        accs_a = [r["final_acc"] for r in results[a]]
        accs_b = [r["final_acc"] for r in results[b]]
        _, p = stats.mannwhitneyu(accs_a, accs_b, alternative="two-sided")
        x1 = ENTANGLEMENTS.index(a) + 1
        x2 = ENTANGLEMENTS.index(b) + 1
        y  = y_max + 0.02 + i * 0.04
        ax2.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="-", color="gray"))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else f"p={p:.2f}"
        ax2.text((x1 + x2) / 2, y + 0.005, sig, ha="center", fontsize=9, color="gray")

    ax2.set_ylabel("Acurácia final (época 60)")
    ax2.set_title(f"H1 — Distribuição por topologia (Mann-Whitney U)\n"
                  f"N={len(results['ring'])} seeds")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0.4, 1.05)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "h1_fix_curves_boxplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_h1_convergence_detail(results, epochs_fix):
    """Foco nas últimas 30 épocas — o que muda além da ep30 original."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for ent in ENTANGLEMENTS:
        curves = np.array([r["history"]["test_acc"] for r in results[ent]])
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        # Só mostra época 20 em diante
        ep = range(20, len(mean) + 1)
        ax.plot(ep, mean[19:], label=ENT_LABELS[ent], color=COLORS[ent], lw=2)
        ax.fill_between(ep, mean[19:] - std[19:], mean[19:] + std[19:],
                        color=COLORS[ent], alpha=0.15)

    ax.axvline(30, color="gray", ls=":", lw=1.5, alpha=0.8, label="ep30 (limite original)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no teste")
    ax.set_title("H1 — Detalhe das épocas 20–60: anel ainda cresce após ep30?")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "h1_fix_convergence_detail.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICOS H3
# ─────────────────────────────────────────────────────────────────────────────

def plot_h3_initial_grad(results):
    """
    Gráfico principal de H3: std(∇θ) inicial × n_qubits por topologia.
    Evidência de barren plateau: queda mais rápida em all_to_all.
    """
    ig = results["initial_grad"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gráfico de linhas com erro
    ax = axes[0]
    for ent in ENTANGLEMENTS:
        means, stds = [], []
        for nq in N_QUBITS_H3:
            gvs = [r["grad_std"] for r in ig[ent][nq]]
            means.append(np.mean(gvs))
            stds.append(np.std(gvs))
        ax.errorbar(N_QUBITS_H3, means, yerr=stds,
                    label=ENT_LABELS[ent], color=COLORS[ent],
                    marker="o", lw=2, capsize=5, capthick=1.5)

    ax.set_xlabel("Número de qubits")
    ax.set_ylabel("std(∇θ) — gradiente inicial (antes de qualquer update)")
    ax.set_title("H3 — Barren plateau: gradiente inicial × n_qubits\n"
                 "(métrica correta: pesos aleatórios U[−π, π])")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Normalizado pela referência em 2q (mostra queda relativa)
    ax2 = axes[1]
    for ent in ENTANGLEMENTS:
        means = []
        ref = np.mean([r["grad_std"] for r in ig[ent][2]])
        for nq in N_QUBITS_H3:
            gvs = np.mean([r["grad_std"] for r in ig[ent][nq]])
            means.append(gvs / ref)
        ax2.plot(N_QUBITS_H3, means, label=ENT_LABELS[ent],
                 color=COLORS[ent], marker="o", lw=2)

    # Referência teórica: decaimento 1/4^n (barren plateau ideal)
    ref_theory = [1, 1/4, 1/16]   # 2q=1, 4q=1/4, 6q=1/16
    ax2.plot(N_QUBITS_H3, ref_theory, ls=":", color="gray",
             lw=1.5, label="Teórico 1/4ⁿ (BP puro)", marker="x")

    ax2.set_xlabel("Número de qubits")
    ax2.set_ylabel("std(∇θ) normalizado (relativo a 2 qubits)")
    ax2.set_title("H3 — Queda relativa do gradiente vs previsão teórica")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "h3_fix_initial_grad_qubits.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_h3_epoch_curves(results):
    """
    Curvas de grad_var × época (primeiras N épocas) por n_qubits e topologia.
    Complementa o gráfico inicial — mostra que o efeito persiste.
    """
    ec = results["epoch_curves"]
    n_epochs = len(list(list(list(ec.values())[0].values())[0])[0])

    fig, axes = plt.subplots(1, len(N_QUBITS_H3), figsize=(14, 4), sharey=False)
    for ax, nq in zip(axes, N_QUBITS_H3):
        for ent in ENTANGLEMENTS:
            curves = np.array(ec[ent][nq])
            if len(curves) == 0:
                continue
            mean = curves.mean(axis=0)
            std  = curves.std(axis=0)
            ep   = range(1, len(mean) + 1)
            ax.plot(ep, mean, label=ENT_LABELS[ent], color=COLORS[ent], lw=2)
            ax.fill_between(ep, mean - std, mean + std,
                            color=COLORS[ent], alpha=0.15)
        ax.set_xlabel("Época")
        ax.set_ylabel("std(∇θ)" if nq == N_QUBITS_H3[0] else "")
        ax.set_title(f"{nq} qubits")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle("H3 — Variância do gradiente por época × topologia × n_qubits\n"
                 "(inicialização U[−π, π], épocas iniciais)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "h3_fix_grad_by_qubits_epochs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_h3_heatmap(results):
    """
    Heatmap: std(∇θ) inicial por topologia × n_qubits.
    Visualização rápida para o artigo.
    """
    ig = results["initial_grad"]
    data = np.zeros((len(ENTANGLEMENTS), len(N_QUBITS_H3)))
    for i, ent in enumerate(ENTANGLEMENTS):
        for j, nq in enumerate(N_QUBITS_H3):
            data[i, j] = np.mean([r["grad_std"] for r in ig[ent][nq]])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="std(∇θ) — gradiente inicial")

    ax.set_xticks(range(len(N_QUBITS_H3)))
    ax.set_xticklabels([f"{nq} qubits" for nq in N_QUBITS_H3])
    ax.set_yticks(range(len(ENTANGLEMENTS)))
    ax.set_yticklabels([ENT_LABELS[e] for e in ENTANGLEMENTS])
    ax.set_title("H3 — Gradiente inicial: maior = mais trainável\n"
                 "verde = maior gradiente, vermelho = barren plateau")

    for i in range(len(ENTANGLEMENTS)):
        for j in range(len(N_QUBITS_H3)):
            ax.text(j, i, f"{data[i,j]:.4f}", ha="center", va="center",
                    fontsize=9, color="black")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "h3_fix_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix H1 e H3")
    parser.add_argument("--h1-only",  action="store_true", help="Só roda H1")
    parser.add_argument("--h3-only",  action="store_true", help="Só roda H3")
    parser.add_argument("--fast",     action="store_true",
                        help="Validação rápida: 5 seeds, 30 épocas (~30 min)")
    parser.add_argument("--plots-only", action="store_true",
                        help="Só regera gráficos dos JSONs já existentes")
    args = parser.parse_args()

    run_h1 = not args.h3_only
    run_h3 = not args.h1_only

    epochs_used = 30 if args.fast else EPOCHS_FIX

    # ── Roda ou carrega H1 ────────────────────────────────────────────────
    h1_path = os.path.join(OUT_DIR, "h1_fix_results.json")
    if run_h1:
        if args.plots_only and os.path.exists(h1_path):
            print(f"  Carregando H1 de {h1_path}...")
            with open(h1_path) as f:
                h1_results = json.load(f)
            # Detecta quantas épocas foram rodadas
            epochs_used = len(h1_results["ring"][0]["history"]["test_acc"])
        else:
            h1_results = run_h1_fix(fast=args.fast)

        print("\n  Gerando gráficos H1...")
        accs = analyze_h1(h1_results)
        plot_h1_curves(h1_results, epochs_used)
        plot_h1_convergence_detail(h1_results, epochs_used)

    # ── Roda ou carrega H3 ────────────────────────────────────────────────
    h3_path = os.path.join(OUT_DIR, "h3_fix_results.json")
    if run_h3:
        if args.plots_only and os.path.exists(h3_path):
            print(f"  Carregando H3 de {h3_path}...")
            with open(h3_path) as f:
                h3_results = json.load(f)
        else:
            h3_results = run_h3_fix(fast=args.fast)

        print("\n  Gerando gráficos H3...")
        analyze_h3(h3_results)
        plot_h3_initial_grad(h3_results)
        plot_h3_epoch_curves(h3_results)
        plot_h3_heatmap(h3_results)

    print(f"\n{'='*55}")
    print(f"  ✓ Concluído. Resultados em: {OUT_DIR}/")
    print(f"  ✓ Figuras em:  {FIG_DIR}/")
    print(f"{'='*55}")