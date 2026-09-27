"""
generality_experiments.py  (v2 — bug fix + parallel workers)
=============================================================
Correções em relação à versão original:
  1. WinError 433 (Google Drive): usa os.path.exists() + try/except em vez de
     pathlib.Path.exists(), e aceita --out-dir para saída local.
  2. Velocidade: --workers N executa N condições em paralelo via
     ProcessPoolExecutor. Cada worker é um processo independente (PennyLane
     safe). O processo principal é o único escritor do CSV (sem race condition).

Uso:
  # Rodar no diretório LOCAL, 4 workers paralelos
  python genalizacao_fixed.py --exp A --out-dir C:/resultados --workers 4

  # Teste rápido (3 seeds, 20 épocas)
  python genalizacao_fixed.py --fast --workers 4

  # Só gráficos
  python genalizacao_fixed.py --plots-only --out-dir C:/resultados
"""

import argparse
import csv
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.datasets import load_digits, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
import pennylane as qml

# ─────────────────────────────────────────
# Configuração global (read-only nos workers)
# ─────────────────────────────────────────
N_QUBITS    = 4
N_LAYERS    = 3
N_SEEDS     = 10
EPOCHS      = 60
LR          = 0.05
TEST_SIZE   = 0.25
POOL_SIZE   = 600
TRAIN_SIZES = [50, 100, 200]
TOPOLOGIES  = ["ring", "all_to_all", "none"]
ENCODINGS   = ["angular", "amplitude"]

MNIST_PAIRS = [(0, 1), (3, 5), (4, 9)]
SYNTH_DSETS = ["circles", "spirals", "moons"]

COLORS     = {"ring": "#534AB7", "all_to_all": "#D85A30", "none": "#1D9E75"}
ENT_LABELS = {"ring": "Ring", "all_to_all": "All-to-all", "none": "No entanglement"}

RAW_FIELDS = [
    "dataset", "topology", "encoding", "train_size", "seed",
    "final_acc", "best_acc", "train_acc", "grad_var_init",
    "train_time_s", "timestamp"
]

# Caminhos definidos em runtime (ver main())
OUT_DIR     = None
FIG_DIR     = None
RAW_CSV     = None
SUMMARY_CSV = None


# ─────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────

def load_mnist_pair(class_a, class_b, train_size, seed):
    digits = load_digits()
    mask   = np.isin(digits.target, [class_a, class_b])
    X, y   = digits.data[mask], digits.target[mask]
    y      = (y == class_b).astype(int)

    rng  = np.random.RandomState(seed)
    pool = min(POOL_SIZE, len(X))
    idx  = rng.choice(len(X), pool, replace=False)
    X, y = X[idx], y[idx]

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    X_tr, _, y_tr, _ = train_test_split(
        X_tv, y_tv, train_size=train_size, stratify=y_tv, random_state=seed
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    def c4x4(Xf):
        return Xf.reshape(-1, 8, 8)[:, 2:6, 2:6].reshape(-1, 16)

    return (
        torch.tensor(c4x4(X_tr), dtype=torch.float64),
        torch.tensor(c4x4(X_te), dtype=torch.float64),
        torch.tensor(y_tr, dtype=torch.float64),
        torch.tensor(y_te, dtype=torch.float64),
    )


def load_synthetic(dataset_name, train_size, seed):
    rng     = np.random.RandomState(seed)
    n_total = POOL_SIZE + TEST_SIZE + 50

    if dataset_name == "circles":
        X, y = make_circles(n_samples=n_total, noise=0.1, factor=0.5,
                            random_state=seed)
    elif dataset_name == "spirals":
        n_half = n_total // 2
        theta  = np.linspace(0, 4 * np.pi, n_half)
        r      = np.linspace(0.1, 1.0, n_half)
        X0     = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        X1     = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
        X0    += rng.normal(0, 0.1, X0.shape)
        X1    += rng.normal(0, 0.1, X1.shape)
        X      = np.vstack([X0, X1])
        y      = np.array([0] * n_half + [1] * n_half)
    else:  # moons
        X, y = make_moons(n_samples=n_total, noise=0.15, random_state=seed)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X      = scaler.fit_transform(X)

    idx  = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    max_train = len(X_tv) - 1
    if train_size >= max_train:
        raise ValueError(
        f"train_size={train_size} exceeds available "
        f"samples ({max_train}) for dataset."
    )
    X_tr, _, y_tr, _ = train_test_split(
        X_tv, y_tv, train_size=train_size, stratify=y_tv, random_state=seed
    )

    def pad16(Xf):
        n_feat = Xf.shape[1]
        if n_feat < 16:
            pad = np.zeros((Xf.shape[0], 16 - n_feat))
            return np.hstack([Xf, pad])
        return Xf[:, :16]

    return (
        torch.tensor(pad16(X_tr), dtype=torch.float64),
        torch.tensor(pad16(X_te), dtype=torch.float64),
        torch.tensor(y_tr, dtype=torch.float64),
        torch.tensor(y_te, dtype=torch.float64),
    )


# ─────────────────────────────────────────
# Circuitos VQC
# ─────────────────────────────────────────

def build_circuit(topology):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        n_feat = x.shape[0]
        for q in range(N_QUBITS):
            qml.RY(x[(q * 4)     % n_feat], wires=q)
            qml.RZ(x[(q * 4 + 2) % n_feat], wires=q)
        for layer in range(N_LAYERS):
            w = weights[layer]
            for q in range(N_QUBITS):
                qml.RY(w[q, 0], wires=q)
                qml.RZ(w[q, 1], wires=q)
            if topology == "ring":
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
            elif topology == "all_to_all":
                for q0, q1 in combinations(range(N_QUBITS), 2):
                    qml.CNOT(wires=[q0, q1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_amplitude_circuit(topology):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        n_amp = 2 ** N_QUBITS
        x_amp = x[:n_amp]
        norm  = torch.norm(x_amp)
        x_amp = x_amp / (norm + 1e-8)
        qml.AmplitudeEmbedding(x_amp, wires=range(N_QUBITS), normalize=False)
        for layer in range(N_LAYERS):
            w = weights[layer]
            for q in range(N_QUBITS):
                qml.RY(w[q, 0], wires=q)
                qml.RZ(w[q, 1], wires=q)
            if topology == "ring":
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
            elif topology == "all_to_all":
                for q0, q1 in combinations(range(N_QUBITS), 2):
                    qml.CNOT(wires=[q0, q1])
        return qml.expval(qml.PauliZ(0))

    return circuit


def predict(circuit, X, weights):
    raw = torch.stack([circuit(x, weights) for x in X])
    return (raw + 1) / 2


# ─────────────────────────────────────────
# Treinamento
# ─────────────────────────────────────────

def train_vqc(X_tr, X_te, y_tr, y_te, topology, encoding, seed, epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if encoding == "angular":
        circuit = build_circuit(topology)
    else:
        circuit = build_amplitude_circuit(topology)

    weights = nn.Parameter(
        torch.rand(N_LAYERS, N_QUBITS, 2, dtype=torch.float64) * 2 * np.pi - np.pi
    )
    optimizer = torch.optim.Adam([weights], lr=LR)
    loss_fn   = nn.BCELoss()

    # Gradiente inicial (antes de qualquer update)
    optimizer.zero_grad()
    preds = predict(circuit, X_tr[:min(20, len(X_tr))], weights)
    loss  = loss_fn(preds, y_tr[:min(20, len(y_tr))])
    loss.backward()
    grad_var = float(weights.grad.detach().flatten().std())
    optimizer.zero_grad()

    best_acc = 0.0
    t0       = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = predict(circuit, X_tr, weights)
        loss  = loss_fn(preds, y_tr)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            te = ((predict(circuit, X_te, weights) > 0.5).float()
                  == y_te).float().mean().item()
            best_acc = max(best_acc, te)

    with torch.no_grad():
        final_acc = ((predict(circuit, X_te, weights) > 0.5).float()
                     == y_te).float().mean().item()
        train_acc = ((predict(circuit, X_tr, weights) > 0.5).float()
                     == y_tr).float().mean().item()

    return {
        "final_acc":     round(final_acc, 6),
        "best_acc":      round(best_acc, 6),
        "train_acc":     round(train_acc, 6),
        "grad_var_init": round(grad_var, 8),
        "train_time_s":  round(time.time() - t0, 1),
    }


# ─────────────────────────────────────────
# Função de worker (nível de módulo — obrigatório para pickle no Windows)
# ─────────────────────────────────────────

def _worker(task):
    """
    Executa um único run VQC.
    task = (dtype, dname, ca, cb, topology, encoding, train_size, seed, epochs)
    Retorna (task_key_dict, result_dict) ou (task_key_dict, None) em caso de erro.
    """
    dtype, dname, ca, cb, topology, encoding, train_size, seed, epochs = task
    try:
        if dtype == "mnist":
            X_tr, X_te, y_tr, y_te = load_mnist_pair(ca, cb, train_size, seed)
        else:
            X_tr, X_te, y_tr, y_te = load_synthetic(dname, train_size, seed)

        result = train_vqc(X_tr, X_te, y_tr, y_te,
                           topology, encoding, seed, epochs)
        return (dname, topology, encoding, train_size, seed), result

    except Exception as e:
        return (dname, topology, encoding, train_size, seed), {"error": str(e)}


# ─────────────────────────────────────────
# Resume — leitura do CSV de progresso
# ─────────────────────────────────────────

def load_done_set(raw_csv):
    """
    Retorna um set de tuplas (dataset, topology, encoding, train_size, seed)
    já presentes no CSV. Usa os.path.exists() para evitar WinError 433
    com o Google Drive virtual filesystem.
    """
    done = set()
    try:
        if not os.path.exists(raw_csv):
            return done
        with open(raw_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((
                    row["dataset"],
                    row["topology"],
                    row["encoding"],
                    row["train_size"],
                    row["seed"],
                ))
    except Exception:
        # Se o arquivo estiver corrompido ou inacessível, começa do zero
        pass
    return done


def write_row(raw_csv, row):
    exists = os.path.exists(raw_csv)
    with open(raw_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────
# Loop principal — paralelo
# ─────────────────────────────────────────

def run_all(args, raw_csv, summary_csv, out_dir, fig_dir):
    epochs = 20 if args.fast else EPOCHS
    seeds  = list(range(3 if args.fast else N_SEEDS))

    # Constrói lista de datasets
    datasets = []
    if args.exp in ["A", "all"]:
        for ca, cb in MNIST_PAIRS:
            if args.pair and sorted([ca, cb]) != sorted(args.pair):
                continue
            datasets.append(("mnist", f"mnist_{ca}v{cb}", ca, cb))
    if args.exp in ["B", "all"]:
        for ds in SYNTH_DSETS:
            if args.dataset and ds != args.dataset:
                continue
            datasets.append(("synth", ds, None, None))

    # Lê progresso já salvo (UMA VEZ)
    done_set = load_done_set(raw_csv)

    # Monta lista de tasks pendentes
    all_tasks   = []
    skip_count  = 0
    for dtype, dname, ca, cb in datasets:
        for topology in TOPOLOGIES:
            for encoding in ENCODINGS:
                for train_size in TRAIN_SIZES:
                    for seed in seeds:
                        key = (dname, topology, encoding,
                               str(train_size), str(seed))
                        if key in done_set:
                            skip_count += 1
                        else:
                            all_tasks.append(
                                (dtype, dname, ca, cb,
                                 topology, encoding, train_size, seed, epochs)
                            )

    total = len(all_tasks) + skip_count
    print(f"\nConfiguração:")
    print(f"  Datasets:   {[d[1] for d in datasets]}")
    print(f"  Topologias: {TOPOLOGIES}")
    print(f"  Encodings:  {ENCODINGS}")
    print(f"  Train sizes:{TRAIN_SIZES}")
    print(f"  Seeds:      {len(seeds)}")
    print(f"  Épocas:     {epochs}")
    print(f"  Workers:    {args.workers}")
    print(f"  Total runs: {total:,}  (pendentes: {len(all_tasks)}, já feitos: {skip_count})")
    print(f"  Saídas em:  {out_dir}\n")

    if not all_tasks:
        print("Nada a fazer — todos os runs já estão no CSV.")
        generate_summary(raw_csv, summary_csv)
        generate_plots(raw_csv, summary_csv, fig_dir)
        return

    done_cnt   = skip_count
    t0_total   = time.time()
    error_cnt  = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, task): task for task in all_tasks}

        for future in as_completed(futures):
            done_cnt += 1
            task = futures[future]
            _, dname, _, _, topology, encoding, train_size, seed, _ = task
            tag  = f"{dname} topo={topology} enc={encoding} n={train_size} seed={seed}"

            try:
                (key_ds, key_topo, key_enc, key_n, key_s), result = future.result()
            except Exception as exc:
                error_cnt += 1
                print(f"  [{done_cnt:4d}/{total}] {tag}  WORKER EXCEPTION: {exc}")
                continue

            if "error" in result:
                error_cnt += 1
                print(f"  [{done_cnt:4d}/{total}] {tag}  ERRO: {result['error']}")
                continue

            # Escritor único = processo principal (sem race condition)
            write_row(raw_csv, {
                "dataset":       key_ds,
                "topology":      key_topo,
                "encoding":      key_enc,
                "train_size":    key_n,
                "seed":          key_s,
                "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
                **result,
            })

            elapsed = time.time() - t0_total
            remaining = total - done_cnt
            eta = (elapsed / max(done_cnt - skip_count, 1)) * remaining
            print(f"  [{done_cnt:4d}/{total}] {tag} "
                  f"acc={result['final_acc']:.3f} "
                  f"t={result['train_time_s']:.0f}s "
                  f"ETA={eta/3600:.1f}h")

    elapsed_h = (time.time() - t0_total) / 3600
    print(f"\nTotal: {elapsed_h:.2f}h  |  Erros: {error_cnt}")
    generate_summary(raw_csv, summary_csv)
    generate_plots(raw_csv, summary_csv, fig_dir)


# ─────────────────────────────────────────
# Sumário
# ─────────────────────────────────────────

def generate_summary(raw_csv, summary_csv):
    if not os.path.exists(raw_csv):
        return
    df = pd.read_csv(raw_csv)
    summary = df.groupby(
        ["dataset", "topology", "encoding", "train_size"]
    ).agg(
        final_acc_mean  = ("final_acc",     "mean"),
        final_acc_std   = ("final_acc",     "std"),
        best_acc_mean   = ("best_acc",      "mean"),
        grad_var_mean   = ("grad_var_init", "mean"),
        n_seeds         = ("seed",          "count"),
    ).reset_index()
    summary.to_csv(summary_csv, index=False)
    print(f"\nSumário salvo: {summary_csv}")

    print("\n=== Acurácia por dataset × encoding (topologia anel, n=100) ===")
    sub = summary[
        (summary["topology"] == "ring") &
        (summary["train_size"] == 100)
    ][["dataset", "encoding", "final_acc_mean", "final_acc_std"]]
    print(sub.to_string(index=False))

    print("\n=== Achado 1: amplitude + none (n=100) ===")
    sub2 = summary[
        (summary["encoding"] == "amplitude") &
        (summary["topology"] == "none") &
        (summary["train_size"] == 100)
    ][["dataset", "final_acc_mean", "final_acc_std"]]
    print(sub2.to_string(index=False))


# ─────────────────────────────────────────
# Gráficos (idêntico ao original)
# ─────────────────────────────────────────

def generate_plots(raw_csv, summary_csv, fig_dir):
    if not os.path.exists(summary_csv):
        generate_summary(raw_csv, summary_csv)
    if not os.path.exists(summary_csv):
        print("Sem dados suficientes para gráficos.")
        return

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)

    # Plot 1: Acurácia por dataset × topologia
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, enc in enumerate(["angular", "amplitude"]):
        ax  = axes[ax_idx]
        sub = df[(df["encoding"] == enc) & (df["train_size"] == 100)]
        datasets_all = sorted(sub["dataset"].unique())
        x   = np.arange(len(datasets_all))
        w   = 0.25
        for i, topo in enumerate(TOPOLOGIES):
            means, errs = [], []
            for ds in datasets_all:
                row = sub[(sub["dataset"] == ds) & (sub["topology"] == topo)]
                means.append(row["final_acc_mean"].values[0] if len(row) else 0)
                errs.append(row["final_acc_std"].values[0] if len(row) else 0)
            ax.bar(x + i * w, means, w, yerr=errs,
                   label=ENT_LABELS[topo], color=COLORS[topo], alpha=0.75, capsize=4)
        ax.set_xticks(x + w)
        ax.set_xticklabels(datasets_all, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Test accuracy (mean ± SD)", fontsize=10)
        ax.set_title(f"{enc.capitalize()} encoding, n=100", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Accuracy by dataset and entanglement topology", fontsize=12)
    fig.tight_layout()
    p = str(Path(fig_dir) / "accuracy_by_dataset.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {p}")

    # Plot 2: VQC vs train_size por dataset
    datasets_all = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, ds in enumerate(datasets_all[:6]):
        ax  = axes[idx]
        sub = df[(df["dataset"] == ds) & (df["encoding"] == "angular") &
                 (df["topology"] == "ring")].sort_values("train_size")
        if not sub.empty:
            ax.errorbar(sub["train_size"], sub["final_acc_mean"],
                        yerr=sub["final_acc_std"],
                        color=COLORS["ring"], marker="o", lw=2, capsize=4,
                        label="VQC (angular, ring)")
        ax.set_xlabel("Training set size", fontsize=9)
        ax.set_ylabel("Test accuracy", fontsize=9)
        ax.set_title(ds, fontsize=10)
        ax.set_xticks(TRAIN_SIZES)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(len(datasets_all), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Accuracy vs training set size by dataset\n(angular, ring)", fontsize=12)
    fig.tight_layout()
    p = str(Path(fig_dir) / "h2_accuracy_vs_trainsize.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {p}")

    # Plot 3: Achado 1 — amplitude none vs ring
    fig, ax = plt.subplots(figsize=(10, 5))
    sub = df[(df["encoding"] == "amplitude") & (df["train_size"] == 100)]
    x   = np.arange(len(datasets_all))
    w   = 0.35
    for i, topo in enumerate(["ring", "none"]):
        means, errs = [], []
        for ds in datasets_all:
            row = sub[(sub["dataset"] == ds) & (sub["topology"] == topo)]
            means.append(row["final_acc_mean"].values[0] if len(row) else 0)
            errs.append(row["final_acc_std"].values[0] if len(row) else 0)
        ax.bar(x + i * w, means, w, yerr=errs,
               label=ENT_LABELS[topo], color=COLORS[topo], alpha=0.75, capsize=4)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6, label="Chance level")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(datasets_all, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Test accuracy (mean ± SD)", fontsize=10)
    ax.set_title("Finding 1: amplitude encoding with vs without entanglement\nn=100, 4 qubits, 3 layers",
                 fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = str(Path(fig_dir) / "finding1_amplitude_entanglement.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {p}")
    print(f"\nTodos os gráficos em: {fig_dir}/")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp",      default="all", choices=["A", "B", "all"])
    p.add_argument("--fast",     action="store_true",
                   help="3 seeds, 20 épocas — teste rápido")
    p.add_argument("--pair",     type=int, nargs=2, default=None,
                   help="Par específico: --pair 3 5")
    p.add_argument("--dataset",  type=str, default=None,
                   help="Dataset sintético: circles, spirals, moons")
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--workers",  type=int, default=1,
                   help="Número de processos paralelos (default: 1). "
                        "Use N = número de CPUs lógicas disponíveis. "
                        "Ex: --workers 4")
    p.add_argument("--out-dir",  type=str, default=None,
                   help="Diretório LOCAL de saída. Se não informado, usa "
                        "'results_generality' no diretório atual do script. "
                        "IMPORTANTE: use um caminho local (não Google Drive) "
                        "para evitar WinError 433.")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve diretório de saída
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Padrão: mesmo diretório do script (não Google Drive)
        out_dir = Path(__file__).parent / "results_generality"

    fig_dir     = out_dir / "figures_generality"
    raw_csv     = str(out_dir / "generality_raw.csv")
    summary_csv = str(out_dir / "generality_summary.csv")

    # Cria diretórios locais (não usa pathlib.exists — usa os.makedirs)
    os.makedirs(str(out_dir), exist_ok=True)
    os.makedirs(str(fig_dir), exist_ok=True)

    print(f"Saídas em: {out_dir}")

    if args.plots_only:
        generate_summary(raw_csv, summary_csv)
        generate_plots(raw_csv, summary_csv, str(fig_dir))
        return

    run_all(args, raw_csv, summary_csv, str(out_dir), str(fig_dir))


# ─────────────────────────────────────────
# Guard obrigatório para multiprocessing no Windows
# ─────────────────────────────────────────
if __name__ == "__main__":
    main()