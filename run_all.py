"""
VQC Pipeline — Circuitos Quânticos Variacionais para Classificação MNIST
========================================================================
Investiga H1, H2, H3 + análises extras B, C, D, F, G

Requisitos:
    pip install pennylane==0.44.* torch scikit-learn numpy matplotlib seaborn

Uso:
    python vqc_pipeline.py                  # Execução completa (~4–8h CPU)
    python vqc_pipeline.py --fast           # Subset rápido para validar (~15 min)
    python vqc_pipeline.py --skip-vqc      # Só baselines clássicos

Saídas (criadas em ./results/):
    results.json          — todos os resultados brutos
    figures/              — todos os gráficos (.png)
    summary_table.csv     — tabela de acurácia consolidada
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
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
TRAIN_SIZES    = [50, 100, 200, 400]        # H2: regime de dados
N_QUBITS_LIST  = [2, 4, 6]                  # D: escalonamento de qubits
N_LAYERS_LIST  = [1, 2, 3]                  # C: profundidade do circuito
N_SEEDS        = 5                          # B: variância por inicialização
EPOCHS         = 30
LR             = 0.05
TEST_SIZE      = 100
POOL_SIZE      = 600                        # amostras por classe disponíveis

# Topologias e encodings investigados
ENTANGLEMENTS  = ["ring", "all_to_all", "none"]
ENCODINGS      = ["angular", "amplitude"]

# Diretório de saída
OUT_DIR = "results"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DADOS
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_binary(classes=(0, 1), test_size=TEST_SIZE, pool_size=POOL_SIZE,
                      train_size=200, seed=SEED):
    """
    Carrega MNIST (via sklearn digits 8×8), filtra duas classes,
    devolve:
      X_train_vqc  (n, 16)  — região central 4×4 normalizada [0, π]
      X_test_vqc   (test_size, 16)
      X_train_cls  (n, 64)  — imagem completa 8×8 para baselines
      X_test_cls   (test_size, 64)
      y_train, y_test  — rótulos {0, 1}
    """
    digits = load_digits()
    mask = np.isin(digits.target, classes)
    X, y = digits.data[mask], digits.target[mask]
    y = (y == classes[1]).astype(int)  # binariza: 0 → 0, 1 → 1

    # pool estratificado
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n_each = pool_size // 2
    idx0 = rng.choice(idx0, min(n_each, len(idx0)), replace=False)
    idx1 = rng.choice(idx1, min(n_each, len(idx1)), replace=False)
    pool_idx = np.concatenate([idx0, idx1])
    rng.shuffle(pool_idx)
    X_pool, y_pool = X[pool_idx], y[pool_idx]

    # split teste fixo (estratificado)
    X_pool_tr, X_test_cls, y_pool_tr, y_test = train_test_split(
        X_pool, y_pool, test_size=test_size, stratify=y_pool,
        random_state=seed
    )

    # sub-amostra do treino
    n_each_tr = train_size // 2
    idx0_tr = np.where(y_pool_tr == 0)[0]
    idx1_tr = np.where(y_pool_tr == 1)[0]
    chosen = np.concatenate([
        rng.choice(idx0_tr, min(n_each_tr, len(idx0_tr)), replace=False),
        rng.choice(idx1_tr, min(n_each_tr, len(idx1_tr)), replace=False),
    ])
    rng.shuffle(chosen)
    X_train_cls = X_pool_tr[chosen]
    y_train = y_pool_tr[chosen]

    # normalização
    scaler_cls = MinMaxScaler()
    X_train_cls = scaler_cls.fit_transform(X_train_cls)
    X_test_cls  = scaler_cls.transform(X_test_cls)

    # VQC: região central 4×4 da imagem 8×8
    def central_4x4(X_flat):
        imgs = X_flat.reshape(-1, 8, 8)
        region = imgs[:, 2:6, 2:6].reshape(-1, 16)
        return region

    X_train_vqc_raw = central_4x4(X_train_cls)
    X_test_vqc_raw  = central_4x4(X_test_cls)

    scaler_vqc = MinMaxScaler(feature_range=(0, np.pi))
    X_train_vqc = scaler_vqc.fit_transform(X_train_vqc_raw)
    X_test_vqc  = scaler_vqc.transform(X_test_vqc_raw)

    return (
        torch.tensor(X_train_vqc, dtype=torch.float64),
        torch.tensor(X_test_vqc,  dtype=torch.float64),
        torch.tensor(X_train_cls, dtype=torch.float32),
        torch.tensor(X_test_cls,  dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float64),
        torch.tensor(y_test,  dtype=torch.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. VQC
# ─────────────────────────────────────────────────────────────────────────────

def build_circuit(n_qubits, n_layers, entanglement, encoding):
    """Devolve função circuit(x, weights) e número de parâmetros."""

    dev = qml.device("default.qubit", wires=n_qubits)
    n_params = n_qubits * 2 * n_layers   # 2 rotações (RY, RZ) por qubit por camada

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, weights):
        # ── Encoding ──────────────────────────────────────────────────────
        if encoding == "angular":
            # RY + RZ por qubit, 4 features por qubit (cicla se necessário)
            n_features = x.shape[0]
            for q in range(n_qubits):
                idx_y = (q * 4) % n_features
                idx_z = (q * 4 + 2) % n_features
                qml.RY(x[idx_y], wires=q)
                qml.RZ(x[idx_z], wires=q)
        else:  # amplitude
            # normaliza e prepara estado de amplitude
            x_norm = x / (torch.norm(x) + 1e-8)
            # preenche até 2^n_qubits
            dim = 2 ** n_qubits
            if x_norm.shape[0] < dim:
                pad = torch.zeros(dim - x_norm.shape[0], dtype=x_norm.dtype)
                x_norm = torch.cat([x_norm, pad])
            else:
                x_norm = x_norm[:dim]
            x_norm = x_norm / (torch.norm(x_norm) + 1e-8)
            qml.StatePrep(x_norm, wires=range(n_qubits))

        # ── Camadas variacionais ───────────────────────────────────────────
        for layer in range(n_layers):
            w = weights[layer]                          # shape (n_qubits, 2)
            for q in range(n_qubits):
                qml.RY(w[q, 0], wires=q)
                qml.RZ(w[q, 1], wires=q)

            # Entanglement
            if entanglement == "ring":
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
            elif entanglement == "all_to_all":
                for q0, q1 in combinations(range(n_qubits), 2):
                    qml.CNOT(wires=[q0, q1])
            # "none": sem portas CNOT

        return qml.expval(qml.PauliZ(0))

    return circuit, n_params


class VQCClassifier:
    def __init__(self, n_qubits=4, n_layers=3, entanglement="ring",
                 encoding="angular", lr=LR, epochs=EPOCHS, seed=SEED):
        self.n_qubits     = n_qubits
        self.n_layers     = n_layers
        self.entanglement = entanglement
        self.encoding     = encoding
        self.lr           = lr
        self.epochs       = epochs
        self.seed         = seed

        self.circuit, self.n_params = build_circuit(
            n_qubits, n_layers, entanglement, encoding
        )

        torch.manual_seed(seed)
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2, dtype=torch.float64) * 0.1
        )
        self.optimizer = torch.optim.Adam([self.weights], lr=lr)
        self.loss_fn = nn.BCELoss()

    def predict_proba(self, X):
        preds = torch.stack([self.circuit(x, self.weights) for x in X])
        return (preds + 1) / 2   # [-1,1] → [0,1]

    def fit(self, X_train, y_train, X_test, y_test):
        history = {
            "train_loss":    [],
            "train_acc":     [],
            "test_acc":      [],
            "grad_var":      [],   # H3: variância do gradiente
            "epoch_time":    [],
        }
        for epoch in range(self.epochs):
            t0 = time.time()
            self.optimizer.zero_grad()
            proba = self.predict_proba(X_train)
            loss  = self.loss_fn(proba, y_train)
            loss.backward()

            # registra variância do gradiente ANTES do step
            grads = self.weights.grad.detach().flatten()
            history["grad_var"].append(float(grads.std()))

            self.optimizer.step()
            epoch_time = time.time() - t0

            # métricas
            with torch.no_grad():
                tr_pred  = (self.predict_proba(X_train) > 0.5).float()
                te_pred  = (self.predict_proba(X_test)  > 0.5).float()
                tr_acc   = (tr_pred == y_train).float().mean().item()
                te_acc   = (te_pred == y_test).float().mean().item()

            history["train_loss"].append(float(loss))
            history["train_acc"].append(tr_acc)
            history["test_acc"].append(te_acc)
            history["epoch_time"].append(epoch_time)

            if (epoch + 1) % 10 == 0:
                print(f"    epoch {epoch+1:3d}/{self.epochs} | "
                      f"loss={loss:.4f} | tr={tr_acc:.3f} | te={te_acc:.3f} | "
                      f"grad_std={history['grad_var'][-1]:.4f}")

        # métricas finais
        with torch.no_grad():
            te_proba = self.predict_proba(X_test).numpy()
            te_pred  = (te_proba > 0.5).astype(int)
        cm = confusion_matrix(y_test.numpy(), te_pred)

        return {
            "history":    history,
            "best_acc":   max(history["test_acc"]),
            "final_acc":  history["test_acc"][-1],
            "train_acc":  history["train_acc"][-1],    # F: gap treino×teste
            "total_time": sum(history["epoch_time"]),
            "n_params":   self.n_params,
            "conf_matrix": cm.tolist(),                # G: matriz de confusão
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. BASELINES CLÁSSICOS
# ─────────────────────────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_cnn(X_train, y_train, X_test, y_test,
              epochs=EPOCHS, lr=LR, seed=SEED):
    torch.manual_seed(seed)
    model = TinyCNN()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_tr = X_train.view(-1, 1, 8, 8).float()
    X_te = X_test.view(-1, 1, 8, 8).float()
    y_tr = y_train.float()
    y_te = y_test.float()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out  = model(X_tr)
        loss = loss_fn(out, y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr_acc = ((model(X_tr) > 0.5).float() == y_tr).float().mean().item()
            te_acc = ((model(X_te) > 0.5).float() == y_te).float().mean().item()
        history["train_loss"].append(float(loss))
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

    model.eval()
    with torch.no_grad():
        te_pred = (model(X_te) > 0.5).float().numpy().astype(int)
    cm = confusion_matrix(y_te.numpy(), te_pred)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "history":    history,
        "best_acc":   max(history["test_acc"]),
        "final_acc":  history["test_acc"][-1],
        "train_acc":  history["train_acc"][-1],
        "total_time": time.time() - t0,
        "n_params":   n_params,
        "conf_matrix": cm.tolist(),
    }


class MLP(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(X_train, y_train, X_test, y_test,
              epochs=EPOCHS, lr=LR, seed=SEED):
    torch.manual_seed(seed)
    model = MLP(in_dim=X_train.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out  = model(X_train)
        loss = loss_fn(out, y_train.float())
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            tr_acc = ((model(X_train) > 0.5).float() == y_train.float()).float().mean().item()
            te_acc = ((model(X_test)  > 0.5).float() == y_test.float()).float().mean().item()
        history["train_loss"].append(float(loss))
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

    model.eval()
    with torch.no_grad():
        te_pred = (model(X_test) > 0.5).float().numpy().astype(int)
    cm = confusion_matrix(y_test.numpy(), te_pred)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "history":    history,
        "best_acc":   max(history["test_acc"]),
        "final_acc":  history["test_acc"][-1],
        "train_acc":  history["train_acc"][-1],
        "total_time": time.time() - t0,
        "n_params":   n_params,
        "conf_matrix": cm.tolist(),
    }


def train_svm(X_train, y_train, X_test, y_test, seed=SEED):
    t0 = time.time()
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    clf.fit(X_train.numpy(), y_train.numpy())
    te_pred = clf.predict(X_test.numpy())
    tr_pred = clf.predict(X_train.numpy())
    cm = confusion_matrix(y_test.numpy(), te_pred)
    return {
        "history":    None,
        "best_acc":   accuracy_score(y_test.numpy(), te_pred),
        "final_acc":  accuracy_score(y_test.numpy(), te_pred),
        "train_acc":  accuracy_score(y_train.numpy(), tr_pred),
        "total_time": time.time() - t0,
        "n_params":   clf.n_support_.sum(),   # vetores de suporte
        "conf_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPERIMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def run_vqc_experiment(train_size, n_qubits, n_layers, entanglement,
                       encoding, seed=SEED):
    print(f"  VQC | n={train_size} | q={n_qubits} | L={n_layers} | "
          f"ent={entanglement} | enc={encoding} | seed={seed}")

    (X_tr_vqc, X_te_vqc, X_tr_cls, X_te_cls,
     y_tr, y_te) = load_mnist_binary(train_size=train_size, seed=seed)

    clf = VQCClassifier(
        n_qubits=n_qubits, n_layers=n_layers,
        entanglement=entanglement, encoding=encoding,
        lr=LR, epochs=EPOCHS, seed=seed,
    )
    result = clf.fit(X_tr_vqc, y_tr, X_te_vqc, y_te)
    result.update({
        "train_size":   train_size,
        "n_qubits":     n_qubits,
        "n_layers":     n_layers,
        "entanglement": entanglement,
        "encoding":     encoding,
        "seed":         seed,
        "model":        "vqc",
    })
    return result


def run_classical_experiments(train_size, seed=SEED):
    print(f"  Baselines | n={train_size} | seed={seed}")
    (X_tr_vqc, X_te_vqc, X_tr_cls, X_te_cls,
     y_tr, y_te) = load_mnist_binary(train_size=train_size, seed=seed)

    results = {}
    results["cnn"] = train_cnn(X_tr_cls, y_tr, X_te_cls, y_te, seed=seed)
    results["mlp"] = train_mlp(X_tr_cls, y_tr, X_te_cls, y_te, seed=seed)
    results["svm"] = train_svm(X_tr_cls, y_tr, X_te_cls, y_te, seed=seed)

    for model_name, r in results.items():
        r.update({
            "train_size": train_size,
            "seed":       seed,
            "model":      model_name,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. ORQUESTRADOR
# ─────────────────────────────────────────────────────────────────────────────

def run_all(fast=False, skip_vqc=False):
    all_results = {"vqc": [], "classical": []}
    total_t0 = time.time()

    # ── Configuração do sweep ──────────────────────────────────────────────
    train_sizes  = [50, 100] if fast else TRAIN_SIZES
    seeds        = [SEED]    if fast else list(range(N_SEEDS))      # B
    n_qubits_l   = [4]       if fast else N_QUBITS_LIST             # D
    n_layers_l   = [3]       if fast else N_LAYERS_LIST             # C
    entangl_l    = ENTANGLEMENTS
    encodings_l  = ENCODINGS

    total_runs = (len(train_sizes) * len(seeds) * len(n_qubits_l)
                  * len(n_layers_l) * len(entangl_l) * len(encodings_l))
    print(f"\n{'='*60}")
    print(f"  VQC Pipeline — {total_runs} runs VQC + baselines clássicos")
    print(f"  fast={fast} | skip_vqc={skip_vqc}")
    print(f"{'='*60}\n")

    # ── VQC sweep ─────────────────────────────────────────────────────────
    if not skip_vqc:
        run_n = 0
        for ts in train_sizes:
            for nq in n_qubits_l:
                for nl in n_layers_l:
                    for ent in entangl_l:
                        for enc in encodings_l:
                            for sd in seeds:
                                run_n += 1
                                print(f"[{run_n}/{total_runs}]", end=" ")
                                result = run_vqc_experiment(
                                    ts, nq, nl, ent, enc, seed=sd
                                )
                                all_results["vqc"].append(result)
                                # salva incrementalmente (seguro contra crash)
                                _save_results(all_results)

    # ── Baselines ─────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("  Baselines clássicos")
    print(f"{'─'*40}")
    for ts in train_sizes:
        for sd in seeds:
            classical = run_classical_experiments(ts, seed=sd)
            for model_name, r in classical.items():
                all_results["classical"].append(r)
            _save_results(all_results)

    elapsed = time.time() - total_t0
    print(f"\n✓ Experimentos concluídos em {elapsed/3600:.1f}h")
    return all_results


def _save_results(results):
    path = os.path.join(OUT_DIR, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ANÁLISE E GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "ring":        "#534AB7",   # purple
    "all_to_all":  "#D85A30",   # coral
    "none":        "#1D9E75",   # teal
    "cnn":         "#378ADD",   # blue
    "mlp":         "#BA7517",   # amber
    "svm":         "#639922",   # green
}

ENT_LABELS = {
    "ring":       "Anel",
    "all_to_all": "All-to-all",
    "none":       "Sem ent.",
}

ENC_LABELS = {
    "angular":   "Angular",
    "amplitude": "Amplitude",
}


def _filter(results_list, **kwargs):
    """Filtra lista de dicts por campos."""
    out = results_list
    for k, v in kwargs.items():
        if v is not None:
            if isinstance(v, list):
                out = [r for r in out if r.get(k) in v]
            else:
                out = [r for r in out if r.get(k) == v]
    return out


def _mean_history(results_list, key):
    """Média de uma métrica de histórico sobre múltiplas seeds."""
    arrays = [r["history"][key] for r in results_list if r.get("history")]
    if not arrays:
        return [], []
    arr = np.array(arrays)
    return arr.mean(axis=0), arr.std(axis=0)


# ── H1: Encoding angular vs entanglement ─────────────────────────────────────

def plot_h1_topology_comparison(results, train_size=100, encoding="angular"):
    """
    H1 — Acurácia de teste por época para as 3 topologias
    Encoding fixo = angular, n = train_size, n_qubits=4, n_layers=3
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for ent in ENTANGLEMENTS:
        runs = _filter(
            results["vqc"],
            train_size=train_size, encoding=encoding,
            entanglement=ent, n_qubits=4, n_layers=3
        )
        mean, std = _mean_history(runs, "test_acc")
        if len(mean) == 0:
            continue
        epochs = range(1, len(mean) + 1)
        ax.plot(epochs, mean, label=ENT_LABELS[ent], color=COLORS[ent], lw=2)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=COLORS[ent], alpha=0.15)

    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia no teste")
    ax.set_title(f"H1 — Topologia de entanglement\n"
                 f"Encoding={ENC_LABELS[encoding]}, n={train_size}, 4 qubits, 3 camadas")
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"h1_topology_n{train_size}_{encoding}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_h1_encoding_comparison(results, train_size=100):
    """
    H1 — Angular vs Amplitude por topologia (subplots)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, ent in zip(axes, ENTANGLEMENTS):
        for enc in ENCODINGS:
            runs = _filter(
                results["vqc"],
                train_size=train_size, encoding=enc,
                entanglement=ent, n_qubits=4, n_layers=3
            )
            mean, std = _mean_history(runs, "test_acc")
            if len(mean) == 0:
                continue
            epochs = range(1, len(mean) + 1)
            ls = "-" if enc == "angular" else "--"
            ax.plot(epochs, mean, label=ENC_LABELS[enc], ls=ls,
                    color=COLORS[ent], lw=2)
            ax.fill_between(epochs, mean - std, mean + std,
                            color=COLORS[ent], alpha=0.12)
        ax.set_title(ENT_LABELS[ent])
        ax.set_xlabel("Época")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.05)
    axes[0].set_ylabel("Acurácia no teste")
    fig.suptitle(f"H1 — Angular vs Amplitude | n={train_size}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"h1_encoding_n{train_size}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── H2: Regime de vantagem ───────────────────────────────────────────────────

def plot_h2_crossover(results):
    """
    H2 — Acurácia final × n para VQC-anel-angular vs baselines
    Ponto de cruzamento onde clássicos ultrapassam VQC
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    train_sizes = sorted(set(r["train_size"] for r in results["vqc"]))

    # VQC anel angular
    vqc_means, vqc_stds = [], []
    for ts in train_sizes:
        runs = _filter(results["vqc"], train_size=ts,
                       entanglement="ring", encoding="angular",
                       n_qubits=4, n_layers=3)
        accs = [r["final_acc"] for r in runs]
        vqc_means.append(np.mean(accs) if accs else np.nan)
        vqc_stds.append(np.std(accs)  if accs else 0)

    ax.errorbar(train_sizes, vqc_means, yerr=vqc_stds,
                label="VQC (anel + angular)", color=COLORS["ring"],
                marker="o", lw=2, capsize=4)

    # Baselines
    for model in ["cnn", "mlp", "svm"]:
        means, stds = [], []
        for ts in train_sizes:
            runs = _filter(results["classical"], train_size=ts, model=model)
            accs = [r["final_acc"] for r in runs]
            means.append(np.mean(accs) if accs else np.nan)
            stds.append(np.std(accs)  if accs else 0)
        ax.errorbar(train_sizes, means, yerr=stds,
                    label=model.upper(), color=COLORS[model],
                    marker="s", ls="--", lw=2, capsize=4)

    ax.set_xlabel("Tamanho do conjunto de treino (n)")
    ax.set_ylabel("Acurácia no teste (época 30)")
    ax.set_title("H2 — Ponto de cruzamento: VQC vs baselines clássicos")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    path = os.path.join(FIG_DIR, "h2_crossover.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_h2_heatmap(results):
    """
    H2 — Tabela de acurácia: modelos × n (heatmap)
    """
    import matplotlib.colors as mcolors

    train_sizes = sorted(set(r["train_size"] for r in results["vqc"]))
    models_vqc  = [(f"VQC-{ENT_LABELS[e]}-{ENC_LABELS[enc]}", e, enc)
                   for e in ENTANGLEMENTS for enc in ENCODINGS]
    models_cls  = [("CNN", "cnn"), ("MLP", "mlp"), ("SVM-RBF", "svm")]

    row_labels = [m[0] for m in models_vqc] + [m[0] for m in models_cls]
    data = []

    for label, ent, enc in models_vqc:
        row = []
        for ts in train_sizes:
            runs = _filter(results["vqc"], train_size=ts,
                           entanglement=ent, encoding=enc,
                           n_qubits=4, n_layers=3)
            accs = [r["final_acc"] for r in runs]
            row.append(np.mean(accs) if accs else np.nan)
        data.append(row)

    for label, model in models_cls:
        row = []
        for ts in train_sizes:
            runs = _filter(results["classical"], train_size=ts, model=model)
            accs = [r["final_acc"] for r in runs]
            row.append(np.mean(accs) if accs else np.nan)
        data.append(row)

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(8, len(row_labels) * 0.55 + 1.5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Acurácia")

    ax.set_xticks(range(len(train_sizes)))
    ax.set_xticklabels([f"n={ts}" for ts in train_sizes])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("H2 — Acurácia (época 30) por modelo × tamanho de treino")

    for i in range(len(row_labels)):
        for j in range(len(train_sizes)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "h2_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── H3: Barren plateaus ──────────────────────────────────────────────────────

def plot_h3_grad_variance(results, train_size=100, encoding="angular"):
    """
    H3 — Variância do gradiente por época para as 3 topologias
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for ent in ENTANGLEMENTS:
        runs = _filter(
            results["vqc"],
            train_size=train_size, encoding=encoding,
            entanglement=ent, n_qubits=4, n_layers=3
        )
        mean, std = _mean_history(runs, "grad_var")
        if len(mean) == 0:
            continue
        epochs = range(1, len(mean) + 1)
        ax.plot(epochs, mean, label=ENT_LABELS[ent],
                color=COLORS[ent], lw=2)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=COLORS[ent], alpha=0.15)

    ax.set_xlabel("Época")
    ax.set_ylabel("std(∇θ)  —  Variância do gradiente")
    ax.set_title(f"H3 — Barren plateaus: variância do gradiente por topologia\n"
                 f"Encoding={ENC_LABELS[encoding]}, n={train_size}, 4 qubits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"h3_grad_var_n{train_size}_{encoding}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_h3_loss_curves(results, train_size=100, encoding="angular"):
    """
    H3 — Loss (BCE) por época: convergência por topologia
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for ent in ENTANGLEMENTS:
        runs = _filter(
            results["vqc"],
            train_size=train_size, encoding=encoding,
            entanglement=ent, n_qubits=4, n_layers=3
        )
        mean, std = _mean_history(runs, "train_loss")
        if len(mean) == 0:
            continue
        epochs = range(1, len(mean) + 1)
        ax.plot(epochs, mean, label=ENT_LABELS[ent],
                color=COLORS[ent], lw=2)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=COLORS[ent], alpha=0.15)

    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title(f"H3 — Convergência do treinamento por topologia\n"
                 f"n={train_size}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"h3_loss_n{train_size}_{encoding}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── B: Variância por inicialização ───────────────────────────────────────────

def plot_b_seed_variance(results, train_size=100, encoding="angular"):
    """
    B — Distribuição de acurácia final por seed (boxplot por topologia)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    data_plot = []
    labels_plot = []
    colors_plot = []

    for ent in ENTANGLEMENTS:
        runs = _filter(
            results["vqc"],
            train_size=train_size, encoding=encoding,
            entanglement=ent, n_qubits=4, n_layers=3
        )
        accs = [r["final_acc"] for r in runs]
        data_plot.append(accs)
        labels_plot.append(ENT_LABELS[ent])
        colors_plot.append(COLORS[ent])

    bp = ax.boxplot(data_plot, labels=labels_plot, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors_plot):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Acurácia no teste (época 30)")
    ax.set_title(f"B — Sensibilidade à inicialização (N={N_SEEDS} seeds)\n"
                 f"n={train_size}, encoding={ENC_LABELS[encoding]}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"b_seed_variance_n{train_size}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── C: Impacto da profundidade ────────────────────────────────────────────────

def plot_c_depth(results, train_size=100, encoding="angular", entanglement="ring"):
    """
    C — Acurácia final × número de camadas
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    layers_available = sorted(set(r["n_layers"] for r in results["vqc"]))
    for ent in [entanglement]:
        means, stds = [], []
        for nl in layers_available:
            runs = _filter(
                results["vqc"],
                train_size=train_size, encoding=encoding,
                entanglement=ent, n_layers=nl, n_qubits=4
            )
            accs = [r["final_acc"] for r in runs]
            means.append(np.mean(accs) if accs else np.nan)
            stds.append(np.std(accs)  if accs else 0)
        ax.errorbar(layers_available, means, yerr=stds,
                    marker="o", lw=2, capsize=4, color=COLORS[ent],
                    label=ENT_LABELS[ent])

    # também plota variância do gradiente no eixo secundário
    ax2 = ax.twinx()
    for ent in [entanglement]:
        grad_means = []
        for nl in layers_available:
            runs = _filter(
                results["vqc"],
                train_size=train_size, encoding=encoding,
                entanglement=ent, n_layers=nl, n_qubits=4
            )
            # média da variância média do gradiente na última época
            gvs = [r["history"]["grad_var"][-1] for r in runs
                   if r.get("history")]
            grad_means.append(np.mean(gvs) if gvs else np.nan)
        ax2.plot(layers_available, grad_means, ls="--", marker="^",
                 color="gray", alpha=0.7, label="Grad std (eixo dir.)")

    ax.set_xlabel("Número de camadas")
    ax.set_ylabel("Acurácia no teste")
    ax2.set_ylabel("std(∇θ) final", color="gray")
    ax.set_title(f"C — Impacto da profundidade | n={train_size}")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"c_depth_n{train_size}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── D: Escalonamento com número de qubits ────────────────────────────────────

def plot_d_qubits(results, train_size=100, encoding="angular", entanglement="ring"):
    """
    D — Acurácia × número de qubits (replica Fig.5 de Yang et al. no paradigma gate-based)
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    qubits_available = sorted(set(r["n_qubits"] for r in results["vqc"]))
    means, stds = [], []
    for nq in qubits_available:
        runs = _filter(
            results["vqc"],
            train_size=train_size, encoding=encoding,
            entanglement=entanglement, n_qubits=nq, n_layers=3
        )
        accs = [r["final_acc"] for r in runs]
        means.append(np.mean(accs) if accs else np.nan)
        stds.append(np.std(accs)  if accs else 0)

    ax.errorbar(qubits_available, means, yerr=stds,
                marker="o", lw=2, capsize=4, color=COLORS[entanglement])
    ax.set_xlabel("Número de qubits")
    ax.set_ylabel("Acurácia no teste")
    ax.set_title(f"D — Escalonamento com qubits | n={train_size}\n"
                 f"(replica gate-based do Fig.5 de Yang et al.)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"d_qubits_n{train_size}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── F: Gap treino × teste (overfitting) ──────────────────────────────────────

def plot_f_overfitting(results):
    """
    F — Gap treino–teste para todos os modelos por n de treino
    """
    train_sizes = sorted(set(r["train_size"] for r in results["vqc"]))
    fig, axes = plt.subplots(1, len(train_sizes), figsize=(14, 5), sharey=True)

    for ax, ts in zip(axes, train_sizes):
        labels, gaps, colors = [], [], []

        # VQC variantes
        for ent in ENTANGLEMENTS:
            runs = _filter(results["vqc"], train_size=ts,
                           entanglement=ent, encoding="angular",
                           n_qubits=4, n_layers=3)
            if runs:
                g = np.mean([r["train_acc"] - r["final_acc"] for r in runs])
                labels.append(f"VQC\n{ENT_LABELS[ent]}")
                gaps.append(g)
                colors.append(COLORS[ent])

        # Baselines
        for model in ["cnn", "mlp", "svm"]:
            runs = _filter(results["classical"], train_size=ts, model=model)
            if runs:
                g = np.mean([r["train_acc"] - r["final_acc"] for r in runs])
                labels.append(model.upper())
                gaps.append(g)
                colors.append(COLORS[model])

        bars = ax.bar(range(len(labels)), gaps, color=colors, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(f"n={ts}")
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Gap acurácia (treino − teste)")
    fig.suptitle("F — Gap de overfitting: treino − teste por modelo e tamanho",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "f_overfitting_gap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── G: Matrizes de confusão ───────────────────────────────────────────────────

def plot_g_confusion_matrices(results, train_size=100):
    """
    G — Matrizes de confusão: VQC-anel-angular vs CNN vs SVM
    """
    configs = [
        ("VQC (anel+angular)", "vqc",
         {"entanglement": "ring", "encoding": "angular",
          "n_qubits": 4, "n_layers": 3}),
        ("CNN", "classical", {"model": "cnn"}),
        ("SVM-RBF", "classical", {"model": "svm"}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, pool, filters) in zip(axes, configs):
        key = "vqc" if pool == "vqc" else "classical"
        runs = _filter(results[key], train_size=train_size, **filters)
        if not runs:
            ax.set_title(f"{title}\n(sem dados)")
            continue

        # agrega matrizes de confusão
        cms = [np.array(r["conf_matrix"]) for r in runs]
        cm_mean = np.mean(cms, axis=0).astype(int)

        im = ax.imshow(cm_mean, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["Real 0", "Real 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm_mean[i, j], ha="center", va="center",
                        fontsize=14, color="white" if cm_mean[i, j] > cm_mean.max()/2 else "black")
        ax.set_title(f"{title}\n(n={train_size})")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("G — Matrizes de confusão (média sobre seeds)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"g_confusion_matrices_n{train_size}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── Custo computacional ───────────────────────────────────────────────────────

def plot_computational_cost(results):
    """
    Tempo de treinamento e razão tempo/acurácia por modelo
    """
    train_sizes = sorted(set(r["train_size"] for r in results["vqc"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Tempo total por configuração de VQC
    ax = axes[0]
    for ent in ENTANGLEMENTS:
        times = []
        for ts in train_sizes:
            runs = _filter(results["vqc"], train_size=ts,
                           entanglement=ent, encoding="angular",
                           n_qubits=4, n_layers=3)
            t = [r["total_time"] for r in runs]
            times.append(np.mean(t) if t else np.nan)
        ax.plot(train_sizes, times, marker="o", lw=2,
                color=COLORS[ent], label=f"VQC {ENT_LABELS[ent]}")

    for model in ["cnn", "mlp"]:
        times = []
        for ts in train_sizes:
            runs = _filter(results["classical"], train_size=ts, model=model)
            t = [r["total_time"] for r in runs]
            times.append(np.mean(t) if t else np.nan)
        ax.plot(train_sizes, times, marker="s", ls="--", lw=2,
                color=COLORS[model], label=model.upper())

    ax.set_xlabel("Tamanho do treino")
    ax.set_ylabel("Tempo total (s)")
    ax.set_title("Custo computacional por modelo")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Razão tempo/acurácia (VQC vs CNN)
    ax2 = axes[1]
    vqc_ratios, cnn_ratios = [], []
    for ts in train_sizes:
        vqc_runs = _filter(results["vqc"], train_size=ts,
                           entanglement="ring", encoding="angular",
                           n_qubits=4, n_layers=3)
        cnn_runs = _filter(results["classical"], train_size=ts, model="cnn")
        if vqc_runs and cnn_runs:
            vt = np.mean([r["total_time"] for r in vqc_runs])
            ct = np.mean([r["total_time"] for r in cnn_runs])
            va = np.mean([r["final_acc"]  for r in vqc_runs])
            ca = np.mean([r["final_acc"]  for r in cnn_runs])
            vqc_ratios.append(vt / max(va, 1e-6))
            cnn_ratios.append(ct / max(ca, 1e-6))

    if vqc_ratios:
        ax2.bar(np.array(range(len(train_sizes))) - 0.15, vqc_ratios,
                0.3, label="VQC (anel+angular)", color=COLORS["ring"], alpha=0.8)
        ax2.bar(np.array(range(len(train_sizes))) + 0.15, cnn_ratios,
                0.3, label="CNN", color=COLORS["cnn"], alpha=0.8)
        ax2.set_xticks(range(len(train_sizes)))
        ax2.set_xticklabels([f"n={ts}" for ts in train_sizes])
        ax2.set_ylabel("Tempo / Acurácia (s/unidade)")
        ax2.set_title("Razão custo-efetividade")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "computational_cost.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ── Summary CSV ───────────────────────────────────────────────────────────────

def generate_summary_csv(results):
    """
    Gera tabela consolidada de resultados para uso no artigo
    """
    import csv

    rows = []
    train_sizes = sorted(set(r["train_size"] for r in results["vqc"]))

    # VQC
    for ent in ENTANGLEMENTS:
        for enc in ENCODINGS:
            for nl in sorted(set(r["n_layers"] for r in results["vqc"])):
                for nq in sorted(set(r["n_qubits"] for r in results["vqc"])):
                    for ts in train_sizes:
                        runs = _filter(results["vqc"], train_size=ts,
                                       entanglement=ent, encoding=enc,
                                       n_layers=nl, n_qubits=nq)
                        if not runs:
                            continue
                        accs  = [r["final_acc"]  for r in runs]
                        baccs = [r["best_acc"]   for r in runs]
                        taccs = [r["train_acc"]  for r in runs]
                        times = [r["total_time"] for r in runs]
                        gvars = [r["history"]["grad_var"][-1]
                                 for r in runs if r.get("history")]
                        rows.append({
                            "model":          f"VQC-{ent}-{enc}",
                            "n_qubits":       nq,
                            "n_layers":       nl,
                            "train_size":     ts,
                            "n_params":       runs[0]["n_params"],
                            "final_acc_mean": f"{np.mean(accs):.4f}",
                            "final_acc_std":  f"{np.std(accs):.4f}",
                            "best_acc_mean":  f"{np.mean(baccs):.4f}",
                            "train_acc_mean": f"{np.mean(taccs):.4f}",
                            "overfit_gap":    f"{np.mean(taccs)-np.mean(accs):.4f}",
                            "time_mean_s":    f"{np.mean(times):.1f}",
                            "grad_var_final": f"{np.mean(gvars):.6f}" if gvars else "NA",
                            "n_seeds":        len(runs),
                        })

    # Baselines
    for model in ["cnn", "mlp", "svm"]:
        for ts in train_sizes:
            runs = _filter(results["classical"], train_size=ts, model=model)
            if not runs:
                continue
            accs  = [r["final_acc"]  for r in runs]
            taccs = [r["train_acc"]  for r in runs]
            times = [r["total_time"] for r in runs]
            rows.append({
                "model":          model.upper(),
                "n_qubits":       "—",
                "n_layers":       "—",
                "train_size":     ts,
                "n_params":       runs[0]["n_params"],
                "final_acc_mean": f"{np.mean(accs):.4f}",
                "final_acc_std":  f"{np.std(accs):.4f}",
                "best_acc_mean":  f"{np.mean(accs):.4f}",
                "train_acc_mean": f"{np.mean(taccs):.4f}",
                "overfit_gap":    f"{np.mean(taccs)-np.mean(accs):.4f}",
                "time_mean_s":    f"{np.mean(times):.1f}",
                "grad_var_final": "NA",
                "n_seeds":        len(runs),
            })

    path = os.path.join(OUT_DIR, "summary_table.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Salvo: {path}")
    return rows


# ── Gerador de todos os gráficos ──────────────────────────────────────────────

def generate_all_plots(results):
    print("\n  Gerando figuras...")
    train_sizes = sorted(set(r["train_size"] for r in results["vqc"]))

    for ts in train_sizes:
        for enc in ENCODINGS:
            plot_h1_topology_comparison(results, train_size=ts, encoding=enc)
            plot_h3_grad_variance(results, train_size=ts, encoding=enc)
            plot_h3_loss_curves(results, train_size=ts, encoding=enc)

        plot_h1_encoding_comparison(results, train_size=ts)
        plot_b_seed_variance(results, train_size=ts)
        plot_c_depth(results, train_size=ts)
        plot_d_qubits(results, train_size=ts)
        plot_g_confusion_matrices(results, train_size=ts)

    plot_h2_crossover(results)
    plot_h2_heatmap(results)
    plot_f_overfitting(results)
    plot_computational_cost(results)
    generate_summary_csv(results)
    print(f"\n  Todas as figuras salvas em: {FIG_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQC Pipeline")
    parser.add_argument("--fast",      action="store_true",
                        help="Roda subset rápido: n=[50,100], 1 seed, 4 qubits, 3 camadas")
    parser.add_argument("--skip-vqc", action="store_true",
                        help="Pula experimentos VQC, só roda baselines clássicos")
    parser.add_argument("--plots-only", action="store_true",
                        help="Apenas regera gráficos a partir de results.json existente")
    args = parser.parse_args()

    results_path = os.path.join(OUT_DIR, "results.json")

    if args.plots_only:
        print(f"  Carregando resultados de {results_path}...")
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = run_all(fast=args.fast, skip_vqc=args.skip_vqc)

    generate_all_plots(results)
    print("\n✓ Pipeline completo.")