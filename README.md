# VQC entanglement, encoding, and data regime

Code and raw results for the article

> **Variational Quantum Circuits for Image Classification: The Roles of Entanglement, Data Encoding, and Training Data Regime**
> M. B. Figueredo, L. S. Morais, T. B. Murari, A. Correia, R. L. S. Monteiro, A. N. Silva, V. Fonseca, E. P. Garrido, J. R. A. Fontoura, M. A. Moret.
> Submitted to *Quantum Machine Intelligence* (Springer Nature).

The study characterizes how data encoding (angle vs. amplitude), entanglement topology (ring, all-to-all, none), circuit width and depth, and training-set size affect gate-based variational quantum classifiers (VQCs) on binary handwritten-digit tasks, and compares them with classical baselines (CNN, SVM-RBF, MLP).

## Data

All experiments use the *Optical Recognition of Handwritten Digits* dataset (UCI Machine Learning Repository, [doi:10.24432/C50P49](https://doi.org/10.24432/C50P49)), loaded through `sklearn.datasets.load_digits` (8×8 grayscale images). No data file needs to be downloaded. The digit pairs appear as `mnist_0v1`, `mnist_3v5`, and `mnist_4v9` in the result files; this is a historical label, and the images are the 8×8 UCI digits, not MNIST.

## Environment

Python 3.13, CPU only.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Scripts and where each result appears in the paper

| Script | Output folder | Paper |
|---|---|---|
| `run_all.py` | `results/` | Full factorial sweep (216 VQC conditions, 5 seeds) and classical baselines: Tables 2–5, 9, 12, 13 |
| `fixh.py` | `results_fix/` | Convergence curves over 60 epochs, N = 15 seeds (Fig. 2) |
| `fix_h1_h3_v2.py` | `results_fix_v2/` | Topology comparison with N = 30 seeds and Mann–Whitney tests (Table 7) |
| `genalizacao.py --exp A` | `results_generality/` | Cross-task replication on 0 vs 1, 3 vs 5, 4 vs 9, N = 10 seeds (Table 11, Fig. 3) |
| `control_experiments.py` | `results_controls/` | E1 per-parameter gradient variance, n = 2–10 (Table 8); E2 readout control (Table 10); E3 input-information control (Table 6) |

`results_confusion_tsne/` holds the per-sample ⟨Z₀⟩ outputs used for the output-distribution figure (Fig. 1).

### Reproducing

```bash
python run_all.py                    # factorial sweep + baselines (several hours on CPU)
python run_all.py --plots-only       # regenerate figures from results/results.json
python fixh.py --h1-only             # convergence curves
python fix_h1_h3_v2.py --h1-only     # topology comparison, 30 seeds
python genalizacao.py --exp A --out-dir results_generality --workers 4
python control_experiments.py        # E1, E2, E3 (use --only E1|E2|E3 to run one)
```

`control_experiments.py` imports the data loading and circuit construction from `run_all.py`, so both must stay in the same folder. Its runs that replicate factorial conditions reproduce the original final accuracies exactly.

## Notes on naming in the raw files

These scripts were written before the final analysis. Some labels in the raw outputs differ from the terminology used in the paper:

- **`n400` / `train_size = 400`** in `results/`: `load_digits` has only 360 images of digits 0 and 1, and 100 are held out for testing, so the largest training set actually contains **260** samples. The paper reports it as n = 260.
- **`grad_var`** in `results/`, `results_fix/`, `results_fix_v2/`, and `results_generality/`: this is the standard deviation of the components of a single gradient vector (σ_g in the paper), not the variance over random initializations. The true per-parameter variance, Var_θ[∂L/∂θ_k], is computed only in `results_controls/E1_grad_variance.json`.
- **`angular`** means angle encoding.
- Some code comments and log messages are in Portuguese.

## License

MIT; see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the article (see [CITATION.cff](CITATION.cff)).
