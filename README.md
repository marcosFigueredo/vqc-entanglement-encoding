# VQC entanglement, encoding, and data regime

Code and raw results for the article

> **Variational Quantum Circuits for Image Classification: The Roles of Entanglement, Data Encoding, and Training Data Regime**
> M. B. Figueredo, L. S. Morais, T. B. Murari, A. R. C. B. da Silva, R. L. S. Monteiro, A. N. Silva, V. Fonseca, E. P. Garrido, J. R. A. Fontoura, M. A. Moret.
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
| `figure1_z0_distributions.py` | `results_z0/` | Per-sample ⟨Z₀⟩ on the test set for six factorial configurations, pooled over 5 seeds (Fig. 1) |
| `revision_fix_crosstask.py` | `results_generality/` | Reruns the cross-task runs that had been evaluated on the wrong test split (see note below) |
| `revision_readout_ceiling.py` | `results_revision/readout_ceiling.csv` | Bloch-vector reference (direction chosen on the training set) and exact test-set bound (maximum over all measurement directions, computed exactly) for circuits without entangling gates (Table 11, Fig. 3) |
| `revision_multireadout.py` | `results_revision/multireadout_*.csv` | Readout ⟨Z₀⟩ vs. the four ⟨Zᵢ⟩ on the three tasks, seeds 0–4 |
| `revision_correlator_readout.py` | `results_revision/correlator_readout_*.csv` | Readouts with ⟨Zᵢ⟩ + ⟨ZᵢZⱼ⟩ and with the full computational-basis distribution, ring vs. no entangling gates (Table 12) |
| `revision_stats.py` | `results_revision/S1–S7_*.csv` | Paired Wilcoxon and TOST equivalence tests, Holm correction, seed-level comparisons, CI of the gradient-decay slope, classical baselines on the VQC features, VQC selection by training accuracy, TOST by readout |
| `revision_cost.py` | `results_revision/S6_compiled_cost.csv` | CNOT count and depth of the compiled circuits, including state preparation |
| `revision_figures.py` | `results_revision/fig2_*.png`, `fig3_*.png` | Figures 2 and 3 |

### Reproducing

```bash
python run_all.py                    # factorial sweep + baselines (several hours on CPU)
python run_all.py --plots-only       # regenerate figures from results/results.json
python fixh.py --h1-only             # convergence curves
python fix_h1_h3_v2.py --h1-only     # topology comparison, 30 seeds
python genalizacao.py --exp A --out-dir results_generality --workers 4
python control_experiments.py        # E1, E2, E3 (use --only E1|E2|E3 to run one)
python figure1_z0_distributions.py   # Fig. 1 (about 5 min); --plots-only to redraw
```

`control_experiments.py` and `figure1_z0_distributions.py` import the data loading and circuit construction from `run_all.py`, so all three must stay in the same folder. Their runs that replicate factorial conditions reproduce the original final accuracies exactly, and `figure1_z0_distributions.py` checks this for each of its 30 runs.

## Notes on naming in the raw files

These scripts were written before the final analysis. Some labels in the raw outputs differ from the terminology used in the paper:

- **`n400` / `train_size = 400`** in `results/`: `load_digits` has only 360 images of digits 0 and 1, and 100 are held out for testing, so the largest training set actually contains **260** samples. The paper reports it as n = 260.
- **`grad_var`** in `results/`, `results_fix/`, `results_fix_v2/`, and `results_generality/`: this is the standard deviation of the components of a single gradient vector (σ_g in the paper), not the variance over random initializations. The true per-parameter variance, Var_θ[∂L/∂θ_k], is computed only in `results_controls/E1_grad_variance.json`.
- **`angular`** means angle encoding; **`none`** means no entangling gates in the variational layers (the amplitude-encoded state itself can be entangled).
- **Cross-task runs on the wrong test split**: in the original cross-task data, all 0 vs 1 runs with ring topology and two 0 vs 1 angle/all-to-all runs had been produced by an earlier script version with a fixed 100-sample test set, while all other cross-task runs use the 25% stratified split of `genalizacao.py`. `revision_fix_crosstask.py` detects these runs (their accuracy is not a multiple of 1/test-size) and reruns all 62 with the 25% split; the original file is kept as `results_generality/generality_raw_original.csv`.
- Some code comments and log messages are in Portuguese.

## License

MIT; see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the article (see [CITATION.cff](CITATION.cff)).
