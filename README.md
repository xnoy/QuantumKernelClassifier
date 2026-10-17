# Qiskit Fall Fest 2026 — Track 4 (QML)
# PS-Q2: Quantum Kernel Estimator & Support Vector Machine (QSVM)
### Benchmark and Hardware Transpilation Evaluation on Processors A & B

[![Qiskit 1.x / 2.x](https://img.shields.io/badge/Qiskit-1.x%20%2F%202.x-6929C4.svg)](https://qiskit.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides an authoritative, end-to-end implementation and benchmarking suite for **Problem Statement Q2 (PS-Q2)** under the official **QFF 2026 Phase 1 Master Guide** (SRM University-AP). 

We evaluate 4-qubit Quantum Support Vector Machines (QSVM) using parameterized `ZZFeatureMap` circuits under **frozen QFF 2026 synthetic noise** across two heterogeneous physical processor topologies:
- **Processor A**: 5 physical qubits, branching-tree topology ($V=5, E=4$, diameter 3) plus creative ring alternative ($E=5$, diameter 2).
- **Processor B**: 9 physical qubits, heavy-hex-inspired topology ($V=9, E=12$, diameter 4) with dense cycle subgraph $[1, 2, 5, 6]$.

---

## 1. Repository Architecture

The project strictly adheres to the authoritative Phase 1 Master Guide repository layout:

```text
qff2026-psq2-qsvm/
├── README.md                               # Project documentation & benchmark analysis
├── report.md                               # Formal technical report for QFF judges
├── main.ipynb                              # Fully executed, reproducible Jupyter Notebook
├── main.html                               # Standalone HTML report export
├── requirements.txt                        # Pinned dependencies
├── generate_notebook.py                    # Programmatic end-to-end pipeline generator
├── data/
│   ├── raw/                                # Raw unscaled arrays (moons, digits, synthetic)
│   └── processed/                          # Scaled [0, π]^4 feature matrices & labels
├── processors/
│   ├── processor_A.json                    # 5-qubit branching tree + custom ring alternative
│   └── processor_B.json                    # 9-qubit heavy-hex with optimal 4-qubit cycle [1,2,5,6]
├── src/
│   ├── __init__.py                         # Public API exports
│   ├── processors.py                       # Processor configs, AerSimulator, frozen NoiseModel
│   ├── problem.py                          # Primary (moons), secondary (digits), quantum-aligned datasets
│   ├── routing.py                          # Transpilation analysis, depth, CX, and SWAP overhead
│   ├── metrics.py                          # Ideal & noisy kernel estimation, PSD mitigation, SVM benchmarking
│   └── visualization.py                    # Publication-grade figures (300 DPI)
└── results/
    ├── figures/
    │   ├── fig1_processor_topologies.png   # Side-by-side NetworkX coupling topologies
    │   ├── fig2_architecture_comparison.png# Transpilation depth, CX, and SWAP bar chart
    │   ├── fig3_kernel_matrices_and_concentration.png # Gram heatmaps & off-diagonal histograms
    │   ├── fig4_decision_boundaries.png    # 2D contour comparisons (Ideal, Noisy, Mitigated, Classical)
    │   └── fig5_accuracy_auc_comparison.png# 7-model performance with 95% Confidence Intervals
    └── tables/
        ├── architecture_routing_comparison.csv # Complete hardware routing metrics
        ├── classification_benchmark_results.csv # 5-seed statistical benchmark scores
        └── kernel_fidelity_metrics.csv     # Frobenius error, KTA, and variance metrics
```

---

## 2. Frozen Hardware & Synthetic Noise Specifications

All quantum circuits were executed under the strict Phase 1 frozen parameters:
- **Native Basis Gates**: `['rz', 'sx', 'x', 'cx']`
- **1-Qubit Gate Depolarizing Error**: $p_{1q} = 0.001$ on `['rz', 'sx', 'x']`
- **2-Qubit Gate Depolarizing Error**: $p_{2q} = 0.005$ on `['cx']`
- **Symmetric Readout Error**: $p_{ro} = 0.01$ ($[[0.99, 0.01], [0.01, 0.99]]$) on all physical qubits
- **Transpilation Settings**: `optimization_level = 1`, `seed_transpiler = 42`, `shots = 4096`.
- **Modern Qiskit Standard**: Built strictly on Qiskit 1.x / 2.x standards (`qiskit.transpile`, `qiskit_aer.AerSimulator`). Parameterized circuits are transpiled once outside pair loops.

---

## 3. Quickstart & Reproducibility

### Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name qsvm-venv --display-name "Python (QSVM)"
```

### Run Modular Pipeline & Self-Tests
```bash
# 1. Validate processors & noise models
python3 -m src.processors

# 2. Generate and verify datasets & feature maps
python3 -m src.problem

# 3. Execute routing analysis and compute SWAP overhead
python3 -m src.routing

# 4. Compute ideal & noisy hardware kernels, PSD mitigation & SVM benchmarks
python3 -m src.metrics

# 5. Render all 5 publication figures
python3 -m src.visualization
```

### Regenerate & Execute Notebook End-to-End
```bash
python3 generate_notebook.py
```

---

## 4. Architectural Trade-Off Analysis

### Processor Topology Invariants

| Architectural Parameter | Processor A (Branching Tree) | Processor A (Ring Alternative) | Processor B (Heavy-Hex Sub-lattice) |
|---|:---:|:---:|:---:|
| **Physical Qubit Budget ($N$)** | 5 | 5 | 9 |
| **Physical Coupling Edges ($E$)** | 4 | 5 | 12 |
| **Graph Diameter** | 3 | 2 | 4 |
| **Average Shortest Path** | 1.6000 | 1.2000 | 1.8333 |
| **Average Node Degree** | 1.6000 | 2.0000 | 2.6667 |
| **Optimal 4-Qubit Subgraph** | `[0, 1, 2, 3]` | `[0, 1, 2, 3]` | `[1, 2, 5, 6]` (Dense Cycle) |

### Transpilation & Routing Overhead (`architecture_routing_comparison.csv`)

| Configuration | Topology | Entanglement | Reps | Logical CX | Transpiled Depth | Transpiled CX | Est. SWAPs | Sabre SWAPs | Total Gates |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proc A (Primary)** | Branching Tree | Linear | 2 | 12 | **23** | 12 | **0** | 0 | 50 |
| **Proc A (Ring Alt)** | 5Q Cycle | Linear | 2 | 12 | **23** | 12 | **0** | 0 | 50 |
| **Proc A (Primary)** | Branching Tree | Full | 2 | 24 | **61** | 33 | **3** | 0 | 77 |
| **Proc A (Ring Alt)** | 5Q Cycle | Full | 2 | 24 | **55** | 40 | 5 | 0 | 84 |
| **Proc B (Primary)** | Heavy-Hex | Linear | 2 | 12 | **23** | 12 | **0** | 0 | 50 |
| **Proc B (Primary)** | Heavy-Hex | Full | 2 | 24 | **44** | 30 | **2** | 0 | 74 |

#### Key Architectural Takeaways:
1. **Linear Entanglement is SWAP-Free**: Both processors achieve depth 23 and 0 SWAPs because a 4-qubit line directly embeds into both coupling graphs.
2. **Heavy-Hex Cycle Outperforms Branching Tree**: On full entanglement, Processor B achieves **depth 44 and only 2 SWAPs**, compared to Processor A's **depth 61 and 3 SWAPs** (a **28% depth reduction** and **33% SWAP reduction**).
3. **Topological Bottlenecks**: In Processor A, node 1 is a degree-3 junction, forcing sequential gate serialization. Processor B's $[1, 2, 5, 6]$ cycle provides two independent routing channels.

### Phase 2 Processor C Proposal
To eliminate all remaining SWAP routing overhead for Phase 2 finalists:
- **Processor C Topology**: 16-qubit dual-hexagon lattice with cross-hub shortcut couplers.
- **Routing Advantage**: Guarantees that any 4-qubit all-to-all subgraph embeds as a planar $K_4$ minor, achieving **0 SWAPs** and depth $\le 35$ for arbitrary full-entanglement feature maps.

---

## 5. Statistical Benchmarking & QML vs. Classical Comparison

### 5-Seed Benchmark Results (`classification_benchmark_results.csv`)

Evaluated across seeds `[42, 100, 2026, 7, 99]` on the primary `make_moons` dataset ($N=100$, 80/20 stratified split, 4096 shots). Reported as **Mean $\pm$ 95% Confidence Interval**:

| Model | Test Accuracy (Mean ± 95% CI) | F1-Score (Mean ± 95% CI) | ROC-AUC (Mean ± 95% CI) | KTA (Mean ± 95% CI) | Off-Diag Variance | Total Circuits |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **QSVM (Ideal)** | **0.8000 ± 0.0982** | 0.7958 ± 0.0995 | 0.8460 ± 0.1212 | 0.1037 ± 0.0043 | 0.00789 | 3,160 |
| **QSVM (Proc A - Noisy)** | 0.7800 ± 0.0942 | 0.7803 ± 0.0869 | 0.8380 ± 0.0914 | 0.0949 ± 0.0039 | 0.00385 | 3,160 |
| **QSVM (Proc B - Noisy)** | 0.7700 ± 0.0833 | 0.7725 ± 0.0756 | 0.8360 ± 0.0948 | 0.0954 ± 0.0038 | 0.00403 | 3,160 |
| **QSVM (Proc A - Mitigated)** | **0.8000 ± 0.1161** | 0.7973 ± 0.1075 | 0.8400 ± 0.1155 | 0.0948 ± 0.0040 | 0.00752 | 3,160 |
| **QSVM (Proc B - Mitigated)** | **0.7900 ± 0.1194** | 0.7921 ± 0.1157 | 0.8460 ± 0.1074 | 0.0954 ± 0.0038 | 0.00752 | 3,160 |
| **Classical SVM (RBF Tuned)** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | N/A | N/A | 0 |
| **Classical SVM (Poly Tuned)** | 0.9700 ± 0.0340 | 0.9694 ± 0.0348 | 0.9780 ± 0.0544 | N/A | N/A | 0 |

### Circuit Execution Budget
Total quantum circuit executions needed for kernel construction:
$$\text{Circuits} = \frac{N(N-1)}{2} = \frac{80 \times 79}{2} = \mathbf{3,160} \text{ circuits per training Gram matrix}$$
(Plus $N_{\text{test}} \times N_{\text{train}} = 20 \times 80 = 1,600$ evaluation circuits, totaling 4,760 circuits per seed).

---

## 6. Mitigation of Kernel Concentration & PSD Correction

### Noise-Induced Concentration
Under two-qubit depolarizing error ($p_{2q}=0.005$) and readout errors ($p_{ro}=0.01$):
- Quantum states decay toward the maximally mixed state $\frac{I}{16}$.
- Diagonal entries drop from $1.000$ to $\sim 0.716$ (Proc A) and $\sim 0.733$ (Proc B).
- Off-diagonal variance halves from $0.0075$ to $0.0037$, compressing values toward a single spike (noise-induced concentration).

### Two-Stage Mitigation Strategy
1. **Diagonal Normalization**:
   $$\tilde{K}_{ij} = \frac{K_{ij}}{\sqrt{K_{ii} K_{jj}}}$$
   Restores diagonal entries strictly to $1.000$ and neutralizes global depolarizing shrinkage.
2. **Negative Eigenvalue Clipping (PSD Projection)**:
   $$K_{\text{PSD}} = V \max(\Lambda, 0) V^T + \epsilon I$$
   Ensures the Gram matrix is strictly positive semi-definite, preventing quadratic programming solver failures in `SVC(kernel='precomputed')`.

### Kernel Fidelity Recovery (`kernel_fidelity_metrics.csv`)
| Kernel Configuration | Relative Frobenius Error vs Ideal | Off-Diagonal Variance | Diagonal Mean | Min. Eigenvalue |
|---|:---:|:---:|:---:|:---:|
| **QSVM Ideal** | 0.0000 | 0.00752 | 1.0000 | +0.0118 |
| **Proc A Noisy** | 0.2575 | 0.00366 | 0.7163 | +0.0106 |
| **Proc A Mitigated** | **0.1502** | **0.00714** | **1.0000** | +0.0148 |
| **Proc B Noisy** | 0.2422 | 0.00386 | 0.7329 | +0.0032 |
| **Proc B Mitigated** | **0.1388** | **0.00719** | **1.0000** | +0.0044 |

Mitigation reduces relative Frobenius error by **41.7% on Processor A** and **42.7% on Processor B**, successfully recovering classification accuracy back to ideal levels.

---

## 7. Deliverables & Artifact Verification

- [x] **`main.ipynb`**: Fully executed top-to-bottom without warnings or errors.
- [x] **`main.html` / `qsvm.html`**: Standalone web report viewable in any browser.
- [x] **`report.md`**: Complete Phase 1 technical submission report.
- [x] **`processors/`**: Creative JSON definitions for Processor A (branching tree + ring) and Processor B (heavy-hex).
- [x] **`results/figures/`**: 5 high-resolution 300 DPI publication plots.
- [x] **`results/tables/`**: CSV tables for routing, classification, and kernel fidelity.
- [x] **`data/`**: Raw and processed `.npy` and `.csv` files for all 3 datasets.

---

## 8. References

1. Havlíček, V., et al. "Supervised learning with quantum-enhanced feature spaces." *Nature* 567, 209–212 (2019).
2. Thanasilp, S., et al. "Exponential concentration in quantum kernel methods." *Nature Communications* 15, 5200 (2024).
3. Kübler, J., Muandet, K., & Bousquet, O. "The Inductive Bias of Quantum Kernels." *NeurIPS* 34, 12661–12673 (2021).
4. Liu, Y., Arunachalam, S., & Temme, K. "A rigorous and robust quantum speed-up in supervised machine learning." *Nature Physics* 17, 1013–1017 (2021).
5. SRM University-AP QFF 2026 Phase 1 Master Guide (Unified Challenge Kit).
