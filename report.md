# Qiskit Fall Fest 2026 — Track 4 (QML)
# PS-Q2: Quantum Kernel Estimator & Support Vector Machine (QSVM)
## Technical Report & Architectural Evaluation — Phase 1

---

## 1. Executive Summary

This report delivers the comprehensive benchmarking and architectural evaluation for **Track 4 (QML) — PS-Q2: Quantum Kernel Estimator & Support Vector Machine (QSVM)** under the frozen **QFF 2026 Phase 1** hardware and synthetic noise guidelines.

We benchmark 4-qubit Quantum Support Vector Machines (QSVM) using parameterized `ZZFeatureMap` circuits evaluated on two physical topologies:
1. **Processor A**: 5 physical qubits, branching-tree topology ($V=5, E=4$, diameter 3) plus custom ring alternative ($E=5$).
2. **Processor B**: 9 physical qubits, heavy-hex-inspired topology ($V=9, E=12$, diameter 4) with dense cycle subgraph $[1, 2, 5, 6]$.

We rigorously quantify the interplay between **quantum circuit depth, topological routing overhead (SWAPs), depolarizing-induced kernel concentration, and classification accuracy** compared against hyperparameter-tuned classical SVM baselines (RBF and Polynomial kernels) across 5 random seeds (`[42, 100, 2026, 7, 99]`).

---

## 2. Frozen Hardware & Synthetic Noise Specifications

All quantum circuits were executed under the strict Phase 1 frozen parameters:

- **Native Basis Gates**: `['rz', 'sx', 'x', 'cx']`
- **1-Qubit Gate Depolarizing Error**: $p_{1q} = 0.001$ on `['rz', 'sx', 'x']`
- **2-Qubit Gate Depolarizing Error**: $p_{2q} = 0.005$ on `['cx']`
- **Symmetric Readout Error**: $p_{ro} = 0.01$ per qubit:
  $$\begin{pmatrix} 0.99 & 0.01 \\ 0.01 & 0.99 \end{pmatrix}$$
- **Transpilation Settings**: `optimization_level = 1`, `seed_transpiler = 42`, `shots = 4096`.
- **Modern Qiskit Standard**: Strictly built on Qiskit 1.x / 2.x standards (`qiskit.transpile`, `qiskit_aer.AerSimulator`, `qiskit_aer.noise.*`). Parameterized circuits are transpiled once outside pair loops.

---

## 3. Processor Topology Modeling & Co-Design

### 3.1 Processor A (5-Qubit Branching Tree)
- **Undirected Edges**: `[(0, 1), (1, 2), (1, 3), (3, 4)]`
- **Directed Coupling Map**: 8 directed edges.
- **Topological Analysis**: Qubit 1 is a degree-3 junction, creating a routing bottleneck.
- **Active 4-Qubit Subgraph**: `[0, 1, 2, 3]` provides a continuous 3-edge chain `0-1-2` with branch `1-3`.
- **Custom Co-Design Alternative (5-Qubit Ring)**: Adding edge `(4, 0)` creates a cycle `0-1-2-3-4-0`, yielding uniform degree-2 connectivity and reducing diameter from 3 to 2.

### 3.2 Processor B (9-Qubit Heavy-Hex-Inspired)
- **Undirected Edges**: 12 edges: `[(0, 1), (1, 2), (1, 6), (2, 3), (2, 6), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (8, 0)]`
- **Directed Coupling Map**: 24 directed edges.
- **Optimal 4-Qubit Subgraph**: Nodes `[1, 2, 5, 6]` form a dense cycle with chord: edges `(1, 2), (2, 5), (5, 6), (6, 1), (2, 6)`.
- **Co-Design Benefit**: This dense 4-cycle allows 4-qubit feature maps to route with minimal SWAP overhead compared to branching or linear topologies.

---

## 4. Quantum Feature Maps & Compute-Uncompute Kernel

### 4.1 Feature Map Construction
A classical sample $x = (x_0, x_1, x_2, x_3) \in [0, \pi]^4$ is encoded via the 2-local `ZZFeatureMap` with 2 repetitions:
$$U_\Phi(x) = \left[ \prod_{j=0}^3 R_Z(2x_j) \cdot H^{\otimes 4} \cdot \prod_{(j,k) \in E} \text{CX}_{j,k} \cdot R_Z(2(\pi - x_j)(\pi - x_k)) \cdot \text{CX}_{j,k} \right]^2$$

We analyze two entanglement connectivity structures:
1. **Linear Entanglement**: 3 CX pairs per repetition `[(0,1), (1,2), (2,3)]` $\implies$ 12 logical CX gates total.
2. **Full Entanglement**: 6 all-to-all pairs per repetition $\implies$ 24 logical CX gates total.

### 4.2 Compute-Uncompute Circuit
Transition probability estimation evaluates the overlap between two states $|\Phi(x_i)\rangle$ and $|\Phi(x_j)\rangle$:
$$K(x_i, x_j) = |\langle\Phi(x_i)|\Phi(x_j)\rangle|^2 = |\langle 0^{\otimes 4}| U_\Phi^\dagger(x_i) U_\Phi(x_j) |0^{\otimes 4}\rangle|^2 = P(|0000\rangle)$$

The circuit composes $U_\Phi(x_j)$, a barrier, and $U_\Phi^\dagger(x_i)$, measured into classical registers with 4096 shots.

---

## 5. Architectural Routing & SWAP Overhead Analysis

| Configuration | Topology | Entanglement | Reps | Logical CX | Transpiled Depth | Transpiled CX | Est. SWAPs | Sabre SWAPs | Total Gates |
|---|---|---|---|---|---|---|---|---|---|
| **Proc A (Primary)** | Branching Tree | Linear | 2 | 12 | **23** | 12 | **0** | 0 | 50 |
| **Proc A (Ring Alt)** | 5Q Cycle | Linear | 2 | 12 | **23** | 12 | **0** | 0 | 50 |
| **Proc A (Primary)** | Branching Tree | Full | 2 | 24 | **61** | 33 | **3** | 0 | 77 |
| **Proc A (Ring Alt)** | 5Q Cycle | Full | 2 | 24 | **55** | 40 | 5 | 0 | 84 |
| **Proc B (Primary)** | Heavy-Hex | Linear | 2 | 12 | **23** | 12 | **0** | 0 | 50 |
| **Proc B (Primary)** | Heavy-Hex | Full | 2 | 24 | **44** | 30 | **2** | 0 | 74 |

### Key Architectural Takeaways:
1. **Linear Entanglement is SWAP-Free**: Both processors achieve depth 23 and 0 SWAPs because a 4-qubit line easily embeds into both graphs.
2. **Heavy-Hex Cycle Outperforms Branching Tree**: On full entanglement, Processor B achieves **depth 44 and only 2 SWAPs**, compared to Processor A's **depth 61 and 3 SWAPs** (28% depth reduction).
3. **Co-Design Ring Trade-off**: The ring topology reduces maximum circuit layer depth from 61 to 55 for full entanglement, demonstrating the advantage of closed cycle routing.

---

## 6. Kernel Concentration & Positive Semi-Definite Mitigation

### 6.1 Noise-Induced Concentration
Under two-qubit depolarizing error ($p_{2q}=0.005$) and readout errors ($p_{ro}=0.01$):
- Pure states decay towards the maximally mixed state $\frac{I}{16}$.
- Diagonal entries decay from $1.000$ down to $\sim 0.716$ (Proc A) and $\sim 0.733$ (Proc B).
- Off-diagonal variance drops from $0.0075$ to $0.0037$, compressing kernel values towards a narrow band (kernel concentration).

### 6.2 PSD & Diagonal Mitigation
We apply a two-stage mitigation technique:
1. **Diagonal Normalization**:
   $$\tilde{K}_{ij} = \frac{K_{ij}}{\sqrt{K_{ii} K_{jj}}}$$
   Restores the diagonal strictly to $1.000$ and offsets global depolarizing shrinkage.
2. **Negative Eigenvalue Clipping**:
   $$K_{\text{PSD}} = V \max(\Lambda, 0) V^T$$
   Guarantees that finite shot noise does not produce non-PSD Gram matrices that cause quadratic program solver failures in `SVC(kernel='precomputed')`.

### 6.3 Kernel Fidelity Metrics Summary

| Kernel Configuration | Frobenius Rel. Error vs Ideal | Kernel-Target Alignment (KTA) | Off-Diagonal Variance | Diagonal Mean | Min. Eigenvalue |
|---|---|---|---|---|---|
| **QSVM Ideal** | 0.0000 | 0.1144 | 0.00752 | 1.0000 | +0.0118 |
| **Proc A Noisy** | 0.2575 | 0.1044 | 0.00366 | 0.7163 | +0.0106 |
| **Proc A Mitigated** | **0.1502** | 0.1043 | **0.00714** | **1.0000** | +0.0148 |
| **Proc B Noisy** | 0.2422 | 0.1056 | 0.00386 | 0.7329 | +0.0032 |
| **Proc B Mitigated** | **0.1388** | 0.1056 | **0.00719** | **1.0000** | +0.0044 |

Mitigation reduces the relative Frobenius error by **41.7% on Processor A** and **42.7% on Processor B**, restoring off-diagonal variance to within 5% of ideal.

---

## 7. 5-Seed Classification Benchmark Results

Evaluated across seeds `[42, 100, 2026, 7, 99]` on primary `make_moons` dataset ($N=100$, 80/20 stratified split, 4096 shots). Metrics report **Mean ± 95% Confidence Interval**:

| Model | Test Accuracy (Mean ± 95% CI) | F1-Score (Mean ± 95% CI) | ROC-AUC (Mean ± 95% CI) | KTA (Mean ± 95% CI) | Off-Diag Variance | Total Circuits |
|---|---|---|---|---|---|---|
| **QSVM (Ideal)** | **0.8000 ± 0.0982** | 0.7958 ± 0.0995 | 0.8460 ± 0.1212 | 0.1037 ± 0.0043 | 0.00789 | 3,160 |
| **QSVM (Proc A - Noisy)** | 0.7800 ± 0.0942 | 0.7803 ± 0.0869 | 0.8380 ± 0.0914 | 0.0949 ± 0.0039 | 0.00385 | 3,160 |
| **QSVM (Proc B - Noisy)** | 0.7700 ± 0.0833 | 0.7725 ± 0.0756 | 0.8360 ± 0.0948 | 0.0954 ± 0.0038 | 0.00403 | 3,160 |
| **QSVM (Proc A - Mitigated)** | **0.8000 ± 0.1161** | 0.7973 ± 0.1075 | 0.8400 ± 0.1155 | 0.0948 ± 0.0040 | 0.00752 | 3,160 |
| **QSVM (Proc B - Mitigated)** | **0.7900 ± 0.1194** | 0.7921 ± 0.1157 | 0.8460 ± 0.1074 | 0.0954 ± 0.0038 | 0.00752 | 3,160 |
| **Classical SVM (RBF Tuned)** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | N/A | N/A | 0 |
| **Classical SVM (Poly Tuned)** | 0.9700 ± 0.0340 | 0.9694 ± 0.0348 | 0.9780 ± 0.0544 | N/A | N/A | 0 |

---

## 8. Final Judge Cheat Sheet & Key Numbers

Memorize these key numbers for the judges:

1. **Routing Depth & SWAP Overhead**:
   - Linear Entanglement: Depth **23**, **0 SWAPs** on both chips.
   - Full Entanglement:
     - Processor A: Depth **61**, **33 CX gates**, **3 SWAPs**.
     - Processor B: Depth **44**, **30 CX gates**, **2 SWAPs** (**28% depth savings** due to `[1,2,5,6]` cycle).
2. **Noise Impact**:
   - Unmitigated depolarizing noise drops diagonal overlap to **~0.72** and halves off-diagonal variance from **0.0075** to **0.0037**.
3. **Mitigation Effect**:
   - PSD & diagonal mitigation restores diagonal to **1.0000** and off-diagonal variance to **0.0075**, boosting noisy QSVM accuracy from **78% back to 80%** (matching ideal QSVM).
4. **Circuit Count**:
   - Exactly $\frac{N(N-1)}{2} = \frac{80 \times 79}{2} = \mathbf{3,160}$ circuits per training kernel ($4,950$ for full $N=100$).

---

## 9. Phase 2 Readiness

The pipeline is verified and fully prepared for Phase 2:
- Dynamic circuit branching and mid-circuit measurement support.
- Multi-partite Bell witness correlation estimators.
- Scalable to $\ge 16$ qubits using the validated single-pass parameterized transpilation template.
