"""
src/metrics.py
==============
QFF 2026 – PS-Q2 Quantum Kernel SVM
Track 4 (QML) · Phase 1 · Kernel Matrices, Mitigation & Benchmarking

Provides:
    ── Kernel computation ────────────────────────────────────────────────────
    compute_ideal_kernel(X1, X2, feature_map)
        Exact Gram matrix via Statevector.from_instruction(). < 0.5 s for N=100.

    compute_noisy_hardware_kernel(X1, X2, feature_map, cfg, shots=4096)
        Noisy kernel via density-matrix AerSimulator Tr(ρᵢρⱼ) with
        multinomial shot-noise sampling. Full 100×100 matrix in < 15 s.

    verify_shot_kernel(X1, X2, kernel_qc_transpiled, noisy_sim, shots=4096)
        Shot-based compute-uncompute verification on a subset.

    ── Mitigation ────────────────────────────────────────────────────────────
    mitigate_kernel_matrix(K)
        Diagonal normalisation + negative-eigenvalue clipping.

    ── SVM benchmarking ──────────────────────────────────────────────────────
    evaluate_classifiers(bundle, cfg_a, cfg_b, seeds, verbose)
        Train QSVM (ideal / noisy A & B / mitigated A & B) + classical
        SVM (RBF + Poly GridSearch) across 5 seeds. Returns DataFrame
        of Mean ± Std ± 95% CI for Accuracy, F1, ROC-AUC, Kernel Alignment,
        Off-Diagonal Variance, Total Circuit Executions.

    save_metrics_csv(df, path)
        Persist to results/tables/classification_metrics.csv.

Qiskit 1.x compliance
---------------------
- NEVER `execute`, `Aer.get_backend`, `IBMQ`, `QuantumInstance`, `opflow`.
- Uses `Statevector.from_instruction`, `AerSimulator`, `transpile`.
- `transpile()` called ONCE per processor, OUTSIDE all (i,j) loops.
"""

from __future__ import annotations

import csv
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

# ── scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize

# ── Qiskit 1.x ────────────────────────────────────────────────────────────────
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ── Qiskit Aer ────────────────────────────────────────────────────────────────
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# ── Project ───────────────────────────────────────────────────────────────────
from src.processors import ProcessorConfig, build_noise_model
from src.problem    import (
    DatasetBundle, build_feature_map,
    build_parametrized_kernel_circuit, NUM_QUBITS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results" / "tables"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = _RESULTS_DIR / "classification_metrics.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SHOTS:   int = 4096
DEFAULT_SEED:    int = 42
EVAL_SEEDS: List[int] = [42, 100, 2026, 7, 99]
TRAIN_FRAC: float     = 0.80       # 80/20 split for per-seed evaluation


# ---------------------------------------------------------------------------
# ─── KERNEL COMPUTATION ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def compute_ideal_kernel(
    X1:          np.ndarray,
    X2:          np.ndarray,
    feature_map: QuantumCircuit,
) -> np.ndarray:
    """
    Compute the **exact** quantum kernel Gram matrix via statevector simulation.

    Formula: ``K[i,j] = |⟨φ(X1[i])|φ(X2[j])⟩|²``

    Implementation
    --------------
    - Binds each sample ``x`` into the feature map via ``assign_parameters``.
    - Extracts the full statevector ``|φ(x)⟩`` with
      ``Statevector.from_instruction(bound_circuit)`` — no measurement needed.
    - Assembles the statevector matrix ``S1`` (shape ``n1 × 2^n``) and
      ``S2`` (shape ``n2 × 2^n``), then computes ``K = |S1 @ S2†|²``
      in one NumPy matmul — O(n²·2^n) but constant-overhead.
    - For n=4, ``2^4 = 16`` amplitudes per sample, making this < 0.5 s for
      N=100 (10 000 samples × 16 amplitudes each).

    Parameters
    ----------
    X1          : np.ndarray, shape (n1, num_features).  Left samples.
    X2          : np.ndarray, shape (n2, num_features).  Right samples.
    feature_map : Parameterized ``QuantumCircuit`` (e.g. from
                  ``build_feature_map()``). Must have ``num_features``
                  free parameters.

    Returns
    -------
    K : np.ndarray, shape (n1, n2), dtype float64.  All entries in [0, 1].
    """
    params = list(feature_map.parameters)   # sorted ParameterView

    def _statevectors(X: np.ndarray) -> np.ndarray:
        n_samples  = X.shape[0]
        dim        = 2 ** feature_map.num_qubits
        S          = np.empty((n_samples, dim), dtype=complex)
        for i, x in enumerate(X):
            bound = feature_map.assign_parameters(dict(zip(params, x)))
            S[i]  = Statevector.from_instruction(bound).data
        return S

    S1 = _statevectors(X1)
    S2 = _statevectors(X2)

    # K[i,j] = |<φ(x_i)|φ(x_j)>|² = |(S1 @ S2†)[i,j]|²
    K = np.abs(S1 @ S2.conj().T) ** 2
    return K.astype(np.float64)


# ---------------------------------------------------------------------------

def _simulate_density_matrices(
    X:              np.ndarray,
    feature_map:    QuantumCircuit,
    noisy_sim:      AerSimulator,
    coupling_map,
    basis_gates:    List[str],
    seed:           int = DEFAULT_SEED,
) -> List[np.ndarray]:
    """
    Simulate the noisy density matrix ``ρ(x)`` for each sample ``x`` in X.

    Transpiles the feature map circuit ONCE (structure pass), then binds
    parameters per-sample with ``assign_parameters`` — never calls transpile
    inside the sample loop.

    Returns
    -------
    List of density matrix arrays, each shape ``(2^n, 2^n)``.
    """
    params = list(feature_map.parameters)

    # ── Transpile parameterized feature map circuit ONCE ─────────────────────
    try:
        qc_template = transpile(
            feature_map,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=1,
            seed_transpiler=seed,
        )
        qc_template.save_density_matrix()

        circuits = [
            qc_template.assign_parameters(dict(zip(params, x)))
            for x in X
        ]
        result = noisy_sim.run(circuits).result()
        try:
            return [np.array(result.data(i)["density_matrix"]) for i in range(len(X))]
        except Exception:
            return [np.array(result.data(qc)["density_matrix"]) for qc in circuits]
    except Exception:
        # Fallback to per-sample execution if batch/parameterized save is unsupported
        density_matrices: List[np.ndarray] = []
        for x in X:
            qc_bound = feature_map.assign_parameters(dict(zip(params, x)))
            qc_t = transpile(
                qc_bound,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                optimization_level=1,
                seed_transpiler=seed,
            )
            qc_t.save_density_matrix()
            res = noisy_sim.run(qc_t).result()
            density_matrices.append(np.array(res.data()["density_matrix"]))
        return density_matrices


def _multinomial_shot_noise(
    rng:         np.random.Generator,
    K_exact:     np.ndarray,
    shots:       int,
) -> np.ndarray:
    """
    Add multinomial shot noise to an exact kernel matrix entry.

    For each K[i,j] ∈ [0,1], the measured kernel estimate is::

        K̃[i,j] ~ Binomial(shots, K[i,j]) / shots

    Parameters
    ----------
    rng      : NumPy random Generator (seeded).
    K_exact  : np.ndarray, exact kernel matrix, entries in [0, 1].
    shots    : int, number of measurement shots.

    Returns
    -------
    np.ndarray of same shape as K_exact with shot noise applied.
    """
    K_noisy = rng.binomial(shots, np.clip(K_exact, 0.0, 1.0)) / shots
    return K_noisy.astype(np.float64)


def compute_noisy_hardware_kernel(
    X1:           np.ndarray,
    X2:           np.ndarray,
    feature_map:  Optional[QuantumCircuit] = None,
    cfg:          Any = None,
    shots:        int  = DEFAULT_SHOTS,
    seed:         int  = DEFAULT_SEED,
    mode:         str  = "density_matrix_plus_shot_sampling",
    transpiled_feature_map_or_kernel_qc: Optional[QuantumCircuit] = None,
    noise_model:  Optional[Any] = None,
) -> np.ndarray:
    """
    Compute the **noisy** quantum kernel matrix on a specified processor or NoiseModel.

    Two computation modes
    --------------------
    ``"density_matrix_plus_shot_sampling"`` *(default, fast)*:
        1. Simulate the noisy density matrix ``ρ(xᵢ)`` via
           ``AerSimulator(method='density_matrix', noise_model=…)`` for each
           unique sample — N forward passes.
        2. Compute the exact noisy overlap ``Tr(ρᵢ·ρⱼ)`` for all (i,j) pairs
           in a single NumPy matrix product: ``K[i,j] = |Tr(ρᵢ·ρⱼ)|``.
        3. Add multinomial shot noise: ``K̃[i,j] ~ Binomial(shots, K[i,j])/shots``.
        Result: full 100×100 noisy kernel in < 15 s on any modern laptop.

    ``"shot_based"`` *(exact, slow)*:
        Runs the full compute-uncompute circuit for every (i,j) pair with
        ``shots`` shots. Use only for verification on small subsets.

    Parameters
    ----------
    X1          : np.ndarray (n1, 4).  Left feature matrix.
    X2          : np.ndarray (n2, 4).  Right feature matrix.
    feature_map : Parameterized 4-qubit QuantumCircuit.
    cfg         : ProcessorConfig or NoiseModel.
    shots       : int, number of measurement shots for shot-noise model.
    seed        : int, RNG seed for reproducible shot noise.
    mode        : "density_matrix_plus_shot_sampling" | "shot_based".
    transpiled_feature_map_or_kernel_qc : Alias for feature_map.
    noise_model : Alias / direct pass for noise model.

    Returns
    -------
    K : np.ndarray, shape (n1, n2), float64.
    """
    fm = transpiled_feature_map_or_kernel_qc if transpiled_feature_map_or_kernel_qc is not None else feature_map
    if fm is None:
        raise ValueError("A feature map circuit must be provided.")

    target = noise_model if noise_model is not None else cfg
    if target is None:
        raise ValueError("A ProcessorConfig or NoiseModel must be provided.")

    if isinstance(target, NoiseModel) or (hasattr(target, "basis_gates") and not hasattr(target, "noise_params")):
        nm = target
        basis_gates = getattr(nm, "basis_gates", ["rz", "sx", "x", "cx"])
        directed_cm = None
        coupling_map = None
    else:
        nm = build_noise_model(target)
        basis_gates = target.basis_gates
        directed_cm = target.directed_coupling_map
        coupling_map = target.coupling_map

    if mode == "density_matrix_plus_shot_sampling":

        # ── Density-matrix simulator ───────────────────────────────────────────
        dm_sim = AerSimulator(
            method="density_matrix",
            noise_model=nm,
            basis_gates=basis_gates,
            coupling_map=directed_cm,
        )

        rho1 = _simulate_density_matrices(
            X1, fm, dm_sim, coupling_map,
            basis_gates, seed=seed,
        )
        rho2 = _simulate_density_matrices(
            X2, fm, dm_sim, coupling_map,
            basis_gates, seed=seed,
        )

        # ── Noisy overlap Tr(ρᵢ ρⱼ) ──────────────────────────────────────────
        # Stack density matrices: shape (n_samples, dim, dim)
        R1 = np.stack(rho1, axis=0)   # (n1, 2^n, 2^n)
        R2 = np.stack(rho2, axis=0)   # (n2, 2^n, 2^n)

        # Tr(ρᵢ ρⱼ) = einsum('iab,jba->', R1, R2) in one shot per row:
        K_dm = np.abs(
            np.einsum("iab,jab->ij", R1, R2.conj())
        ).real.astype(np.float64)
        K_dm = np.clip(K_dm, 0.0, 1.0)

        # ── Add multinomial shot noise ─────────────────────────────────────────
        rng = np.random.default_rng(seed)
        K   = _multinomial_shot_noise(rng, K_dm, shots)
        return K

    elif mode == "shot_based":
        # Full shot-based mode — intended for small subsets / verification
        kernel_qc = build_parametrized_kernel_circuit(fm)
        noisy_sim = AerSimulator(
            noise_model=nm,
            basis_gates=basis_gates,
            coupling_map=directed_cm,
        )
        # Transpile the kernel circuit ONCE
        kernel_qc_t = transpile(
            kernel_qc,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=1,
            seed_transpiler=seed,
        )
        return _shot_kernel_matrix(X1, X2, kernel_qc, kernel_qc_t,
                                   noisy_sim, shots, seed)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Choose 'density_matrix_plus_shot_sampling' or 'shot_based'."
        )


# ---------------------------------------------------------------------------

def _shot_kernel_matrix(
    X1:          np.ndarray,
    X2:          np.ndarray,
    kernel_qc:   QuantumCircuit,
    kernel_qc_t: QuantumCircuit,
    noisy_sim:   AerSimulator,
    shots:       int,
    seed:        int,
) -> np.ndarray:
    """
    Compute a noisy kernel matrix via shot-based compute-uncompute circuits.

    Uses the pre-transpiled ``kernel_qc_t`` (transpiled ONCE outside this
    function). Parameters are bound via ``assign_parameters`` per (i,j) pair.

    Returns
    -------
    K : np.ndarray, shape (n1, n2).
    """
    params   = sorted(kernel_qc.parameters, key=lambda p: p.name)
    x_params = [p for p in params if p.name.startswith("x[")]
    y_params = [p for p in params if p.name.startswith("y[")]

    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2), dtype=float)
    all_zeros = "0" * kernel_qc.num_qubits

    for i, xi in enumerate(X1):
        for j, xj in enumerate(X2):
            param_bind = (
                {p: float(v) for p, v in zip(y_params, xi)}
                | {p: float(v) for p, v in zip(x_params, xj)}
            )
            bound = kernel_qc_t.assign_parameters(param_bind)
            result = noisy_sim.run(bound, shots=shots,
                                   seed_simulator=seed).result()
            counts = result.get_counts()
            K[i, j] = counts.get(all_zeros, 0) / shots

    return K


# ---------------------------------------------------------------------------

def verify_shot_kernel(
    X1:          np.ndarray,
    X2:          np.ndarray,
    feature_map: QuantumCircuit,
    cfg:         ProcessorConfig,
    subset_size: int  = 5,
    shots:       int  = DEFAULT_SHOTS,
    seed:        int  = DEFAULT_SEED,
) -> Dict[str, np.ndarray]:
    """
    Verification function: compare density-matrix kernel vs shot-based kernel
    on a small subset to confirm statistical consistency.

    Parameters
    ----------
    X1, X2       : Feature matrices (will be subsampled to ``subset_size``).
    feature_map  : Parameterized feature map.
    cfg          : ProcessorConfig.
    subset_size  : Number of samples from X1 and X2 to compare.
    shots        : Shots for the shot-based kernel.
    seed         : RNG seed.

    Returns
    -------
    dict with keys:
        "dm_kernel"    : density-matrix-based kernel (subset_size × subset_size)
        "shot_kernel"  : shot-based kernel (subset_size × subset_size)
        "max_abs_diff" : max absolute difference between the two.
        "mean_abs_diff": mean absolute difference.
    """
    n1 = min(subset_size, len(X1))
    n2 = min(subset_size, len(X2))
    X1s, X2s = X1[:n1], X2[:n2]

    K_dm   = compute_noisy_hardware_kernel(
        X1s, X2s, feature_map, cfg, shots=shots, seed=seed,
        mode="density_matrix_plus_shot_sampling",
    )
    K_shot = compute_noisy_hardware_kernel(
        X1s, X2s, feature_map, cfg, shots=shots, seed=seed,
        mode="shot_based",
    )

    diff = np.abs(K_dm - K_shot)
    return {
        "dm_kernel":     K_dm,
        "shot_kernel":   K_shot,
        "max_abs_diff":  float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


# ---------------------------------------------------------------------------
# ─── PSD & DIAGONAL MITIGATION ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def mitigate_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """
    Apply PSD and diagonal mitigation to a raw quantum kernel matrix.

    Two-step procedure
    ------------------
    **Step 1 — Diagonal normalisation**::

        K̃[i,j] = K[i,j] / sqrt(K[i,i] · K[j,j])

    This maps the diagonal to 1 and suppresses global scale shifts from noise.

    **Step 2 — Negative eigenvalue clipping**::

        K̃_psd = V · max(Λ, 0) · Vᵀ   (eigendecomposition)

    Clips any negative eigenvalues to 0, guaranteeing the matrix is positive
    semi-definite as required by SVC(kernel='precomputed').

    Parameters
    ----------
    K : np.ndarray, shape (n, n).  Raw (possibly non-PSD) kernel matrix.

    Returns
    -------
    K_mit : np.ndarray, shape (n, n).  Mitigated, PSD kernel matrix.
    """
    K = np.array(K, dtype=float)

    # ── Step 1: diagonal normalisation ────────────────────────────────────────
    diag   = np.sqrt(np.diag(K))
    # Avoid division by zero for any zero diagonal entries
    diag   = np.where(diag > 1e-12, diag, 1.0)
    K_norm = K / np.outer(diag, diag)
    K_norm = np.clip(K_norm, -1.0, 1.0)  # numerical safety

    # ── Step 2: negative eigenvalue clipping ──────────────────────────────────
    # Symmetrise first (numerical noise can break symmetry slightly)
    K_sym = (K_norm + K_norm.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(K_sym)
    eigvals_clipped  = np.maximum(eigvals, 0.0)
    K_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # Re-clip to [0, 1] after reconstruction
    K_psd = np.clip(K_psd, 0.0, 1.0)
    return K_psd


# ---------------------------------------------------------------------------
# ─── KERNEL ALIGNMENT ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the **Kernel-Target Alignment (KTA)** between kernel matrix K
    and binary label vector y.

    KTA(K, yy^T) = <K, yy^T>_F / (||K||_F · ||yy^T||_F)

    where y ∈ {-1, +1} and ``<·,·>_F`` is the Frobenius inner product.

    Parameters
    ----------
    K : np.ndarray (n, n).  Kernel (Gram) matrix.
    y : np.ndarray (n,).    Binary labels in {0, 1} or {-1, 1}.

    Returns
    -------
    float  Kernel-target alignment in [-1, 1].
    """
    y_ = np.where(y == 0, -1, 1).astype(float)
    T  = np.outer(y_, y_)   # ideal kernel
    num   = np.sum(K * T)
    denom = np.linalg.norm(K, "fro") * np.linalg.norm(T, "fro")
    return float(num / denom) if denom > 1e-12 else 0.0


def off_diagonal_variance(K: np.ndarray) -> float:
    """
    Compute the variance of the **off-diagonal** kernel entries.

    High off-diagonal variance ↔ kernel is discriminative (not concentrated).
    Low variance ↔ kernel concentration (all entries similar → poor SVM).

    Parameters
    ----------
    K : np.ndarray (n, n).  Kernel matrix.

    Returns
    -------
    float  Variance of off-diagonal entries.
    """
    n = K.shape[0]
    mask   = ~np.eye(n, dtype=bool)
    return float(np.var(K[mask]))


# ---------------------------------------------------------------------------
# ─── SVM BENCHMARKING ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@dataclass
class ClassifierResult:
    """Per-seed result for one classifier."""
    seed:       int
    clf_name:   str
    accuracy:   float
    f1:         float
    roc_auc:    float
    kta:        float        # Kernel-Target Alignment
    od_var:     float        # Off-Diagonal Kernel Variance
    n_circuits: int          # Total circuit executions = N*(N-1)//2 for train kernel


class ClassifierResultsDict(dict):
    """Dictionary holding aggregated evaluation results with prompt alias support."""
    _ALIASES = {
        "QSVM (Ideal)": "QSVM_Ideal",
        "QSVM (Processor A - Noisy)": "QSVM_ProcA_Noisy",
        "QSVM (Processor B - Noisy)": "QSVM_ProcB_Noisy",
        "QSVM (Processor A - Mitigated)": "QSVM_ProcA_Mitigated",
        "QSVM (Processor B - Mitigated)": "QSVM_ProcB_Mitigated",
        "Classical SVM (RBF - GridSearchCV tuned C and gamma)": "Classical_SVM_RBF",
        "Classical SVM (Polynomial - GridSearchCV tuned degree, C, gamma)": "Classical_SVM_Poly",
    }

    def __getitem__(self, key):
        canonical = self._ALIASES.get(key, key)
        return super().__getitem__(canonical)

    def get(self, key, default=None):
        canonical = self._ALIASES.get(key, key)
        return super().get(canonical, default)

    def __contains__(self, key):
        canonical = self._ALIASES.get(key, key)
        return super().__contains__(canonical)


def _train_qsvm(
    K_train:  np.ndarray,
    y_train:  np.ndarray,
    K_test:   np.ndarray,
    y_test:   np.ndarray,
    clf_name: str,
    seed:     int,
    K_full:   Optional[np.ndarray] = None,
    y_full:   Optional[np.ndarray] = None,
) -> ClassifierResult:
    """
    Train and evaluate a precomputed-kernel SVM, compute all metrics.

    Parameters
    ----------
    K_train  : (n_train, n_train) precomputed kernel.
    y_train  : (n_train,) binary labels.
    K_test   : (n_test, n_train) test-vs-train kernel block.
    y_test   : (n_test,) binary labels.
    clf_name : Label string for this classifier.
    seed     : Random seed (for SVC reproducibility).
    K_full   : Optional full (n_train+n_test, n_train+n_test) kernel for KTA.
    y_full   : Optional full label vector for KTA.

    Returns
    -------
    ClassifierResult
    """
    svc = SVC(kernel="precomputed", probability=True, random_state=seed)
    svc.fit(K_train, y_train)

    y_pred  = svc.predict(K_test)
    y_prob  = svc.predict_proba(K_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    # KTA on training kernel
    kta    = kernel_alignment(K_train, y_train)
    od_var = off_diagonal_variance(K_train)

    n_tr   = len(y_train)
    n_circ = n_tr * (n_tr - 1) // 2     # unique (i,j) pairs for train kernel

    return ClassifierResult(
        seed=seed, clf_name=clf_name,
        accuracy=acc, f1=f1, roc_auc=roc_auc,
        kta=kta, od_var=od_var, n_circuits=n_circ,
    )


def _train_classical_svm(
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    clf_name: str,
    seed:     int,
    kernel:   str = "rbf",
) -> ClassifierResult:
    """
    Train and evaluate a classical SVM with GridSearchCV.

    Kernels:
    - ``"rbf"``:  searches C ∈ {0.1,1,10,100}, gamma ∈ {'scale','auto',0.01,0.1,1}.
    - ``"poly"``: searches C ∈ {0.1,1,10}, degree ∈ {2,3,4},
                  gamma ∈ {'scale','auto'}, coef0 ∈ {0,1}.
    """
    if kernel == "rbf":
        param_grid = {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        }
    else:   # poly
        param_grid = {
            "C":      [0.1, 1, 10],
            "degree": [2, 3, 4],
            "gamma":  ["scale", "auto"],
            "coef0":  [0.0, 1.0],
        }

    base_svc = SVC(kernel=kernel, probability=True, random_state=seed)
    cv_inner = StratifiedShuffleSplit(n_splits=3, test_size=0.2,
                                      random_state=seed)
    gscv = GridSearchCV(base_svc, param_grid, cv=cv_inner,
                        scoring="accuracy", n_jobs=-1)
    gscv.fit(X_train, y_train)

    best = gscv.best_estimator_
    y_pred  = best.predict(X_test)
    y_prob  = best.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    return ClassifierResult(
        seed=seed, clf_name=clf_name,
        accuracy=acc, f1=f1, roc_auc=roc_auc,
        kta=float("nan"), od_var=float("nan"), n_circuits=0,
    )


def _aggregate_results(
    results: List[ClassifierResult],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-seed ClassifierResults into mean ± std ± 95% CI per metric.

    Returns
    -------
    dict  clf_name → {metric: value, …} with _mean / _std / _ci95 suffixes.
    """
    from collections import defaultdict

    by_clf: Dict[str, List[ClassifierResult]] = defaultdict(list)
    for r in results:
        by_clf[r.clf_name].append(r)

    agg: Dict[str, Dict[str, float]] = ClassifierResultsDict()
    for clf_name, rs in by_clf.items():
        metrics_vals: Dict[str, List[float]] = {
            "accuracy":   [r.accuracy   for r in rs],
            "f1":         [r.f1         for r in rs],
            "roc_auc":    [r.roc_auc    for r in rs],
            "kta":        [r.kta        for r in rs],
            "od_var":     [r.od_var     for r in rs],
            "n_circuits": [float(r.n_circuits) for r in rs],
        }
        row: Dict[str, float] = {"clf_name": clf_name}
        for metric, vals in metrics_vals.items():
            arr  = np.array(vals, dtype=float)
            mean = float(np.nanmean(arr))
            std  = float(np.nanstd(arr, ddof=min(1, len(arr) - 1)))
            # 95% CI via t-distribution (two-sided, df = n-1)
            n = len(arr)
            if n > 1 and not np.all(np.isnan(arr)):
                se   = std / np.sqrt(n)
                t95  = float(sp_stats.t.ppf(0.975, df=n - 1))
                ci95 = float(t95 * se)
            else:
                ci95 = float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"]  = std
            row[f"{metric}_ci95"] = ci95
        agg[clf_name] = row

    return agg


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_classifiers(
    bundle:   DatasetBundle,
    cfg_a:    ProcessorConfig,
    cfg_b:    ProcessorConfig,
    seeds:    Sequence[int] = EVAL_SEEDS,
    shots:    int           = DEFAULT_SHOTS,
    verbose:  bool          = True,
    reps:     int           = 2,
    entanglement: str       = "full",
) -> Dict[str, Dict[str, float]]:
    """
    Train and benchmark **7 classifiers** across ``len(seeds)`` random splits.

    Classifiers evaluated
    ---------------------
    a) QSVM – Ideal (exact statevector kernel)
    b) QSVM – Processor A Noisy (density-matrix + shot noise)
    c) QSVM – Processor B Noisy
    d) QSVM – Processor A Mitigated (diagonal norm + eigenvalue clip)
    e) QSVM – Processor B Mitigated
    f) Classical SVM – RBF (GridSearchCV over C, gamma)
    g) Classical SVM – Polynomial (GridSearchCV over degree, C, gamma, coef0)

    Per-seed metrics collected
    --------------------------
    - Accuracy, F1-Score, ROC-AUC (test set)
    - Kernel-Target Alignment (KTA) on training kernel
    - Off-Diagonal Kernel Variance (concentration measure)
    - Total Circuit Executions = N_train*(N_train-1)//2

    Statistical output
    ------------------
    Mean, Standard Deviation, and 95% Confidence Interval for every metric,
    computed over all seeds via t-distribution (df = len(seeds) - 1).

    Parameters
    ----------
    bundle       : DatasetBundle from ``src.problem``.
    cfg_a        : ProcessorConfig for Processor A.
    cfg_b        : ProcessorConfig for Processor B.
    seeds        : List of random seeds for train/test splits (default: 5 seeds).
    shots        : Measurement shots for noisy kernels (default 4096).
    verbose      : Print per-seed progress.
    reps         : ZZFeatureMap reps (default 2).
    entanglement : ZZFeatureMap entanglement (default "full").

    Returns
    -------
    Dict[clf_name, Dict[metric_key, value]]
        Aggregated results. Each value key has _mean / _std / _ci95 suffixes.
    """
    fm     = build_feature_map(num_qubits=NUM_QUBITS, reps=reps,
                               entanglement=entanglement)
    X_all  = np.vstack([bundle.X_train, bundle.X_test])
    y_all  = np.concatenate([bundle.y_train, bundle.y_test])

    all_results: List[ClassifierResult] = []

    for seed in seeds:
        if verbose:
            print(f"\n  ── Seed {seed} ──────────────────────────────────────")

        # ── Stratified 80/20 split ─────────────────────────────────────────────
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - TRAIN_FRAC,
                                     random_state=seed)
        tr_idx, te_idx = next(sss.split(X_all, y_all))
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # ── Ideal kernel ───────────────────────────────────────────────────────
        if verbose: print("    [1/7] QSVM Ideal …", end=" ", flush=True)
        t0 = time.perf_counter()
        K_ideal_tr = compute_ideal_kernel(X_tr, X_tr, fm)
        K_ideal_te = compute_ideal_kernel(X_te, X_tr, fm)
        if verbose: print(f"{time.perf_counter()-t0:.1f}s")
        r = _train_qsvm(K_ideal_tr, y_tr, K_ideal_te, y_te,
                        "QSVM_Ideal", seed)
        all_results.append(r)

        # ── Noisy Processor A ──────────────────────────────────────────────────
        if verbose: print("    [2/7] QSVM Processor A Noisy …", end=" ", flush=True)
        t0 = time.perf_counter()
        K_a_tr = compute_noisy_hardware_kernel(X_tr, X_tr, fm, cfg_a, shots, seed)
        K_a_te = compute_noisy_hardware_kernel(X_te, X_tr, fm, cfg_a, shots, seed)
        if verbose: print(f"{time.perf_counter()-t0:.1f}s")
        all_results.append(_train_qsvm(K_a_tr, y_tr, K_a_te, y_te,
                                       "QSVM_ProcA_Noisy", seed))

        # ── Noisy Processor B ──────────────────────────────────────────────────
        if verbose: print("    [3/7] QSVM Processor B Noisy …", end=" ", flush=True)
        t0 = time.perf_counter()
        K_b_tr = compute_noisy_hardware_kernel(X_tr, X_tr, fm, cfg_b, shots, seed)
        K_b_te = compute_noisy_hardware_kernel(X_te, X_tr, fm, cfg_b, shots, seed)
        if verbose: print(f"{time.perf_counter()-t0:.1f}s")
        all_results.append(_train_qsvm(K_b_tr, y_tr, K_b_te, y_te,
                                       "QSVM_ProcB_Noisy", seed))

        # ── Mitigated Processor A ──────────────────────────────────────────────
        if verbose: print("    [4/7] QSVM Processor A Mitigated …", end=" ", flush=True)
        K_a_tr_mit = mitigate_kernel_matrix(K_a_tr)
        # For the test block, we use a row-wise normalisation to be consistent
        diag_a     = np.sqrt(np.diag(K_a_tr))
        diag_a     = np.where(diag_a > 1e-12, diag_a, 1.0)
        diag_te_a  = np.sqrt(np.diag(
            compute_noisy_hardware_kernel(X_te, X_te, fm, cfg_a, shots, seed)
        ))
        diag_te_a  = np.where(diag_te_a > 1e-12, diag_te_a, 1.0)
        K_a_te_mit = K_a_te / np.outer(diag_te_a, diag_a)
        K_a_te_mit = np.clip(K_a_te_mit, 0.0, 1.0)
        if verbose: print("done")
        all_results.append(_train_qsvm(K_a_tr_mit, y_tr, K_a_te_mit, y_te,
                                       "QSVM_ProcA_Mitigated", seed))

        # ── Mitigated Processor B ──────────────────────────────────────────────
        if verbose: print("    [5/7] QSVM Processor B Mitigated …", end=" ", flush=True)
        K_b_tr_mit = mitigate_kernel_matrix(K_b_tr)
        diag_b     = np.sqrt(np.diag(K_b_tr))
        diag_b     = np.where(diag_b > 1e-12, diag_b, 1.0)
        diag_te_b  = np.sqrt(np.diag(
            compute_noisy_hardware_kernel(X_te, X_te, fm, cfg_b, shots, seed)
        ))
        diag_te_b  = np.where(diag_te_b > 1e-12, diag_te_b, 1.0)
        K_b_te_mit = K_b_te / np.outer(diag_te_b, diag_b)
        K_b_te_mit = np.clip(K_b_te_mit, 0.0, 1.0)
        if verbose: print("done")
        all_results.append(_train_qsvm(K_b_tr_mit, y_tr, K_b_te_mit, y_te,
                                       "QSVM_ProcB_Mitigated", seed))

        # ── Classical SVM – RBF ───────────────────────────────────────────────
        if verbose: print("    [6/7] Classical SVM RBF …", end=" ", flush=True)
        t0 = time.perf_counter()
        all_results.append(_train_classical_svm(
            X_tr, y_tr, X_te, y_te, "Classical_SVM_RBF", seed, kernel="rbf"
        ))
        if verbose: print(f"{time.perf_counter()-t0:.1f}s")

        # ── Classical SVM – Poly ──────────────────────────────────────────────
        if verbose: print("    [7/7] Classical SVM Poly …", end=" ", flush=True)
        t0 = time.perf_counter()
        all_results.append(_train_classical_svm(
            X_tr, y_tr, X_te, y_te, "Classical_SVM_Poly", seed, kernel="poly"
        ))
        if verbose: print(f"{time.perf_counter()-t0:.1f}s")

    # ── Aggregate across seeds ─────────────────────────────────────────────────
    aggregated = _aggregate_results(all_results)

    if verbose:
        print("\n" + "=" * 65)
        print(f"  Results (mean ± 95% CI) over {len(seeds)} seeds:")
        print("=" * 65)
        for clf, row in aggregated.items():
            print(f"  {clf:<30}  "
                  f"acc={row['accuracy_mean']:.4f}±{row['accuracy_ci95']:.4f}  "
                  f"f1={row['f1_mean']:.4f}  "
                  f"auc={row['roc_auc_mean']:.4f}  "
                  f"kta={row.get('kta_mean', float('nan')):.4f}")

    return aggregated


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def save_metrics_csv(
    results: Dict[str, Dict[str, float]],
    path:    pathlib.Path = METRICS_CSV,
) -> pathlib.Path:
    """
    Persist aggregated classifier results to a CSV file.

    Columns: clf_name, accuracy_mean, accuracy_std, accuracy_ci95,
             f1_mean, f1_std, f1_ci95, roc_auc_mean, …, kta_mean, …,
             od_var_mean, …, n_circuits_mean.

    Parameters
    ----------
    results : Dict returned by :func:`evaluate_classifiers`.
    path    : Output path (default ``results/tables/classification_metrics.csv``).

    Returns
    -------
    pathlib.Path  written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "f1", "roc_auc", "kta", "od_var", "n_circuits"]
    suffixes = ["mean", "std", "ci95"]

    fieldnames = ["clf_name"]
    for m in metrics:
        for s in suffixes:
            fieldnames.append(f"{m}_{s}")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for clf_name, row in results.items():
            out = {"clf_name": clf_name}
            for m in metrics:
                for s in suffixes:
                    key = f"{m}_{s}"
                    out[key] = f"{row.get(key, float('nan')):.6f}"
            writer.writerow(out)

    print(f"  ✓ Metrics CSV saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":   # pragma: no cover
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 65)
    print("  src/metrics.py  —  kernel & SVM self-test  (fast subset)")
    print("=" * 65)

    from src.problem    import make_dataset_moons
    from src.processors import load_processor_A, load_processor_B

    bundle = make_dataset_moons(n_samples=20, random_state=42)
    cfg_a  = load_processor_A()
    cfg_b  = load_processor_B()
    fm     = build_feature_map(num_qubits=4, reps=2, entanglement="full")

    X_tr, X_te = bundle.X_train[:10], bundle.X_test[:5]
    y_tr, y_te = bundle.y_train[:10], bundle.y_test[:5]

    # Ideal kernel
    print("\n▶ Ideal kernel (10×10) …", end=" ", flush=True)
    t0 = time.perf_counter()
    K_id = compute_ideal_kernel(X_tr, X_tr, fm)
    print(f"{time.perf_counter()-t0:.2f}s")
    assert K_id.shape == (10, 10)
    assert np.allclose(np.diag(K_id), 1.0, atol=1e-6), "Diagonal should be 1"
    print(f"  shape={K_id.shape}  diag≈1  ✓")

    # Noisy kernel – Processor A
    print("▶ Noisy kernel Proc A (10×10) …", end=" ", flush=True)
    t0 = time.perf_counter()
    K_a = compute_noisy_hardware_kernel(X_tr, X_tr, fm, cfg_a, shots=256, seed=42)
    print(f"{time.perf_counter()-t0:.2f}s")
    assert K_a.shape == (10, 10)
    print(f"  shape={K_a.shape}  ✓")

    # Mitigation
    print("▶ PSD mitigation …", end=" ")
    K_mit = mitigate_kernel_matrix(K_a)
    eigvals = np.linalg.eigvalsh(K_mit)
    assert eigvals.min() >= -1e-9, f"Non-PSD after mitigation: {eigvals.min()}"
    print(f"  min_eigenvalue={eigvals.min():.2e}  ✓")

    # KTA
    kta = kernel_alignment(K_id[:len(y_tr),:len(y_tr)], y_tr)
    print(f"▶ KTA (ideal, train) = {kta:.4f}  ✓")

    # Verify shot kernel (tiny subset)
    print("▶ Verification: dm vs shot kernel (3×3) …", end=" ", flush=True)
    t0 = time.perf_counter()
    vfy = verify_shot_kernel(X_tr[:3], X_tr[:3], fm, cfg_a,
                             subset_size=3, shots=512, seed=42)
    print(f"{time.perf_counter()-t0:.2f}s")
    print(f"  max_abs_diff={vfy['max_abs_diff']:.4f}  "
          f"mean_abs_diff={vfy['mean_abs_diff']:.4f}  ✓")

    # Mini evaluate_classifiers (2 seeds, tiny data)
    print("\n▶ evaluate_classifiers (2 seeds, 20 samples) …")
    agg = evaluate_classifiers(bundle, cfg_a, cfg_b,
                                seeds=[42, 7], shots=256, verbose=True)
    assert len(agg) == 7, f"Expected 7 classifiers, got {len(agg)}"
    csv_path = save_metrics_csv(agg)
    assert csv_path.exists()

    print("\n" + "=" * 65)
    print("  All assertions passed.")
    print("=" * 65 + "\n")
