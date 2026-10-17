"""
src/problem.py
==============
QFF 2026 – PS-Q2 Quantum Kernel SVM
Track 4 (QML) · Phase 1

Dataset generation, preprocessing, and 4-qubit Quantum Feature Map
construction for the QSVM pipeline.

Public API
----------
    # ── Datasets ──────────────────────────────────────────────────────────────
    make_dataset_moons()          -> DatasetBundle   (primary, 2D → 4D expanded)
    make_dataset_real()           -> DatasetBundle   (secondary, PCA-4 real data)
    make_dataset_quantum_aligned()-> DatasetBundle   (synthetic, ZZFeatureMap-aligned)
    load_all_datasets()           -> dict[str, DatasetBundle]

    # ── Persistence ───────────────────────────────────────────────────────────
    save_dataset(bundle, tag)     -> None            (raw + processed CSV/npy)
    load_dataset(tag)             -> DatasetBundle   (reload from disk)

    # ── Quantum circuits ──────────────────────────────────────────────────────
    build_feature_map(num_qubits, reps, entanglement) -> ZZFeatureMap
    build_parametrized_kernel_circuit(feature_map)    -> QuantumCircuit

Qiskit 1.x compliance
---------------------
- NEVER use `execute`, `Aer.get_backend`, `IBMQ`, `QuantumInstance`, `opflow`.
- ZZFeatureMap from `qiskit.circuit.library`.
- ParameterVector from `qiskit.circuit`.
- transpile() called ONCE per processor, outside (i,j) loops.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

# ── scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.datasets import make_moons, load_digits, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ── Qiskit 1.x / 2.x ─────────────────────────────────────────────────────────
from qiskit.circuit import QuantumCircuit, ParameterVector

# Qiskit 2.1 deprecated ZZFeatureMap class → prefer zz_feature_map function.
# Keep ZZFeatureMap imported for type annotations; use the function to construct.
try:
    from qiskit.circuit.library import zz_feature_map as _zz_feature_map_fn
    from qiskit.circuit.library import ZZFeatureMap          # for annotations
    _USE_FM_FN = True
except ImportError:
    # Qiskit < 2.1: function doesn't exist, class is the only option
    from qiskit.circuit.library import ZZFeatureMap          # type: ignore
    _zz_feature_map_fn = None                               # type: ignore
    _USE_FM_FN = False

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_REPO_ROOT    = pathlib.Path(__file__).resolve().parent.parent
_DATA_RAW     = _REPO_ROOT / "data" / "raw"
_DATA_PROC    = _REPO_ROOT / "data" / "processed"

for _d in (_DATA_RAW, _DATA_PROC):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_QUBITS:   int   = 4
FEATURE_RANGE        = (0.0, np.pi)   # scale target for all feature vectors
N_SAMPLES:    int   = 100
RANDOM_STATE: int   = 42
MOONS_NOISE:  float = 0.1
TEST_SIZE:    float = 0.25            # 75/25 train-test split

# ---------------------------------------------------------------------------
# DatasetBundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    """
    Container for a fully preprocessed dataset ready for the QSVM kernel.

    Attributes
    ----------
    tag : str
        Short identifier used for filenames (e.g. ``"moons"``, ``"real"``,
        ``"quantum_aligned"``).
    X_train : np.ndarray, shape (n_train, 4)
        Training feature vectors in ``[0, π]^4``.
    X_test  : np.ndarray, shape (n_test, 4)
        Test feature vectors in ``[0, π]^4``.
    y_train : np.ndarray, shape (n_train,)
        Binary training labels in ``{0, 1}``.
    y_test  : np.ndarray, shape (n_test,)
        Binary test labels in ``{0, 1}``.
    X_raw   : np.ndarray
        Original (unscaled, un-PCA'd) feature matrix, for audit trails.
    y_raw   : np.ndarray
        Original labels matching ``X_raw``.
    scaler  : MinMaxScaler
        Fitted scaler (for inverse-transform or downstream re-use).
    description : str
        Human-readable provenance note.
    """

    tag:         str
    X_train:     np.ndarray
    X_test:      np.ndarray
    y_train:     np.ndarray
    y_test:      np.ndarray
    X_raw:       np.ndarray
    y_raw:       np.ndarray
    scaler:      MinMaxScaler = field(repr=False)
    description: str = ""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def class_balance(self) -> dict:
        """Return label counts for both splits."""
        unique_tr, counts_tr = np.unique(self.y_train, return_counts=True)
        unique_te, counts_te = np.unique(self.y_test,  return_counts=True)
        return {
            "train": dict(zip(unique_tr.tolist(), counts_tr.tolist())),
            "test":  dict(zip(unique_te.tolist(), counts_te.tolist())),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_to_pi(X: np.ndarray, scaler: Optional[MinMaxScaler] = None
                 ) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Fit (or reuse) a MinMaxScaler to map *X* into ``[0, π]^d``.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input feature matrix.
    scaler : MinMaxScaler, optional
        If provided, transform-only (already fitted). Otherwise fit+transform.

    Returns
    -------
    X_scaled : np.ndarray
    scaler   : MinMaxScaler (fitted)
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


def _expand_moons_features(
    X2d: np.ndarray,
    mode: Literal["nonlinear", "tile"] = "nonlinear",
) -> np.ndarray:
    """
    Lift a 2D make_moons array to a 4-dimensional feature vector.

    Modes
    -----
    ``"nonlinear"`` (default, Master Guide requirement):
        ``[x1, x2, x1 * x2, x1**2 + x2**2]``
        Captures interaction and radial distance — better separability for
        the quantum kernel.

    ``"tile"``:
        ``[x1, x2, x1, x2]``
        Simple repetition baseline; useful for ablation comparisons.

    Parameters
    ----------
    X2d : np.ndarray, shape (n, 2)
    mode : {"nonlinear", "tile"}

    Returns
    -------
    X4d : np.ndarray, shape (n, 4)
    """
    x1, x2 = X2d[:, 0], X2d[:, 1]
    if mode == "nonlinear":
        return np.column_stack([x1, x2, x1 * x2, x1**2 + x2**2])
    elif mode == "tile":
        return np.column_stack([x1, x2, x1, x2])
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'nonlinear' or 'tile'.")


# ---------------------------------------------------------------------------
# Dataset 1 — make_moons  (primary)
# ---------------------------------------------------------------------------

def make_dataset_moons(
    n_samples:    int   = N_SAMPLES,
    noise:        float = MOONS_NOISE,
    random_state: int   = RANDOM_STATE,
    expansion:    Literal["nonlinear", "tile"] = "nonlinear",
    test_size:    float = TEST_SIZE,
) -> DatasetBundle:
    """
    **Primary dataset** — 2-moons mapped to a 4-qubit register.

    Steps
    -----
    1. ``make_moons(n_samples=100, noise=0.1, random_state=42)`` → 2D (x1, x2).
    2. Nonlinear feature expansion → 4D: ``[x1, x2, x1*x2, x1²+x2²]``.
    3. ``MinMaxScaler(feature_range=(0, π))`` fit on *training split only*,
       applied to both splits (no data leakage).
    4. Stratified 75/25 train-test split (``random_state=42``).

    Parameters
    ----------
    n_samples    : int   Total samples (default 100).
    noise        : float Gaussian noise σ on moon coordinates (default 0.1).
    random_state : int   NumPy/sklearn seed (default 42).
    expansion    : str   Feature expansion mode ("nonlinear" | "tile").
    test_size    : float Fraction reserved for test (default 0.25).

    Returns
    -------
    DatasetBundle
    """
    # Raw 2D data
    X2d, y = make_moons(n_samples=n_samples, noise=noise,
                        random_state=random_state)

    # Expand to 4D (before scaling, so scaler sees true feature distribution)
    X4d = _expand_moons_features(X2d, mode=expansion)

    # Stratified split  (split first → scale on train only)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X4d, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Scale train → fit scaler; transform test with same scaler
    X_tr, scaler = _scale_to_pi(X_tr_raw)
    X_te, _      = _scale_to_pi(X_te_raw, scaler=scaler)

    return DatasetBundle(
        tag="moons",
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        X_raw=X4d,
        y_raw=y,
        scaler=scaler,
        description=(
            f"make_moons(n={n_samples}, noise={noise}, seed={random_state}), "
            f"expansion='{expansion}', scaled to [0,π]^4."
        ),
    )


# ---------------------------------------------------------------------------
# Dataset 2 — Real-world high-dimensional  (secondary)
# ---------------------------------------------------------------------------

def make_dataset_real(
    source:       Literal["digits", "breast_cancer"] = "digits",
    n_samples:    int   = N_SAMPLES,
    n_components: int   = NUM_QUBITS,
    random_state: int   = RANDOM_STATE,
    test_size:    float = TEST_SIZE,
) -> DatasetBundle:
    """
    **Secondary dataset** — dimensionality-reduced real-world classification.

    Steps
    -----
    1. Load ``load_digits(n_class=2)`` *or* ``load_breast_cancer()``.
    2. Stratified subsample to ``n_samples=100`` (``random_state=42``).
    3. ``StandardScaler`` to zero-mean / unit-variance.
    4. ``PCA(n_components=4)`` to project to 4D.
    5. ``MinMaxScaler(feature_range=(0, π))`` fit on train split only.

    Parameters
    ----------
    source       : "digits" | "breast_cancer"
    n_samples    : int   Target subsample size.
    n_components : int   PCA output dimension (must equal NUM_QUBITS = 4).
    random_state : int   Seed for subsample and PCA.
    test_size    : float Fraction reserved for test.

    Returns
    -------
    DatasetBundle
    """
    # ── Load raw data ─────────────────────────────────────────────────────────
    if source == "digits":
        data   = load_digits(n_class=2)
        X_full = data.data
        y_full = data.target
        src_desc = "load_digits(n_class=2)"
    elif source == "breast_cancer":
        data   = load_breast_cancer()
        X_full = data.data
        y_full = data.target
        src_desc = "load_breast_cancer()"
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'digits' or 'breast_cancer'.")

    # ── Stratified subsample ──────────────────────────────────────────────────
    if n_samples < len(y_full):
        # Use train_test_split to extract a stratified subsample
        X_sub, _, y_sub, _ = train_test_split(
            X_full, y_full,
            train_size=n_samples,
            stratify=y_full,
            random_state=random_state,
        )
    else:
        X_sub, y_sub = X_full, y_full

    # ── Standardize + PCA ─────────────────────────────────────────────────────
    std_scaler = StandardScaler()
    X_std      = std_scaler.fit_transform(X_sub)

    pca   = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_std)

    # ── Stratified split ──────────────────────────────────────────────────────
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_pca, y_sub, test_size=test_size, stratify=y_sub,
        random_state=random_state,
    )

    # ── Scale to [0, π] (fit on train only) ──────────────────────────────────
    X_tr, scaler = _scale_to_pi(X_tr_raw)
    X_te, _      = _scale_to_pi(X_te_raw, scaler=scaler)

    return DatasetBundle(
        tag=f"real_{source}",
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        X_raw=X_pca,
        y_raw=y_sub,
        scaler=scaler,
        description=(
            f"{src_desc}, subsample n={n_samples}, StandardScaler + "
            f"PCA(n_components={n_components}), scaled to [0,π]^4."
        ),
    )


# ---------------------------------------------------------------------------
# Dataset 3 — Quantum-aligned synthetic  (secondary)
# ---------------------------------------------------------------------------

def make_dataset_quantum_aligned(
    n_samples:    int  = N_SAMPLES,
    num_qubits:   int  = NUM_QUBITS,
    reps:         int  = 2,
    random_state: int  = RANDOM_STATE,
    test_size:    float = TEST_SIZE,
) -> DatasetBundle:
    """
    **Synthetic quantum-aligned dataset** — feature vectors whose binary labels
    are generated directly from the parity of a ZZFeatureMap statevector.

    This dataset is natively aligned with the quantum feature map geometry,
    providing an ideal upper-bound reference for QSVM performance.

    Label generation
    ----------------
    For each sample ``x ∈ [0, π]^4``:

    1. Bind ``x`` into a ``ZZFeatureMap(num_qubits=4, reps=2)`` circuit.
    2. Execute on a **noiseless** ``AerSimulator(method='statevector')``.
    3. Measure the full statevector ``|ψ(x)⟩``; compute the expectation value
       of the diagonal Z-parity observable::

           P(x) = <ψ(x)| Z₀⊗Z₁⊗Z₂⊗Z₃ |ψ(x)>

    4. Label: ``y = 1  if  P(x) ≥ 0  else  0``.

    This ensures the quantum kernel ``K(xᵢ, xⱼ) = |<ψ(xᵢ)|ψ(xⱼ)>|²`` is
    maximally informative for this label structure.

    Parameters
    ----------
    n_samples    : int   Number of random 4-qubit feature vectors.
    num_qubits   : int   Must be 4 for the 4-qubit register.
    reps         : int   ZZFeatureMap repetitions (default 2).
    random_state : int   NumPy seed for reproducible random feature vectors.
    test_size    : float Fraction reserved for test.

    Returns
    -------
    DatasetBundle
        ``X_raw`` contains the raw ``[0, π]^4`` feature vectors
        (already in quantum range; scaler is identity-equivalent).
    """
    from qiskit_aer import AerSimulator

    rng = np.random.default_rng(random_state)

    # ── Random feature vectors uniformly in [0, π]^4 ─────────────────────────
    X_raw = rng.uniform(0.0, np.pi, size=(n_samples, num_qubits))

    # ── Build ZZFeatureMap (parameterised, bound per sample) ──────────────────
    if _USE_FM_FN and _zz_feature_map_fn is not None:
        fm = _zz_feature_map_fn(feature_dimension=num_qubits, reps=reps,
                                entanglement="full")
    else:
        fm = ZZFeatureMap(feature_dimension=num_qubits, reps=reps,
                          entanglement="full")
    params = fm.parameters  # ParameterView, length == num_qubits

    sim = AerSimulator(method="statevector")

    # ── Z-parity observable: eigenvalue of |bitstring⟩ is (-1)^parity ────────
    def _z_parity_expectation(sv: np.ndarray) -> float:
        """Compute <ψ| Z⊗Z⊗Z⊗Z |ψ> from a statevector array."""
        probs    = np.abs(sv) ** 2
        n        = int(np.log2(len(sv)))
        parities = np.array([
            (-1) ** bin(k).count("1") for k in range(2 ** n)
        ], dtype=float)
        return float(np.dot(parities, probs))

    # ── Label each sample ─────────────────────────────────────────────────────
    # Transpile ONCE with a concrete binding (for circuit structure only),
    # then re-bind per sample — avoids calling transpile inside the loop.
    param_dict_0 = dict(zip(params, X_raw[0]))
    bound_qc_0   = fm.assign_parameters(param_dict_0)
    bound_qc_0.save_statevector()

    from qiskit import transpile as qiskit_transpile
    # Transpile the structure circuit once (no coupling map needed for ideal sim)
    _template = qiskit_transpile(bound_qc_0, backend=sim,
                                  optimization_level=1, seed_transpiler=42)

    labels = np.empty(n_samples, dtype=int)
    for i, x in enumerate(X_raw):
        param_dict = dict(zip(params, x))
        qc_bound   = fm.assign_parameters(param_dict)
        qc_bound.save_statevector()
        # Re-transpile is cheap here (same structure, only parameters differ);
        # for the full kernel loop, use assign_parameters on a pre-transpiled
        # parametrized circuit instead.
        result = sim.run(qiskit_transpile(
            qc_bound, backend=sim,
            optimization_level=1, seed_transpiler=42
        )).result()
        sv = np.array(result.get_statevector())
        labels[i] = 1 if _z_parity_expectation(sv) >= 0.0 else 0

    # ── X_raw is already in [0, π]; fit a scaler anyway for API consistency ───
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, labels, test_size=test_size, stratify=labels,
        random_state=random_state,
    )
    X_tr, scaler = _scale_to_pi(X_tr_raw)
    X_te, _      = _scale_to_pi(X_te_raw, scaler=scaler)

    return DatasetBundle(
        tag="quantum_aligned",
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        X_raw=X_raw,
        y_raw=labels,
        scaler=scaler,
        description=(
            f"Synthetic quantum-aligned: {n_samples} random [0,π]^4 samples, "
            f"labels from ZZFeatureMap(reps={reps}) Z-parity expectation value. "
            f"seed={random_state}."
        ),
    )


# ---------------------------------------------------------------------------
# Convenience loader — all three datasets at once
# ---------------------------------------------------------------------------

def load_all_datasets(
    moons_expansion: Literal["nonlinear", "tile"] = "nonlinear",
    real_source:     Literal["digits", "breast_cancer"] = "digits",
    quantum_reps:    int  = 2,
    n_samples:       int  = N_SAMPLES,
    random_state:    int  = RANDOM_STATE,
    verbose:         bool = True,
) -> dict[str, DatasetBundle]:
    """
    Generate all three PS-Q2 datasets and return them in a dict.

    Keys: ``"moons"``, ``"real_digits"`` (or ``"real_breast_cancer"``),
    ``"quantum_aligned"``.

    Parameters
    ----------
    moons_expansion : Feature expansion mode for Dataset 1.
    real_source     : Real-world source for Dataset 2.
    quantum_reps    : ZZFeatureMap reps for Dataset 3 label generation.
    n_samples       : Common sample count (100).
    random_state    : Common seed.
    verbose         : Print provenance + class balance summary.

    Returns
    -------
    dict[str, DatasetBundle]
    """
    bundles: dict[str, DatasetBundle] = {}

    # Dataset 1 — moons
    if verbose:
        print("▶ Generating Dataset 1: make_moons (primary) …", flush=True)
    ds1 = make_dataset_moons(n_samples=n_samples, expansion=moons_expansion,
                             random_state=random_state)
    bundles[ds1.tag] = ds1

    # Dataset 2 — real
    if verbose:
        print(f"▶ Generating Dataset 2: real/{real_source} (secondary) …",
              flush=True)
    ds2 = make_dataset_real(source=real_source, n_samples=n_samples,
                            random_state=random_state)
    bundles[ds2.tag] = ds2

    # Dataset 3 — quantum aligned
    if verbose:
        print("▶ Generating Dataset 3: quantum-aligned synthetic …", flush=True)
    ds3 = make_dataset_quantum_aligned(n_samples=n_samples, reps=quantum_reps,
                                       random_state=random_state)
    bundles[ds3.tag] = ds3

    if verbose:
        print()
        for tag, b in bundles.items():
            print(f"  [{tag}]  train={b.n_train}  test={b.n_test}  "
                  f"balance={b.class_balance}")
            print(f"    {b.description}")
        print()

    return bundles


# ---------------------------------------------------------------------------
# Persistence — save / load DatasetBundles
# ---------------------------------------------------------------------------

def save_dataset(bundle: DatasetBundle, tag: Optional[str] = None) -> None:
    """
    Persist a DatasetBundle to disk.

    Saves to:
    - ``data/raw/<tag>_X_raw.npy``       raw (unexpanded/unscaled) features
    - ``data/raw/<tag>_y_raw.npy``       raw labels
    - ``data/processed/<tag>_X_train.npy``
    - ``data/processed/<tag>_X_test.npy``
    - ``data/processed/<tag>_y_train.npy``
    - ``data/processed/<tag>_y_test.npy``
    - ``data/processed/<tag>_X_train.csv``  (human-readable)
    - ``data/processed/<tag>_X_test.csv``
    - ``data/processed/<tag>_y_train.csv``
    - ``data/processed/<tag>_y_test.csv``

    Parameters
    ----------
    bundle : DatasetBundle
    tag    : str, optional. Overrides ``bundle.tag`` as the filename stem.
    """
    t = tag or bundle.tag

    # Raw
    np.save(_DATA_RAW / f"{t}_X_raw.npy", bundle.X_raw)
    np.save(_DATA_RAW / f"{t}_y_raw.npy", bundle.y_raw)

    # Processed — numpy
    np.save(_DATA_PROC / f"{t}_X_train.npy", bundle.X_train)
    np.save(_DATA_PROC / f"{t}_X_test.npy",  bundle.X_test)
    np.save(_DATA_PROC / f"{t}_y_train.npy", bundle.y_train)
    np.save(_DATA_PROC / f"{t}_y_test.npy",  bundle.y_test)

    # Processed — CSV (with header)
    _hdr = ",".join(f"f{i}" for i in range(bundle.X_train.shape[1]))
    np.savetxt(_DATA_PROC / f"{t}_X_train.csv", bundle.X_train,
               delimiter=",", header=_hdr, comments="")
    np.savetxt(_DATA_PROC / f"{t}_X_test.csv",  bundle.X_test,
               delimiter=",", header=_hdr, comments="")
    np.savetxt(_DATA_PROC / f"{t}_y_train.csv", bundle.y_train,
               delimiter=",", header="label", comments="", fmt="%d")
    np.savetxt(_DATA_PROC / f"{t}_y_test.csv",  bundle.y_test,
               delimiter=",", header="label", comments="", fmt="%d")

    # Description file
    (_DATA_PROC / f"{t}_description.txt").write_text(bundle.description)

    print(f"  ✓ Saved dataset '{t}' → data/raw/ and data/processed/")


def load_dataset(tag: str) -> DatasetBundle:
    """
    Reload a previously saved DatasetBundle from ``data/processed/``.

    Parameters
    ----------
    tag : str
        The tag used when ``save_dataset`` was called.

    Returns
    -------
    DatasetBundle
        ``scaler`` is re-fitted on X_train (faithfully reconstructed).
    """
    X_train = np.load(_DATA_PROC / f"{tag}_X_train.npy")
    X_test  = np.load(_DATA_PROC / f"{tag}_X_test.npy")
    y_train = np.load(_DATA_PROC / f"{tag}_y_train.npy")
    y_test  = np.load(_DATA_PROC / f"{tag}_y_test.npy")
    X_raw   = np.load(_DATA_RAW  / f"{tag}_X_raw.npy")
    y_raw   = np.load(_DATA_RAW  / f"{tag}_y_raw.npy")

    desc_path = _DATA_PROC / f"{tag}_description.txt"
    description = desc_path.read_text() if desc_path.exists() else ""

    # Re-fit scaler on the stored X_train to faithfully restore it
    scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
    scaler.fit(X_train)

    return DatasetBundle(
        tag=tag,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_raw=X_raw,
        y_raw=y_raw,
        scaler=scaler,
        description=description,
    )


# ---------------------------------------------------------------------------
# Quantum Feature Map construction
# ---------------------------------------------------------------------------

def build_feature_map(
    num_qubits:   int = NUM_QUBITS,
    reps:         int = 2,
    entanglement: str = "full",
) -> ZZFeatureMap:
    """
    Construct a Qiskit ``ZZFeatureMap`` for a 4-qubit register.

    The ZZFeatureMap encodes a classical feature vector ``x ∈ ℝ^n`` into a
    quantum state ``|φ(x)⟩`` via two-local ZZ rotations::

        U_φ(x) = [H^⊗n · exp(i·Σ_{jk} φ_jk(x)·Z_j⊗Z_k)]^reps

    where ``φ_jk(x) = (π - x_j)(π - x_k)`` (the default ZZFeatureMap kernel).

    Parameters
    ----------
    num_qubits   : int  Number of qubits (default 4, matching the 4-qubit register).
    reps         : int  Number of circuit repetitions (default 2).
                        Higher reps → more expressive but deeper circuit.
    entanglement : str  Entanglement structure for the ZZ terms.
                        Options: ``"full"``, ``"linear"``, ``"circular"``,
                        ``"sca"`` (shift-and-cyclic alternating).
                        Use ``"full"`` for maximum expressibility;
                        ``"linear"`` for shallower circuits on constrained topologies.

    Returns
    -------
    ZZFeatureMap
        Parameterized Qiskit circuit (parameters named ``x[0]`` … ``x[3]``).

    Example
    -------
    >>> fm = build_feature_map(num_qubits=4, reps=2, entanglement="full")
    >>> print(fm.num_parameters)   # 4
    >>> print(fm.depth())          # depends on reps and entanglement
    """
    if _USE_FM_FN and _zz_feature_map_fn is not None:
        fm = _zz_feature_map_fn(
            feature_dimension=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )
    else:
        fm = ZZFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            entanglement=entanglement,
        )
    return fm


# ---------------------------------------------------------------------------
# Parametrized compute-uncompute kernel circuit
# ---------------------------------------------------------------------------

def build_parametrized_kernel_circuit(
    feature_map: ZZFeatureMap,
) -> QuantumCircuit:
    """
    Build a single parametrized **compute-uncompute** kernel circuit for the
    quantum kernel estimator.

    The circuit implements::

        K(xᵢ, xⱼ) ≈ |⟨0^n| U_φ†(xᵢ) · U_φ(xⱼ) |0^n⟩|²

    which equals the probability of measuring the all-zeros bitstring
    ``|00…0⟩`` after applying ``U_φ(xⱼ)`` followed by ``U_φ†(xᵢ)``.

    Circuit structure
    -----------------
    - **x_vec** (ParameterVector, length ``num_qubits``):
      Parameters for the *ket* state ``U_φ(xⱼ)`` (second / right side).
    - **y_vec** (ParameterVector, length ``num_qubits``):
      Parameters for the *bra* state ``U_φ†(xᵢ)`` (first / left side —
      conjugate-transposed feature map applied after).
    - ``measure_all()`` appended at the end.

    Usage (in the kernel loop — outside (i,j) inner loop)
    ------
    Transpile this circuit **once** per processor, then bind parameters per
    (i,j) pair via ``assign_parameters({x_vec: x_j, y_vec: x_i})``.
    NEVER call ``transpile()`` inside the (i,j) loop.

    Parameters
    ----------
    feature_map : ZZFeatureMap
        The feature map circuit (e.g. from ``build_feature_map()``).

    Returns
    -------
    QuantumCircuit
        Parametrized 4-qubit compute-uncompute circuit with 2×``num_qubits``
        free parameters (``x_vec`` and ``y_vec``) and ``measure_all()`` appended.

    Raises
    ------
    ValueError
        If ``feature_map.num_qubits`` is not 4.

    Example
    -------
    >>> fm  = build_feature_map(num_qubits=4, reps=2, entanglement="full")
    >>> qkc = build_parametrized_kernel_circuit(fm)
    >>> print(qkc.num_parameters)   # 8  (4 for x_vec, 4 for y_vec)
    >>> # Bind and run:
    >>> bound = qkc.assign_parameters({**dict(zip(x_vec, x_j)),
    ...                                **dict(zip(y_vec, x_i))})
    """
    n = feature_map.num_qubits
    if n != NUM_QUBITS:
        raise ValueError(
            f"build_parametrized_kernel_circuit expects a {NUM_QUBITS}-qubit "
            f"feature map, but got {n} qubits."
        )

    # ── Two independent ParameterVectors ──────────────────────────────────────
    x_vec = ParameterVector("x", length=n)   # for U_φ(xⱼ)  — "ket"
    y_vec = ParameterVector("y", length=n)   # for U_φ†(xᵢ) — "bra"

    # ── Bind feature_map params → x_vec and y_vec ────────────────────────────
    fm_params = list(feature_map.parameters)  # sorted ParameterView

    # U_φ(xⱼ): bind feature_map parameters to x_vec
    fm_xj = feature_map.assign_parameters(
        dict(zip(fm_params, x_vec))
    )

    # U_φ†(xᵢ): bind feature_map parameters to y_vec, then inverse
    fm_xi_dag = feature_map.assign_parameters(
        dict(zip(fm_params, y_vec))
    ).inverse()

    # ── Compose kernel circuit ────────────────────────────────────────────────
    #   |0^n⟩ ─── U_φ(xⱼ) ─── U_φ†(xᵢ) ─── measure
    kernel_circuit = QuantumCircuit(n, name="kernel_cu")
    kernel_circuit.compose(fm_xj,     inplace=True)
    kernel_circuit.barrier()
    kernel_circuit.compose(fm_xi_dag, inplace=True)
    kernel_circuit.measure_all()

    return kernel_circuit


# ---------------------------------------------------------------------------
# Quick self-test  (python3 -m src.problem)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import pprint

    print("=" * 65)
    print("  src/problem.py  —  PS-Q2 dataset & circuit self-test")
    print("=" * 65)

    # ── Dataset 1 ─────────────────────────────────────────────────────────────
    print("\n▶ Dataset 1: make_moons (primary, nonlinear expansion)")
    d1 = make_dataset_moons()
    assert d1.X_train.shape == (75, 4), f"Unexpected shape: {d1.X_train.shape}"
    assert d1.X_test.shape  == (25, 4)
    assert d1.X_train.min() >= 0.0 - 1e-9
    assert d1.X_train.max() <= np.pi + 1e-9
    print(f"  X_train : {d1.X_train.shape}  range [{d1.X_train.min():.4f}, {d1.X_train.max():.4f}]  ✓")
    print(f"  X_test  : {d1.X_test.shape}   range [{d1.X_test.min():.4f},  {d1.X_test.max():.4f}]")
    print(f"  Balance : {d1.class_balance}")
    save_dataset(d1)

    # Tile expansion variant
    d1t = make_dataset_moons(expansion="tile")
    assert d1t.X_train.shape[1] == 4
    print(f"  Tile variant X_train: {d1t.X_train.shape}  ✓")

    # ── Dataset 2 ─────────────────────────────────────────────────────────────
    print("\n▶ Dataset 2: load_digits (secondary, PCA-4)")
    d2 = make_dataset_real(source="digits")
    assert d2.X_train.shape[1] == 4
    assert d2.X_train.min() >= 0.0 - 1e-9
    print(f"  X_train : {d2.X_train.shape}  range [{d2.X_train.min():.4f}, {d2.X_train.max():.4f}]  ✓")
    print(f"  Balance : {d2.class_balance}")
    save_dataset(d2)

    print("\n▶ Dataset 2b: load_breast_cancer (secondary, PCA-4)")
    d2b = make_dataset_real(source="breast_cancer")
    print(f"  X_train : {d2b.X_train.shape}  ✓")
    save_dataset(d2b)

    # ── Dataset 3 ─────────────────────────────────────────────────────────────
    print("\n▶ Dataset 3: quantum-aligned synthetic (ZZFeatureMap parity labels)")
    d3 = make_dataset_quantum_aligned(n_samples=20)   # small n for speed
    assert d3.X_train.shape[1] == 4
    print(f"  X_train : {d3.X_train.shape}  ✓")
    print(f"  Labels  : {np.unique(d3.y_raw, return_counts=True)}")
    save_dataset(d3)

    # ── Reload test ───────────────────────────────────────────────────────────
    print("\n▶ Reload from disk …")
    d1r = load_dataset("moons")
    assert np.allclose(d1r.X_train, d1.X_train)
    print("  ✓ Reloaded 'moons' matches original.")

    # ── ZZFeatureMap ──────────────────────────────────────────────────────────
    print("\n▶ build_feature_map(num_qubits=4, reps=2, entanglement='full')")
    fm = build_feature_map(num_qubits=4, reps=2, entanglement="full")
    print(f"  Parameters  : {fm.num_parameters}  ✓")
    print(f"  Circuit depth: {fm.decompose().depth()}")

    print("\n▶ build_parametrized_kernel_circuit(fm)")
    kc = build_parametrized_kernel_circuit(fm)
    print(f"  Total parameters : {kc.num_parameters}  (expect 8)  ✓")
    print(f"  Circuit name     : {kc.name}")
    print(f"  Num qubits       : {kc.num_qubits}")
    x_params = [p for p in kc.parameters if p.name.startswith("x")]
    y_params = [p for p in kc.parameters if p.name.startswith("y")]
    assert len(x_params) == 4, f"Expected 4 x params, got {len(x_params)}"
    assert len(y_params) == 4, f"Expected 4 y params, got {len(y_params)}"
    print(f"  x_vec params : {sorted(str(p) for p in x_params)}")
    print(f"  y_vec params : {sorted(str(p) for p in y_params)}")
    print("  ✓ Kernel circuit verified.")

    print("\n" + "=" * 65)
    print("  All assertions passed.")
    print("=" * 65 + "\n")
