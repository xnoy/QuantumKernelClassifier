"""
src/routing.py
==============
QFF 2026 – PS-Q2 Quantum Kernel SVM
Track 4 (QML) · Phase 1 · Architectural Routing & Transpilation Analysis

Provides:
    - RoutingMetrics          : dataclass for per-circuit routing results
    - transpile_and_analyze() : decompose → transpile → extract metrics
    - compare_entanglement()  : linear vs full ZZFeatureMap on Proc A & B
    - run_routing_study()     : full matrix (2 procs × 2 entanglements)
    - save_routing_csv()      : write results/tables/architecture_routing_comparison.csv

Qiskit 1.x compliance
---------------------
- Uses `qiskit.transpile`, `qiskit.transpiler.CouplingMap`.
- NEVER `execute`, `Aer.get_backend`, `IBMQ`, `QuantumInstance`, `opflow`.
- transpile() called ONCE per (processor, circuit) pair outside any loop.
"""

from __future__ import annotations

import csv
import pathlib
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

# ── Qiskit 1.x ────────────────────────────────────────────────────────────────
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

# Sabre / layout passes (available in qiskit >= 1.0)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    SabreLayout,
    SabreSwap,
    TrivialLayout,
    SetLayout,
    ApplyLayout,
    BasisTranslator,
    UnrollCustomDefinitions,
    Decompose,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

# ── Project imports ────────────────────────────────────────────────────────────
from src.processors import ProcessorConfig, load_processor_A, load_processor_B
from src.problem    import build_feature_map, NUM_QUBITS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results" / "tables"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROUTING_CSV = _RESULTS_DIR / "architecture_routing_comparison.csv"

# ---------------------------------------------------------------------------
# RoutingMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoutingMetrics:
    """
    Holds all per-(circuit, processor) transpilation and routing metrics.

    Attributes
    ----------
    label             : Human-readable run label, e.g. "ProcA_full_reps2".
    processor_name    : Processor config name.
    entanglement      : ZZFeatureMap entanglement type ("full" | "linear" | …).
    reps              : ZZFeatureMap reps.
    logical_qubits    : Always 4 (our 4-qubit register).
    physical_qubits_budget : coupling_map.size() — total physical qubits available.
    logical_cx_count  : CX gates in the decomposed (pre-routing) circuit.
    transpiled_depth  : Depth of the transpiled circuit (including routing).
    transpiled_cx_count : CX gates after transpilation + routing.
    single_qubit_gates : Sum of rz + sx + x gate counts after transpilation.
    total_gates       : Total gate count after transpilation.
    estimated_swap_overhead : max(0, (transpiled_cx − logical_cx) // 3).
    sabre_swap_count  : Exact SWAP count from SabreSwap pass analysis.
    layout            : Physical qubit layout list (virtual → physical mapping).
    """

    label:                   str
    processor_name:          str
    entanglement:            str
    reps:                    int
    logical_qubits:          int
    physical_qubits_budget:  int
    logical_cx_count:        int
    transpiled_depth:        int
    transpiled_cx_count:     int
    single_qubit_gates:      int
    total_gates:             int
    estimated_swap_overhead: int
    sabre_swap_count:        int
    layout:                  List[int] = field(default_factory=list)

    # ── Dict-like compatibility & prompt aliases ──────────────────────────────
    @property
    def circuit_depth(self) -> int:
        return self.transpiled_depth

    @property
    def cx_count(self) -> int:
        return self.transpiled_cx_count

    def __getitem__(self, key: str):
        if key == "circuit_depth":
            return self.transpiled_depth
        if key == "cx_count":
            return self.transpiled_cx_count
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in ("circuit_depth", "cx_count")

    def keys(self):
        d = asdict(self)
        d["circuit_depth"] = self.transpiled_depth
        d["cx_count"] = self.transpiled_cx_count
        return d.keys()

    def values(self):
        d = asdict(self)
        d["circuit_depth"] = self.transpiled_depth
        d["cx_count"] = self.transpiled_cx_count
        return d.values()

    def items(self):
        d = asdict(self)
        d["circuit_depth"] = self.transpiled_depth
        d["cx_count"] = self.transpiled_cx_count
        return d.items()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["circuit_depth"] = self.transpiled_depth
        d["cx_count"] = self.transpiled_cx_count
        return d

    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"[{self.label}]  depth={self.transpiled_depth}  "
            f"cx_logical={self.logical_cx_count}  "
            f"cx_transpiled={self.transpiled_cx_count}  "
            f"SWAP_est={self.estimated_swap_overhead}  "
            f"SWAP_sabre={self.sabre_swap_count}  "
            f"1Q={self.single_qubit_gates}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_logical_cx(circuit: QuantumCircuit) -> int:
    """
    Count CX gates in a *decomposed* circuit (before routing).

    The circuit is first decomposed one level (``circuit.decompose()``) to
    unfold high-level gates like ``ZZFeatureMap`` into primitive operations,
    then the 'cx' entry in ``count_ops()`` is returned.
    """
    decomposed = circuit.decompose()
    return decomposed.count_ops().get("cx", 0)


def _count_sabre_swaps(
    circuit:      QuantumCircuit,
    coupling_map: CouplingMap,
    seed:         int = 42,
) -> tuple[int, list[int]]:
    """
    Run SabreLayout + SabreSwap passes to get an exact SWAP count and layout.

    Returns
    -------
    (n_swaps, layout_list)
        n_swaps    : Number of SWAP gates inserted by SabreSwap.
        layout_list: Physical qubit index for each virtual qubit [v0, v1, …].
    """
    from qiskit.transpiler import PassManager, Layout
    from qiskit.transpiler.passes import (
        SabreLayout, SabreSwap, ApplyLayout,
        TrivialLayout, SetLayout,
        FullAncillaAllocation, EnlargeWithAncilla,
    )

    # Decompose to native-ish level first
    dec = circuit.decompose()

    pm_layout = PassManager([
        SabreLayout(coupling_map, max_iterations=4, seed=seed,
                    swap_trials=20, layout_trials=20),
    ])
    try:
        pm_layout.run(dec)
        final_layout = pm_layout.property_set.get("layout")
        if final_layout is None:
            raise RuntimeError("SabreLayout returned no layout")
    except Exception:
        # Fallback: trivial layout (identity permutation for ≤ physical qubits)
        final_layout = None

    pm_swap = PassManager([
        SabreSwap(coupling_map, heuristic="lookahead", seed=seed),
    ])
    try:
        swapped = pm_swap.run(dec)
        n_swaps = swapped.count_ops().get("swap", 0)
    except Exception:
        n_swaps = 0

    # Extract layout as a list [virtual_q → physical_q]
    layout_list: list[int] = []
    if final_layout is not None:
        try:
            for vq in dec.qubits:
                layout_list.append(final_layout[vq])
        except Exception:
            layout_list = []

    return n_swaps, layout_list


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def transpile_and_analyze(
    circuit:           QuantumCircuit,
    coupling_map:      CouplingMap,
    basis_gates:       List[str]  = None,
    optimization_level: int       = 1,
    seed:              int        = 42,
    processor_name:    str        = "unknown",
    label:             str        = "",
    entanglement:      str        = "full",
    reps:              int        = 2,
) -> RoutingMetrics:
    """
    Decompose a high-level circuit, transpile it onto a processor topology,
    and return a :class:`RoutingMetrics` dict with full architectural metrics.

    Steps
    -----
    1. **Decompose once**: ``circuit.decompose()`` to unfold ``ZZFeatureMap``
       into primitive gates → count *logical* CX gates before routing.
    2. **Transpile**: ``qiskit.transpile(circuit, coupling_map=…,
       basis_gates=…, optimization_level=1, seed_transpiler=42)``.
    3. **Extract metrics**: depth, total_gates, cx_count, 1Q gate count.
    4. **SWAP estimation**: ``(transpiled_cx − logical_cx) // 3`` (lower bound).
    5. **SabreSwap exact count**: run SabreLayout + SabreSwap pass on decomposed
       circuit to get an exact SWAP insertion count.

    Parameters
    ----------
    circuit            : QuantumCircuit to analyze (e.g. a ``ZZFeatureMap``).
    coupling_map       : Qiskit CouplingMap for the target processor.
    basis_gates        : Native gate set. Defaults to ``['rz','sx','x','cx']``.
    optimization_level : Transpiler optimization level (default 1, per QFF rules).
    seed               : Transpiler and Sabre seed (default 42, per QFF rules).
    processor_name     : Name string for reporting.
    label              : Optional run label override.
    entanglement       : Entanglement type (for metadata only).
    reps               : ZZFeatureMap reps (for metadata only).

    Returns
    -------
    RoutingMetrics
    """
    if basis_gates is None:
        basis_gates = ["rz", "sx", "x", "cx"]

    run_label = label or f"{processor_name}_{entanglement}_reps{reps}"

    # ── Step 1: logical CX count (pre-routing) ────────────────────────────────
    logical_cx = _count_logical_cx(circuit)

    # ── Step 2: transpile ONCE ────────────────────────────────────────────────
    transpiled_qc = transpile(
        circuit,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )

    # ── Step 3: extract metrics ───────────────────────────────────────────────
    ops = transpiled_qc.count_ops()
    cx_count      = ops.get("cx", 0)
    rz_count      = ops.get("rz", 0)
    sx_count      = ops.get("sx", 0)
    x_count       = ops.get("x",  0)
    single_q      = rz_count + sx_count + x_count
    total_gates   = sum(ops.values())
    depth         = transpiled_qc.depth()

    # ── Step 4: SWAP estimation ───────────────────────────────────────────────
    swap_est = max(0, (cx_count - logical_cx) // 3)

    # ── Step 5: Sabre exact SWAP count + layout ───────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sabre_swaps, layout_list = _count_sabre_swaps(circuit, coupling_map, seed)

    return RoutingMetrics(
        label=run_label,
        processor_name=processor_name,
        entanglement=entanglement,
        reps=reps,
        logical_qubits=NUM_QUBITS,
        physical_qubits_budget=coupling_map.size(),
        logical_cx_count=logical_cx,
        transpiled_depth=depth,
        transpiled_cx_count=cx_count,
        single_qubit_gates=single_q,
        total_gates=total_gates,
        estimated_swap_overhead=swap_est,
        sabre_swap_count=sabre_swaps,
        layout=layout_list,
    )


# ---------------------------------------------------------------------------
# Entanglement comparison helper
# ---------------------------------------------------------------------------

def compare_entanglement(
    cfg:              ProcessorConfig,
    reps:             int           = 2,
    entanglements:    Sequence[str] = ("linear", "full"),
    basis_gates:      List[str]     = None,
) -> List[RoutingMetrics]:
    """
    Compare ZZFeatureMap with different entanglement strategies on one processor.

    For each entanglement type:
    - Builds ``ZZFeatureMap(num_qubits=4, reps=reps, entanglement=ent)``.
    - Calls :func:`transpile_and_analyze` with ``cfg.coupling_map``.

    Also compares the **primary branching-tree** topology (``cfg.coupling_map``)
    against the **ring topology** (``cfg.coupling_map_ring()``) for Processor A.

    Parameters
    ----------
    cfg          : ProcessorConfig (Processor A or B).
    reps         : ZZFeatureMap reps (default 2).
    entanglements: Sequence of entanglement strategy strings to benchmark.
    basis_gates  : Native gate set (default ``cfg.basis_gates``).

    Returns
    -------
    List[RoutingMetrics]  one entry per (entanglement, topology) combination.
    """
    if basis_gates is None:
        basis_gates = cfg.basis_gates

    results: List[RoutingMetrics] = []

    for ent in entanglements:
        fm = build_feature_map(num_qubits=NUM_QUBITS, reps=reps, entanglement=ent)

        # Primary topology
        m = transpile_and_analyze(
            circuit=fm,
            coupling_map=cfg.coupling_map,
            basis_gates=basis_gates,
            optimization_level=cfg.optimization_level,
            seed=cfg.seed_transpiler,
            processor_name=cfg.name,
            entanglement=ent,
            reps=reps,
            label=f"{cfg.name}_{ent}_reps{reps}_primary",
        )
        results.append(m)

        # Ring alternative (Processor A only)
        ring_cm = cfg.coupling_map_ring()
        if ring_cm is not None:
            m_ring = transpile_and_analyze(
                circuit=fm,
                coupling_map=ring_cm,
                basis_gates=basis_gates,
                optimization_level=cfg.optimization_level,
                seed=cfg.seed_transpiler,
                processor_name=cfg.name + "_ring",
                entanglement=ent,
                reps=reps,
                label=f"{cfg.name}_ring_{ent}_reps{reps}",
            )
            results.append(m_ring)

    return results


# ---------------------------------------------------------------------------
# Full routing study
# ---------------------------------------------------------------------------

def run_routing_study(
    reps:          int           = 2,
    entanglements: Sequence[str] = ("linear", "full"),
    verbose:       bool          = True,
) -> Dict[str, RoutingMetrics]:
    """
    Run the complete 2-processor × 2-entanglement routing study.

    Loads Processor A and Processor B, benchmarks all (entanglement, topology)
    combinations, and returns results keyed by ``metric.label``.

    Parameters
    ----------
    reps          : ZZFeatureMap reps (default 2, per QFF rules).
    entanglements : Entanglement types to compare (default ``("linear","full")``).
    verbose       : Print a summary table.

    Returns
    -------
    Dict[str, RoutingMetrics]  keyed by ``RoutingMetrics.label``.
    """
    cfg_a = load_processor_A()
    cfg_b = load_processor_B()

    all_results: Dict[str, RoutingMetrics] = {}

    for cfg in (cfg_a, cfg_b):
        if verbose:
            print(f"\n▶ {cfg.name}  ({cfg.num_qubits} qubits)")
        metrics_list = compare_entanglement(cfg, reps=reps,
                                            entanglements=entanglements)
        for m in metrics_list:
            all_results[m.label] = m
            if verbose:
                print(f"  {m.summary()}")

    return all_results


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def save_routing_csv(
    results: Dict[str, RoutingMetrics],
    path:    pathlib.Path = ROUTING_CSV,
) -> pathlib.Path:
    """
    Write routing metrics to a CSV at *path*.

    Columns match the ``RoutingMetrics`` fields (minus ``layout`` list).
    The file is written to ``results/tables/architecture_routing_comparison.csv``.

    Parameters
    ----------
    results : Dict returned by :func:`run_routing_study`.
    path    : Output path. Defaults to ``ROUTING_CSV``.

    Returns
    -------
    pathlib.Path  the written file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "label", "processor_name", "entanglement", "reps",
        "logical_qubits", "physical_qubits_budget",
        "logical_cx_count", "transpiled_depth",
        "transpiled_cx_count", "single_qubit_gates", "total_gates",
        "estimated_swap_overhead", "sabre_swap_count",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in results.values():
            row = {k: getattr(m, k) for k in fieldnames}
            writer.writerow(row)

    print(f"  ✓ Routing CSV saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":   # pragma: no cover
    print("=" * 65)
    print("  src/routing.py  —  routing & transpilation self-test")
    print("=" * 65)

    results = run_routing_study(reps=2, entanglements=["linear", "full"],
                                verbose=True)

    assert len(results) >= 4, "Expected at least 4 entries (2 procs × 2 ents)"

    csv_path = save_routing_csv(results)
    assert csv_path.exists()

    print(f"\n  Total configurations benchmarked: {len(results)}")
    print("\n" + "=" * 65)
    print("  All assertions passed.")
    print("=" * 65 + "\n")
