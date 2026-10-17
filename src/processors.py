"""
src/processors.py
=================
QFF 2026 – PS-Q2 Quantum Kernel SVM
Track 4 (QML) · Phase 1 · Processors A & B

Provides:
    - ProcessorConfig  : dataclass holding topology & configuration data
    - load_processor   : parses a processor JSON → ProcessorConfig
    - build_noise_model: constructs the frozen QFF 2026 synthetic NoiseModel
    - get_simulators   : returns (ideal_sim, noisy_sim) AerSimulator instances

Qiskit 1.x compliance
----------------------
- NO `execute`, `Aer.get_backend`, `IBMQ`, `QuantumInstance`, `opflow`.
- Uses `qiskit.transpile`, `qiskit_aer.AerSimulator`, `qiskit_aer.noise.*`.
- `transpile()` must be called ONCE per processor outside any (i,j) loop.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx

# ── Qiskit 1.x ────────────────────────────────────────────────────────────────
from qiskit.transpiler import CouplingMap

# ── Qiskit Aer ────────────────────────────────────────────────────────────────
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve the repo root as two levels above this file:  src/ → qff2026/
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PROCESSORS_DIR = _REPO_ROOT / "processors"

PROCESSOR_A_JSON = _PROCESSORS_DIR / "processor_A.json"
PROCESSOR_B_JSON = _PROCESSORS_DIR / "processor_B.json"


# ---------------------------------------------------------------------------
# ProcessorConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProcessorConfig:
    """
    Holds all topology and configuration data for a single processor.

    Attributes
    ----------
    name : str
        Human-readable processor name (e.g. "Processor_A_5Q_Branching_Tree").
    num_qubits : int
        Total number of physical qubits.
    undirected_edges : List[List[int]]
        Undirected edge list (each edge stored once).
    directed_coupling_map : List[List[int]]
        Fully directed edge list (both [u,v] and [v,u] for every undirected edge).
    basis_gates : List[str]
        Native gate set (e.g. ["rz", "sx", "x", "cx"]).
    noise_params : dict
        Frozen QFF 2026 noise parameters:
        ``p_1q``, ``p_2q``, ``p_ro``.
    transpile_config : dict
        Transpilation settings:
        ``optimization_level``, ``seed``, ``shots``.
    coupling_map : CouplingMap
        Qiskit CouplingMap built from ``directed_coupling_map``.
    graph : nx.Graph
        NetworkX undirected graph built from ``undirected_edges``.
    custom_ring_edges : Optional[List[List[int]]]
        Alternative ring topology (Processor A only). ``None`` for Processor B.
    optimal_4q_subgraph : Optional[List[int]]
        Recommended 4-qubit subgraph nodes for ZZFeatureMap embedding.
    """

    name: str
    num_qubits: int
    undirected_edges: List[List[int]]
    directed_coupling_map: List[List[int]]
    basis_gates: List[str]
    noise_params: dict
    transpile_config: dict
    coupling_map: CouplingMap = field(repr=False)
    graph: nx.Graph = field(repr=False)
    custom_ring_edges: Optional[List[List[int]]] = None
    optimal_4q_subgraph: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def shots(self) -> int:
        """Shortcut for ``transpile_config["shots"]``."""
        return self.transpile_config["shots"]

    @property
    def optimization_level(self) -> int:
        """Shortcut for ``transpile_config["optimization_level"]``."""
        return self.transpile_config["optimization_level"]

    @property
    def seed_transpiler(self) -> int:
        """Shortcut for ``transpile_config["seed"]``."""
        return self.transpile_config["seed"]

    @property
    def p_1q(self) -> float:
        """1-qubit gate depolarizing error probability."""
        return self.noise_params["p_1q"]

    @property
    def p_2q(self) -> float:
        """2-qubit gate depolarizing error probability."""
        return self.noise_params["p_2q"]

    @property
    def p_ro(self) -> float:
        """Readout (measurement) error probability."""
        return self.noise_params["p_ro"]

    def coupling_map_ring(self) -> Optional[CouplingMap]:
        """
        Returns a Qiskit CouplingMap for the custom ring topology,
        or ``None`` if this processor has no ring alternative.
        """
        if self.custom_ring_edges is None:
            return None
        ring_directed = []
        for u, v in self.custom_ring_edges:
            ring_directed.append([u, v])
            ring_directed.append([v, u])
        return CouplingMap(couplinglist=ring_directed)


# ---------------------------------------------------------------------------
# load_processor
# ---------------------------------------------------------------------------

def load_processor(json_path: pathlib.Path | str) -> ProcessorConfig:
    """
    Parse a processor JSON file and return a fully initialised
    :class:`ProcessorConfig`.

    Parameters
    ----------
    json_path : pathlib.Path | str
        Absolute or relative path to a processor JSON file
        (e.g. ``PROCESSOR_A_JSON`` or ``PROCESSOR_B_JSON``).

    Returns
    -------
    ProcessorConfig
        Populated dataclass including the Qiskit ``CouplingMap`` and
        NetworkX ``Graph``.

    Raises
    ------
    FileNotFoundError
        If *json_path* does not exist.
    KeyError
        If a required field is missing from the JSON.
    """
    json_path = pathlib.Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"Processor config not found: {json_path}\n"
            f"  Expected location: {json_path.resolve()}"
        )

    with json_path.open("r", encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    # ── Build Qiskit CouplingMap ───────────────────────────────────────────
    directed = raw["directed_coupling_map"]
    coupling_map = CouplingMap(couplinglist=directed)

    # ── Build NetworkX undirected Graph ───────────────────────────────────
    graph = nx.Graph()
    graph.add_nodes_from(range(raw["num_qubits"]))
    graph.add_edges_from(raw["undirected_edges"])

    return ProcessorConfig(
        name=raw["name"],
        num_qubits=raw["num_qubits"],
        undirected_edges=raw["undirected_edges"],
        directed_coupling_map=directed,
        basis_gates=raw["basis_gates"],
        noise_params=raw["noise_params"],
        transpile_config=raw["transpile_config"],
        coupling_map=coupling_map,
        graph=graph,
        custom_ring_edges=raw.get("custom_ring_edges"),       # Processor A only
        optimal_4q_subgraph=raw.get("optimal_4q_subgraph"),  # Both processors
    )


# ---------------------------------------------------------------------------
# Convenience loaders for A and B
# ---------------------------------------------------------------------------

def load_processor_A() -> ProcessorConfig:
    """Load the canonical Processor A (5-qubit branching-tree) config."""
    return load_processor(PROCESSOR_A_JSON)


def load_processor_B() -> ProcessorConfig:
    """Load the canonical Processor B (9-qubit heavy-hex) config."""
    return load_processor(PROCESSOR_B_JSON)


# ---------------------------------------------------------------------------
# build_noise_model
# ---------------------------------------------------------------------------

def build_noise_model(cfg: ProcessorConfig) -> NoiseModel:
    """
    Construct the **frozen QFF 2026 synthetic noise model** from a
    :class:`ProcessorConfig`.

    Noise specification (immutable for Phase 1):
    - 1-qubit depolarizing  p = cfg.p_1q  on gates ``['rz', 'sx', 'x']``
    - 2-qubit depolarizing  p = cfg.p_2q  on gate  ``['cx']``
    - Symmetric readout error p = cfg.p_ro on every qubit::

        [[1-p_ro, p_ro],
         [p_ro,   1-p_ro]]

    Parameters
    ----------
    cfg : ProcessorConfig
        Processor configuration supplying ``p_1q``, ``p_2q``, ``p_ro``,
        ``num_qubits``, and ``basis_gates``.

    Returns
    -------
    qiskit_aer.noise.NoiseModel
        Fully constructed noise model ready to pass to ``AerSimulator``.
    """
    noise_model = NoiseModel(basis_gates=cfg.basis_gates)

    # ── 1-qubit depolarizing error ─────────────────────────────────────────
    err_1q = depolarizing_error(cfg.p_1q, 1)
    for gate in ["rz", "sx", "x"]:
        noise_model.add_all_qubit_quantum_error(err_1q, gate)

    # ── 2-qubit depolarizing error ─────────────────────────────────────────
    err_2q = depolarizing_error(cfg.p_2q, 2)
    noise_model.add_all_qubit_quantum_error(err_2q, "cx")

    # ── Symmetric readout error  ───────────────────────────────────────────
    p_ro = cfg.p_ro
    ro_matrix = [[1 - p_ro, p_ro],
                 [p_ro,     1 - p_ro]]
    err_ro = ReadoutError(ro_matrix)
    for qubit in range(cfg.num_qubits):
        noise_model.add_readout_error(err_ro, [qubit])

    return noise_model


# ---------------------------------------------------------------------------
# get_simulators
# ---------------------------------------------------------------------------

def get_simulators(
    cfg: ProcessorConfig,
) -> Tuple[AerSimulator, AerSimulator]:
    """
    Return a pair of ``AerSimulator`` instances for *cfg*:

    - **ideal_sim** : noiseless statevector / density-matrix backend.
    - **noisy_sim** : ``AerSimulator`` seeded with the frozen QFF 2026
      noise model, configured to run up to ``cfg.shots`` shots.

    Both simulators are configured with the processor's ``basis_gates``
    so that Qiskit's transpiler uses the correct native gate set.

    Parameters
    ----------
    cfg : ProcessorConfig
        Processor configuration used to build the noise model and set
        coupling constraints.

    Returns
    -------
    ideal_sim : AerSimulator
        Noiseless simulator (statevector method).
    noisy_sim : AerSimulator
        Noisy simulator (automatic method selection) with QFF 2026 noise.

    Example
    -------
    >>> from src.processors import load_processor_A, get_simulators
    >>> cfg_a = load_processor_A()
    >>> ideal, noisy = get_simulators(cfg_a)
    >>> result = noisy.run(transpiled_qc, shots=cfg_a.shots).result()
    """
    # ── Ideal (noiseless) simulator ────────────────────────────────────────
    ideal_sim = AerSimulator(method="statevector")

    # ── Noisy simulator ────────────────────────────────────────────────────
    noise_model = build_noise_model(cfg)
    noisy_sim = AerSimulator(
        noise_model=noise_model,
        basis_gates=cfg.basis_gates,
        coupling_map=cfg.directed_coupling_map,
    )

    return ideal_sim, noisy_sim


# ---------------------------------------------------------------------------
# Quick self-test (python -m src.processors)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import pprint

    for label, loader in [("A", load_processor_A), ("B", load_processor_B)]:
        print(f"\n{'='*60}")
        print(f"  Processor {label}")
        print(f"{'='*60}")
        cfg = loader()
        print(f"  Name         : {cfg.name}")
        print(f"  Num qubits   : {cfg.num_qubits}")
        print(f"  Basis gates  : {cfg.basis_gates}")
        print(f"  Noise params : p_1q={cfg.p_1q}, p_2q={cfg.p_2q}, p_ro={cfg.p_ro}")
        print(f"  Transpile    : opt={cfg.optimization_level}, "
              f"seed={cfg.seed_transpiler}, shots={cfg.shots}")
        print(f"  Undirected edges ({len(cfg.undirected_edges)}):")
        pprint.pprint(cfg.undirected_edges, indent=4)
        print(f"  Directed edges ({len(cfg.directed_coupling_map)}):")
        pprint.pprint(cfg.directed_coupling_map, indent=4)
        if cfg.custom_ring_edges:
            print(f"  Ring alt edges ({len(cfg.custom_ring_edges)}):")
            pprint.pprint(cfg.custom_ring_edges, indent=4)
        print(f"  Optimal 4Q subgraph : {cfg.optimal_4q_subgraph}")

        # Validate CouplingMap
        cm = cfg.coupling_map
        print(f"  CouplingMap  : {cm.size()} nodes, "
              f"{len(list(cm.get_edges()))} directed edges  ✓")

        # Validate NetworkX graph
        g = cfg.graph
        print(f"  nx.Graph     : {g.number_of_nodes()} nodes, "
              f"{g.number_of_edges()} edges  ✓")

        # Build noise model
        nm = build_noise_model(cfg)
        print(f"  NoiseModel   : basis_gates={nm.basis_gates}  ✓")

        # Get simulators
        ideal, noisy = get_simulators(cfg)
        print(f"  Ideal sim    : {ideal}  ✓")
        print(f"  Noisy sim    : {noisy}  ✓")

    print(f"\n{'='*60}")
    print("  All processors loaded and validated successfully.")
    print(f"{'='*60}\n")
