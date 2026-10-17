"""
QFF 2026 – PS-Q2 QSVM Source Package
======================================
Track 4 (QML) · Processor A (5-qubit) & Processor B (9-qubit) · Phase 1

Public API re-exports for convenient top-level imports:

    from src import load_processor, build_noise_model, get_simulators
    from src import ProcessorConfig
    from src import DatasetBundle, make_dataset_moons, make_dataset_real
    from src import make_dataset_quantum_aligned, load_all_datasets
    from src import save_dataset, load_dataset
    from src import build_feature_map, build_parametrized_kernel_circuit
"""

from src.processors import (
    ProcessorConfig,
    load_processor,
    load_processor_A,
    load_processor_B,
    build_noise_model,
    get_simulators,
)

from src.routing import (
    RoutingMetrics,
    transpile_and_analyze,
    compare_entanglement,
    run_routing_study,
    save_routing_csv,
)

from src.metrics import (
    compute_ideal_kernel,
    compute_noisy_hardware_kernel,
    verify_shot_kernel,
    mitigate_kernel_matrix,
    kernel_alignment,
    off_diagonal_variance,
    evaluate_classifiers,
    save_metrics_csv,
    ClassifierResult,
)

from src.problem import (
    DatasetBundle,
    make_dataset_moons,
    make_dataset_real,
    make_dataset_quantum_aligned,
    load_all_datasets,
    save_dataset,
    load_dataset,
    build_feature_map,
    build_parametrized_kernel_circuit,
    NUM_QUBITS,
    FEATURE_RANGE,
)

from src.visualization import (
    plot_processor_topologies,
    plot_architecture_metrics,
    plot_kernel_heatmaps,
    plot_decision_boundaries,
    plot_classification_performance,
    render_all_figures,
)

__all__ = [
    # processors
    "ProcessorConfig",
    "load_processor",
    "load_processor_A",
    "load_processor_B",
    "build_noise_model",
    "get_simulators",
    # problem / datasets
    "DatasetBundle",
    "make_dataset_moons",
    "make_dataset_real",
    "make_dataset_quantum_aligned",
    "load_all_datasets",
    "save_dataset",
    "load_dataset",
    # quantum circuits
    "build_feature_map",
    "build_parametrized_kernel_circuit",
    # routing & metrics
    "RoutingMetrics",
    "transpile_and_analyze",
    "compare_entanglement",
    "run_routing_study",
    "save_routing_csv",
    "compute_ideal_kernel",
    "compute_noisy_hardware_kernel",
    "verify_shot_kernel",
    "mitigate_kernel_matrix",
    "kernel_alignment",
    "off_diagonal_variance",
    "evaluate_classifiers",
    "save_metrics_csv",
    "ClassifierResult",
    # visualization
    "plot_processor_topologies",
    "plot_architecture_metrics",
    "plot_kernel_heatmaps",
    "plot_decision_boundaries",
    "plot_classification_performance",
    "render_all_figures",
    # constants
    "NUM_QUBITS",
    "FEATURE_RANGE",
]

__version__ = "1.0.0"
__author__  = "QFF 2026 – PS-Q2 Team"
