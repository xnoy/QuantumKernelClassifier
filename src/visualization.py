"""
src/visualization.py
====================
QFF 2026 – PS-Q2 Quantum Kernel SVM
Track 4 (QML) · Phase 1 · Publication-quality Figures

Provides 5 plotting functions, each auto-saving 300 DPI PNG figures to
``results/figures/``:

    plot_processor_topologies(proc_A, proc_B, ...)
        Side-by-side NetworkX graph: Proc A branching tree + ring overlay,
        Proc B heavy-hex with optimal 4-qubit cycle [1,2,5,6] highlighted.

    plot_architecture_metrics(routing_df, ...)
        Grouped bar chart: depth / CX count / SWAP overhead across
        Linear vs Full entanglement on Proc A (tree & ring) and Proc B.

    plot_kernel_heatmaps(K_ideal, K_procA, K_procB, ...)
        4-panel: ideal kernel, Proc-A noisy kernel, Proc-B noisy kernel,
        off-diagonal similarity histograms showing kernel concentration.

    plot_decision_boundaries(X, y, models_dict, ...)
        2D contour plots for Ideal QSVM, Proc-A QSVM, Proc-B QSVM,
        and Classical RBF SVM on make_moons.

    plot_classification_performance(results_df, ...)
        Grouped bar chart + 95% CI error bars for Accuracy, F1, ROC-AUC
        across all 7 classifiers × all datasets.

Global style
------------
- ``set_publication_style()`` applies a clean seaborn + matplotlib rcParams
  theme with QFF 2026 colour palette (no dependency on LaTeX).
- Every figure is saved at DPI=300 with tight bounding box.
- ``matplotlib.use('Agg')`` is called at import to allow headless execution
  in notebook-less environments.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # headless — safe for script & notebook alike

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import numpy as np
import networkx as nx
import seaborn as sns

# ── Project imports (lazy – only used inside functions) ─────────────────────
from src.processors import ProcessorConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
_FIGURES_DIR = _REPO_ROOT / "results" / "figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# QFF 2026 Colour Palette & Global Style
# ---------------------------------------------------------------------------

# Named colour constants for consistent use across all figures
C_PROC_A     = "#2196F3"   # blue   – Processor A (branching tree)
C_PROC_A_RING= "#64B5F6"   # light blue – Proc A ring topology
C_PROC_B     = "#E91E63"   # pink/magenta – Processor B (heavy-hex)
C_IDEAL      = "#4CAF50"   # green  – ideal / noiseless
C_MITIGATED  = "#FF9800"   # amber  – mitigated
C_CLASSICAL  = "#9C27B0"   # purple – classical SVMs
C_HIGHLIGHT  = "#FFD700"   # gold   – subgraph highlight
C_NEUTRAL    = "#607D8B"   # blue-grey – neutral / background nodes
C_BACKGROUND = "#FAFAFA"   # near-white figure background

# Classifier display name → colour map (used in bar charts and legends)
CLF_COLOURS: Dict[str, str] = {
    "QSVM_Ideal":            C_IDEAL,
    "QSVM_ProcA_Noisy":      C_PROC_A,
    "QSVM_ProcB_Noisy":      C_PROC_B,
    "QSVM_ProcA_Mitigated":  C_MITIGATED,
    "QSVM_ProcB_Mitigated":  "#FF5722",   # deep-orange
    "Classical_SVM_RBF":     C_CLASSICAL,
    "Classical_SVM_Poly":    "#673AB7",   # deep-purple
}

CLF_LABELS: Dict[str, str] = {
    "QSVM_Ideal":            "QSVM – Ideal",
    "QSVM_ProcA_Noisy":      "QSVM – Proc A Noisy",
    "QSVM_ProcB_Noisy":      "QSVM – Proc B Noisy",
    "QSVM_ProcA_Mitigated":  "QSVM – Proc A Mitigated",
    "QSVM_ProcB_Mitigated":  "QSVM – Proc B Mitigated",
    "Classical_SVM_RBF":     "Classical SVM (RBF)",
    "Classical_SVM_Poly":    "Classical SVM (Poly)",
}

DPI = 300


def set_publication_style() -> None:
    """
    Apply a clean, publication-ready matplotlib/seaborn style.

    - Seaborn ``whitegrid`` base with custom overrides.
    - Font sizes appropriate for a two-column conference figure.
    - No LaTeX dependency (``text.usetex = False``).
    - Tight layout enabled by default.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    plt.rcParams.update({
        # Figure
        "figure.facecolor":      C_BACKGROUND,
        "axes.facecolor":        C_BACKGROUND,
        "figure.dpi":            100,          # screen; saved at DPI=300
        "savefig.dpi":           DPI,
        "savefig.bbox":          "tight",
        "savefig.facecolor":     C_BACKGROUND,
        # Fonts
        "font.family":           "DejaVu Sans",
        "font.size":             11,
        "axes.titlesize":        13,
        "axes.titleweight":      "bold",
        "axes.labelsize":        11,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "legend.fontsize":       9,
        "legend.framealpha":     0.9,
        # Lines & spines
        "axes.linewidth":        0.8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "grid.linewidth":        0.5,
        "grid.alpha":            0.6,
        # Text – no LaTeX
        "text.usetex":           False,
    })


def _save(fig: plt.Figure, path: str | pathlib.Path, *, verbose: bool = True) -> pathlib.Path:
    """Save *fig* to *path* at 300 DPI with tight bounding box, then close."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=C_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"  ✓ Saved → {p}")
    return p


# ---------------------------------------------------------------------------
# Figure 1 — Processor Topology Graphs
# ---------------------------------------------------------------------------

def plot_processor_topologies(
    proc_A:    ProcessorConfig,
    proc_B:    ProcessorConfig,
    save_path: str | pathlib.Path = _FIGURES_DIR / "fig1_processor_topologies.png",
) -> pathlib.Path:
    """
    Side-by-side NetworkX graph visualisation of Processor A and Processor B.

    Left panel — Processor A (5 qubits, branching tree):
    - Primary branching-tree edges drawn in solid blue.
    - Ring alternative edges (``custom_ring_edges``) overlaid as dashed
      light-blue arcs to illustrate the creative topology alternative.
    - Optimal 4-qubit nodes (0-1-2-3) highlighted in gold.

    Right panel — Processor B (9 qubits, heavy-hex):
    - All 12 heavy-hex edges in magenta.
    - Optimal 4-qubit cycle subgraph nodes [1, 2, 5, 6] highlighted in gold
      with thicker gold edges connecting them.

    Parameters
    ----------
    proc_A    : ProcessorConfig for Processor A.
    proc_B    : ProcessorConfig for Processor B.
    save_path : Output file path.

    Returns
    -------
    pathlib.Path  Absolute path to the saved figure.
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             facecolor=C_BACKGROUND)
    fig.suptitle("QFF 2026 · Processor Topologies (Phase 1)",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── Panel A: Processor A ─────────────────────────────────────────────────
    ax_a = axes[0]
    ax_a.set_title(f"Processor A — {proc_A.num_qubits} Qubits\n"
                   "(Branching Tree + Ring Alt.)",
                   fontsize=12, fontweight="bold")

    G_a = proc_A.graph.copy()
    # Spring layout seeded for reproducibility
    pos_a = nx.spring_layout(G_a, seed=42, k=1.2)

    # Node colours: highlight optimal 4Q subgraph in gold
    opt_a  = set(proc_A.optimal_4q_subgraph or [0, 1, 2, 3])
    n_col_a = [C_HIGHLIGHT if n in opt_a else C_PROC_A for n in G_a.nodes()]
    n_size_a= [700 if n in opt_a else 500 for n in G_a.nodes()]

    # Draw primary branching-tree edges
    nx.draw_networkx_edges(G_a, pos_a, ax=ax_a,
                           edge_color=C_PROC_A, width=2.2,
                           alpha=0.85, style="solid")
    # Draw ring-alternative edges (dashed overlay)
    if proc_A.custom_ring_edges:
        ring_edges = [(u, v) for u, v in proc_A.custom_ring_edges
                      if not G_a.has_edge(u, v)]  # only truly new edges
        all_ring   = list(proc_A.custom_ring_edges)
        nx.draw_networkx_edges(G_a, pos_a, edgelist=all_ring, ax=ax_a,
                               edge_color=C_PROC_A_RING, width=1.5,
                               alpha=0.60, style="dashed",
                               connectionstyle="arc3,rad=0.25")

    nx.draw_networkx_nodes(G_a, pos_a, ax=ax_a,
                           node_color=n_col_a, node_size=n_size_a,
                           edgecolors="#333333", linewidths=1.5)
    nx.draw_networkx_labels(G_a, pos_a, ax=ax_a,
                            font_size=11, font_weight="bold",
                            font_color="white")

    # Legend
    leg_a = [
        mpatches.Patch(color=C_PROC_A,      label="Branching-tree edge"),
        mpatches.Patch(color=C_PROC_A_RING, label="Ring alt. edge (dashed)", alpha=0.7),
        mpatches.Patch(color=C_HIGHLIGHT,   label=f"Optimal 4Q nodes {list(opt_a)}"),
        mpatches.Patch(color=C_PROC_A,      label="Other nodes"),
    ]
    ax_a.legend(handles=leg_a, loc="lower left", fontsize=8.5,
                framealpha=0.92, edgecolor="#cccccc")
    ax_a.set_axis_off()

    # Annotation: SWAP-free condition for linear ZZFeatureMap
    ax_a.annotate("Linear ZZFeatureMap: 0 SWAPs on tree\n"
                  "Full ZZFeatureMap: 3 est. SWAPs on tree",
                  xy=(0.5, -0.05), xycoords="axes fraction",
                  ha="center", fontsize=8.5, color="#444444",
                  style="italic")

    # ── Panel B: Processor B ─────────────────────────────────────────────────
    ax_b = axes[1]
    ax_b.set_title(f"Processor B — {proc_B.num_qubits} Qubits\n"
                   "(12-Edge Heavy-Hex-Inspired)",
                   fontsize=12, fontweight="bold")

    G_b = proc_B.graph.copy()
    pos_b = nx.spring_layout(G_b, seed=7, k=1.5)

    opt_b = set(proc_B.optimal_4q_subgraph or [1, 2, 5, 6])
    n_col_b  = [C_HIGHLIGHT if n in opt_b else C_PROC_B for n in G_b.nodes()]
    n_size_b = [700 if n in opt_b else 500 for n in G_b.nodes()]

    # All heavy-hex edges
    nx.draw_networkx_edges(G_b, pos_b, ax=ax_b,
                           edge_color=C_PROC_B, width=2.0, alpha=0.7)

    # Highlight optimal 4Q subgraph edges in gold
    opt_edges_b = [(u, v) for u, v in G_b.edges()
                   if u in opt_b and v in opt_b]
    if opt_edges_b:
        nx.draw_networkx_edges(G_b, pos_b, edgelist=opt_edges_b, ax=ax_b,
                               edge_color=C_HIGHLIGHT, width=4.0, alpha=0.95)

    nx.draw_networkx_nodes(G_b, pos_b, ax=ax_b,
                           node_color=n_col_b, node_size=n_size_b,
                           edgecolors="#333333", linewidths=1.5)
    nx.draw_networkx_labels(G_b, pos_b, ax=ax_b,
                            font_size=11, font_weight="bold",
                            font_color="white")

    leg_b = [
        mpatches.Patch(color=C_PROC_B,    label="Heavy-hex edge"),
        mpatches.Patch(color=C_HIGHLIGHT, label=f"Optimal 4Q cycle {sorted(opt_b)}"),
        Line2D([0], [0], color=C_HIGHLIGHT, linewidth=3.5,
               label="Optimal 4Q cycle edges"),
        mpatches.Patch(color=C_PROC_B,    label="Other nodes"),
    ]
    ax_b.legend(handles=leg_b, loc="lower left", fontsize=8.5,
                framealpha=0.92, edgecolor="#cccccc")
    ax_b.set_axis_off()

    ax_b.annotate("Optimal subgraph [1,2,5,6] enables\n"
                  "2 est. SWAPs for Full ZZFeatureMap",
                  xy=(0.5, -0.05), xycoords="axes fraction",
                  ha="center", fontsize=8.5, color="#444444",
                  style="italic")

    fig.tight_layout(pad=2.5)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 2 — Architecture Metrics (Grouped Bar Chart)
# ---------------------------------------------------------------------------

def plot_architecture_metrics(
    routing_df,
    save_path: str | pathlib.Path = _FIGURES_DIR / "fig2_architecture_comparison.png",
) -> pathlib.Path:
    """
    Grouped bar charts comparing circuit depth, CX count, and SWAP overhead.

    Expects *routing_df* to be a pandas DataFrame (or dict-of-dicts) with
    columns/keys:
    ``label``, ``processor_name``, ``entanglement``, ``transpiled_depth``,
    ``transpiled_cx_count``, ``logical_cx_count``, ``estimated_swap_overhead``,
    ``sabre_swap_count``.

    Layout: 3 sub-panels side by side.
      (a) Transpiled Circuit Depth
      (b) Total 2-Qubit CX Gate Count (logical vs transpiled)
      (c) Inserted SWAP Overhead (estimated & Sabre)

    Parameters
    ----------
    routing_df : pandas.DataFrame or list[dict] of RoutingMetrics.asdict().
    save_path  : Output path.

    Returns
    -------
    pathlib.Path
    """
    import pandas as pd

    set_publication_style()

    # Accept either a DataFrame or a list/dict of RoutingMetrics
    if isinstance(routing_df, dict):
        from dataclasses import asdict
        rows = []
        for v in routing_df.values():
            if hasattr(v, "__dataclass_fields__"):
                rows.append(asdict(v))
            elif isinstance(v, dict):
                rows.append(v)
        df = pd.DataFrame(rows)
    elif not hasattr(routing_df, "columns"):
        df = pd.DataFrame(list(routing_df))
    else:
        df = routing_df.copy()

    # Shorten labels for display
    def _short(label: str) -> str:
        label = label.replace("Processor_A_5Q_Branching_Tree", "ProcA")
        label = label.replace("Processor_B_9Q_Heavy_Hex", "ProcB")
        label = label.replace("_primary", "")
        label = label.replace("_reps2", "")
        label = label.replace("_reps1", "")
        return label

    df["short_label"] = df["label"].apply(_short)

    # Colour by processor + entanglement
    def _bar_colour(row) -> str:
        if "ring" in row["label"]:
            return C_PROC_A_RING
        if "ProcA" in row["short_label"] or "Branching" in row.get("processor_name", ""):
            return C_PROC_A if "full" in row["entanglement"] else "#90CAF9"
        return C_PROC_B if "full" in row["entanglement"] else "#F48FB1"

    df["colour"] = df.apply(_bar_colour, axis=1)

    labels = df["short_label"].tolist()
    x      = np.arange(len(labels))
    width  = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=C_BACKGROUND)
    fig.suptitle("QFF 2026 · Architectural Routing Analysis\n"
                 "Processor A vs B · Linear vs Full Entanglement",
                 fontsize=14, fontweight="bold", y=1.03)

    # Sub-panel (a) — Circuit Depth
    ax = axes[0]
    bars = ax.bar(x, df["transpiled_depth"], width=width,
                  color=df["colour"], edgecolor="#333333",
                  linewidth=0.7, zorder=3)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8.5, fontweight="bold")
    ax.set_title("(a) Transpiled Circuit Depth", fontweight="bold")
    ax.set_ylabel("Gate depth (layers)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(0, df["transpiled_depth"].max() * 1.25)
    ax.grid(axis="y", alpha=0.5, zorder=0)

    # Sub-panel (b) — CX Gate Count (logical vs transpiled)
    ax = axes[1]
    w2 = width / 2 - 0.02
    b1 = ax.bar(x - w2/2, df["logical_cx_count"],   width=w2,
                color="#B0BEC5", edgecolor="#555", linewidth=0.7,
                label="Logical CX",     zorder=3)
    b2 = ax.bar(x + w2/2, df["transpiled_cx_count"], width=w2,
                color=df["colour"],     edgecolor="#333", linewidth=0.7,
                label="Transpiled CX",  zorder=3)
    ax.bar_label(b1, fmt="%d", padding=3, fontsize=7.5)
    ax.bar_label(b2, fmt="%d", padding=3, fontsize=7.5, fontweight="bold")
    ax.set_title("(b) 2-Qubit CX Gate Count", fontweight="bold")
    ax.set_ylabel("CX gate count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(0, max(df["transpiled_cx_count"].max(),
                       df["logical_cx_count"].max()) * 1.3)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.5, zorder=0)

    # Sub-panel (c) — SWAP Overhead
    ax = axes[2]
    b3 = ax.bar(x - w2/2, df["estimated_swap_overhead"], width=w2,
                color="#FFCC80", edgecolor="#555", linewidth=0.7,
                label="Est. SWAP ((CX_t-CX_l)//3)", zorder=3)
    b4 = ax.bar(x + w2/2, df["sabre_swap_count"],        width=w2,
                color="#FF7043", edgecolor="#333", linewidth=0.7,
                label="Sabre SWAP (exact)",           zorder=3)
    ax.bar_label(b3, fmt="%d", padding=3, fontsize=7.5)
    ax.bar_label(b4, fmt="%d", padding=3, fontsize=7.5, fontweight="bold")
    ax.set_title("(c) Inserted SWAP Gate Overhead", fontweight="bold")
    ax.set_ylabel("SWAP gate count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(0, max(df[["estimated_swap_overhead",
                            "sabre_swap_count"]].max().max() + 1, 3))
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.5, zorder=0)

    # Global colour legend (processor / entanglement)
    colour_legend = [
        mpatches.Patch(color="#90CAF9",    label="Proc A · Linear (primary)"),
        mpatches.Patch(color=C_PROC_A,     label="Proc A · Full (primary)"),
        mpatches.Patch(color=C_PROC_A_RING,label="Proc A · Ring alt."),
        mpatches.Patch(color="#F48FB1",    label="Proc B · Linear"),
        mpatches.Patch(color=C_PROC_B,     label="Proc B · Full"),
    ]
    fig.legend(handles=colour_legend, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.08),
               fontsize=8.5, framealpha=0.9)

    fig.tight_layout(pad=2.5)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 3 — Kernel Heatmaps + Concentration Histogram
# ---------------------------------------------------------------------------

def plot_kernel_heatmaps(
    K_ideal:   np.ndarray,
    K_procA:   np.ndarray,
    K_procB:   np.ndarray,
    save_path: str | pathlib.Path = _FIGURES_DIR / "fig3_kernel_matrices_and_concentration.png",
    y:         Optional[np.ndarray] = None,
) -> pathlib.Path:
    """
    4-panel figure: 3 kernel heatmaps + off-diagonal concentration histogram.

    Panels
    ------
    (1) Ideal Quantum Kernel K — exact statevector Gram matrix.
    (2) Processor A Noisy Kernel — density-matrix + shot noise.
    (3) Processor B Noisy Kernel — density-matrix + shot noise.
    (4) Off-diagonal kernel value histogram comparing ideal vs Proc A vs Proc B,
        illustrating depolarizing-induced kernel concentration.

    If *y* is supplied, the first three heatmaps add label-sorted axis ticks
    and a dashed divider line to visually separate the two classes.

    Parameters
    ----------
    K_ideal  : np.ndarray (n, n). Ideal Gram matrix.
    K_procA  : np.ndarray (n, n). Processor A noisy kernel.
    K_procB  : np.ndarray (n, n). Processor B noisy kernel.
    save_path: Output path.
    y        : Optional binary label array for class-sorted display.

    Returns
    -------
    pathlib.Path
    """
    # Allow calling as (K_ideal, K_procA, K_procB, y, save_path)
    if isinstance(save_path, np.ndarray) and y is None:
        y, save_path = save_path, _FIGURES_DIR / "fig3_kernel_matrices_and_concentration.png"

    set_publication_style()

    # Optionally sort by class label
    if y is not None:
        sort_idx = np.argsort(y)
        K_ideal = K_ideal[np.ix_(sort_idx, sort_idx)]
        K_procA = K_procA[np.ix_(sort_idx, sort_idx)]
        K_procB = K_procB[np.ix_(sort_idx, sort_idx)]
        y_sorted = y[sort_idx]
        class_boundary = int(np.sum(y_sorted == 0))
    else:
        class_boundary = None

    fig = plt.figure(figsize=(16, 5.5), facecolor=C_BACKGROUND)
    fig.suptitle("QFF 2026 · Quantum Kernel Matrices & Noise-Induced Concentration",
                 fontsize=14, fontweight="bold", y=1.02)

    gs = GridSpec(1, 4, figure=fig, wspace=0.38)

    cmap   = sns.diverging_palette(230, 20, s=80, l=55, as_cmap=True)
    n      = K_ideal.shape[0]
    vmin, vmax = 0.0, 1.0

    def _heatmap(ax, K, title, colour):
        im = ax.imshow(K, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_title(title, fontweight="bold", fontsize=11, pad=8)
        ax.set_xlabel("Sample index j", fontsize=9)
        ax.set_ylabel("Sample index i", fontsize=9)

        # Class boundary divider
        if class_boundary is not None and 0 < class_boundary < n:
            for b in (ax.axhline, ax.axvline):
                b(class_boundary - 0.5, color="#FF5722",
                  linewidth=1.5, linestyle="--", alpha=0.85)

        # Colour bar
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("K(xᵢ, xⱼ)", fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)

        # Diagonal stats annotation
        diag_mean = np.mean(np.diag(K))
        od_mask   = ~np.eye(n, dtype=bool)
        od_mean   = np.mean(K[od_mask])
        od_std    = np.std(K[od_mask])
        ax.annotate(
            f"diaḡ={diag_mean:.3f}\n"
            f"off-diaḡ={od_mean:.3f} ± {od_std:.3f}",
            xy=(0.02, 0.01), xycoords="axes fraction",
            fontsize=7.5, color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=colour, alpha=0.85, lw=1.2),
        )

    _heatmap(fig.add_subplot(gs[0]), K_ideal,
             "(1) Ideal Quantum Kernel K", C_IDEAL)
    _heatmap(fig.add_subplot(gs[1]), K_procA,
             "(2) Proc A Noisy Kernel\n(branching tree, p₂q=0.005)", C_PROC_A)
    _heatmap(fig.add_subplot(gs[2]), K_procB,
             "(3) Proc B Noisy Kernel\n(heavy-hex, p₂q=0.005)", C_PROC_B)

    # ── Panel 4: Off-diagonal histogram ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    od_mask = ~np.eye(n, dtype=bool)

    for K, colour, label in [
        (K_ideal,  C_IDEAL,   "Ideal"),
        (K_procA,  C_PROC_A,  "Proc A Noisy"),
        (K_procB,  C_PROC_B,  "Proc B Noisy"),
    ]:
        od_vals = K[od_mask].flatten()
        ax4.hist(od_vals, bins=30, density=True,
                 alpha=0.55, color=colour, label=label,
                 edgecolor="none")
        ax4.axvline(od_vals.mean(), color=colour,
                    linestyle="--", linewidth=1.8, alpha=0.9)

    ax4.set_title("(4) Off-Diagonal K Distribution\n(noise → concentration)",
                  fontweight="bold", fontsize=11, pad=8)
    ax4.set_xlabel("K(xᵢ, xⱼ)  i ≠ j", fontsize=9)
    ax4.set_ylabel("Density", fontsize=9)
    ax4.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    # Annotate concentration direction
    ax4.annotate("← more concentrated\n(depolarising noise)",
                 xy=(0.55, 0.88), xycoords="axes fraction",
                 fontsize=7.5, color="#555555", ha="center",
                 style="italic",
                 arrowprops=dict(arrowstyle="->", color="#555555"),
                 xytext=(0.55, 0.75))

    fig.tight_layout(pad=2.0)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 4 — Decision Boundaries (2D make_moons PCA projection)
# ---------------------------------------------------------------------------

def plot_decision_boundaries(
    X:           np.ndarray,
    y:           np.ndarray,
    models_dict: Dict[str, object],
    save_path:   str | pathlib.Path = _FIGURES_DIR / "fig4_decision_boundaries.png",
    feature_names: Tuple[str, str]  = ("f₀  (x₁, [0,π])", "f₁  (x₂, [0,π])"),
    grid_res:    int                = 120,
) -> pathlib.Path:
    """
    2D contour decision boundaries on the first two feature dimensions.

    Displays a filled contour plot for each classifier in *models_dict* on
    the make_moons dataset (projected onto features 0 and 1).

    Parameters
    ----------
    X            : np.ndarray (n, ≥2).  Feature matrix (all 4 dims; plot uses [:, :2]).
    y            : np.ndarray (n,).     Binary labels {0, 1}.
    models_dict  : Dict[label_str → fitted sklearn SVC with predict()].
                   Expected keys (any subset of):
                   ``"QSVM_Ideal"``, ``"QSVM_ProcA_Noisy"``,
                   ``"QSVM_ProcB_Noisy"``, ``"Classical_SVM_RBF"``.
    save_path    : Output path.
    feature_names: Display names for the x and y axes.
    grid_res     : Number of grid points per axis (default 120).

    Returns
    -------
    pathlib.Path
    """
    set_publication_style()

    n_models = len(models_dict)
    ncols    = min(n_models, 4)
    nrows    = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.8 * nrows),
                             facecolor=C_BACKGROUND,
                             squeeze=False)
    fig.suptitle("QFF 2026 · Decision Boundaries (make_moons, 2D projection)",
                 fontsize=14, fontweight="bold", y=1.01)

    # Build 2D grid using first 2 features only
    X2 = X[:, :2]
    pad = 0.15
    x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
    y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )

    scatter_kw = dict(s=55, edgecolors="black", linewidths=0.6, zorder=5)
    cmap_bg    = mcolors.LinearSegmentedColormap.from_list(
        "qff_bg", ["#BBDEFB", "#FFCDD2"], N=256
    )

    for idx, (clf_key, clf) in enumerate(models_dict.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        display_name = CLF_LABELS.get(clf_key, clf_key)
        colour       = CLF_COLOURS.get(clf_key, "#607D8B")

        # Build prediction grid — pad remaining features with their mean
        if X.shape[1] > 2:
            X_mean_rest = X[:, 2:].mean(axis=0)
            grid_rest   = np.tile(X_mean_rest, (xx.ravel().shape[0], 1))
            grid_full   = np.hstack([
                np.c_[xx.ravel(), yy.ravel()], grid_rest
            ])
        else:
            grid_full = np.c_[xx.ravel(), yy.ravel()]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                Z = clf.predict(grid_full).reshape(xx.shape)
            except Exception:
                # If clf is a precomputed-kernel SVC
                try:
                    from src.metrics import compute_ideal_kernel
                    from src.problem import build_feature_map
                    fm = getattr(clf, "feature_map", None) or build_feature_map()
                    X_tr = getattr(clf, "_X_train", X)
                    K_grid = compute_ideal_kernel(grid_full, X_tr, fm)
                    Z = clf.predict(K_grid).reshape(xx.shape)
                except Exception:
                    Z = np.zeros_like(xx)

        ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5])
        ax.contour( xx, yy, Z, levels=[0.5], colors=[colour], linewidths=1.8)

        # Scatter training points
        for cls, marker, label in [(0, "o", "Class 0"), (1, "^", "Class 1")]:
            mask = y == cls
            ax.scatter(X2[mask, 0], X2[mask, 1],
                       c=[C_PROC_A if cls == 0 else C_PROC_B],
                       marker=marker, label=label, **scatter_kw)

        ax.set_title(display_name, fontweight="bold",
                     fontsize=11, color=colour, pad=6)
        ax.set_xlabel(feature_names[0], fontsize=9)
        ax.set_ylabel(feature_names[1], fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if idx == 0:
            ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout(pad=2.5)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Figure 5 — Classification Performance Bar Chart + CI Error Bars
# ---------------------------------------------------------------------------

def plot_classification_performance(
    results_df,
    save_path: str | pathlib.Path  = _FIGURES_DIR / "fig5_accuracy_auc_comparison.png",
    datasets:  Optional[List[str]] = None,
) -> pathlib.Path:
    # Allow calling as (results_df, datasets, save_path)
    if isinstance(save_path, (list, tuple)) and datasets is None:
        datasets, save_path = save_path, _FIGURES_DIR / "fig5_accuracy_auc_comparison.png"
    """
    Grouped bar chart with 95% CI error bars for Accuracy, F1, and ROC-AUC.

    Accepts the dict returned by ``evaluate_classifiers()`` or a pandas
    DataFrame with columns ``clf_name``, ``accuracy_mean``, ``f1_mean``,
    ``roc_auc_mean``, and corresponding ``_ci95`` columns.

    Panels
    ------
    (a) Accuracy    ± 95% CI
    (b) F1-Score    ± 95% CI
    (c) ROC-AUC     ± 95% CI

    All bars are grouped by classifier and coloured consistently with the
    QFF 2026 colour palette.

    Parameters
    ----------
    results_df : dict (from ``evaluate_classifiers``) or pandas.DataFrame.
    datasets   : Optional list of dataset labels to add as a figure subtitle.
    save_path  : Output path.

    Returns
    -------
    pathlib.Path
    """
    import pandas as pd

    set_publication_style()

    # ── Normalise input to DataFrame ──────────────────────────────────────────
    if isinstance(results_df, dict):
        # Dict returned by evaluate_classifiers: {clf_name: {metric_key: val}}
        rows = []
        for clf_name, row in results_df.items():
            r = dict(row)
            r["clf_name"] = clf_name
            rows.append(r)
        df = pd.DataFrame(rows)
    else:
        df = results_df.copy()

    if "clf_name" not in df.columns:
        raise ValueError("Input must have a 'clf_name' column.")

    # Ensure required columns have defaults
    for metric in ["accuracy", "f1", "roc_auc"]:
        for suf in ["_mean", "_ci95"]:
            col = f"{metric}{suf}"
            if col not in df.columns:
                df[col] = 0.0

    clf_names = df["clf_name"].tolist()
    colours   = [CLF_COLOURS.get(c, C_NEUTRAL) for c in clf_names]
    labels    = [CLF_LABELS.get(c, c) for c in clf_names]

    x     = np.arange(len(clf_names))
    width = 0.62

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor=C_BACKGROUND)
    subtitle = (f"Datasets: {', '.join(datasets)}" if datasets
                else "All tested datasets")
    fig.suptitle(
        "QFF 2026 · Classification Performance — Quantum vs Classical SVM\n"
        f"{subtitle}  ·  5-seed mean ± 95% CI",
        fontsize=13, fontweight="bold", y=1.03
    )

    metrics = [
        ("accuracy", "Accuracy",  "(a) Accuracy ± 95% CI",  [0.0, 1.05]),
        ("f1",       "F1-Score",  "(b) F1-Score ± 95% CI",  [0.0, 1.05]),
        ("roc_auc",  "ROC-AUC",   "(c) ROC-AUC ± 95% CI",   [0.0, 1.05]),
    ]

    for ax, (metric, ylabel, title, ylim) in zip(axes, metrics):
        means  = df[f"{metric}_mean"].values.astype(float)
        ci95s  = df[f"{metric}_ci95"].fillna(0).values.astype(float)

        bars = ax.bar(x, means, width=width, color=colours,
                      edgecolor="#333333", linewidth=0.7, zorder=3,
                      yerr=ci95s, capsize=4,
                      error_kw=dict(elinewidth=1.5, ecolor="#333333",
                                    capthick=1.5, zorder=6))

        # Value labels above each bar
        for xi, (mean, ci) in enumerate(zip(means, ci95s)):
            offset = mean + ci + 0.012
            if not np.isnan(mean):
                ax.text(xi, min(offset, ylim[1] - 0.04),
                        f"{mean:.3f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold",
                        color="#222222")

        # Reference lines
        ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.1,
                   label="Random baseline", zorder=2)

        # Best performer highlight
        if len(means) > 0:
            best_idx = int(np.nanargmax(means))
            ax.get_children()[best_idx].set_edgecolor("#FFD700")
            ax.get_children()[best_idx].set_linewidth(2.5)

        ax.set_title(title, fontweight="bold", fontsize=12, pad=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=38, ha="right",
                           fontsize=8.5)
        ax.set_ylim(ylim)
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.grid(axis="y", alpha=0.5, zorder=0)
        ax.axhline(0.5, color="#AAAAAA", linestyle=":", linewidth=1.0, zorder=1)

    # ── Global legend with classifier colours ─────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=CLF_COLOURS.get(c, C_NEUTRAL),
                       edgecolor="#333333", linewidth=0.7,
                       label=CLF_LABELS.get(c, c))
        for c in clf_names
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.10),
               fontsize=8.5, framealpha=0.9, edgecolor="#cccccc",
               title="Classifier", title_fontsize=9)

    fig.tight_layout(pad=2.5)
    return _save(fig, save_path)


# ---------------------------------------------------------------------------
# Convenience: render all 5 figures from pre-computed results
# ---------------------------------------------------------------------------

def render_all_figures(
    proc_A:      ProcessorConfig,
    proc_B:      ProcessorConfig,
    routing_results,
    K_ideal:     np.ndarray,
    K_procA:     np.ndarray,
    K_procB:     np.ndarray,
    y_kernel:    Optional[np.ndarray],
    X_moon:      np.ndarray,
    y_moon:      np.ndarray,
    models_dict: Dict[str, object],
    perf_results,
    verbose:     bool = True,
) -> Dict[str, pathlib.Path]:
    """
    Render all 5 publication figures and return a dict of saved paths.

    Convenience wrapper — calls all five plot functions in sequence.

    Returns
    -------
    dict  {"fig1": Path, "fig2": Path, …, "fig5": Path}
    """
    paths: Dict[str, pathlib.Path] = {}

    if verbose:
        print("▶ Rendering Fig 1: Processor topologies …")
    paths["fig1"] = plot_processor_topologies(proc_A, proc_B)

    if verbose:
        print("▶ Rendering Fig 2: Architecture metrics …")
    paths["fig2"] = plot_architecture_metrics(routing_results)

    if verbose:
        print("▶ Rendering Fig 3: Kernel heatmaps …")
    paths["fig3"] = plot_kernel_heatmaps(K_ideal, K_procA, K_procB, y=y_kernel)

    if verbose:
        print("▶ Rendering Fig 4: Decision boundaries …")
    paths["fig4"] = plot_decision_boundaries(X_moon, y_moon, models_dict)

    if verbose:
        print("▶ Rendering Fig 5: Classification performance …")
    paths["fig5"] = plot_classification_performance(perf_results)

    if verbose:
        print(f"\n  All 5 figures saved to {_FIGURES_DIR.resolve()}")

    return paths


# ---------------------------------------------------------------------------
# Self-test (synthetic data — no real SVM or kernels needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":   # pragma: no cover
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 65)
    print("  src/visualization.py  —  figure self-test (synthetic data)")
    print("=" * 65)

    from src.processors import load_processor_A, load_processor_B
    from src.problem    import make_dataset_moons

    np.random.seed(42)
    N = 30

    # Synthetic kernels
    K_id = np.random.rand(N, N)
    K_id = (K_id + K_id.T) / 2
    np.fill_diagonal(K_id, 1.0)
    K_a  = K_id * 0.9 + 0.05 * np.random.rand(N, N)
    K_b  = K_id * 0.85 + 0.08 * np.random.rand(N, N)
    np.fill_diagonal(K_a, 1.0); np.fill_diagonal(K_b, 1.0)

    y_fake = np.array([0] * (N // 2) + [1] * (N - N // 2))

    # Synthetic dataset
    bundle = make_dataset_moons(n_samples=60, random_state=42)
    X_all  = np.vstack([bundle.X_train, bundle.X_test])
    y_all  = np.concatenate([bundle.y_train, bundle.y_test])

    # Fake trained SVMs (train on 2D projection for decision boundary)
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    models: Dict[str, object] = {}
    for name, kernel, C in [
        ("QSVM_Ideal",        "rbf",  1.0),
        ("QSVM_ProcA_Noisy",  "rbf",  0.5),
        ("QSVM_ProcB_Noisy",  "rbf",  0.5),
        ("Classical_SVM_RBF", "rbf",  10.0),
    ]:
        clf = SVC(kernel=kernel, C=C, probability=True)
        clf.fit(X_all, y_all)
        models[name] = clf

    # Synthetic performance results
    perf = {
        name: {
            "clf_name":       name,
            "accuracy_mean":  np.random.uniform(0.6, 0.95),
            "accuracy_std":   0.03,
            "accuracy_ci95":  0.05,
            "f1_mean":        np.random.uniform(0.58, 0.94),
            "f1_std":         0.03,
            "f1_ci95":        0.04,
            "roc_auc_mean":   np.random.uniform(0.60, 0.98),
            "roc_auc_std":    0.03,
            "roc_auc_ci95":   0.05,
            "kta_mean":       np.random.uniform(0.1, 0.5),
            "od_var_mean":    np.random.uniform(0.01, 0.05),
        }
        for name in CLF_LABELS
    }

    proc_a = load_processor_A()
    proc_b = load_processor_B()

    # Fig 1
    print("\n▶ Fig 1: Processor topologies …", end=" ", flush=True)
    p = plot_processor_topologies(proc_a, proc_b)
    assert p.exists(); print("✓")

    # Fig 2 (need routing results)
    from src.routing import run_routing_study
    routing = run_routing_study(reps=2, entanglements=["linear", "full"],
                                verbose=False)
    print("▶ Fig 2: Architecture metrics …", end=" ", flush=True)
    p = plot_architecture_metrics(routing)
    assert p.exists(); print("✓")

    # Fig 3
    print("▶ Fig 3: Kernel heatmaps …", end=" ", flush=True)
    p = plot_kernel_heatmaps(K_id, K_a, K_b, y=y_fake)
    assert p.exists(); print("✓")

    # Fig 4
    print("▶ Fig 4: Decision boundaries …", end=" ", flush=True)
    p = plot_decision_boundaries(X_all, y_all, models)
    assert p.exists(); print("✓")

    # Fig 5
    print("▶ Fig 5: Classification performance …", end=" ", flush=True)
    p = plot_classification_performance(perf)
    assert p.exists(); print("✓")

    print("\n" + "=" * 65)
    print("  All 5 figures generated successfully.")
    print(f"  Saved to: {_FIGURES_DIR.resolve()}")
    print("=" * 65 + "\n")
