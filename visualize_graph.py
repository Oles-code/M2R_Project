"""
visualize_graph
===============
Annotated causal-graph figure for the recovered LiNGAM structure.

The figure encodes four things on top of the bare DAG so the report can show
"what the model claims AND how much we believe each claim":

    edge colour     verdict from the interventional validation
                      green   : confirmed (predicted edge, MW-significant after BH)
                      red     : refuted   (predicted edge, MW NOT significant)
                      orange  : cyclic    (confirmed but reverse also significant)
                      grey    : unstable  (predicted but bootstrap freq < threshold)

    edge thickness  proportional to |b_ij| from the point-estimate B —
                    visually weights how strong the SEM coefficient is.

    edge opacity    proportional to bootstrap selection frequency — visually
                    distinguishes "near-universal in resamples" from "barely
                    above threshold".

A legend is added as a separate subgraph cluster so the colour mapping is
self-documenting in the saved PNG.

Optional secondary figure: scatter of Wasserstein distance vs bootstrap
frequency for predicted edges, which separates strong + stable edges from
weak ones at a glance.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from lingam_model import LingamFit, SelectionResult
from validate_edges import CONFIRMED, REFUTED, CYCLIC
from plot_style import VERDICT_COLOURS, apply_style


# ── verdict-to-colour palette ────────────────────────────────────────────────
# Sourced from the shared style module so the graph, the scatter, and every
# other figure agree on green=confirmed / red=refuted / orange=cyclic /
# grey=unstable.

EDGE_COLOURS = {
    CONFIRMED:  VERDICT_COLOURS["confirmed"],
    REFUTED:    VERDICT_COLOURS["refuted"],
    CYCLIC:     VERDICT_COLOURS["cyclic"],
    "unstable": VERDICT_COLOURS["unstable"],
}


def _opacity_to_hex(o: float) -> str:
    """Map [0, 1] opacity to a two-char hex alpha suffix usable by graphviz."""
    o = max(0.15, min(1.0, float(o)))   # clamp so the edge is never invisible
    return f"{int(round(o * 255)):02X}"


def _edge_width(b: float, b_max: float, w_min: float = 0.6, w_max: float = 4.0) -> float:
    """Map |b| to a graphviz pen width on a linear scale capped at w_max."""
    if not np.isfinite(b) or b_max <= 0:
        return w_min
    return w_min + (w_max - w_min) * (abs(b) / b_max)


def _pick_subset_nodes(
    df: pd.DataFrame,
    gene_names: List[str],
    max_nodes: int = 15,
) -> List[str]:
    """Pick a readable subset of nodes: prefer those touching confirmed edges,
    then refuted, then any predicted edge, until we hit `max_nodes`.

    Returns gene names in `gene_names` order to keep colour-independent
    layout stable across runs.
    """
    touched: list = []
    seen = set()
    priority_order = [CONFIRMED, CYCLIC, REFUTED]
    for verdict in priority_order:
        sub = df[df["verdict"] == verdict]
        for _, r in sub.iterrows():
            for g in (r["cause_i"], r["effect_j"]):
                if g not in seen:
                    seen.add(g)
                    touched.append(g)
                    if len(touched) >= max_nodes:
                        return [g for g in gene_names if g in seen]
    # Pad with any remaining genes (preserving the canonical ordering) so the
    # graph still has a sensible number of nodes even if no edges were found.
    for g in gene_names:
        if g not in seen:
            seen.add(g)
            touched.append(g)
            if len(touched) >= max_nodes:
                break
    return [g for g in gene_names if g in seen]


def render_causal_graph(
    fit: LingamFit,
    sel: SelectionResult,
    df: pd.DataFrame,
    out_dir: str,
    *,
    max_nodes: int = 15,
    freq_threshold: float = 0.8,
    edge_threshold: float = 0.01,
    label_short: bool = True,
    file_stem: str = "k562_causal_graph",
) -> dict:
    """Build the annotated causal graph and save PNG, SVG, and .gv source.

    Returns a dict with the saved paths and the chosen node subset, so the
    orchestrator can reference them in SUMMARY.md.
    """
    import graphviz

    os.makedirs(out_dir, exist_ok=True)

    # Choose a readable subset and the indices into fit.B that hit it.
    subset = _pick_subset_nodes(df, sel.gene_names, max_nodes=max_nodes)
    idx_map = {g: i for i, g in enumerate(sel.gene_names)}
    subset_idx = [idx_map[g] for g in subset]

    # Verdict lookup keyed by (cause, effect) for the subset.
    verdict_lookup = {
        (r["cause_i"], r["effect_j"]): r for _, r in df.iterrows()
        if r["cause_i"] in idx_map and r["effect_j"] in idx_map
    }

    # `fdp` is a force-directed engine (supports the legend cluster, unlike
    # neato) that packs the nodes together instead of stretching them along a
    # rank axis as `dot`+rankdir=LR did. `K` is the spring rest-length: small
    # K + overlap=false gives a compact graph where the (enlarged) nodes are
    # visually prominent and edges no longer dominate the figure.
    g = graphviz.Digraph("causal", format="png", engine="fdp")
    g.attr(overlap="false", splines="true", K="0.6", sep="+6",
           label="Causal Graph", labelloc="t", fontsize="16",
           fontname="Helvetica")
    # Node defaults: large filled ellipses so nodes read as the primary
    # objects and the edges between them stay short relative to node size.
    g.attr("node", shape="ellipse", style="filled", fillcolor="#ecf0f1",
           color="#7f8c8d", fontname="Helvetica", fontsize="14",
           width="0.9", height="0.7", fixedsize="false")

    # Truncated labels (last 5 chars of the Ensembl ID) make a 15-node graph
    # readable. The full IDs go into the .gv source as comments so they are
    # recoverable.
    def _label(name: str) -> str:
        if not label_short:
            return name
        return name[-5:]   # last 5 chars of ENSG…

    for name in subset:
        # Node styling comes from the graph-level node defaults set above; we
        # only supply the (truncated) label and keep the full Ensembl ID as a
        # comment so the truncation doesn't lose information in the .gv source.
        g.node(name, label=_label(name), comment=name)

    # Pick the global |b| scale across drawn edges so widths are comparable.
    drawn = []
    for ci_name in subset:
        for ei_name in subset:
            if ci_name == ei_name:
                continue
            ci = idx_map[ci_name]; ei = idx_map[ei_name]
            b = float(fit.B[ei, ci])
            if abs(b) <= edge_threshold:
                continue
            drawn.append((ci_name, ei_name, b))
    b_max = max((abs(b) for _, _, b in drawn), default=1.0)

    for ci_name, ei_name, b in drawn:
        row = verdict_lookup.get((ci_name, ei_name))
        verdict = row["verdict"] if row is not None else "unknown"
        freq = float(row["bootstrap_freq"]) if row is not None else 0.0

        # Colour: greyed-out below the bootstrap stability threshold,
        # otherwise follow the verdict palette.
        if freq < freq_threshold:
            base = EDGE_COLOURS["unstable"]
        else:
            base = EDGE_COLOURS.get(verdict, "#34495e")  # dark grey fallback

        color = base + _opacity_to_hex(freq)
        width = _edge_width(b, b_max)

        # Edge label = signed coefficient to 2 dp; useful for the report.
        edge_label = f"{b:+.2f}"
        g.edge(ci_name, ei_name, color=color, penwidth=f"{width:.2f}",
               label=edge_label, fontsize="8", fontcolor=base)

    # Legend as a separate subgraph, so it doesn't interact with the layout.
    # We use individual filled boxes rather than an HTML-table label because
    # graphviz's HTML-label parser is fussy and varies between versions —
    # plain box nodes work in every graphviz install.
    legend_entries = [
        ("confirmed",            EDGE_COLOURS[CONFIRMED],
         "predicted, MW-sig (BH)"),
        ("refuted",              EDGE_COLOURS[REFUTED],
         "predicted, NOT MW-sig"),
        ("cyclic",               EDGE_COLOURS[CYCLIC],
         "confirmed + reverse-dir MW-sig"),
        ("unstable",             EDGE_COLOURS["unstable"],
         f"bootstrap freq < {freq_threshold:.2f}"),
    ]
    with g.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Legend (edge colour = verdict)",
                    style="rounded", fontsize="9", labeljust="l")
        prev_id = None
        for name, colour, desc in legend_entries:
            node_id = f"legend_{name}"
            legend.node(
                node_id,
                label=f"{name}\\n{desc}",
                shape="box", style="filled",
                fillcolor=colour, fontcolor="white",
                fontsize="9", fontname="Helvetica",
            )
            if prev_id is not None:
                # Invisible edges stack the legend nodes vertically so they
                # don't get rearranged by the main graph layout.
                legend.edge(prev_id, node_id, style="invis")
            prev_id = node_id

    base_path = os.path.join(out_dir, file_stem)
    # The .gv source is the canonical reproducible artefact; the PNG / SVG
    # are renderings of it. We write the .gv unconditionally so the output
    # survives even when the system `dot` binary is missing (the Python
    # `graphviz` package is just a wrapper around the CLI, so PNG/SVG
    # rendering needs `brew install graphviz` or equivalent on PATH).
    gv_src_path = base_path + ".gv"
    with open(gv_src_path, "w") as f:
        f.write(g.source)

    png_path = None
    svg_path = None
    try:
        # Both renders go through `g` so they use the fdp engine set above;
        # rendering the SVG via a bare graphviz.Source would silently fall back
        # to dot and produce a differently-laid-out figure.
        png_path = g.render(filename=file_stem, directory=out_dir,
                            format="png", cleanup=False)
        svg_path = g.render(filename=file_stem + "_svg", directory=out_dir,
                            format="svg", cleanup=True)
    except graphviz.backend.execute.ExecutableNotFound:
        # Fall back to a matplotlib + networkx render so we still produce
        # a viewable figure when the `dot` CLI isn't installed.
        png_path = _matplotlib_fallback(
            fit, sel, df, subset, subset_idx, out_dir,
            freq_threshold=freq_threshold, edge_threshold=edge_threshold,
            file_stem=file_stem,
        )

    return {
        "subset_nodes": subset,
        "subset_idx": subset_idx,
        "gv_source": gv_src_path,
        "png": png_path,
        "svg": svg_path,
    }


def _matplotlib_fallback(
    fit, sel, df, subset, subset_idx, out_dir,
    *, freq_threshold, edge_threshold, file_stem,
):
    """Render the same graph with matplotlib + networkx when `dot` is missing.

    Less pretty than graphviz but doesn't require an extra system binary.
    Same colour / width / opacity encoding as the graphviz version.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    idx_map = {g: i for i, g in enumerate(sel.gene_names)}
    G = nx.DiGraph()
    for g in subset:
        G.add_node(g)

    verdict_lookup = {
        (r["cause_i"], r["effect_j"]): r for _, r in df.iterrows()
        if r["cause_i"] in idx_map and r["effect_j"] in idx_map
    }

    edges, edge_colors, edge_widths = [], [], []
    b_max = 0.0
    for ci_name in subset:
        for ei_name in subset:
            if ci_name == ei_name:
                continue
            ci = idx_map[ci_name]; ei = idx_map[ei_name]
            b = float(fit.B[ei, ci])
            if abs(b) <= edge_threshold:
                continue
            row = verdict_lookup.get((ci_name, ei_name))
            verdict = row["verdict"] if row is not None else "unknown"
            freq = float(row["bootstrap_freq"]) if row is not None else 0.0
            if freq < freq_threshold:
                base = EDGE_COLOURS["unstable"]
            else:
                base = EDGE_COLOURS.get(verdict, "#34495e")
            G.add_edge(ci_name, ei_name)
            edges.append((ci_name, ei_name))
            edge_colors.append(base)
            edge_widths.append(abs(b))
            b_max = max(b_max, abs(b))

    if b_max > 0:
        edge_widths = [0.6 + 3.4 * (w / b_max) for w in edge_widths]
    else:
        edge_widths = [0.6 for _ in edges]

    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(G, seed=0, k=1.8 / max(1, len(G) ** 0.5))
    nx.draw_networkx_nodes(G, pos, node_color="#ecf0f1",
                           edgecolors="#7f8c8d", node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={n: n[-5:] for n in G.nodes},
                            font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                           width=edge_widths, arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.05", ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=EDGE_COLOURS[CONFIRMED], lw=3, label="confirmed"),
        Line2D([0], [0], color=EDGE_COLOURS[REFUTED],   lw=3, label="refuted"),
        Line2D([0], [0], color=EDGE_COLOURS[CYCLIC],    lw=3, label="cyclic"),
        Line2D([0], [0], color=EDGE_COLOURS["unstable"], lw=3,
               label=f"unstable (freq < {freq_threshold:.2f})"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
    ax.set_title("Causal Graph")
    ax.set_axis_off()
    plt.tight_layout()
    out = os.path.join(out_dir, file_stem + "_fallback.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_wasserstein_vs_freq_scatter(
    df: pd.DataFrame,
    out_dir: str,
    *,
    freq_threshold: float = 0.8,
    edge_threshold: float = 0.01,
    file_stem: str = "k562_w_vs_freq",
) -> str:
    """Optional secondary figure: Wasserstein vs bootstrap frequency for
    predicted edges. Strong + stable edges sit in the top right; weak or
    unstable predictions sit in the bottom or left.
    """
    import matplotlib.pyplot as plt

    apply_style()
    os.makedirs(out_dir, exist_ok=True)
    pred = df[df["lingam_b_ij"].abs() > edge_threshold].copy()
    if pred.empty:
        # Nothing to plot but still produce a placeholder so the orchestrator
        # has a stable filename to reference.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no predicted edges", ha="center", va="center")
        ax.set_axis_off()
        out = os.path.join(out_dir, file_stem + ".png")
        fig.savefig(out, dpi=150); plt.close(fig)
        return out

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colour_map = {
        CONFIRMED:      EDGE_COLOURS[CONFIRMED],
        REFUTED:        EDGE_COLOURS[REFUTED],
        CYCLIC:         EDGE_COLOURS[CYCLIC],
    }
    for verdict, group in pred.groupby("verdict"):
        c = colour_map.get(verdict, "#7f8c8d")
        ax.scatter(group["bootstrap_freq"], group["wasserstein"],
                   s=30 + 100 * group["lingam_b_ij"].abs(),
                   alpha=0.7, c=c, edgecolors="white", linewidths=0.6,
                   label=f"{verdict} (n={len(group)})")

    ax.axvline(freq_threshold, color="black", linestyle="--", linewidth=0.7,
               label=f"freq = {freq_threshold:.2f}")
    ax.set_xlabel("bootstrap selection frequency")
    ax.set_ylabel("Wasserstein distance ($D_{int}$ vs $D_{obs}$)")
    ax.set_title("Edge Stability vs Interventional Shift")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out = os.path.join(out_dir, file_stem + ".png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
