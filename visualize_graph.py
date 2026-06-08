"""
visualize_graph
===============
Annotated causal-graph figure for the recovered LiNGAM structure, plus a
secondary Wasserstein-vs-bootstrap-frequency scatter.

The graph is rendered with matplotlib (node positions come from graphviz's
`neato` engine via pydot when available, else a deterministic kamada–kawai
layout). matplotlib is used for the drawing itself because it gives precise
control over the things a report needs: a compact corner legend, dashed
"recede" styling for unstable edges, |b|-scaled edge widths, an on-figure
footnote, and 300-DPI PNG + SVG export.

Edge encoding
-------------
    colour      verdict from interventional validation, for *stable* edges
                  green   confirmed   (predicted, MW-significant after BH)
                  red     refuted     (predicted, NOT MW-significant)
                  orange  cyclic      (confirmed + reverse direction also sig)
    grey dashed unstable (bootstrap frequency below `freq_threshold`) — drawn
                thin and dashed so it recedes behind the confirmed structure.
    width       proportional to |b_ij|, floored so coloured edges stay bold.
    hidden      edges below `hide_below_freq` are dropped entirely and counted
                in a footnote, so the densest noise doesn't swamp the figure.

A `.gv` DOT source mirroring the drawn graph is also written, so the structure
remains available as a text artefact (and SUMMARY.md's link stays valid).
"""

from __future__ import annotations

import os
from typing import List

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


def _pick_subset_nodes(
    df: pd.DataFrame,
    gene_names: List[str],
    max_nodes: int = 15,
) -> List[str]:
    """Pick a readable subset of nodes: prefer those touching confirmed edges,
    then cyclic, then refuted, until we hit `max_nodes`.

    Returns gene names in `gene_names` order to keep the node set stable across
    runs regardless of which verdict introduced a node.
    """
    seen = set()
    touched = 0
    for verdict in (CONFIRMED, CYCLIC, REFUTED):
        sub = df[df["verdict"] == verdict]
        for _, r in sub.iterrows():
            for g in (r["cause_i"], r["effect_j"]):
                if g not in seen:
                    seen.add(g)
                    touched += 1
                    if touched >= max_nodes:
                        return [g for g in gene_names if g in seen]
    # Pad with any remaining genes (preserving the canonical ordering) so the
    # graph still has a sensible number of nodes even if few edges were found.
    for g in gene_names:
        if g not in seen:
            seen.add(g)
            touched += 1
            if touched >= max_nodes:
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
    hide_below_freq: float = 0.50,
    label_short: bool = True,
    file_stem: str = "k562_causal_graph",
) -> dict:
    """Render the annotated causal graph (PNG @ 300 DPI + SVG) and a `.gv`
    source mirroring it. Returns a dict of saved paths + the chosen node subset.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    apply_style()
    os.makedirs(out_dir, exist_ok=True)

    subset = _pick_subset_nodes(df, sel.gene_names, max_nodes=max_nodes)
    idx_map = {g: i for i, g in enumerate(sel.gene_names)}
    subset_idx = [idx_map[g] for g in subset]

    def _label(name: str) -> str:
        return name[-5:] if label_short else name

    # Verdict / frequency lookup keyed by (cause, effect).
    verdict_lookup = {
        (r["cause_i"], r["effect_j"]): r for _, r in df.iterrows()
        if r["cause_i"] in idx_map and r["effect_j"] in idx_map
    }

    # Collect every point-estimate edge among the subset nodes. B[ei, ci] is
    # the effect of cause ci on effect ei, i.e. the directed edge ci → ei.
    cand = []  # (cause, effect, b, verdict, freq)
    for ci_name in subset:
        for ei_name in subset:
            if ci_name == ei_name:
                continue
            b = float(fit.B[idx_map[ei_name], idx_map[ci_name]])
            if abs(b) <= edge_threshold:
                continue
            row = verdict_lookup.get((ci_name, ei_name))
            verdict = row["verdict"] if row is not None else "unknown"
            freq = float(row["bootstrap_freq"]) if row is not None else 0.0
            cand.append((ci_name, ei_name, b, verdict, freq))

    b_max = max((abs(c[2]) for c in cand), default=1.0)

    # Drop the lowest-frequency edges entirely; they are mostly resampling
    # noise and only clutter the figure. Their count goes in a footnote.
    shown = [c for c in cand if c[4] >= hide_below_freq]
    n_hidden = len(cand) - len(shown)

    # Split into stable (coloured, solid, bold) and unstable (grey, dashed,
    # thin) so the believable structure stands out from the fragile edges.
    def _width(b):
        # Floor coloured edges at 1.5 so they read as bold; scale up with |b|.
        return max(1.5, 1.5 + 3.0 * (abs(b) / b_max))

    stable, unstable = [], []
    for c in shown:
        (stable if c[4] >= freq_threshold else unstable).append(c)

    # Build the graph for layout (all subset nodes, only the shown edges).
    G = nx.DiGraph()
    G.add_nodes_from(subset)
    for ci_name, ei_name, *_ in shown:
        G.add_edge(ci_name, ei_name)

    # neato (force-directed) spreads a dense graph far better than dot's ranked
    # layout. pydot bridges to the graphviz binary; fall back to a deterministic
    # kamada–kawai layout if either is unavailable.
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog="neato")
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(16, 12))
    node_size = 2400
    nx.draw_networkx_nodes(
        G, pos, node_color="#f7f9fa", edgecolors="#34495e",
        linewidths=1.2, node_size=node_size, ax=ax)
    nx.draw_networkx_labels(
        G, pos, labels={n: _label(n) for n in G.nodes},
        font_size=13, font_family="sans-serif", ax=ax)

    if stable:
        nx.draw_networkx_edges(
            G, pos, edgelist=[(c[0], c[1]) for c in stable],
            edge_color=[EDGE_COLOURS.get(c[3], "#34495e") for c in stable],
            width=[_width(c[2]) for c in stable],
            style="solid", arrows=True, arrowsize=20, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08", node_size=node_size, ax=ax)
        # Effect-size labels only on the stable edges (small font), so the
        # numbers annotate the believable structure without cluttering.
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(c[0], c[1]): f"{c[2]:+.2f}" for c in stable},
            font_size=8, label_pos=0.5, rotate=False,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                      alpha=0.6), ax=ax)
    if unstable:
        nx.draw_networkx_edges(
            G, pos, edgelist=[(c[0], c[1]) for c in unstable],
            edge_color=EDGE_COLOURS["unstable"], width=0.8,
            style="dashed", arrows=True, arrowsize=10, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08", node_size=node_size,
            alpha=0.7, ax=ax)

    # Compact, matplotlib-style legend pinned to the top-right corner.
    handles = [
        Line2D([0], [0], color=EDGE_COLOURS[CONFIRMED], lw=2.5,
               label="confirmed (MW-sig)"),
        Line2D([0], [0], color=EDGE_COLOURS[REFUTED], lw=2.5,
               label="refuted (not MW-sig)"),
        Line2D([0], [0], color=EDGE_COLOURS[CYCLIC], lw=2.5,
               label="cyclic (bidirectional MW-sig)"),
        Line2D([0], [0], color=EDGE_COLOURS["unstable"], lw=1.2, ls="--",
               label=f"unstable (bootstrap < {freq_threshold:.2f})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9,
              frameon=True, framealpha=0.92, borderpad=0.6,
              handlelength=1.6, title="edge colour = verdict",
              title_fontsize=9)

    if n_hidden:
        ax.text(0.01, 0.01,
                f"{n_hidden} edges with bootstrap frequency "
                f"< {hide_below_freq:.2f} hidden",
                transform=ax.transAxes, fontsize=8, color="#666666",
                ha="left", va="bottom")

    ax.set_title("Causal Graph")
    ax.set_axis_off()
    ax.margins(0.08)
    fig.tight_layout()

    png_path = os.path.join(out_dir, file_stem + ".png")
    svg_path = os.path.join(out_dir, file_stem + "_svg.svg")
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)

    gv_src_path = _write_gv_source(
        subset, shown, b_max, out_dir, file_stem,
        freq_threshold=freq_threshold, label_short=label_short)

    return {
        "subset_nodes": subset,
        "subset_idx": subset_idx,
        "gv_source": gv_src_path,
        "png": png_path,
        "svg": svg_path,
        "n_hidden": n_hidden,
    }


def _write_gv_source(
    subset, shown, b_max, out_dir, file_stem, *, freq_threshold, label_short,
) -> str:
    """Write a DOT source mirroring the rendered graph (structure + verdict
    colours). This is a text artefact only; the PNG/SVG are the figures."""
    import graphviz

    g = graphviz.Digraph("causal", engine="neato")
    g.attr(overlap="false", splines="true", label="Causal Graph",
           labelloc="t", fontsize="16", fontname="Helvetica")
    g.attr("node", shape="ellipse", style="filled", fillcolor="#f7f9fa",
           color="#34495e", fontname="Helvetica", fontsize="14")
    for name in subset:
        g.node(name, label=(name[-5:] if label_short else name), comment=name)
    for ci_name, ei_name, b, verdict, freq in shown:
        if freq >= freq_threshold:
            colour = EDGE_COLOURS.get(verdict, "#34495e")
            style, pw = "solid", max(1.5, 1.5 + 3.0 * (abs(b) / b_max))
        else:
            colour, style, pw = EDGE_COLOURS["unstable"], "dashed", 0.8
        g.edge(ci_name, ei_name, color=colour, style=style,
               penwidth=f"{pw:.2f}", label=f"{b:+.2f}", fontsize="8",
               fontcolor=colour)
    gv_src_path = os.path.join(out_dir, file_stem + ".gv")
    with open(gv_src_path, "w") as f:
        f.write(g.source)
    return gv_src_path


def render_wasserstein_vs_freq_scatter(
    df: pd.DataFrame,
    out_dir: str,
    *,
    freq_threshold: float = 0.8,
    edge_threshold: float = 0.01,
    file_stem: str = "k562_w_vs_freq",
) -> str:
    """Secondary figure: Wasserstein vs bootstrap frequency for predicted
    edges. Strong + stable edges sit top-right; weak or unstable ones to the
    bottom or left.
    """
    import matplotlib.pyplot as plt

    apply_style()
    os.makedirs(out_dir, exist_ok=True)
    pred = df[df["lingam_b_ij"].abs() > edge_threshold].copy()
    if pred.empty:
        # Still produce a placeholder so the orchestrator has a stable filename.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no predicted edges", ha="center", va="center")
        ax.set_axis_off()
        out = os.path.join(out_dir, file_stem + ".png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    fig, ax = plt.subplots(figsize=(6, 4.5))
    colour_map = {
        CONFIRMED: EDGE_COLOURS[CONFIRMED],
        REFUTED:   EDGE_COLOURS[REFUTED],
        CYCLIC:    EDGE_COLOURS[CYCLIC],
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
