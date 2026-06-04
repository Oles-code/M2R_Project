"""
run_lingam_analysis
===================
Thin orchestrator for the full LiNGAM + interventional-validation pipeline.

It is deliberately a script (not a function) so the configuration at the top
is the first thing you see when you open the file. Each section delegates to
a single function in `lingam_model`, `validate_edges`, or `visualize_graph`,
so the actual logic lives in modules that a notebook can also import.

Outputs (all under OUTPUT_DIR):

    results.csv          per-ordered-pair validation table
    SUMMARY.md           auto-generated headline numbers + caveats
    *_causal_graph.png   annotated graph (also .gv + .svg)
    *_w_vs_freq.png      Wasserstein-vs-bootstrap-frequency scatter
"""

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR   = "./causalbench_data"
OUTPUT_DIR = "./causalbench_plots"
DATASET    = "weissmann_k562"
NPZ_NAME   = "dataset_k562.npz"

# Gene selection
GENE_K      = None    # post-filter cap; None => take every active-everywhere ∩
                      # has-knockdown-env gene (33 for the default K562 pipeline).

# Fit + bootstrap
N_BOOTSTRAP    = 100
EDGE_THRESHOLD = 0.01    # |b_ij| below this == "no direct edge"
FREQ_THRESHOLD = 0.80    # bootstrap frequency below this => "unstable"

# Validation
BH_ALPHA       = 0.05
MIN_INT_CELLS  = 50

# Visualisation
MAX_NODES_IN_FIG = 15
FILE_STEM        = "k562_causal_graph"
SCATTER_STEM     = "k562_w_vs_freq"

# Reproducibility — all stochastic steps key off this single seed.
SEED = 0


# ─── Imports ────────────────────────────────────────────────────────────────

import os
import time

from causalbench_loader import preprocess, download_raw_data
from lingam_model import select_genes, fit_lingam
from validate_edges import validate_edges
from visualize_graph import render_causal_graph, render_wasserstein_vs_freq_scatter


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz_path = os.path.join(DATA_DIR, NPZ_NAME)

    # Idempotent: download_raw_data is a no-op if the .h5ad is already on disk.
    # Reuses the cached preprocessing .npz if it exists.
    if not os.path.exists(npz_path):
        download_raw_data(DATA_DIR, files=["k562.h5ad"])
        preprocess(DATA_DIR, DATASET)

    print(f"\n{'='*70}\n  Step 1 · Gene selection\n{'='*70}", flush=True)
    t0 = time.time()
    sel = select_genes(npz_path, k=GENE_K, seed=SEED)
    print(f"  ({time.time()-t0:.1f}s) {sel.note}", flush=True)
    print(f"  selected genes: {sel.gene_names}", flush=True)

    print(f"\n{'='*70}\n  Step 2 · DirectLiNGAM fit + bootstrap (n_bootstrap={N_BOOTSTRAP})\n{'='*70}", flush=True)
    t0 = time.time()
    fit = fit_lingam(sel.X_obs, n_bootstrap=N_BOOTSTRAP, seed=SEED, min_effect=EDGE_THRESHOLD)
    n_pred = int(((abs(fit.B) > EDGE_THRESHOLD)).sum())
    n_stable = int(((abs(fit.B) > EDGE_THRESHOLD) & (fit.freq_matrix >= FREQ_THRESHOLD)).sum())
    print(f"  ({time.time()-t0:.1f}s) point estimate B: {n_pred} edges above threshold,"
          f" {n_stable} of them stable (freq ≥ {FREQ_THRESHOLD:.2f}).", flush=True)

    print(f"\n{'='*70}\n  Step 3 · Interventional validation\n{'='*70}", flush=True)
    t0 = time.time()
    df, summary = validate_edges(
        npz_path, fit, sel,
        bh_alpha=BH_ALPHA,
        min_int_cells=MIN_INT_CELLS,
        freq_threshold=FREQ_THRESHOLD,
        edge_threshold=EDGE_THRESHOLD,
        seed=SEED,
    )
    print(f"  ({time.time()-t0:.1f}s) tested {summary.n_pairs_tested} ordered pairs;"
          f" {summary.n_predicted_edges} predicted edges,"
          f" {summary.n_confirmed} confirmed,"
          f" {summary.n_refuted} refuted,"
          f" {summary.n_cyclic} cyclic.", flush=True)
    print(f"  confirmed-edge rate        : {summary.confirmed_rate:.2%}", flush=True)
    print(f"  mean Wasserstein (predicted): {summary.mean_wasserstein_predicted:.4f}", flush=True)
    print(f"  false omission rate (BH)   : {summary.false_omission_rate:.2%}"
          f"  over {summary.n_no_path_pairs} no-path pairs", flush=True)

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  → wrote {csv_path}", flush=True)

    print(f"\n{'='*70}\n  Step 4 · Visualisation\n{'='*70}", flush=True)
    t0 = time.time()
    viz = render_causal_graph(
        fit, sel, df, OUTPUT_DIR,
        max_nodes=MAX_NODES_IN_FIG,
        freq_threshold=FREQ_THRESHOLD,
        edge_threshold=EDGE_THRESHOLD,
        file_stem=FILE_STEM,
    )
    scatter_path = render_wasserstein_vs_freq_scatter(
        df, OUTPUT_DIR,
        freq_threshold=FREQ_THRESHOLD,
        edge_threshold=EDGE_THRESHOLD,
        file_stem=SCATTER_STEM,
    )
    print(f"  ({time.time()-t0:.1f}s) wrote {viz['png']}, {viz['svg']}, "
          f"{viz['gv_source']}, {scatter_path}", flush=True)

    # ─── SUMMARY.md ──────────────────────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "SUMMARY.md")
    write_summary_md(summary_path, sel, fit, summary, viz, scatter_path,
                     csv_path=csv_path, dataset=DATASET, seed=SEED,
                     n_bootstrap=N_BOOTSTRAP, freq_threshold=FREQ_THRESHOLD,
                     edge_threshold=EDGE_THRESHOLD, bh_alpha=BH_ALPHA,
                     min_int_cells=MIN_INT_CELLS)
    print(f"\n  → wrote {summary_path}\n", flush=True)


def write_summary_md(
    path, sel, fit, summary, viz, scatter_path, *,
    csv_path, dataset, seed, n_bootstrap, freq_threshold, edge_threshold,
    bh_alpha, min_int_cells,
):
    """Auto-generate SUMMARY.md with the headline numbers and the explicit
    caveats required by the brief."""
    n_edges = int((abs(fit.B) > edge_threshold).sum())
    n_stable = int(((abs(fit.B) > edge_threshold) & (fit.freq_matrix >= freq_threshold)).sum())

    rel = lambda p: os.path.relpath(p, start=os.path.dirname(path))

    lines = []
    lines.append(f"# LiNGAM + interventional validation — {dataset}\n")
    lines.append(
        "Auto-generated by `run_lingam_analysis.py`. Numbers and figure paths are "
        "ready to lift straight into the report.\n"
    )
    lines.append("## Configuration\n")
    lines.append(f"- dataset            : `{dataset}`")
    lines.append(f"- seed               : `{seed}`")
    lines.append(f"- bootstrap reps     : `{n_bootstrap}`")
    lines.append(f"- edge threshold     : `|b_ij| > {edge_threshold}`")
    lines.append(f"- stability threshold: `bootstrap freq ≥ {freq_threshold}`")
    lines.append(f"- BH α               : `{bh_alpha}`")
    lines.append(f"- min int cells/env  : `{min_int_cells}`\n")

    lines.append("## Gene set\n")
    lines.append(f"- {sel.note}")
    lines.append(f"- p = {len(sel.gene_names)} selected genes (Ensembl IDs):")
    cols = 3
    for k in range(0, len(sel.gene_names), cols):
        line = "  - " + "  ".join(f"`{g}`" for g in sel.gene_names[k:k+cols])
        lines.append(line)
    lines.append("")

    lines.append("## Headline numbers\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| edges in point estimate B (|b| > {edge_threshold}) | {n_edges} |")
    lines.append(f"| stable edges (bootstrap freq ≥ {freq_threshold}) | {n_stable} |")
    lines.append(f"| ordered pairs tested | {summary.n_pairs_tested} |")
    lines.append(f"| predicted edges (powered tests) | {summary.n_predicted_edges} |")
    lines.append(f"| **confirmed** | {summary.n_confirmed} |")
    lines.append(f"| refuted | {summary.n_refuted} |")
    lines.append(f"| cyclic | {summary.n_cyclic} |")
    lines.append(f"| **confirmed-edge rate** | **{summary.confirmed_rate:.2%}** |")
    lines.append(f"| **mean Wasserstein (predicted)** | **{summary.mean_wasserstein_predicted:.4f}** |")
    lines.append(f"| no-path pairs (powered, negative pool) | {summary.n_no_path_pairs} |")
    lines.append(f"| false omissions in that pool | {summary.n_false_omissions} |")
    lines.append(f"| **false omission rate** | **{summary.false_omission_rate:.2%}** |")
    lines.append("")

    lines.append("## Artefacts\n")
    lines.append(f"- per-pair table: [`{rel(csv_path)}`]({rel(csv_path)})")
    lines.append(f"- causal graph (PNG): [`{rel(viz['png'])}`]({rel(viz['png'])})")
    lines.append(f"- causal graph (SVG): [`{rel(viz['svg'])}`]({rel(viz['svg'])})")
    lines.append(f"- causal graph (DOT source): [`{rel(viz['gv_source'])}`]({rel(viz['gv_source'])})")
    lines.append(f"- W vs bootstrap freq: [`{rel(scatter_path)}`]({rel(scatter_path)})\n")

    lines.append("## Caveats (read alongside any quoted number)\n")
    lines.append(
        "- **CRISPRi ≠ deletion**: every knockdown is `do(x_i = low)`, a *soft* "
        "intervention. The direction of the predicted shift is meaningful but "
        "the magnitude is not a calibrated SEM effect size.\n"
    )
    lines.append(
        "- **Out-of-support (Schultheiss §5)**: knocked-down cells often sit "
        "outside the observational support, so the comparison probes an "
        "extrapolation regime. Treat the Wasserstein magnitudes as a ranking, "
        "not a fitted effect.\n"
    )
    lines.append(
        f"- **Multiple testing**: ~p² Mann–Whitney tests were Benjamini–Hochberg "
        f"adjusted at α = {bh_alpha}. The CSV exposes both raw and adjusted p-values.\n"
    )
    lines.append(
        "- **Self-edges**: ordered pairs (i, i) are never tested — knocking out "
        "i trivially shifts i.\n"
    )
    lines.append(
        f"- **Small environments**: pairs with fewer than {min_int_cells} "
        "interventional cells for the cause gene are flagged `underpowered` "
        "and excluded from the confirmed-edge and false-omission rates.\n"
    )
    lines.append(
        f"- **No-direct-edge ≠ no-path**: a zero in B is only a claim of no "
        "*direct* edge. Indirect paths i → k → j would still shift j; the "
        "false-omission test deliberately samples only ordered pairs with no "
        "directed path of any length in the recovered DAG.\n"
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
