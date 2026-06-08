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

    # ─── validation_report.md (the comprehensive, report-ready version) ──────
    report_path = os.path.join(OUTPUT_DIR, "validation_report.md")
    write_validation_report(report_path, sel, df, summary, dataset=DATASET)
    print(f"  → wrote {report_path}\n", flush=True)


# ── verdict metadata for the structured report ───────────────────────────────

# Ordered as in the brief. Each entry: (verdict key, display name, blurb).
VERDICT_INFO = [
    ("confirmed", "Confirmed",
     "Predicted edges (|b| > threshold) whose forward direction is "
     "Mann–Whitney significant after BH correction, and whose reverse "
     "direction is not. The interventional data agrees with both the existence "
     "and orientation of the edge — these are the model's successes."),
    ("refuted", "Refuted",
     "Predicted edges whose forward Mann–Whitney test is NOT significant after "
     "BH correction. The SEM coefficient is non-zero but knocking down the "
     "cause produces no detectable shift in the effect, so the interventional "
     "data does not support the edge."),
    ("cyclic", "Cyclic",
     "Predicted edges that are forward-significant but whose reverse direction "
     "is ALSO BH-significant. Both i→j and j→i shift each other's "
     "distributions — a signature of feedback or a hidden common cause that a "
     "single acyclic ordering cannot represent (see the methodology note)."),
    ("false_omission", "False omission",
     "Ordered pairs with NO predicted directed path, yet knocking down the "
     "cause DOES produce a BH-significant shift. These are the negative-edge "
     "failures: a real interventional effect the recovered DAG missed."),
    ("silent_negative", "Silent negative",
     "Ordered pairs with no predicted directed path and no significant "
     "interventional shift — the model correctly predicts the absence of an "
     "effect. These are the true negatives."),
    ("indirect_path", "Indirect path",
     "Pairs with no direct edge but a directed path of length ≥ 2 in the "
     "recovered DAG. LiNGAM makes no direct-edge claim here, so they are "
     "reported for transparency rather than scored as positive or negative."),
]


def _p(x):
    """Format a p-value compactly; em-dash for NaN."""
    import math
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if x == 0:
        return "0"
    return f"{x:.1e}"


def _verdict_table(sub, max_rows=30):
    """Render a per-verdict edge table sorted by Wasserstein (desc)."""
    import numpy as np
    sub = sub.sort_values("wasserstein", ascending=False)
    n_total = len(sub)
    shown = sub.head(max_rows)
    lines = [
        "| source | target | b_ij | boot freq | fwd MW p (BH) | "
        "rev MW p (BH) | Wasserstein |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in shown.iterrows():
        w = r["wasserstein"]
        w_str = "—" if (isinstance(w, float) and np.isnan(w)) else f"{w:.4f}"
        lines.append(
            f"| `{r['cause_i'][-5:]}` | `{r['effect_j'][-5:]}` "
            f"| {r['lingam_b_ij']:+.3f} | {r['bootstrap_freq']:.2f} "
            f"| {_p(r['mw_p_bh'])} | {_p(r['mw_p_bh_rev'])} | {w_str} |")
    note = ""
    if n_total > max_rows:
        note = (f"\n*Showing top {max_rows} of {n_total} by Wasserstein; "
                f"the full list is in `results.csv`.*\n")
    return "\n".join(lines) + "\n" + note


def write_validation_report(path, sel, df, summary, *, dataset):
    """Write the comprehensive six-verdict validation report."""
    n_genes = len(sel.gene_names)
    n_obs = sel.X_obs.shape[0]
    n_pairs = len(df)
    counts = {k: int((df["verdict"] == k).sum()) for k, _, _ in VERDICT_INFO}

    confirmed = df[df["verdict"] == "confirmed"]
    mean_w_conf = float(confirmed["wasserstein"].mean()) if len(confirmed) else float("nan")
    med_freq_conf = float(confirmed["bootstrap_freq"].median()) if len(confirmed) else float("nan")

    def pct(n):
        return f"{n} ({n / max(n_pairs, 1):.1%})"

    L = []
    L.append(f"# Validation report — {dataset}\n")
    L.append(
        "Auto-generated by `run_lingam_analysis.py`. Every ordered gene pair "
        "(i ≠ j) from the recovered DirectLiNGAM structure is validated against "
        "the held-out interventional (CRISPRi) data and assigned one of six "
        "verdicts. Percentages below are of the "
        f"{n_pairs} tested ordered pairs unless stated otherwise. Gene labels "
        "are the last five characters of the Ensembl ID; the full IDs are in "
        "`SUMMARY.md` and `results.csv`.\n")

    # 1. Headline table
    L.append("## Headline summary\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Genes in model | {n_genes} |")
    L.append(f"| Observational cells | {n_obs:,} |")
    L.append(f"| Predicted edges (powered) | {summary.n_predicted_edges} |")
    L.append(f"| Confirmed | {pct(counts['confirmed'])} |")
    L.append(f"| Refuted | {pct(counts['refuted'])} |")
    L.append(f"| Cyclic | {pct(counts['cyclic'])} |")
    L.append(f"| False omission | {pct(counts['false_omission'])} |")
    L.append(f"| Silent negative | {pct(counts['silent_negative'])} |")
    L.append(f"| Indirect path | {pct(counts['indirect_path'])} |")
    L.append(f"| Confirmed-edge rate (of predicted) | {summary.confirmed_rate:.2%} |")
    L.append(f"| Mean Wasserstein (confirmed) | {mean_w_conf:.4f} |")
    L.append(f"| Median bootstrap frequency (confirmed) | {med_freq_conf:.2f} |")
    L.append(f"| False omission rate (FOR) | {summary.false_omission_rate:.2%} |")
    L.append("")

    # 2. Per-verdict breakdown
    L.append("## Per-verdict breakdown\n")
    for key, name, blurb in VERDICT_INFO:
        sub = df[df["verdict"] == key]
        L.append(f"### {name} ({len(sub)})\n")
        L.append(blurb + "\n")
        if len(sub):
            L.append(_verdict_table(sub))
        else:
            L.append("*No pairs in this category.*\n")

    # 3. Top edges
    L.append("## Top 10 confirmed edges by Wasserstein distance\n")
    top = confirmed.sort_values("wasserstein", ascending=False).head(10)
    if len(top):
        L.append(_verdict_table(top, max_rows=10))
    else:
        L.append("*No confirmed edges.*\n")

    # 4. Methodology note
    L.append("## Methodology note — what \"cyclic\" means here\n")
    L.append(
        "DirectLiNGAM always returns a directed **acyclic** graph; the recovered "
        "structure cannot contain a cycle by construction. The `cyclic` verdict "
        "is therefore *not* a violation of the model's output. It is a property "
        "of the biological system revealed by the interventional validation: "
        "when knocking down i shifts j AND knocking down j shifts i, the "
        "data carry bidirectional significance that a single acyclic ordering "
        "cannot represent. This is the expected signature of a feedback loop or "
        "a hidden common cause. The acyclicity constraint forces LiNGAM to pick "
        "one direction; the validation recovers the bidirectionality the model "
        "had to discard. See `bootstrap_summary.md` for a complementary "
        "bootstrap-instability diagnostic of the same phenomenon.\n")

    with open(path, "w") as f:
        f.write("\n".join(L))


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
