"""
threshold_sensitivity
=====================
Read-only sensitivity analysis of the interventional verdicts to the
edge-presence cutoff `|b_ij| > EDGE_THRESHOLD`.

The main pipeline (`run_lingam_analysis.py`) fixes `EDGE_THRESHOLD = 0.01`,
which is simply the `lingam` library default and is never swept. This script
re-runs the interventional validation at several thresholds so the report can
state how sensitive the headline verdicts are to that one arbitrary number.

It reuses the existing pipeline verbatim — the same config constants, the same
idempotent data-prep, and `select_genes` / `fit_lingam` / `validate_edges`. The
point estimate B and the bootstrap do NOT depend on the threshold, so the fit is
run ONCE; only `validate_edges` is re-run per threshold (it is the threshold,
via `edge_threshold`, that decides which |b_ij| count as predicted edges).

Outputs (all under OUTPUT_DIR):

    threshold_sensitivity.csv   per-threshold breakdown, one row per threshold
    threshold_sensitivity.tex   booktabs summary table for the report
                                (also printed to stdout)

This script writes only those two new files; it modifies nothing existing.
"""

# ─── Configuration (mirrors run_lingam_analysis.py) ──────────────────────────

DATA_DIR   = "./causalbench_data"
OUTPUT_DIR = "./causalbench_plots"
DATASET    = "weissmann_k562"
NPZ_NAME   = "dataset_k562.npz"

# Gene selection
GENE_K = None    # post-filter cap; None => take every active-everywhere ∩
                 # has-knockdown-env gene.

# Fit + bootstrap
N_BOOTSTRAP    = 100
FREQ_THRESHOLD = 0.80    # bootstrap frequency below this => "unstable"

# Validation
BH_ALPHA      = 0.05
MIN_INT_CELLS = 50

# Reproducibility — all stochastic steps key off this single seed.
SEED = 0

# The sweep itself. The default pipeline uses 0.01; we bracket it.
THRESHOLDS = [0.005, 0.01, 0.02, 0.05]


# ─── Imports ─────────────────────────────────────────────────────────────────

import os
import time

import pandas as pd

from causalbench_loader import preprocess, download_raw_data
from lingam_model import select_genes, fit_lingam
from validate_edges import validate_edges


# Verdict labels in a fixed display order, so the per-threshold value_counts
# line up into stable CSV columns regardless of which appear at a threshold.
VERDICT_ORDER = [
    "confirmed", "refuted", "cyclic", "false_omission",
    "silent_negative", "indirect_path", "underpowered", "unverifiable",
]


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz_path = os.path.join(DATA_DIR, NPZ_NAME)

    # Idempotent: reuse the cached preprocessing .npz if it exists; only
    # download + preprocess when it is missing (matches the orchestrator).
    if not os.path.exists(npz_path):
        download_raw_data(DATA_DIR, files=["k562.h5ad"])
        preprocess(DATA_DIR, DATASET)

    print(f"\n{'='*70}\n  Step 1 · Gene selection (once)\n{'='*70}", flush=True)
    t0 = time.time()
    sel = select_genes(npz_path, k=GENE_K, seed=SEED)
    print(f"  ({time.time()-t0:.1f}s) {sel.note}", flush=True)

    # The point estimate B and the bootstrap are threshold-independent, so we
    # fit ONCE. min_effect is pinned to the smallest swept threshold purely so
    # the bootstrap frequency annotation is consistent across the sweep — the
    # frequency only drives the (unused-here) is_stable flag, never a verdict.
    min_thr = min(THRESHOLDS)
    print(f"\n{'='*70}\n  Step 2 · DirectLiNGAM fit + bootstrap "
          f"(n_bootstrap={N_BOOTSTRAP}, min_effect={min_thr})\n{'='*70}", flush=True)
    t0 = time.time()
    fit = fit_lingam(sel.X_obs, n_bootstrap=N_BOOTSTRAP, seed=SEED, min_effect=min_thr)
    print(f"  ({time.time()-t0:.1f}s) fit complete; sweeping {len(THRESHOLDS)} "
          f"thresholds without refitting.", flush=True)

    print(f"\n{'='*70}\n  Step 3 · Validation across thresholds\n{'='*70}", flush=True)
    records = []
    for thr in THRESHOLDS:
        t0 = time.time()
        df, summary = validate_edges(
            npz_path, fit, sel,
            bh_alpha=BH_ALPHA,
            min_int_cells=MIN_INT_CELLS,
            freq_threshold=FREQ_THRESHOLD,
            edge_threshold=thr,
            seed=SEED,
        )
        vc = df["verdict"].value_counts().to_dict()
        silent = summary.n_no_path_pairs - summary.n_false_omissions

        rec = {
            "edge_threshold": thr,
            "n_predicted_edges": summary.n_predicted_edges,
            "n_confirmed": summary.n_confirmed,
            "n_refuted": summary.n_refuted,
            "n_cyclic": summary.n_cyclic,
            "confirmed_rate": summary.confirmed_rate,
            "mean_wasserstein_predicted": summary.mean_wasserstein_predicted,
            "n_no_path_pairs": summary.n_no_path_pairs,
            "n_false_omissions": summary.n_false_omissions,
            "false_omission_rate": summary.false_omission_rate,
            "silent": silent,
            "n_pairs_tested": summary.n_pairs_tested,
        }
        # Full verdict value_counts as their own columns (0 if absent here).
        for v in VERDICT_ORDER:
            rec[f"verdict_{v}"] = int(vc.get(v, 0))
        # Any verdict not in the known order (defensive — should not happen).
        for v, n in vc.items():
            if v not in VERDICT_ORDER:
                rec[f"verdict_{v}"] = int(n)
        records.append(rec)

        print(f"  ({time.time()-t0:.1f}s) thr={thr:<6} "
              f"predicted={summary.n_predicted_edges:<3} "
              f"confirmed={summary.n_confirmed:<3} "
              f"rate={summary.confirmed_rate:6.2%} "
              f"meanW={summary.mean_wasserstein_predicted:.4f} "
              f"FOR={summary.false_omission_rate:6.2%}", flush=True)

    out = pd.DataFrame(records)

    # ─── CSV (full per-threshold breakdown) ──────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "threshold_sensitivity.csv")
    out.to_csv(csv_path, index=False)

    # ─── stdout summary table ────────────────────────────────────────────────
    _print_summary(out)

    # ─── LaTeX booktabs table ────────────────────────────────────────────────
    tex_path = os.path.join(OUTPUT_DIR, "threshold_sensitivity.tex")
    tex = _booktabs_table(out)
    with open(tex_path, "w") as f:
        f.write(tex)

    print(f"\n  → wrote {csv_path}", flush=True)
    print(f"  → wrote {tex_path}\n", flush=True)


def _print_summary(out: pd.DataFrame) -> None:
    """Print the per-threshold summary table to stdout, plus a per-threshold
    verdict-count line and a sanity check that the counts sum to n_pairs_tested
    (some pairs get marked underpowered/unverifiable, so they will not all be
    confirmed/refuted/cyclic)."""
    print(f"\n{'='*70}\n  Threshold sensitivity summary\n{'='*70}", flush=True)
    header = (f"  {'thr':>6} {'pred':>5} {'conf':>5} {'refu':>5} {'cyc':>5} "
              f"{'conf%':>7} {'meanW':>8} {'no_path':>8} {'FO':>4} "
              f"{'silent':>7} {'FOR%':>7}")
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for _, r in out.iterrows():
        print(f"  {r['edge_threshold']:>6.3f} "
              f"{int(r['n_predicted_edges']):>5} "
              f"{int(r['n_confirmed']):>5} "
              f"{int(r['n_refuted']):>5} "
              f"{int(r['n_cyclic']):>5} "
              f"{r['confirmed_rate']:>7.2%} "
              f"{r['mean_wasserstein_predicted']:>8.4f} "
              f"{int(r['n_no_path_pairs']):>8} "
              f"{int(r['n_false_omissions']):>4} "
              f"{int(r['silent']):>7} "
              f"{r['false_omission_rate']:>7.2%}", flush=True)

    # Per-threshold full verdict breakdown + sum sanity check.
    verdict_cols = [c for c in out.columns if c.startswith("verdict_")]
    print(f"\n  Full verdict counts per threshold "
          f"(should sum to n_pairs_tested):", flush=True)
    for _, r in out.iterrows():
        parts = [f"{c.replace('verdict_', '')}={int(r[c])}"
                 for c in verdict_cols if int(r[c]) > 0]
        total = sum(int(r[c]) for c in verdict_cols)
        flag = "" if total == int(r["n_pairs_tested"]) else "  <-- MISMATCH"
        print(f"  thr={r['edge_threshold']:.3f}: " + ", ".join(parts)
              + f"  [sum={total}, tested={int(r['n_pairs_tested'])}]{flag}",
              flush=True)


def _booktabs_table(out: pd.DataFrame) -> str:
    """Render the report-ready booktabs table.

    Columns: Threshold, Predicted edges, Confirmed, Confirmed rate (%),
    Mean W (predicted), FOR (%). Requires \\usepackage{booktabs}.
    """
    L = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        (r"Threshold & Predicted edges & Confirmed & Confirmed rate (\%) & "
         r"Mean W (predicted) & FOR (\%) \\"),
        r"\midrule",
    ]
    for _, r in out.iterrows():
        L.append(
            f"{r['edge_threshold']:.3f} & "
            f"{int(r['n_predicted_edges'])} & "
            f"{int(r['n_confirmed'])} & "
            f"{r['confirmed_rate'] * 100:.1f} & "
            f"{r['mean_wasserstein_predicted']:.4f} & "
            f"{r['false_omission_rate'] * 100:.1f} \\\\"
        )
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
