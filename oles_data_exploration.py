"""
CausalBench Data Exploration
=============================
Exploratory checks for the LiNGAM pipeline on the K562 Perturb-seq data.

This script is deliberately scoped to the two figures the report actually
uses, both computed on the *same* observational submatrix that DirectLiNGAM
is fit on (the 33 active-everywhere genes from `lingam_model.select_genes`):

  1. Gene–gene correlation matrix.
       Motivates the modelling choice: linear correlations alone leave the
       causal direction unidentified, which is exactly what LiNGAM resolves
       using residual non-Gaussianity.

  2. Residual non-Gaussianity check (the LiNGAM identifiability assumption).
       LiNGAM is only identifiable if the structural residuals e_i in
       x_i = sum_j b_ij x_j + e_i are non-Gaussian. We don't know the true
       structure yet, so as a proxy we run pairwise OLS regressions between
       the selected genes and test whether *those* residuals depart from
       Gaussianity (excess kurtosis, skewness, Shapiro–Wilk).

Data loading goes through `causalbench_loader.py` (cached .npz; nothing is
re-downloaded). Gene selection is imported from `lingam_model` so the figures
describe the identical gene set used downstream.

Run:  python oles_data_exploration.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from causalbench_loader import download_raw_data, preprocess
from lingam_model import select_genes
from plot_style import apply_style

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = "./causalbench_data"
OUTPUT_DIR = "./causalbench_plots"
DATASET    = "weissmann_k562"
NPZ_NAME   = "dataset_k562.npz"

# Every stochastic step keys off this single seed so the figures are
# reproducible across runs and machines.
SEED = 0


def _ols_residuals(x: np.ndarray, y: np.ndarray):
    """OLS y ~ x with intercept. Returns (residuals, [slope, intercept])."""
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef, coef


def main() -> None:
    apply_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npz_path = os.path.join(DATA_DIR, NPZ_NAME)

    # Idempotent: no-op if the cached .npz already exists.
    if not os.path.exists(npz_path):
        download_raw_data(DATA_DIR, files=["k562.h5ad"])
        preprocess(DATA_DIR, DATASET)

    # The selected genes ARE the LiNGAM input — both figures below describe
    # exactly this submatrix, so the EDA and the model agree on their object.
    sel = select_genes(npz_path, k=None, seed=SEED)
    X = sel.X_obs                      # (n_obs, p), log1p-normalised observational
    names = sel.gene_names
    n_obs, p = X.shape
    # Last-5 of the Ensembl ID keeps 33 axis labels legible; full IDs are in
    # the gene set printed by the pipeline / SUMMARY.
    short = [g[-5:] for g in names]

    print(f"Selected gene set: p = {p} genes, n_obs = {n_obs} cells")
    print(f"  {sel.note}")

    # ── 1. Gene–gene correlation matrix ──────────────────────────────────────
    corr = np.corrcoef(X.T)            # (p, p) Pearson correlations

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(p)); ax.set_yticks(range(p))
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.grid(False)                     # no gridlines over a heatmap
    ax.set_title("Gene–Gene Correlation Matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", fontsize=9)
    cbar.outline.set_visible(False)
    corr_path = os.path.join(OUTPUT_DIR, "k562_gene_correlation.png")
    fig.savefig(corr_path)
    plt.close(fig)
    print(f"  saved correlation matrix → {corr_path}")

    # ── 2. Residual non-Gaussianity check ────────────────────────────────────
    # Every ordered pair of selected genes: regress one on the other, inspect
    # the residual. Aggregating over all pairs gives the population view; a
    # single representative pair gives the per-residual diagnostic panels.
    pairs = [(i, j) for i in range(p) for j in range(p) if i != j]
    ex_kurts = np.empty(len(pairs))
    skews    = np.empty(len(pairs))

    # Shapiro–Wilk is only defined for n ≤ 5000; subsample once and reuse the
    # same cells for every pair so the test is comparable across pairs.
    rng = np.random.default_rng(SEED)
    n_sw = min(5000, n_obs)
    sw_idx = rng.choice(n_obs, size=n_sw, replace=False)
    sw_pvals = np.empty(len(pairs))

    for k, (i, j) in enumerate(pairs):
        r, _ = _ols_residuals(X[:, i], X[:, j])
        ex_kurts[k] = stats.kurtosis(r)        # Fisher: 0 == Gaussian
        skews[k]    = stats.skew(r)
        sw_pvals[k] = stats.shapiro(r[sw_idx]).pvalue

    n_reject = int((sw_pvals < 0.05).sum())
    print(f"  median excess kurtosis     : {np.median(ex_kurts):+.3f}  (Gaussian = 0)")
    print(f"  median |skewness|          : {np.median(np.abs(skews)):.3f}")
    print(f"  Shapiro rejections @0.05   : {n_reject}/{len(pairs)} "
          f"(low p ⇒ non-Gaussian ⇒ good for LiNGAM)")

    # Representative pair = residuals CLOSEST to Gaussian (smallest |excess
    # kurtosis|). Showing the hardest-to-reject case means the easier pairs
    # are non-Gaussian a fortiori.
    rep = int(np.argmin(np.abs(ex_kurts)))
    i_rep, j_rep = pairs[rep]
    x_rep, y_rep = X[:, i_rep], X[:, j_rep]
    r_rep, coef_rep = _ols_residuals(x_rep, y_rep)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # 2a. Scatter + fitted line for the representative pair.
    ax = axes[0, 0]
    samp = rng.choice(n_obs, size=min(3000, n_obs), replace=False)
    ax.scatter(x_rep[samp], y_rep[samp], s=4, alpha=0.3, color="#0072B2")
    xs = np.linspace(x_rep.min(), x_rep.max(), 100)
    ax.plot(xs, coef_rep[0] * xs + coef_rep[1], color="#D55E00", lw=1.4,
            label=f"OLS fit (slope {coef_rep[0]:+.2f})")
    ax.set_xlabel(short[i_rep]); ax.set_ylabel(short[j_rep])
    ax.set_title("Representative pair")
    ax.legend(loc="upper left")

    # 2b. Residual histogram + matched Gaussian.
    ax = axes[0, 1]
    ax.hist(r_rep, bins=80, density=True, color="#009E73",
            edgecolor="white", alpha=0.85)
    xs = np.linspace(r_rep.min(), r_rep.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, r_rep.mean(), r_rep.std()),
            color="#D55E00", ls="--", lw=1.5, label="matched Normal")
    ax.set_xlabel("residual"); ax.set_ylabel("density")
    ax.set_title(f"Residual distribution "
                 f"(skew {stats.skew(r_rep):+.2f}, "
                 f"ex. kurt {stats.kurtosis(r_rep):+.2f})")
    ax.legend()

    # 2c. Q–Q plot vs Normal.
    ax = axes[1, 0]
    stats.probplot(r_rep, dist="norm", plot=ax)
    pts = ax.get_lines()
    pts[0].set_markersize(2.5); pts[0].set_color("#0072B2"); pts[0].set_alpha(0.6)
    pts[1].set_color("#D55E00")
    ax.set_title("Normal Q–Q plot")

    # 2d. Population view: excess kurtosis across every pairwise residual.
    ax = axes[1, 1]
    ax.hist(ex_kurts, bins=30, color="#CC79A7", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#333333", lw=0.8, label="Gaussian (0)")
    ax.axvline(np.median(ex_kurts), color="#D55E00", ls="--", lw=1.2,
               label=f"median {np.median(ex_kurts):+.2f}")
    ax.set_xlabel("excess kurtosis"); ax.set_ylabel(f"# of {len(pairs)} pairs")
    ax.set_title("Residual excess kurtosis (all pairs)")
    ax.legend()

    fig.tight_layout()
    ng_path = os.path.join(OUTPUT_DIR, "k562_residual_nongaussianity.png")
    fig.savefig(ng_path)
    plt.close(fig)
    print(f"  saved non-Gaussianity check → {ng_path}")
    print("✓ Exploration complete.")


if __name__ == "__main__":
    main()
