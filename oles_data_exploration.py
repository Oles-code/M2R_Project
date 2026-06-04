"""
CausalBench Data Exploration
=============================
Covers the two datasets:
  - weissmann_k562  (K562 cell line, day 6)
  - weissmann_rpe1  (RPE1 cell line, day 7)

Run sections top-to-bottom, or jump to the section you need.
Change DATA_DIR and OUTPUT_DIR to wherever you want data cached / plots saved.

Data loading goes through ./causalbench_loader.py, which works around two
bugs in causalscbench 1.1.x (broken gdown call against a WAF-gated host).
See that module's docstring for the gory details.
"""

# ── 0. Paths – edit these ──────────────────────────────────────────────────
DATA_DIR   = "./causalbench_data"   # where the benchmark caches downloaded + processed data
OUTPUT_DIR = "./causalbench_plots"  # where plots are saved

DATASET    = "weissmann_k562"       # "weissmann_k562" | "weissmann_rpe1"
REGIME     = "full_interventional"  # "observational" | "partial_interventional" | "full_interventional"
SUBSET     = 1.0                    # fraction of training data to load (0.0–1.0); use < 1.0 to iterate faster
PARTIAL_FRACTION = 0.5              # only used when REGIME == "partial_interventional"
# Note: "observational" returns control cells only, so sections 4/5b/6/8
# (which compare perturbed vs. control) degenerate to no-ops.


# ── 1. Imports ──────────────────────────────────────────────────────────────
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from causalbench_loader import download_raw_data, preprocess, load_split

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 2. Load data ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Loading: {DATASET}  |  regime: {REGIME}  |  subset: {SUBSET}")
print('='*60)

# Step 2a — make sure the raw .h5ad for the chosen dataset is on disk.
# (Resumable + size-validated; safe to re-run.)
h5ad_for_dataset = {"weissmann_k562": "k562.h5ad", "weissmann_rpe1": "rpe1.h5ad"}
download_raw_data(DATA_DIR, files=[h5ad_for_dataset[DATASET]])

# Step 2b — preprocess (normalize + log1p + filter rarely-perturbed genes).
# Cached in DATA_DIR as a .npz, so subsequent runs are instant.
print("\nPreprocessing (normalize + log1p + filter rarely-perturbed genes)…")
npz_path = preprocess(DATA_DIR, DATASET)
print(f"  processed dataset → {npz_path}")

# Step 2c — apply the training regime and unpack the three arrays we
# actually want for exploration.
expression_matrix, interventions, gene_names = load_split(
    npz_path,
    regime=REGIME,
    subset_data=SUBSET,
    partial_fraction=PARTIAL_FRACTION,
)

# expression_matrix : np.ndarray  shape (n_cells, n_genes)
# interventions     : list[str]   one entry per cell – the gene that was knocked out,
#                                 "non-targeting" for control cells, or "excluded"
#                                 for cells whose perturbation appears < 100× in raw data
# gene_names        : list[str]   length n_genes

n_cells, n_genes = expression_matrix.shape
print(f"\nexpression_matrix shape : {n_cells:,} cells  ×  {n_genes:,} genes")
print(f"interventions           : {len(interventions):,} entries")
print(f"gene_names              : {n_genes} genes")


# ── 3. Basic summary ─────────────────────────────────────────────────────────
print("\n── 3. Expression matrix summary ──")
expr_df = pd.DataFrame(expression_matrix, columns=gene_names)

print(expr_df.describe().T[["mean", "std", "min", "max"]].head(10).to_string())
print("  ... (showing first 10 genes)")

sparsity = (expression_matrix == 0).mean()
print(f"\nOverall sparsity (fraction of zeros): {sparsity:.1%}")
print(f"Mean expression per cell            : {expression_matrix.mean(axis=1).mean():.3f}")
print(f"Mean expression per gene            : {expression_matrix.mean(axis=0).mean():.3f}")


# ── 4. Intervention distribution ─────────────────────────────────────────────
# "excluded" is a synthetic label that preprocessing assigns to cells whose
# perturbation appears in fewer than 100 cells (see causalscbench preprocessing
# step 4). It typically dominates the count, so we drop it from the plots below
# but report its size here for context.
print("\n── 4. Intervention distribution ──")
intervention_counts = Counter(interventions)
n_unique = len(intervention_counts)
n_control = intervention_counts.get("non-targeting", 0)
n_excluded = intervention_counts.get("excluded", 0)
n_real_pert = n_cells - n_control - n_excluded
print(f"Unique intervention targets (incl. control + 'excluded'): {n_unique}")
print(f"Control cells ('non-targeting')                          : {n_control:,}  ({n_control/n_cells:.1%})")
print(f"Cells with rare perturbation ('excluded' label)          : {n_excluded:,}  ({n_excluded/n_cells:.1%})")
print(f"Cells under a real, kept perturbation                    : {n_real_pert:,}  ({n_real_pert/n_cells:.1%})")

# Counts used for plotting — drop the "excluded" bucket so it doesn't
# dominate the distributions. Section 4's printout keeps it visible.
intervention_counts_plot = Counter({
    k: v for k, v in intervention_counts.items() if k != "excluded"
})

# Top 20 most common interventions (after dropping "excluded")
top20 = intervention_counts_plot.most_common(20)
print("\nTop 20 most common interventions (excluding 'excluded'):")
for name, count in top20:
    print(f"  {name:<25} {count:>6,} cells")


# ── 5. Plots ──────────────────────────────────────────────────────────────────
print("\n── 5. Generating plots ──")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"{DATASET}  –  {REGIME}", fontsize=14, fontweight="bold")


# 5a. Cells per intervention (top 30) — 'excluded' bucket dropped
ax = axes[0, 0]
top30_names  = [x[0] for x in intervention_counts_plot.most_common(30)]
top30_counts = [x[1] for x in intervention_counts_plot.most_common(30)]
colors = ["#e74c3c" if n == "non-targeting" else "#3498db" for n in top30_names]
ax.barh(top30_names[::-1], top30_counts[::-1], color=colors[::-1])
ax.set_xlabel("Number of cells")
ax.set_title("Cells per intervention (top 30)\n  red = control · 'excluded' dropped")
ax.tick_params(axis="y", labelsize=7)


# 5b. Distribution of cells-per-intervention (control + 'excluded' dropped)
ax = axes[0, 1]
per_gene_counts = [
    v for k, v in intervention_counts_plot.items() if k != "non-targeting"
]
if per_gene_counts:
    ax.hist(per_gene_counts, bins=40, color="#2ecc71", edgecolor="white")
    ax.set_xlabel("Cells per perturbed gene")
    ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of cells per perturbed gene\n(control + 'excluded' dropped)")
    ax.axvline(np.median(per_gene_counts), color="black", linestyle="--",
               label=f"Median = {int(np.median(per_gene_counts))}")
    ax.legend()
else:
    ax.text(0.5, 0.5, f"No perturbed cells in regime\n{REGIME!r}",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


# 5c. Per-gene mean expression (log scale)
ax = axes[1, 0]
gene_means = expression_matrix.mean(axis=0)
ax.hist(np.log1p(gene_means), bins=60, color="#9b59b6", edgecolor="white")
ax.set_xlabel("log(1 + mean expression)")
ax.set_ylabel("Number of genes")
ax.set_title("Distribution of mean expression per gene")


# 5d. Per-cell total counts (library size)
ax = axes[1, 1]
cell_totals = expression_matrix.sum(axis=1)
ax.hist(np.log1p(cell_totals), bins=60, color="#e67e22", edgecolor="white")
ax.set_xlabel("log(1 + total counts per cell)")
ax.set_ylabel("Number of cells")
ax.set_title("Library-size distribution (per cell)")

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, f"{DATASET}_{REGIME}_overview.png")
plt.savefig(save_path, dpi=150)
print(f"  Saved overview plot → {save_path}")
plt.show()


# ── 6. Control vs perturbed: mean expression comparison ──────────────────────
print("\n── 6. Control vs perturbed expression ──")

ctrl_mask     = np.array([x == "non-targeting" for x in interventions])
excluded_mask = np.array([x == "excluded"      for x in interventions])
# "perturbed" = real, kept perturbations only — drop the synthetic 'excluded'
# bucket so it doesn't drown the MA plot.
pert_mask = ~ctrl_mask & ~excluded_mask

if not pert_mask.any() or not ctrl_mask.any():
    print(f"  skipped — regime {REGIME!r} has no perturbed and/or no control cells")
else:
    ctrl_mean = expression_matrix[ctrl_mask].mean(axis=0)   # shape (n_genes,)
    pert_mean = expression_matrix[pert_mask].mean(axis=0)

    log2fc = np.log2((pert_mean + 1e-6) / (ctrl_mean + 1e-6))

    top_up   = np.argsort(log2fc)[-10:][::-1]
    top_down = np.argsort(log2fc)[:10]

    print("\nTop 10 genes UP in perturbed vs control:")
    for i in top_up:
        print(f"  {gene_names[i]:<20}  log2FC = {log2fc[i]:+.3f}")

    print("\nTop 10 genes DOWN in perturbed vs control:")
    for i in top_down:
        print(f"  {gene_names[i]:<20}  log2FC = {log2fc[i]:+.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(np.log1p(ctrl_mean), log2fc, s=3, alpha=0.4, color="#3498db")
    threshold = 0.5
    ax.axhline( threshold, color="red",  linestyle="--", linewidth=0.8)
    ax.axhline(-threshold, color="red",  linestyle="--", linewidth=0.8)
    ax.axhline(0,          color="black", linestyle="-",  linewidth=0.5)
    # Annotate top movers
    for i in list(top_up[:5]) + list(top_down[:5]):
        ax.annotate(gene_names[i], (np.log1p(ctrl_mean[i]), log2fc[i]),
                    fontsize=6, color="darkred", ha="center")
    ax.set_xlabel("log(1 + mean expression in control)")
    ax.set_ylabel("log2 fold-change  (perturbed / control)")
    ax.set_title("MA-style plot: perturbed vs control")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{DATASET}_{REGIME}_MA_plot.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n  Saved MA plot → {save_path}")
    plt.show()


# ── 7. Correlation heatmap of a random subset of genes ───────────────────────
print("\n── 7. Gene–gene correlation heatmap (random 40 genes) ──")

rng = random.Random(42)
sample_genes = rng.sample(range(n_genes), min(40, n_genes))
sample_names = [gene_names[i] for i in sample_genes]
sample_expr  = expression_matrix[:, sample_genes]

corr_matrix = np.corrcoef(sample_expr.T)   # shape (40, 40)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    xticklabels=sample_names,
    yticklabels=sample_names,
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    linewidths=0.3, ax=ax
)
ax.set_title(f"Pairwise gene correlations (random 40 genes)\n{DATASET}")
plt.xticks(fontsize=5, rotation=90)
plt.yticks(fontsize=5)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, f"{DATASET}_{REGIME}_gene_corr.png")
plt.savefig(save_path, dpi=150)
print(f"  Saved correlation heatmap → {save_path}")
plt.show()


# ── 8. Quick peek at a specific perturbation ─────────────────────────────────
# Pick the most-common real perturbation (skip "non-targeting" and "excluded").
FOCUS_GENE = next(
    (name for name, _ in top20 if name not in ("non-targeting", "excluded")),
    None,
)

print(f"\n── 8. Spotlight on perturbation: {FOCUS_GENE} ──")
if FOCUS_GENE is None or not ctrl_mask.any():
    print(f"  skipped — regime {REGIME!r} has no real perturbations to spotlight")
else:
    focus_mask = np.array([x == FOCUS_GENE for x in interventions])
    focus_expr = expression_matrix[focus_mask]
    ctrl_expr  = expression_matrix[ctrl_mask]

    # Find the 10 genes most differentially expressed under this perturbation
    diff = focus_expr.mean(axis=0) - ctrl_expr.mean(axis=0)
    top_diff_idx = np.argsort(np.abs(diff))[-10:][::-1]

    print(f"Cells with {FOCUS_GENE} knocked out: {focus_mask.sum()}")
    print(f"\nTop 10 most affected genes:")
    for i in top_diff_idx:
        print(f"  {gene_names[i]:<20}  mean_diff = {diff[i]:+.4f}")


# ── 9. Residual non-Gaussianity check (LiNGAM assumption) ───────────────────
# LiNGAM identifiability rests on the *residual* noise e_i in
#     x_i = sum_{k(j) < k(i)} b_{ij} x_j + e_i
# being non-Gaussian. The marginal distribution of x_i (and the histogram
# of per-gene mean expressions in section 5c) is a different object and
# can look perfectly Gaussian even when the residuals are heavy-tailed,
# or vice versa.
#
# We don't know the causal structure yet, so as a proxy we run pairwise
# linear regressions between the top high-variance genes and check
# whether *those* residuals depart from Gaussianity. If they do, LiNGAM's
# assumption is at least plausible on this data; if they look Gaussian,
# the model's identifiability guarantees collapse.
print("\n── 9. Residual non-Gaussianity check (LiNGAM assumption) ──")

N_GENES_RES = 10  # top high-variance genes to pair up

gene_var_all = expression_matrix.var(axis=0)
top_var_idx  = np.argsort(gene_var_all)[-N_GENES_RES:][::-1]
top_var_names = [gene_names[i] for i in top_var_idx]


def _ols_residuals(x, y):
    """OLS y ~ x with intercept. Returns (residuals, [slope, intercept])."""
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef, coef


pairs = [(i, j) for i in range(N_GENES_RES)
                for j in range(N_GENES_RES) if i != j]
ex_kurts = np.empty(len(pairs))
skews    = np.empty(len(pairs))
sw_pvals = np.empty(len(pairs))

# Shapiro–Wilk is defined for n ∈ [3, 5000]; subsample once and reuse.
n_for_sw = min(5000, expression_matrix.shape[0])
rng_np   = np.random.default_rng(0)
sw_idx   = rng_np.choice(expression_matrix.shape[0], size=n_for_sw, replace=False)

for k, (i, j) in enumerate(pairs):
    x = expression_matrix[:, top_var_idx[i]]
    y = expression_matrix[:, top_var_idx[j]]
    r, _ = _ols_residuals(x, y)
    ex_kurts[k] = stats.kurtosis(r)          # Fisher: 0 = Gaussian
    skews[k]    = stats.skew(r)              # 0 = symmetric
    sw_pvals[k] = stats.shapiro(r[sw_idx]).pvalue

print(f"\nOrdered pairs analyzed       : {len(pairs)}  (top {N_GENES_RES} high-variance genes)")
print(f"Median excess kurtosis       : {np.median(ex_kurts):+.3f}  (Gaussian = 0)")
print(f"Median |skewness| of resids  : {np.median(np.abs(skews)):+.3f}  (Gaussian = 0)")
print(f"Shapiro–Wilk rejections @α=0.05 : {(sw_pvals < 0.05).sum()}/{len(pairs)}  (low p ⇒ non-Gaussian ⇒ good for LiNGAM)")

# Concrete pair to visualise: pick the one whose residuals are CLOSEST
# to Gaussian (smallest |excess kurtosis|). Showing the hardest case
# means a clean rejection here implies the easier pairs are fine too.
rep_idx = int(np.argmin(np.abs(ex_kurts)))
i_rep, j_rep = pairs[rep_idx]
x_rep = expression_matrix[:, top_var_idx[i_rep]]
y_rep = expression_matrix[:, top_var_idx[j_rep]]
r_rep, coef_rep = _ols_residuals(x_rep, y_rep)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(
    f"Pairwise residual non-Gaussianity   ({DATASET}, {REGIME})",
    fontsize=13, fontweight="bold")

# 9a. Scatter + fitted line for the representative pair
ax = axes[0, 0]
plot_sample = rng_np.choice(len(x_rep), size=min(3000, len(x_rep)), replace=False)
ax.scatter(x_rep[plot_sample], y_rep[plot_sample], s=3, alpha=0.3, color="#3498db")
xs = np.linspace(x_rep.min(), x_rep.max(), 100)
ax.plot(xs, coef_rep[0]*xs + coef_rep[1], "r-", linewidth=1.2,
        label=f"y = {coef_rep[0]:+.2f}·x {coef_rep[1]:+.2f}")
ax.set_xlabel(f"{top_var_names[i_rep]}  (x)")
ax.set_ylabel(f"{top_var_names[j_rep]}  (y)")
ax.set_title("Representative pair (hardest to reject Gaussianity)")
ax.legend(loc="upper left")

# 9b. Histogram of residuals + Gaussian overlay (same pair)
ax = axes[0, 1]
ax.hist(r_rep, bins=80, density=True, color="#2ecc71",
        edgecolor="white", alpha=0.85)
xs = np.linspace(r_rep.min(), r_rep.max(), 300)
ax.plot(xs, stats.norm.pdf(xs, r_rep.mean(), r_rep.std()), "r--",
        linewidth=1.5, label=f"N({r_rep.mean():.2f}, {r_rep.std():.2f}²)")
sk = stats.skew(r_rep)
kt = stats.kurtosis(r_rep)
ax.set_xlabel("residual")
ax.set_ylabel("density")
ax.set_title(f"Residual histogram\nskew = {sk:+.2f},  excess kurt = {kt:+.2f}")
ax.legend()

# 9c. Q–Q plot of residuals vs. Normal (same pair)
ax = axes[1, 0]
stats.probplot(r_rep, dist="norm", plot=ax)
ln = ax.get_lines()
ln[0].set_markersize(2.5)
ln[0].set_color("#3498db")
ln[0].set_alpha(0.6)
ax.set_title("Q–Q plot of residuals vs. Normal")

# 9d. Distribution of excess kurtosis across all pairs (the population
# view — one bar in 9a–9c is a single example, this is everyone).
ax = axes[1, 1]
ax.hist(ex_kurts, bins=30, color="#9b59b6", edgecolor="white", alpha=0.85)
ax.axvline(0, color="black", linewidth=0.6, label="Gaussian (kurt = 0)")
ax.axvline(np.median(ex_kurts), color="red", linestyle="--", linewidth=1,
           label=f"median = {np.median(ex_kurts):+.2f}")
ax.set_xlabel("excess kurtosis")
ax.set_ylabel(f"# of {len(pairs)} ordered pairs")
ax.set_title("Excess kurtosis across all pairwise residuals\n(|kurt| ≫ 0 ⇒ non-Gaussian ⇒ good for LiNGAM)")
ax.legend()

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, f"{DATASET}_{REGIME}_residual_nongaussianity.png")
plt.savefig(save_path, dpi=150)
print(f"\n  Saved residual non-Gaussianity plot → {save_path}")
plt.show()


print("\n✓ Exploration complete.")
