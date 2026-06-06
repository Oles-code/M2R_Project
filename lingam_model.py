"""
lingam_model
============
Gene selection + DirectLiNGAM fit + bootstrap stability assessment for the
K562 Perturb-seq data.

The pipeline here implements the *observational* half of the analysis:

  (1) Reduce the ~1.2k preprocessed genes to a tractable working set on which
      DirectLiNGAM is identifiable (n >> p), using the active-everywhere
      cleaning rule from Schultheiss & Bühlmann (2024, §5).

  (2) Fit `DirectLiNGAM(measure='pwling')` to the observational submatrix.
      Bootstrap the fit to get per-edge selection frequencies — this is what
      separates "stable structural edges" from "noisy regression artefacts"
      under finite-sample non-Gaussianity.

The interventional (`full_interventional`) regime is NEVER touched here; the
brief reserves it entirely for validation in `validate_edges.py`.

Public API
----------
    select_genes(npz_path, k=None, seed=0)
        Schultheiss §5 cleaning + (optional) variance top-k. Returns a
        SelectionResult with the observational submatrix, gene indices,
        and the set of perturbation labels that have an interventional env.

    fit_lingam(X, n_bootstrap=100, seed=0, min_effect=0.01)
        DirectLiNGAM fit + bootstrap. Returns a LingamFit with B, A,
        causal order, and bootstrap frequency/IQR matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from causalbench_loader import load_split


# ── selection ────────────────────────────────────────────────────────────────


@dataclass
class SelectionResult:
    """Outcome of the gene-selection step.

    Attributes
    ----------
    selected_idx
        Column indices into the *original* `gene_names` order, in the order
        used for the LiNGAM fit. Length p.
    gene_names
        Ensembl IDs corresponding to `selected_idx`, in the same order. The
        column ordering of `X_obs` and of every B / A / freq matrix that
        follows is keyed on this list.
    X_obs
        Observational expression submatrix (n_obs x p), log1p-normalised,
        already restricted to the selected genes.
    has_intervention_env
        Boolean array of length p. True ⇒ this gene has at least one cell in
        `full_interventional` where it was the knockdown target (i.e. the
        edge "i → ?" is checkable). With the active-everywhere ∩
        knockdown-available selection used by default this is all-True.
    n_genes_original
        Number of gene columns before selection (for the n vs p note).
    n_active_everywhere
        Number that survived the Schultheiss §5 cleaning rule, pre-variance.
    note
        Free-text note on n vs p, printed to stdout in the orchestrator.
    """

    selected_idx: np.ndarray
    gene_names: List[str]
    X_obs: np.ndarray
    has_intervention_env: np.ndarray
    n_genes_original: int
    n_active_everywhere: int
    note: str = ""


def select_genes(
    npz_path: str,
    k: Optional[int] = None,
    seed: int = 0,
) -> SelectionResult:
    """Pick a working gene set for DirectLiNGAM.

    Two filters are applied in order, both motivated by the brief:

    (i)  Active-everywhere cleaning (Schultheiss & Bühlmann 2024, §5):
         keep only genes whose log1p-normalised expression is strictly
         positive in *every* observational cell. Genes that are sometimes
         zero are dominated by the zero-inflation of single-cell data — for
         those, a linear SEM "x_i = sum b_ij x_j + e_i" is implausible and
         the LiNGAM residuals would be dominated by the count-distribution
         spike at zero rather than the structural noise.

    (ii) Knockdown availability: an edge i → j is only validatable if we
         can intervene on i. So drop any "active everywhere" gene that has
         no knockdown environment in `full_interventional`. (On K562 with
         the default pipeline this filter is a no-op — every retained gene
         already has an environment — but we apply it defensively.)

    `k` is the post-filter cap, kept as a knob for the report. If None or
    larger than the survivors, every surviving gene is used. When `k` is
    smaller, the top-k by *observational variance* are kept (high-variance
    genes carry more identifying signal for DirectLiNGAM's pairwise
    independence comparisons).
    """
    # Observational expression submatrix is what we fit on. The
    # interventional split is loaded only to find which genes have a
    # knockdown environment — its expression values are NOT used here.
    X_obs, _, gene_names = load_split(npz_path, regime="observational")
    _, int_full, gene_names_full = load_split(npz_path, regime="full_interventional")

    # The column ordering must match between regimes, otherwise we couldn't
    # use a single set of column indices downstream.
    assert list(gene_names) == list(gene_names_full), (
        "gene_names mismatch between observational and full_interventional "
        "splits — the column ordering is no longer shared and downstream "
        "indexing would silently misalign."
    )
    gene_names = [str(g) for g in gene_names]  # np.str_ → plain str

    n_obs, n_genes_original = X_obs.shape

    # Filter (i): active in every observational cell.
    active_everywhere = (X_obs > 0).all(axis=0)
    n_active = int(active_everywhere.sum())

    # Filter (ii): has a knockdown environment in full_interventional.
    # The two synthetic buckets are not real perturbations and must be dropped.
    env_targets = {str(t) for t in int_full if t not in ("non-targeting", "excluded")}
    has_env = np.array([g in env_targets for g in gene_names], dtype=bool)

    keep_mask = active_everywhere & has_env
    keep_idx = np.where(keep_mask)[0]

    # Order by descending observational variance. Variance is the only
    # ranking we use — it's the cheapest proxy for "carries enough signal
    # for LiNGAM to identify an ordering" and avoids re-introducing
    # interventional information into the selection step.
    var_obs = X_obs[:, keep_idx].var(axis=0)
    order = np.argsort(var_obs)[::-1]
    keep_idx = keep_idx[order]

    if k is not None and k < len(keep_idx):
        keep_idx = keep_idx[:k]

    X_obs_sel = X_obs[:, keep_idx].astype(np.float64, copy=True)
    gene_names_sel = [gene_names[i] for i in keep_idx]
    has_env_sel = has_env[keep_idx]

    p = len(keep_idx)
    note = (
        f"selection: {n_genes_original} → {n_active} active-everywhere "
        f"→ {int(keep_mask.sum())} ∩ has-env → {p} after top-k. "
        f"n_obs = {n_obs}, p = {p}, n/p = {n_obs / max(p, 1):.1f}."
    )

    return SelectionResult(
        selected_idx=keep_idx,
        gene_names=gene_names_sel,
        X_obs=X_obs_sel,
        has_intervention_env=has_env_sel,
        n_genes_original=n_genes_original,
        n_active_everywhere=n_active,
        note=note,
    )


# ── fit + bootstrap ──────────────────────────────────────────────────────────


@dataclass
class LingamFit:
    """Outcome of the DirectLiNGAM fit + bootstrap.

    All matrices are indexed (effect, cause). I.e. B[i, j] is the direct
    linear effect of gene j on gene i, and likewise for A, b_median, etc.

    Attributes
    ----------
    B
        Adjacency matrix from the single full-sample fit. Treat this as the
        *point estimate* of the SEM coefficients.
    causal_order
        Indices into the selected gene set, earliest cause first.
    A
        Total effects, (I − B)^{-1}. Interpretation: a unit perturbation of
        gene j propagates through every path to gene i and produces an
        expected change of A[i, j] (the sum of all directed-path products).
        For knockdown read it as: "if I knock j down by Δ, gene i should
        move by ≈ -|A[i, j]| · Δ in expectation, *assuming* the SEM is
        well-specified and the intervention is hard. With CRISPRi the
        intervention is soft, so this is a directionally-correct guideline,
        not a calibrated effect size."
    freq_matrix
        Per-edge bootstrap selection frequency: fraction of resamples on
        which |B_ij| > min_effect. This is what we threshold on for
        stability: an edge with freq < ~0.8 is fragile to resampling and
        we flag it as "unstable" downstream.
    b_median, b_iqr_low, b_iqr_high
        Per-edge median and IQR (25th, 75th percentile) of B_ij across the
        bootstrap resamples. Useful to read alongside `freq_matrix`: a
        high-frequency edge with a tight IQR is the safest claim.
    n_bootstrap
        Number of bootstrap resamples actually run.
    bootstrap_result
        Raw `lingam.bootstrap.BootstrapResult`, kept in case the orchestrator
        wants additional summaries (paths, DAG counts, etc.).
    """

    B: np.ndarray
    causal_order: np.ndarray
    A: np.ndarray
    freq_matrix: np.ndarray
    b_median: np.ndarray
    b_iqr_low: np.ndarray
    b_iqr_high: np.ndarray
    n_bootstrap: int
    bootstrap_result: object = field(repr=False, default=None)


def fit_lingam(
    X: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 0,
    min_effect: float = 0.01,
) -> LingamFit:
    """Fit DirectLiNGAM with the pairwise tanh independence measure and bootstrap.

    Why `measure='pwling'`?
        The `lingam` library exposes two pairwise independence criteria. The
        default mutual-information option is consistent but expensive. The
        `pwling` measure is the pairwise tanh nonlinear-correlation statistic
        from Shimizu (2012, §3), which is what the high-dimensional LiNGAM
        recipe explicitly recommends: it uses the non-Gaussianity of the
        residuals via a tanh nonlinearity (rather than a Gaussian-friendly
        covariance), so it actually *exploits* the heavy tails the EDA
        confirmed are present in this data. That non-Gaussianity is also
        what gives DirectLiNGAM a *unique* causal ordering — covariance-only
        methods can only ever pin down a Markov equivalence class.

    Why bootstrap?
        With ~30 genes and ~8.5k observational cells we are comfortably in
        the n >> p regime, but the residual heavy-tails mean a small handful
        of outlier cells can flip the sign of a marginal regression. The
        bootstrap absorbs that: any edge whose presence depends on a
        specific resample falls below the frequency threshold and gets
        flagged. The full-sample B is the point estimate; freq_matrix is
        the confidence ribbon.

    Parameters
    ----------
    X
        Observational expression submatrix, shape (n_obs, p). Standardisation
        is not required — DirectLiNGAM is scale-equivariant under affine
        transformations of the columns.
    n_bootstrap
        Number of bootstrap resamples. 100 is the conventional default in
        the lingam library; the bootstrap is O(n_bootstrap × n × p^3) so
        push this up only for the final report run.
    seed
        Reproducibility. Surfaced so the orchestrator can record it.
    min_effect
        Coefficients with |B_ij| < min_effect on a given bootstrap resample
        do not count toward that edge's frequency. 0.01 is the library
        default and corresponds to "structurally absent" in log1p-normalised
        expression units.
    """
    import lingam

    p = X.shape[1]
    model = lingam.DirectLiNGAM(random_state=seed, measure="pwling")

    # The point estimate. `B` is what we report as "the recovered structure";
    # everything else is uncertainty around it.
    model.fit(X)
    B = np.array(model.adjacency_matrix_, copy=True)
    causal_order = np.array(model.causal_order_, copy=True)

    # Total effects. (I - B) is lower-triangular under the recovered order
    # so the inverse always exists. For LiNGAM identifiability B must be
    # acyclic (so I - B is non-singular); we still guard with try/except
    # in case a particularly pathological B slips through.
    try:
        A = np.linalg.inv(np.eye(p) - B)
    except np.linalg.LinAlgError:
        A = np.full_like(B, np.nan)

    # Bootstrap. We deliberately do NOT use lingam.bootstrap here.
    # Why: lingam 1.12.2's `BootstrapMixin.bootstrap` not only refits per
    # resample but also calls `calculate_total_effect` for every pair
    # (cause, effect) in the recovered order. That helper internally calls
    # `find_all_paths`, which is exponential in the number of edges. On
    # K562 the recovered B is dense enough that one bootstrap iteration
    # took >30 seconds (vs ~6 s for the actual fit). The total-effects
    # estimate isn't useful to us — we report A = (I - B)^-1 from the
    # point estimate, and we only need the per-edge frequency + IQR of
    # B_ij from the bootstrap. So we run our own resampling loop.
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    stacked = np.zeros((n_bootstrap, p, p), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)              # standard non-parametric bootstrap
        bs_model = lingam.DirectLiNGAM(random_state=seed + 1 + b, measure="pwling")
        bs_model.fit(X[idx])
        stacked[b] = bs_model.adjacency_matrix_
    # Keep a lightweight stand-in so callers that want the raw matrices have access
    # but we never expose the heavyweight BootstrapResult.
    bs_result = stacked

    '''check this bootsrap code is all ok'''

    # Per-edge frequency = fraction of resamples where the coefficient
    # exceeded min_effect in magnitude.
    freq_matrix = (np.abs(stacked) > min_effect).mean(axis=0)

    # Per-edge median + IQR across resamples (includes the zero resamples
    # so the IQR reflects how often LiNGAM picks the edge as well as the
    # spread of the picked values).
    b_median = np.median(stacked, axis=0)
    b_iqr_low = np.percentile(stacked, 25, axis=0)
    b_iqr_high = np.percentile(stacked, 75, axis=0)

    return LingamFit(
        B=B,
        causal_order=causal_order,
        A=A,
        freq_matrix=freq_matrix,
        b_median=b_median,
        b_iqr_low=b_iqr_low,
        b_iqr_high=b_iqr_high,
        n_bootstrap=n_bootstrap,
        bootstrap_result=bs_result,
    )


# ── reachability helper (used by validate_edges.py too) ──────────────────────


def reachability(B: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Return the p × p boolean matrix `R[i, j]` = "there is a directed path j ⇝ i".

    Uses the convention of `B` in this module: B[i, j] ≠ 0 iff there is a
    direct edge j → i. The "has a path" question matters for the false-
    omission test in `validate_edges`: a zero in B is only a claim of no
    *direct* edge — an indirect path j → k → i would still propagate an
    interventional shift, so the negative-edge test must exclude such pairs.

    `eps` is the same threshold idea as `min_effect` in `fit_lingam` —
    edges with |B_ij| ≤ eps are treated as absent. With eps=0 we use B's
    exact zero/non-zero pattern.
    """
    p = B.shape[0]
    adj = (np.abs(B) > eps)  # adj[i, j] = direct edge j → i
    # Warshall-style transitive closure. p ≤ ~50 so the cubic cost is fine.
    R = adj.copy()
    for k in range(p):
        # path ... → k → ... : reach[i, j] |= reach[i, k] & reach[k, j]
        R |= R[:, [k]] & R[[k], :]
    # A node is trivially reachable from itself only along a cycle; for the
    # negative-edge test we want "no path *at all*", so we exclude the
    # diagonal — self-pairs are handled separately upstream (never tested).
    np.fill_diagonal(R, False)
    return R
