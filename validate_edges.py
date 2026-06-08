"""
validate_edges
==============
Interventional validation of a recovered LiNGAM structure against the
full-interventional regime.

Validation principle (Schultheiss & Bühlmann 2024, §5 + CausalBench
mean-Wasserstein / false-omission-rate definitions, Chevalley et al. 2025):

    If i → j is a true causal edge, knocking down gene i should shift the
    distribution of gene j relative to its observational distribution.

    Symmetrically, if there is NO directed path i ⇝ j, knocking down i
    should leave j's distribution unchanged.

We make this operational with two distribution-shift tests on each ordered
pair (i, j):

    • Mann–Whitney U two-sided p-value
          The qualitative "did the distribution shift at all" test —
          robust to the heavy tails this data has, and what Schultheiss §5
          uses. After Benjamini–Hochberg correction this becomes the
          significance verdict.

    • 1-D Wasserstein distance
          The quantitative "how big is the shift" measure used by
          CausalBench. Scales with effect size in the units of the data
          (log1p-normalised expression), so the *mean* Wasserstein across
          predicted edges is the headline quality metric.

Verdicts are assigned per the brief's rubric:

    confirmed       : predicted edge, MW-significant after BH, large W
    refuted         : predicted edge, MW NOT significant
    cyclic          : confirmed but the reverse direction j → i is also
                       MW-significant (Schultheiss §5 asymmetry check)
    false_omission  : no directed path predicted, yet MW IS significant —
                       a CausalBench false-omission, this is the negative-
                       edge failure mode
    underpowered    : interventional sample size for gene i is below
                       MIN_INT_CELLS, so any result is unreliable
    unverifiable    : no knockdown environment for gene i, so we cannot
                       test any (i, ·) pair

Caveats explicitly preserved through to the output:

  * CRISPRi is do(x_i = low), a *soft* intervention. Effect-size readings
    against B are directionally correct but not quantitatively calibrated.
  * Knockdown cells often sit outside the observational support
    (Schultheiss §5). Large W with small MW p-value is still informative;
    treat the exact W magnitude as a ranking, not an effect size.
  * Multiple testing: we run O(p^2) Mann–Whitney tests. Both raw and
    Benjamini–Hochberg-adjusted p-values are kept.
  * Self-pairs (i, i) are skipped — knocking i out trivially shifts i.

Public API
----------
    validate_edges(npz_path, fit, sel, *, ...) -> pd.DataFrame
        Run the full validation. Returns a per-ordered-pair DataFrame and
        the summary headline numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from causalbench_loader import load_split
from lingam_model import LingamFit, SelectionResult, reachability


# ── verdict labels ───────────────────────────────────────────────────────────

CONFIRMED = "confirmed"
REFUTED = "refuted"
CYCLIC = "cyclic"
FALSE_OMISSION = "false_omission"
SILENT_NEGATIVE = "silent_negative"  # no path predicted, no shift observed: as expected
UNDERPOWERED = "underpowered"
UNVERIFIABLE = "unverifiable"


@dataclass
class ValidationSummary:
    """Headline numbers extracted from the per-pair table."""

    n_pairs_tested: int
    n_predicted_edges: int
    n_confirmed: int
    n_refuted: int
    n_cyclic: int
    n_no_path_pairs: int
    n_false_omissions: int
    confirmed_rate: float           # CausalBench positive-edge headline
    mean_wasserstein_predicted: float
    false_omission_rate: float      # CausalBench negative-edge headline
    bh_alpha: float
    min_int_cells: int


# ── benjamini-hochberg ───────────────────────────────────────────────────────


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values, NaN-tolerant.

    Implementation: rank the non-NaN p-values from smallest to largest, scale
    each by m / rank, then enforce the running-minimum monotonicity from the
    top down. NaN inputs (e.g. pairs with no interventional cells, or self-
    pairs) propagate through as NaN — those pairs were never tested so they
    don't enter the multiple-comparisons pool.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    if not valid.any():
        return out
    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    ranks = np.empty(m)
    ranks[order] = np.arange(1, m + 1)
    adj = pv * m / ranks
    # Enforce monotonicity from largest p downwards (the standard BH
    # adjustment is the running minimum of the scaled p-values from the top).
    sorted_adj = adj[order]
    sorted_adj = np.minimum.accumulate(sorted_adj[::-1])[::-1]
    adj_final = np.empty(m)
    adj_final[order] = np.clip(sorted_adj, 0, 1)
    out[valid] = adj_final
    return out


# ── distribution tests on one (i, j) pair ────────────────────────────────────


def _mw_and_wasserstein(d_int: np.ndarray, d_obs: np.ndarray) -> Dict[str, float]:
    """Compute MW two-sided p-value and Wasserstein distance for one pair.

    NaN-safe: if either distribution is empty or constant the MW test would
    raise; we return NaN for the affected statistic but still try to compute
    the Wasserstein (which only needs both samples non-empty).
    """
    if len(d_int) == 0 or len(d_obs) == 0:
        return {"mw_p": float("nan"), "w": float("nan"), "n_int": int(len(d_int))}

    # Wasserstein-1 is the integral of |F_int - F_obs|. Cheap and robust;
    # well-defined even when both samples are constant.
    w = float(stats.wasserstein_distance(d_int, d_obs))

    # Mann–Whitney is undefined when both samples are identical constants;
    # in that case the answer is "no shift" so we return p = 1.0.
    if np.allclose(d_int.std(), 0) and np.allclose(d_obs.std(), 0) and np.isclose(
        d_int.mean(), d_obs.mean()
    ):
        mw_p = 1.0
    else:
        try:
            mw_p = float(stats.mannwhitneyu(d_int, d_obs, alternative="two-sided").pvalue)
        except ValueError:
            mw_p = float("nan")

    return {"mw_p": mw_p, "w": w, "n_int": int(len(d_int))}


# ── main entry point ─────────────────────────────────────────────────────────


def validate_edges(
    npz_path: str,
    fit: LingamFit,
    sel: SelectionResult,
    *,
    bh_alpha: float = 0.05,
    min_int_cells: int = 50,
    freq_threshold: float = 0.8,
    edge_threshold: float = 0.01,
    seed: int = 0,
) -> tuple[pd.DataFrame, ValidationSummary]:
    """Validate the recovered LiNGAM B against the full-interventional regime.

    Parameters
    ----------
    npz_path
        Cached `.npz` produced by `causalbench_loader.preprocess`.
    fit
        Result of `lingam_model.fit_lingam` on the observational submatrix.
    sel
        The `SelectionResult` that produced `fit` — used for the gene-name
        ordering, which must match `fit.B`'s row/column ordering.
    bh_alpha
        Family-wise rejection threshold on BH-adjusted Mann–Whitney p-values.
        0.05 is the convention; the verdict labels are computed against this.
    min_int_cells
        Below this many interventional cells for gene i, every pair
        (i, ·) is marked `underpowered` rather than confirmed/refuted.
    freq_threshold
        Bootstrap selection frequency below which an edge is treated as
        unstable. The verdict still gets assigned (so the table is complete)
        but `is_stable_edge` flips to False so the visualisation can grey
        out the edge.
    edge_threshold
        Magnitude below which |B_ij| is treated as "no direct edge" for the
        purpose of separating predicted edges (b_ij ≠ 0) from absent ones.
        Same idea as the bootstrap's `min_effect`.
    seed
        For any future use; currently no stochastic step in this function.

    Returns
    -------
    df : pd.DataFrame
        One row per ordered pair (i, j), i ≠ j. Columns described inline.
    summary : ValidationSummary
        Headline numbers ready for the report.
    """
    # The observational control population for D_obs: pulled from the
    # full_interventional split's "non-targeting" cells, which guarantees
    # the preprocessing pipeline is the same as for the perturbed cells
    # we'll compare against. (Using the observational regime's controls
    # would be near-identical here because preprocessing is the same, but
    # taking both samples from the same split avoids any subtle
    # subset_data / seed mismatch creeping in.)
    X_full, int_full, gene_names_full = load_split(npz_path, regime="full_interventional")
    int_full = [str(t) for t in int_full]

    # Restrict to the selected gene columns.
    X_sel = X_full[:, sel.selected_idx]
    p = X_sel.shape[1]
    assert p == fit.B.shape[0], (
        f"selected gene count ({p}) doesn't match fit.B shape ({fit.B.shape}) — "
        "did the SelectionResult come from a different run?"
    )

    # Pre-build per-gene masks. "non-targeting" is the control reference.
    # "excluded" is the synthetic rare-perturbation bucket and must be
    # dropped from every analysis (it conflates many distinct knockdowns).
    int_arr = np.array(int_full)
    ctrl_mask = (int_arr == "non-targeting")
    # Per-cause masks keyed on selected gene name. Cells in "excluded"
    # never match any selected gene name so they're naturally excluded.
    cause_masks: Dict[str, np.ndarray] = {
        g: (int_arr == g) for g in sel.gene_names
    }
    n_int_per_cause = np.array([cause_masks[g].sum() for g in sel.gene_names])

    D_obs = X_sel[ctrl_mask]  # shape (n_ctrl, p)
    n_ctrl = D_obs.shape[0]

    # Reachability of B's zero/non-zero pattern. has_path[i, j] = True iff
    # there is a directed path from cause j to effect i in the recovered DAG.
    # NB: `reachability` uses the same (effect, cause) convention as B.
    has_path = reachability(fit.B, eps=edge_threshold)

    # Bookkeeping for the per-pair table.
    rows: List[dict] = []
    n_pairs = 0
    # Run all the raw tests first so BH can be applied across the full
    # tested-pair pool in one pass.
    mw_p_raw_list: List[float] = []
    w_list: List[float] = []
    row_indices_for_bh: List[int] = []

    for ci, cause in enumerate(sel.gene_names):       # cause = column j (loops over CAUSES)
        cause_cells = cause_masks[cause]
        n_int = int(cause_cells.sum())

        for ei, effect in enumerate(sel.gene_names):  # effect = row i
            if ci == ei:
                continue  # never test self-edges (see brief: trivial shift)

            b_ij = float(fit.B[ei, ci])
            is_predicted = abs(b_ij) > edge_threshold
            freq = float(fit.freq_matrix[ei, ci])
            is_stable_edge = (freq >= freq_threshold) if is_predicted else False
            path = bool(has_path[ei, ci])
            b_med = float(fit.b_median[ei, ci])
            b_iqr = (float(fit.b_iqr_low[ei, ci]), float(fit.b_iqr_high[ei, ci]))

            # If there is no knockdown environment for `cause`, the pair is
            # unverifiable and we record NaN tests but keep the row.
            if not sel.has_intervention_env[ci]:
                row = _row(
                    cause, effect, b_ij, b_med, b_iqr, freq, is_stable_edge,
                    path, n_int, float("nan"), float("nan"), UNVERIFIABLE,
                )
                rows.append(row)
                n_pairs += 1
                continue

            d_int = X_sel[cause_cells, ei]
            d_obs = D_obs[:, ei]
            stats_pair = _mw_and_wasserstein(d_int, d_obs)
            mw_p = stats_pair["mw_p"]
            w = stats_pair["w"]

            # Underpowered if the interventional cell count is too small.
            # We still RECORD the test results (for diagnostic plots) but
            # the verdict is forced to underpowered. (We don't run BH on
            # these p-values to avoid them inflating m for stronger tests.)
            if n_int < min_int_cells:
                row = _row(
                    cause, effect, b_ij, b_med, b_iqr, freq, is_stable_edge,
                    path, n_int, mw_p, w, UNDERPOWERED,
                )
                rows.append(row)
                n_pairs += 1
                continue

            row = _row(
                cause, effect, b_ij, b_med, b_iqr, freq, is_stable_edge,
                path, n_int, mw_p, w, "PENDING",
            )
            rows.append(row)
            row_indices_for_bh.append(len(rows) - 1)
            mw_p_raw_list.append(mw_p)
            w_list.append(w)
            n_pairs += 1

    # BH adjustment over the powered tests only.
    mw_p_raw = np.array(mw_p_raw_list, dtype=float)
    mw_p_bh = _bh_adjust(mw_p_raw)
    for k, ridx in enumerate(row_indices_for_bh):
        rows[ridx]["mw_p_bh"] = float(mw_p_bh[k])

    # Build the dataframe so we can vectorise the reverse-direction lookup.
    df = pd.DataFrame(rows)

    # Resolve verdicts that need to look up the reverse direction (cyclic
    # detection) — needs the BH-adjusted p of (j, i) to be known too, so
    # we do this in a second pass.
    bh_lookup = {
        (r["cause_i"], r["effect_j"]): r["mw_p_bh"] for r in rows
    }

    def _resolve(row: dict) -> str:
        v = row["verdict"]
        if v != "PENDING":
            return v
        pred = abs(row["lingam_b_ij"]) > edge_threshold
        has_p = row["has_path"]
        bh = row["mw_p_bh"]
        sig = (not np.isnan(bh)) and (bh < bh_alpha)

        if pred:
            if not sig:
                return REFUTED
            # Check the reverse direction for asymmetry. j → i would be the
            # row keyed (effect, cause).
            rev_bh = bh_lookup.get((row["effect_j"], row["cause_i"]), float("nan"))
            rev_sig = (not np.isnan(rev_bh)) and (rev_bh < bh_alpha)
            if rev_sig:
                return CYCLIC
            return CONFIRMED
        else:
            # No predicted direct edge. If there is also no directed path,
            # this is a "negative pair" for the false-omission test.
            if has_p:
                # Indirect path predicted — no claim either way. The pair
                # carries no LiNGAM prediction about distribution shift, so
                # we leave it as a "silent" row (kept for transparency).
                return "indirect_path"
            return FALSE_OMISSION if sig else SILENT_NEGATIVE

    df["verdict"] = df.apply(_resolve, axis=1)

    # Attach the reverse-direction p-values as their own columns. The verdict
    # logic already consults the reverse direction (for the cyclic check); we
    # surface it here so both the CSV and the validation report can show the
    # forward/reverse asymmetry side by side without a second join.
    raw_lookup = {
        (r["cause_i"], r["effect_j"]): r["mw_pvalue_raw"] for r in rows
    }
    df["mw_pvalue_raw_rev"] = [
        raw_lookup.get((e, c), float("nan"))
        for c, e in zip(df["cause_i"], df["effect_j"])
    ]
    df["mw_p_bh_rev"] = [
        bh_lookup.get((e, c), float("nan"))
        for c, e in zip(df["cause_i"], df["effect_j"])
    ]

    # Headline numbers.
    n_predicted = int(((df["lingam_b_ij"].abs() > edge_threshold)
                       & (df["verdict"] != UNDERPOWERED)
                       & (df["verdict"] != UNVERIFIABLE)).sum())
    n_confirmed = int((df["verdict"] == CONFIRMED).sum())
    n_refuted = int((df["verdict"] == REFUTED).sum())
    n_cyclic = int((df["verdict"] == CYCLIC).sum())
    # Negative-pair pool = ordered pairs with NO directed path AND a powered MW test.
    no_path_powered = df[
        (~df["has_path"])
        & (df["lingam_b_ij"].abs() <= edge_threshold)
        & (df["verdict"].isin([FALSE_OMISSION, SILENT_NEGATIVE]))
    ]
    n_no_path = len(no_path_powered)
    n_fo = int((df["verdict"] == FALSE_OMISSION).sum())

    confirmed_rate = (n_confirmed / max(n_predicted, 1))
    mean_w_pred = float(df.loc[
        (df["lingam_b_ij"].abs() > edge_threshold)
        & (df["verdict"].isin([CONFIRMED, REFUTED, CYCLIC])),
        "wasserstein",
    ].mean()) if n_predicted > 0 else float("nan")
    for_rate = (n_fo / max(n_no_path, 1)) if n_no_path > 0 else float("nan")

    summary = ValidationSummary(
        n_pairs_tested=n_pairs,
        n_predicted_edges=n_predicted,
        n_confirmed=n_confirmed,
        n_refuted=n_refuted,
        n_cyclic=n_cyclic,
        n_no_path_pairs=n_no_path,
        n_false_omissions=n_fo,
        confirmed_rate=confirmed_rate,
        mean_wasserstein_predicted=mean_w_pred,
        false_omission_rate=for_rate,
        bh_alpha=bh_alpha,
        min_int_cells=min_int_cells,
    )

    # Sort the dataframe for human reading: predicted edges first, then by
    # |b_ij| descending so the strongest claims sit at the top.
    df["abs_b"] = df["lingam_b_ij"].abs()
    df = df.sort_values(
        ["abs_b", "wasserstein"], ascending=[False, False]
    ).drop(columns=["abs_b"]).reset_index(drop=True)

    return df, summary


def _row(
    cause: str,
    effect: str,
    b_ij: float,
    b_med: float,
    b_iqr: tuple,
    freq: float,
    is_stable: bool,
    has_p: bool,
    n_int: int,
    mw_p_raw: float,
    w: float,
    verdict: str,
) -> dict:
    return {
        "cause_i": cause,
        "effect_j": effect,
        "lingam_b_ij": b_ij,
        "b_median_bootstrap": b_med,
        "b_iqr_low": b_iqr[0],
        "b_iqr_high": b_iqr[1],
        "bootstrap_freq": freq,
        "is_stable_edge": is_stable,
        "has_path": has_p,
        "n_int_cells": n_int,
        "mw_pvalue_raw": mw_p_raw,
        "mw_p_bh": float("nan"),  # filled later
        "wasserstein": w,
        "verdict": verdict,
    }
