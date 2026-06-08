"""
bootstrap_experiments
======================
Systematic comparison of bootstrap *aggregation* methods and hyperparameters
for the K562 DirectLiNGAM fit, scored against the interventional validation.

What it does
------------
1. Fit DirectLiNGAM once → point-estimate adjacency `B`, and run 200 bootstrap
   resamples → a stacked (200, p, p) array of per-resample adjacencies. The
   stacked array is cached so re-runs (e.g. while tweaking the report) are
   instant. The first 50/100 resamples ARE the n=50 / n=100 runs (same per-
   resample seeds), so we slice rather than re-bootstrap.

2. Run the interventional validation ONCE to obtain a *fixed* per-ordered-pair
   ground truth: whether knocking down the cause produces a BH-significant
   Mann–Whitney shift in the effect (forward direction), and the Wasserstein
   magnitude of that shift. Every aggregation method is then scored against
   this same ground truth, which is the whole point — it isolates the effect of
   the *selection* rule from the *validation*.

   Scoring of a candidate directed edge set:
     n_edges            number of edges the method selects
     n_confirmed        of those, how many have a powered, forward
                        BH-significant MW test (the interventional data
                        confirms a shift in the claimed direction; this is the
                        union of the "confirmed" and "cyclic" verdicts)
     confirmation_rate  n_confirmed / n_edges
     mean_wasserstein   mean Wasserstein over the confirmed edges

3. Four aggregation methods, a hyperparameter grid for the baseline, and a
   bootstrap-cyclic detection heuristic — all written to `bootstrap_summary.md`
   and `bootstrap_comparison.csv`.

Index convention (inherited from lingam_model): every adjacency matrix is
indexed [effect, cause]. So `M[e, c] != 0` is the directed edge cause→effect,
and `stacked[:, e, c]` is that edge's coefficient across resamples.

Run:  python bootstrap_experiments.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from causalbench_loader import download_raw_data, preprocess
from lingam_model import LingamFit, select_genes, fit_lingam
from validate_edges import validate_edges, CYCLIC

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR   = "./causalbench_data"
NPZ_NAME   = "dataset_k562.npz"
DATASET    = "weissmann_k562"
SUMMARY_MD = "bootstrap_summary.md"
COMPARE_CSV = "bootstrap_comparison.csv"

N_BOOT_MAX     = 200          # the largest n_sampling we study; smaller = slices
EDGE_THRESHOLD = 0.01         # |b_ij| below this == "no direct edge"
BH_ALPHA       = 0.05
MIN_INT_CELLS  = 50
SEED           = 0

# Threshold grids per the brief.
FREQ_THRESHOLDS   = [0.50, 0.60, 0.70, 0.80, 0.90]   # methods 1 & 2
MEDIAN_THRESHOLDS = [0.01, 0.02, 0.05]               # method 3
RATIO_THRESHOLDS  = [0.60, 0.70, 0.80, 0.90]         # method 4
N_SAMPLING_GRID   = [50, 100, 200]                   # hyperparameter sweep
CYCLIC_THRESHOLDS = [0.20, 0.30, 0.40]               # cyclic heuristic

CACHE = os.path.join(DATA_DIR, f"bootstrap_k562_n{N_BOOT_MAX}_seed{SEED}.npz")


# ── bootstrap matrices (cached) ──────────────────────────────────────────────


def get_B_and_stacked(X: np.ndarray):
    """Return (B, stacked) where stacked is (N_BOOT_MAX, p, p). Cached on disk
    because the 200-resample run is the expensive step (minutes); the cache is
    keyed on n_boot+seed in the filename and lives under the git-ignored data
    dir."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        print(f"  loaded cached bootstrap from {CACHE}")
        return d["B"], d["stacked"]
    print(f"  fitting + {N_BOOT_MAX} bootstraps (no cache yet)…", flush=True)
    fit = fit_lingam(X, n_bootstrap=N_BOOT_MAX, seed=SEED, min_effect=EDGE_THRESHOLD)
    B = fit.B
    stacked = fit.bootstrap_result
    np.savez_compressed(CACHE, B=B, stacked=stacked)
    print(f"  cached bootstrap → {CACHE}")
    return B, stacked


def build_fit(B: np.ndarray, stacked_n: np.ndarray) -> LingamFit:
    """Assemble a LingamFit from a bootstrap slice so we can call
    `validate_edges`. Only `B` drives the verdict logic; the frequency/median
    matrices are recomputed from the slice for completeness."""
    freq = (np.abs(stacked_n) > EDGE_THRESHOLD).mean(axis=0)
    p = B.shape[0]
    try:
        A = np.linalg.inv(np.eye(p) - B)
    except np.linalg.LinAlgError:
        A = np.full_like(B, np.nan)
    return LingamFit(
        B=B,
        causal_order=np.arange(p),
        A=A,
        freq_matrix=freq,
        b_median=np.median(stacked_n, axis=0),
        b_iqr_low=np.percentile(stacked_n, 25, axis=0),
        b_iqr_high=np.percentile(stacked_n, 75, axis=0),
        n_bootstrap=stacked_n.shape[0],
        bootstrap_result=stacked_n,
    )


# ── ground-truth lookup ──────────────────────────────────────────────────────


def build_ground_truth(df: pd.DataFrame, alpha: float):
    """Map (cause, effect) → interventional ground truth for that ordered pair.

    `sig` = the pair is powered AND forward BH-adjusted MW p < alpha, i.e. the
    interventional data confirms a distribution shift in the cause→effect
    direction. This is exactly the union of the `confirmed` and `cyclic`
    verdicts, and is the criterion every aggregation method is scored on."""
    gt = {}
    for _, r in df.iterrows():
        bh = r["mw_p_bh"]
        powered = (not np.isnan(bh))
        gt[(r["cause_i"], r["effect_j"])] = {
            "sig": bool(powered and bh < alpha),
            "w": float(r["wasserstein"]),
            "verdict": r["verdict"],
        }
    return gt


def score(edges, gt):
    """Score a directed edge set (iterable of (cause, effect) name tuples)."""
    edges = list(edges)
    n = len(edges)
    confirmed_w = [gt[e]["w"] for e in edges if e in gt and gt[e]["sig"]]
    n_conf = len(confirmed_w)
    rate = (n_conf / n) if n else float("nan")
    mean_w = float(np.mean(confirmed_w)) if confirmed_w else float("nan")
    return n, n_conf, rate, mean_w


# ── edge-set builders for each aggregation method ────────────────────────────


def edges_point_estimate(B, names):
    """Method 1 baseline: |B_ij| > threshold from the single full-data fit."""
    p = B.shape[0]
    return {
        (names[c], names[e])
        for e in range(p) for c in range(p)
        if e != c and abs(B[e, c]) > EDGE_THRESHOLD
    }


def edges_freq(freq, names, t):
    """Method 2: bootstrap selection frequency >= t."""
    p = freq.shape[0]
    return {
        (names[c], names[e])
        for e in range(p) for c in range(p)
        if e != c and freq[e, c] >= t
    }


def edges_median(stacked_n, names, t):
    """Method 3: |median b_ij over present-only resamples| > t. 'Present' means
    the edge cleared EDGE_THRESHOLD on that resample, so this asks whether the
    edge — when it appears at all — is consistently non-trivial in magnitude."""
    n, p, _ = stacked_n.shape
    out = set()
    for e in range(p):
        for c in range(p):
            if e == c:
                continue
            vals = stacked_n[:, e, c]
            present = np.abs(vals) > EDGE_THRESHOLD
            if present.any() and abs(np.median(vals[present])) > t:
                out.add((names[c], names[e]))
    return out


def edges_directional(stacked_n, names, t):
    """Method 4: for each unordered pair, include the majority direction iff
    max(fwd, rev) / (fwd + rev) > t, where fwd/rev count the resamples in which
    each direction's edge cleared EDGE_THRESHOLD."""
    n, p, _ = stacked_n.shape
    present = np.abs(stacked_n) > EDGE_THRESHOLD     # (n, p, p), [effect, cause]
    out = set()
    for a in range(p):
        for b in range(a + 1, p):
            # a→b lives at [b, a]; b→a lives at [a, b].
            c_ab = int(present[:, b, a].sum())
            c_ba = int(present[:, a, b].sum())
            tot = c_ab + c_ba
            if tot == 0:
                continue
            ratio = max(c_ab, c_ba) / tot
            if ratio > t:
                if c_ab >= c_ba:
                    out.add((names[a], names[b]))   # cause a → effect b
                else:
                    out.add((names[b], names[a]))
    return out


# ── cyclic detection heuristic ───────────────────────────────────────────────


def cyclic_heuristic(stacked_n, names, df, thresholds):
    """For each unordered pair, flag 'bootstrap-cyclic' if BOTH directions
    appear in >= min_frac of resamples. Compare against interventional cyclic
    verdicts (a pair is interventionally cyclic if either ordered direction was
    given the CYCLIC verdict)."""
    n, p, _ = stacked_n.shape
    present = np.abs(stacked_n) > EDGE_THRESHOLD

    # Interventional cyclic pairs as a set of frozenset({cause, effect}).
    int_cyclic = set()
    for _, r in df.iterrows():
        if r["verdict"] == CYCLIC:
            int_cyclic.add(frozenset((r["cause_i"], r["effect_j"])))

    rows = []
    for min_frac in thresholds:
        flagged = set()
        for a in range(p):
            for b in range(a + 1, p):
                frac_ab = present[:, b, a].mean()
                frac_ba = present[:, a, b].mean()
                if frac_ab >= min_frac and frac_ba >= min_frac:
                    flagged.add(frozenset((names[a], names[b])))
        overlap = flagged & int_cyclic
        false_pos = flagged - int_cyclic
        rows.append({
            "min_frac": min_frac,
            "n_flagged": len(flagged),
            "overlap_with_interventional": len(overlap),
            "false_positives": len(false_pos),
            "interventional_cyclic_total": len(int_cyclic),
            "recall": (len(overlap) / len(int_cyclic)) if int_cyclic else float("nan"),
        })
    return rows, len(int_cyclic)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    npz_path = os.path.join(DATA_DIR, NPZ_NAME)
    if not os.path.exists(npz_path):
        download_raw_data(DATA_DIR, files=["k562.h5ad"])
        preprocess(DATA_DIR, DATASET)

    print("Step 1 · gene selection + bootstrap")
    sel = select_genes(npz_path, k=None, seed=SEED)
    names = sel.gene_names
    B, stacked = get_B_and_stacked(sel.X_obs)
    p = B.shape[0]

    # Slices for the n_sampling sweep; n=100 is the canonical slice used for
    # methods 2-4 and the cyclic heuristic (matches the main pipeline).
    slices = {n: stacked[:n] for n in N_SAMPLING_GRID}
    stacked100 = slices[100]
    freq100 = (np.abs(stacked100) > EDGE_THRESHOLD).mean(axis=0)

    print("Step 2 · interventional validation (fixed ground truth)")
    fit100 = build_fit(B, stacked100)
    df, summary = validate_edges(
        npz_path, fit100, sel,
        bh_alpha=BH_ALPHA, min_int_cells=MIN_INT_CELLS,
        freq_threshold=0.80, edge_threshold=EDGE_THRESHOLD, seed=SEED,
    )
    gt = build_ground_truth(df, BH_ALPHA)

    print("Step 3 · scoring aggregation methods")
    comp_rows = []

    def add(method, thr, edges):
        n, n_conf, rate, mean_w = score(edges, gt)
        comp_rows.append({
            "method": method, "threshold": thr,
            "n_edges": n, "n_confirmed": n_conf,
            "confirmation_rate": rate, "mean_wasserstein_confirmed": mean_w,
        })

    # Method 1 — point-estimate baseline (one row; threshold is the |b| cut).
    add("point_estimate", EDGE_THRESHOLD, edges_point_estimate(B, names))
    # Method 2 — edge frequency thresholding (n=100).
    for t in FREQ_THRESHOLDS:
        add("edge_frequency", t, edges_freq(freq100, names, t))
    # Method 3 — median effect size (present-only), n=100.
    for t in MEDIAN_THRESHOLDS:
        add("median_effect", t, edges_median(stacked100, names, t))
    # Method 4 — directional stability ratio (n=100).
    for t in RATIO_THRESHOLDS:
        add("directional_ratio", t, edges_directional(stacked100, names, t))

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(COMPARE_CSV, index=False)
    print(f"  wrote {COMPARE_CSV}")

    # Hyperparameter sweep: stable edges = point-estimate edges that also clear
    # freq >= threshold under each n_sampling slice.
    print("Step 4 · hyperparameter sweep")
    base_edges = edges_point_estimate(B, names)
    base_idx = {(names[c], names[e]): (e, c)
                for e in range(p) for c in range(p) if e != c}
    hyper_rows = []
    for n in N_SAMPLING_GRID:
        freq_n = (np.abs(slices[n]) > EDGE_THRESHOLD).mean(axis=0)
        for t in FREQ_THRESHOLDS:
            stable = {edge for edge in base_edges
                      if freq_n[base_idx[edge]] >= t}
            ne, nc, rate, mw = score(stable, gt)
            hyper_rows.append({
                "n_sampling": n, "stability_threshold": t,
                "n_stable_edges": ne, "n_confirmed": nc,
                "confirmation_rate": rate,
            })

    print("Step 5 · cyclic heuristic")
    cyc_rows, n_int_cyclic = cyclic_heuristic(
        stacked100, names, df, CYCLIC_THRESHOLDS)

    write_summary(comp_df, hyper_rows, cyc_rows, n_int_cyclic, summary, p)
    print(f"wrote {SUMMARY_MD}")


# ── report ───────────────────────────────────────────────────────────────────


def _fmt(x, nd=4):
    if isinstance(x, float) and np.isnan(x):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_summary(comp_df, hyper_rows, cyc_rows, n_int_cyclic, summary, p):
    L = []
    L.append("# Bootstrap aggregation & hyperparameter study — K562\n")
    L.append(
        f"Auto-generated by `bootstrap_experiments.py` (seed = {SEED}, "
        f"p = {p} genes, edge threshold |b| > {EDGE_THRESHOLD}, "
        f"BH α = {BH_ALPHA}).\n")

    # 1. Overview
    L.append("## 1. Overview\n")
    L.append(
        "DirectLiNGAM defines edges from a single point-estimate adjacency "
        "matrix; the bootstrap is currently used only to annotate those edges "
        "with a selection frequency. This study asks whether a different way of "
        "*aggregating* the bootstrap would select a better edge set. We compare "
        "four selection rules and a hyperparameter sweep against a fixed "
        "interventional ground truth: an edge cause→effect is **confirmed** iff "
        "knocking down the cause produces a powered, BH-significant "
        "Mann–Whitney shift in the effect (the union of the `confirmed` and "
        "`cyclic` validation verdicts). Because the ground truth is held fixed, "
        "differences in confirmation rate reflect the *selection rule*, not the "
        "validation. The bootstrap is run to 200 resamples once; n=50/100 are "
        "leading slices of the same run.\n")

    # 2. Aggregation method comparison
    L.append("## 2. Aggregation method comparison\n")
    L.append("| Method | Threshold | Edges | Confirmed | Confirmation rate | "
             "Mean Wasserstein (confirmed) |")
    L.append("|---|---|---|---|---|---|")
    for _, r in comp_df.iterrows():
        L.append(
            f"| {r['method']} | {_fmt(r['threshold'], 2)} | {int(r['n_edges'])} "
            f"| {int(r['n_confirmed'])} | {_fmt(r['confirmation_rate'], 3)} "
            f"| {_fmt(r['mean_wasserstein_confirmed'])} |")
    L.append("")
    L.append(
        "*`point_estimate` is the current baseline (threshold = the |b| cut). "
        "Higher thresholds in every method trade fewer edges for a higher "
        "confirmation rate — the precision/recall knob.*\n")

    # 3. Hyperparameter sensitivity
    L.append("## 3. Hyperparameter sensitivity (baseline method)\n")
    L.append(
        "Point-estimate edges filtered to the *stable* subset (bootstrap "
        "frequency ≥ stability threshold) under each `n_sampling`. The raw "
        "point-estimate edge set is fixed; what changes is how many survive the "
        "stability filter and their confirmation rate.\n")
    L.append("| n_sampling | Stability threshold | Stable edges | Confirmed | "
             "Confirmation rate |")
    L.append("|---|---|---|---|---|")
    for r in hyper_rows:
        L.append(
            f"| {r['n_sampling']} | {_fmt(r['stability_threshold'], 2)} "
            f"| {r['n_stable_edges']} | {r['n_confirmed']} "
            f"| {_fmt(r['confirmation_rate'], 3)} |")
    L.append("")

    # 4. Cyclic detection heuristic
    L.append("## 4. Cyclic detection heuristic\n")
    L.append(
        f"For each unordered pair {{A, B}} we count, across the 100 bootstrap "
        f"adjacencies, how many resamples placed A→B versus B→A, and flag the "
        f"pair as *bootstrap-cyclic* when BOTH directions appear in at least "
        f"`min_frac` of resamples. The interventional validation independently "
        f"labelled **{n_int_cyclic}** unordered pairs `cyclic` (both directions "
        f"BH-significant).\n")
    L.append("| min_frac | Pairs flagged | Overlap w/ interventional cyclic | "
             "False positives | Recall of interventional cyclic |")
    L.append("|---|---|---|---|---|")
    for r in cyc_rows:
        L.append(
            f"| {_fmt(r['min_frac'], 2)} | {r['n_flagged']} "
            f"| {r['overlap_with_interventional']} | {r['false_positives']} "
            f"| {_fmt(r['recall'], 3)} |")
    L.append("")
    L.append(
        "**Caveat — this is a heuristic, not a test.** DirectLiNGAM is derived "
        "under acyclicity, so reading ordering-instability across resamples as "
        "evidence for a cycle steps outside the model's theoretical guarantees. "
        "A pair can flip direction across resamples for reasons unrelated to "
        "feedback (near-tied independence scores, finite-sample noise under the "
        "heavy tails of this data). The interventional validation remains the "
        "gold standard for cyclicity; the bootstrap flag is best read as a "
        "**complementary diagnostic for potential feedback loops** worth "
        "examining, not a verdict.\n")

    # 5. Recommendations
    L.append("## 5. Recommendations\n")
    # Pick the best non-baseline method/threshold by confirmation rate among
    # rows that still retain a non-trivial number of edges (>= 10), so we don't
    # recommend a degenerate 1-edge set.
    cand = comp_df[(comp_df["method"] != "point_estimate")
                   & (comp_df["n_edges"] >= 10)].copy()
    best = (cand.sort_values("confirmation_rate", ascending=False).iloc[0]
            if not cand.empty else None)
    base = comp_df[comp_df["method"] == "point_estimate"].iloc[0]
    if best is not None:
        L.append(
            f"- The baseline `point_estimate` selects {int(base['n_edges'])} "
            f"edges at a {_fmt(base['confirmation_rate'], 3)} confirmation rate "
            f"— high recall, low precision (it keeps every non-zero coefficient, "
            f"including resampling artefacts).\n")
        L.append(
            f"- The best precision/recall trade-off among the aggregation rules "
            f"that keep ≥ 10 edges is **{best['method']} @ "
            f"{_fmt(best['threshold'], 2)}**: {int(best['n_edges'])} edges at a "
            f"{_fmt(best['confirmation_rate'], 3)} confirmation rate. "
            f"Frequency- and directional-stability thresholds concentrate the "
            f"edge set on reproducible structure, which is what the "
            f"interventional data rewards.\n")
    L.append(
        "- For the report we keep the point-estimate edge set (it is the "
        "model's actual output and preserves recall for the false-omission "
        "analysis) but annotate every edge with bootstrap frequency, and we use "
        "the directional/frequency thresholds above as a stability filter when "
        "a higher-precision sub-graph is wanted.\n")

    with open(SUMMARY_MD, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
