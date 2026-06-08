"""
plot_style
==========
One place to fix the visual identity of every matplotlib figure in the
project, so the report's figures look like they belong to the same document.

Importing this module and calling `apply_style()` sets a clean,
publication-oriented baseline:

  * `seaborn-v0_8-whitegrid` as the base style (light gridlines, no heavy
    chart-junk), overridden per-figure where a grid would distract (e.g.
    heatmaps call `ax.grid(False)`).
  * A single sans-serif font family across all text so titles, ticks and
    annotations are typographically consistent.
  * A restrained, colour-blind-friendly qualitative palette for categorical
    series, exposed as `PALETTE` so callers reuse the same colours.

The verdict-colour map (`VERDICT_COLOURS`) lives here too, so the causal
graph, the scatter plot and any future figure all agree on
green=confirmed / red=refuted / orange=cyclic / grey=unstable.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical palette — Okabe-Ito-inspired, distinguishable in greyscale and
# for the common forms of colour blindness. Index it positionally.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
]

# Verdict → colour, shared with validate_edges' verdict labels. Kept as plain
# strings (not importing validate_edges) so this module has no project-internal
# dependencies and can be imported from anywhere without import cycles.
VERDICT_COLOURS = {
    "confirmed":      "#009E73",  # green
    "refuted":        "#D55E00",  # vermillion
    "cyclic":         "#E69F00",  # orange
    "false_omission": "#CC79A7",  # purple
    "silent_negative": "#56B4E9", # sky blue
    "indirect_path":  "#999999",  # neutral grey
    "unstable":       "#95a5a6",  # grey (used when bootstrap freq < threshold)
}


def apply_style() -> None:
    """Set the project-wide matplotlib rcParams. Idempotent; call once at the
    top of any plotting script."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        # DejaVu Sans ships with matplotlib, so figures render identically on
        # any machine without relying on system fonts being installed.
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
