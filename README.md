# M2R Project — Learning Causal Structure and Effects from Observational Data

An M2R research project (Imperial College London) applying **LiNGAM-based
causal discovery** to single-cell **Perturb-seq** data from the **CausalBench
K562** cell line (Replogle et al. 2022, packaged by Chevalley et al. 2025).

The pipeline fits `DirectLiNGAM` to the *observational* expression of a set of
active-everywhere genes, bootstraps the fit to assess edge stability, and then
**validates the recovered structure against held-out interventional
(CRISPRi-knockdown) data**: for each predicted edge i → j, knocking down gene i
should shift the distribution of gene j. Each ordered gene pair receives one of
six verdicts — *confirmed, refuted, cyclic, false omission, silent negative,
indirect path* — from a Mann–Whitney U test (BH-corrected) plus a 1-D
Wasserstein effect-size measure.

---

## Repository layout

| File | Purpose |
|---|---|
| `causalbench_loader.py` | Download + preprocess the CausalBench K562 data into a cached `.npz`; expose `load_split`. *(Do not modify — import from it.)* |
| `oles_data_exploration.py` | EDA on the selected gene set: gene–gene correlation matrix + residual non-Gaussianity check (the LiNGAM assumption). |
| `lingam_model.py` | Gene selection (Schultheiss §5 active-everywhere rule), `DirectLiNGAM(measure='pwling')` fit, and the bootstrap stability assessment. |
| `validate_edges.py` | Interventional validation: per-pair Mann–Whitney U + Wasserstein, BH correction, and the six-category verdict logic. |
| `visualize_graph.py` | Annotated causal-graph figure (edges coloured by verdict) + the Wasserstein-vs-bootstrap-frequency scatter. |
| `run_lingam_analysis.py` | Orchestrator: runs the full pipeline and writes `results.csv`, `validation_report.md`, `SUMMARY.md`, and the figures. |
| `bootstrap_experiments.py` | Compares bootstrap aggregation methods + hyperparameters; writes `bootstrap_summary.md` and `bootstrap_comparison.csv`. |
| `plot_style.py` | Shared matplotlib style + verdict colour palette used by every figure. |
| `requirements.txt` | Pinned Python dependencies. |
| `setup.md` | Python environment setup instructions. |

### Generated outputs

| Artefact | Produced by |
|---|---|
| `data_summary.md` | Correlation summary statistics over the selected gene set. |
| `causalbench_plots/results.csv` | One row per ordered gene pair (effect size, bootstrap frequency, MW p-values, Wasserstein, verdict). |
| `causalbench_plots/validation_report.md` | Structured six-verdict validation report. |
| `causalbench_plots/SUMMARY.md` | Compact auto-generated headline numbers + caveats. |
| `bootstrap_summary.md` / `bootstrap_comparison.csv` | Bootstrap aggregation/hyperparameter comparison. |
| `causalbench_plots/*.png`, `*.svg`, `*.gv` | Figures (causal graph, correlation matrix, non-Gaussianity, scatter). |

---

## Running the pipeline

```bash
python run_lingam_analysis.py     # full LiNGAM fit + interventional validation
python oles_data_exploration.py   # the two EDA figures
python bootstrap_experiments.py   # bootstrap aggregation/hyperparameter study
```

The cached dataset lives in `causalbench_data/` (git-ignored); nothing is
re-downloaded once it exists. All stochastic steps key off a single `SEED = 0`
for reproducibility.

For environment setup, see **[setup.md](setup.md)**.

---

## Key references

- **Shimizu, S. (2012).** *DirectLiNGAM: A direct method for learning a linear
  non-Gaussian structural equation model.* (DirectLiNGAM, `pwling` pairwise
  measure.)
- **Schultheiss, C. & Bühlmann, P. (2024).** *Assessing the overall and
  partial causal well-specification of nonlinear additive noise models.*
  (Active-everywhere cleaning + interventional well-specification checks.)
- **Chevalley, M., Schwab, P. & Mehrjou, A. (2025).** *CausalBench: A
  large-scale benchmark for network inference from single-cell perturbation
  data.* (Benchmark, mean-Wasserstein / false-omission-rate metrics.)
