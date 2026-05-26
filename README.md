# M2R Project
**Learning Causal Structure and Effects from Observational Data**

M2R project for Imperial College London. Uses the CausalBench benchmark
(Chevalley et al., 2025) on the Replogle 2022 Perturb-seq datasets and
the LiNGAM library for causal discovery.

---

## 1. Setup

Requires Python 3.12.

```bash
python3.12 -m venv m2r_venv
source m2r_venv/bin/activate
pip install -r requirements.txt
```

If `pip install -r requirements.txt` fails, install the two critical
packages manually:

```bash
pip install setuptools
pip install causalbench --use-deprecated=legacy-resolver
pip install lingam
```

---

## 2. Data pipeline

The benchmark is built on two single-cell Perturb-seq datasets from
Replogle et al. 2022:

| Dataset           | Cell line | Day | Raw size  | Source       |
| ----------------- | --------- | --- | --------- | ------------ |
| `weissmann_k562`  | K562      | 6   | 10.66 GB  | Figshare 35773219 |
| `weissmann_rpe1`  | RPE1      | 7   |  8.70 GB  | Figshare 35775606 |

⚠️ **Why we don't just call CausalBench's loader.** The shipping version
(`causalbench` 1.1.2) has two bugs that prevent the data from
downloading at all:

1. Its downloader calls `gdown.download(url, ...)` against Figshare URLs,
   but `gdown` only handles Google Drive. The download silently fails
   and leaves 0-byte `.h5ad` files on disk.
2. The URLs it points at (`plus.figshare.com/...`) sit behind an AWS-WAF
   challenge that blocks all non-browser traffic with an HTTP 202.

We work around both in [causalbench_loader.py](causalbench_loader.py),
which downloads from the working host (`ndownloader.figshare.com`) with a
resumable, size-validated, auto-retrying HTTP loop, then hands the files
to CausalBench's own preprocessing pipeline.

### One-shot pipeline

After setup, the whole pipeline is three calls:

```python
from causalbench_loader import download_raw_data, preprocess, load_split

# 1. Download raw .h5ad files into ./causalbench_data/ (~19 GB total).
#    Resumable: Ctrl-C and re-run continues from where it stopped.
download_raw_data("./causalbench_data")

# 2. Preprocess one cell line (normalize + log1p + filter rare perts).
#    Cached as ./causalbench_data/dataset_<line>.npz.
npz_path = preprocess("./causalbench_data", "weissmann_k562")

# 3. Apply the training regime and unpack the arrays.
expression_matrix, interventions, gene_names = load_split(
    npz_path,
    regime="observational",       # or "partial_interventional" / "full_interventional"
    subset_data=1.0,
)
```

Or just run the script directly:

```bash
python oles_data_exploration.py
```

### What to expect

| Step                    | First run             | Subsequent runs |
| ----------------------- | --------------------- | --------------- |
| `download_raw_data`     | 30–60 min, ~19 GB     | instant (size check) |
| `preprocess` (one line) | 2–5 min, ~6 GB RAM    | instant (.npz cache) |
| `load_split`            | < 1 s                 | < 1 s |
| `oles_data_exploration.py` end-to-end | 10–30 min | < 5 s after step 3 |

Both `./causalbench_data/` and `./causalbench_plots/` are gitignored.

### Troubleshooting

| Symptom | What's wrong | Fix |
| ------- | ------------ | --- |
| `OSError: Unable to ... open file (truncated file: eof = ...)` | An `.h5ad` is partial (e.g. you ran preprocess before download finished) | Re-run `download_raw_data` — it auto-resumes. The loader's size check now catches this before it reaches h5py. |
| `RuntimeError: ... still short after 100 attempts` | Network too unstable for Figshare's CDN | Re-run; the file on disk is kept and the next call resumes from there. |
| `ModuleNotFoundError: causalscbench.data_access.utils.loading` | Old code calling a function that doesn't exist in `causalscbench` 1.1.2 | Replace with the `causalbench_loader` API shown above. |

---

## 3. Project files

There are three Python entry points in the repo:

### [causalbench_loader.py](causalbench_loader.py)
Small wrapper around CausalBench's data pipeline that fixes the two
upstream bugs above. Single source of truth for the download + preprocess
+ split workflow — both the script and the notebook import from it, so
any fix lands in one place. Public API: `download_raw_data`, `preprocess`,
`load_split`.

### [oles_data_exploration.py](oles_data_exploration.py)
Non-interactive exploration script. Runs the full pipeline against one
dataset/regime (configured at the top of the file) and writes a set of
overview plots to `./causalbench_plots/`:

- intervention distribution (cells per knockout, top 30)
- per-gene mean-expression histogram
- per-cell library-size histogram
- MA-style plot of perturbed vs. control fold-change
- 40-gene random-subset correlation heatmap
- spotlight diff for one specific perturbed gene

Run with `python oles_data_exploration.py`.

> **Note on the `"excluded"` label.** CausalBench's preprocessing assigns
> the synthetic label `"excluded"` to every cell whose perturbation target
> appears in fewer than 100 cells (preprocessing step 4 in
> [causalbench_loader.py](causalbench_loader.py)). On K562 this bucket is
> typically the single biggest "intervention" by cell count and dominates
> any plot that ranks interventions or aggregates over "perturbed" cells.
> All plots in this script therefore exclude that bucket: section 5a/5b
> drop it from the top-30 bar and cells-per-perturbed-gene histogram, and
> section 6 builds `pert_mask` from real perturbations only (`~ctrl_mask
> & ~excluded_mask`). Section 4's printout still reports `n_excluded` so
> you can see how many cells the filter removed.

#### Explanation of produced plots
Overview plot (2x2 grid):
top left:
 - Each bar is one CRISPR knockout experiemnt (independent variable), measured against no. of cells 6 days after transduction (dependent variable). All CRISPR knockout experiments show significantly fewer cells counted after 6 days of transduction, suggesting targeted genes play essential roles in growth, metabolism or reproduction. Baseline is non-targeting where no knockout occurs
top right:
 - Essentially a histogram of the left plot, groups genes based off how many cells they ended up with at the end of the 6 days. Low median of 132 means we are in low-sample regime, therefore will be lots of uncertainty in recovered causal ordering; i.e. need to ensure use of bootstrapping
bottom left:
 - Ri

### [oles_playground.ipynb](oles_playground.ipynb)
Notebook for interactive prototyping. Five cells:

1. Download raw data (`download_raw_data`)
2. Preprocess (`preprocess`)
3. Apply regime + unpack arrays (`load_split`)
4. Import `lingam`
5. Toy `DirectLiNGAM` run on the first 100 cells × top-10 highest-variance
   genes — a smoke test of the model on the pipeline output

---

## 4. References

- **CausalBench**: Chevalley, Roohani, Mehrjou, Leskovec, Schwab (2025).
  *A large-scale benchmark for network inference from single-cell
  perturbation data.* Communications Biology 8:412.
  https://doi.org/10.1038/s42003-025-07764-y
- **Perturb-seq data**: Replogle et al. (2022). *Mapping
  information-rich genotype-phenotype landscapes with genome-scale
  Perturb-seq.* Cell 185(14).
- **LiNGAM**: Shimizu et al. (2011). *DirectLiNGAM: A direct method for
  learning a linear non-Gaussian structural equation model.* JMLR 12.
