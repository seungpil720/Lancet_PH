# Lancet_PH: Climate, diet, and disease-burden analysis code

Analysis code supporting the manuscript submitted to *Lancet Public Health* (author-provided title/DOI to be added on acceptance). This repository contains the five scripts used to generate the main-text figures, the merged analytical dataset, and the outputs referenced in the Supplementary Appendix (Reproducibility Appendix section).

## Contents

| File | Produces |
|---|---|
| `data_merged_quarters.csv` | Input dataset: 1,281 rows (183 countries x 7 time points: 1990, 1995, 2000, 2005, 2010, 2015, 2018), 147 columns. |
| `lancetph_figure1.py` | Main-text **Figure 1** — PCA biplots of dietary (GDD1-GDD5) and disease-burden (GBD1-GBD5) dimensions, and climate-specific GDD x GBD correlation heatmaps. |
| `lancetph_figure2.py` | Main-text **Figure 2** — CARE-DDI construction pipeline (CEI, HVI-NCD, HVI-ID, DAC, SBC subindices; four model specifications) and CND-type/country-trajectory panels. Also produces the country-level results summarised in Supplementary Results S7. |
| `lancetph_figure3.py` | Main-text **Figure 3** — climate-specific temporal network analysis (early/late-period correlation networks, edge classification, coupling-strength trends). Parameters documented in Supplementary Methods S4. |
| `lancetph_figure4.py` | Main-text **Figure 4** — geographic distribution of climate profiles and income-stratified trajectories. |
| `lancetph_figure5.py` | Main-text **Figure 5** — country fixed-effects interaction models (8 specifications x 5 dietary patterns x 5 disease outcomes = 200 tests). Full output reproduced in Supplementary Results S8 with Benjamini-Hochberg FDR-adjusted q-values. |
| `lancetph_clustering.py` | Upstream clustering step: derives `Climate_cluster_5`, `GDD1-GDD5`/`GDD_cluster_5`, and `GBD1-GBD5`/`GBD_cluster_5` from raw variables in `data_merged_quarters.csv` (see "Clustering step" below). |
| `requirements.txt` | Python package dependencies for all six scripts. |

A Streamlit dashboard (`app.py`) for interactive per-country exploration may also be present in this repository; it is not required to reproduce any main-text or supplementary figure and has its own optional dependencies (see `requirements.txt`).

## Note on this release

The five scripts above were originally developed as separate Google Colab notebooks and have been cleaned up for this repository:

- **Output filenames now match the current main-text figure numbers.** In the Colab originals, several scripts wrote files using an earlier internal figure numbering (e.g. `lancetph_figure3.py` wrote files prefixed `figure4A_`/`figure4B_`, and `lancetph_figure5.py` wrote files prefixed `figure6A_`). These have been corrected to `figure3A_`/`figure3B_` and `figure5A_` respectively (and equivalently for `lancetph_figure1.py` and the tail of `lancetph_figure2.py`), so that filenames, in-code comments, and the manuscript are all consistent.
- **Colab-specific shell syntax has been removed.** Leftover `!pip install ...` lines (valid only inside a Colab/Jupyter cell) in `lancetph_figure3.py` and `lancetph_figure5.py` have been commented out; use `requirements.txt` instead.
- **A duplicated `from __future__ import annotations` statement** (a side effect of concatenating multiple Colab cells into one file) has been removed from `lancetph_figure3.py`, `lancetph_figure4.py`, and `lancetph_figure5.py`, which previously made those files fail to even parse as standalone `.py` files.
- **Hardcoded Google Drive paths have been replaced with a portable `BASE_DIR`.** Every script previously read/wrote files under `/content/drive/MyDrive/LancetPH/...`. Each script now defines:

  ```python
  BASE_DIR = Path(os.environ.get("LANCETPH_DATA_DIR", str(Path(__file__).resolve().parent)))
  ```

  By default this resolves to the directory the script itself is in, so placing `data_merged_quarters.csv` alongside the scripts (i.e. at the repository root) is enough to run them with no edits. To read/write elsewhere, set the `LANCETPH_DATA_DIR` environment variable rather than editing the scripts.

## Usage

```bash
pip install -r requirements.txt

# place data_merged_quarters.csv in the same directory as the scripts
# (or: export LANCETPH_DATA_DIR=/path/to/data)

python lancetph_figure1.py   # -> Figure1_outputs/
python lancetph_figure2.py   # -> CARE_DDI_rewritten_results_EVC_CNDcluster/
python lancetph_figure3.py   # -> figure/ (figure3A_*, figure3B_*)
python lancetph_figure4.py   # -> figure/ (figure4A_*, figure4B_*; requires a Natural Earth country-boundary file, see script header)
python lancetph_figure5.py   # -> figure/ (figure5A_*)

python lancetph_clustering.py   # -> clustering_outputs/ (see "Clustering step" below)
```

Each script can be run independently; none of them depend on the outputs of another (`lancetph_clustering.py` can be run either before or after the figure scripts -- it reads the same raw variable columns from `data_merged_quarters.csv`, it does not require the figure scripts to have run first). `lancetph_figure2.py` additionally writes `CARE_DDI_four_model_results_EVC_CNDcluster.xlsx`, the workbook underlying Supplementary Results S7.

None of the five figure scripts sets an explicit random seed; this does not affect their outputs (PCA in `lancetph_figure1.py`, correlations in `lancetph_figure3.py`, and OLS in `lancetph_figure5.py` are all deterministic given fixed input). `lancetph_clustering.py` does set a seed (`RANDOM_SEED = 123`, `n_init = 50`), since K-means is stochastic -- see below.

## Clustering step (`lancetph_clustering.py`)

This script derives the cluster-membership and composite-dimension columns that all five figure scripts read as pre-existing input (`Climate_cluster_5`, `GDD1-GDD5`, `GDD_cluster_5`, `GBD1-GBD5`, `GBD_cluster_5`). It was added after `preprocessing.r` (an earlier R draft of this step, also in this repository) was reviewed and found to only partially describe the pipeline that actually produced these columns.

- **Climate**: the 5 raw climate variables (PM25, MeanT, RH, PR, Heat_index) are standardised and clustered directly with K-means (k=5); no dimension reduction is applied, since there are only 5 climate inputs. Verified against `Climate_cluster_5` already in `data_merged_quarters.csv`: Adjusted Rand Index = 0.98.
- **Dietary (GDD)**: `GDD1-GDD5` are **not** scores from a single joint PCA across all 19 dietary variables (an earlier attempt at reconstructing them that way only reached ~0.77 mean correlation and is documented in this script's git history / prior commits for transparency). They are (weighted) means of five pre-defined food-group subsets (`EXPECTED_GDD_GROUPS` in the script), with Coffee and Tea converted to a gram-equivalent basis (`BEVERAGE_CONVERSION_G = 266.8`) before being averaged with the other variables in their group, since the source data reports them in different units. Reconstructing `GDD1-GDD5` this way and comparing to the columns already in `data_merged_quarters.csv` gives an **exact match**: correlation = 1.0, maximum absolute difference ≈ 1e-13 (floating-point noise). K-means on the reconstructed `GDD1-GDD5` then reproduces `GDD_cluster_5` with ARI = 1.00.
- **Disease burden (GBD)**: same approach, using 18 raw disease-burden variables grouped into `EXPECTED_GBD_GROUPS` (no unit conversion needed). Reconstruction of `GBD1-GBD5` is again an exact match (correlation = 1.0, diff ≈ 1e-12). K-means on the reconstructed `GBD1-GBD5` reproduces `GBD_cluster_5` with ARI = 0.98.
- A PCA diagnostic (`run_pca`) is also run on the raw variables in each domain, purely to characterise how they load onto latent components; it is not the mechanism used to build `GDD1-GDD5`/`GBD1-GBD5` (see `PCA_max_loading_vs_expected_dimensions.csv` in the output, which shows the pure-PCA grouping agrees with the actual group definitions on 66-68% of variables -- a useful diagnostic, not a discrepancy).

Running `lancetph_clustering.py` reproduces all of the above automatically and writes the full set of validation tables to `clustering_outputs/` (see the script's module docstring for a description of each output file).

## Known gaps (documented here for transparency)

- **`lancetph_figure2.py` contains two candidate CARE-DDI model specifications used in different parts of the script.** The script's `build_four_care_ddi_scores()` function computes four models (`baseline_equal`, `pca_weighted_linear`, `interaction_enhanced`, `conservative_interaction`) and labels `conservative_interaction` as the "recommended main model," but the script's own default plotting/export calls at the bottom of the file use `interaction_enhanced` for the combined Figure 2 panel. Which of these two was used to generate the Figure 2 published in the manuscript should be confirmed and made consistent before the final release.
- **18 vs. 19 dietary variables.** `lancetph_figure1.py` uses all 19 raw dietary variables listed in Supplementary Table S2 directly in the PCA/biplot step. An earlier draft of the Supplementary Appendix stated that cluster analysis used 18; this repository's code does not support that explanation, and the discrepancy is flagged as unresolved in the Supplementary Appendix (Methods S2).

## Citing / reproducibility mapping

A full figure-by-script-by-output-file mapping, package list, and step-by-step reproduction instructions are provided in the Reproducibility Appendix of the manuscript's Supplementary Appendix.
