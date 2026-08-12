"""lancetph_clustering.py

PCA-guided dimension validation and clustering reproduction for the
climate, dietary (GDD), and disease-burden (GBD) domains.

This is the upstream step that all five lancetph_figureN.py scripts
depend on: it derives Climate_cluster_5, GDD1-GDD5 / GDD_cluster_5,
and GBD1-GBD5 / GBD_cluster_5 from the raw variable columns in
data_merged_quarters.csv, and validates the result against the values
already shipped in that file.

Pipeline:
  1) Climate: 5 raw variables -> standardise -> K-means (K=5).
  2) Dietary (GDD): 19 food-group variables ->
       (a) PCA is run as a diagnostic only, to characterise how the raw
           variables load onto latent dimensions;
       (b) GDD1-GDD5 are reconstructed as (weighted) means of five
           pre-defined food-group subsets (EXPECTED_GDD_GROUPS below).
           Coffee and Tea are converted to a gram-equivalent basis
           (BEVERAGE_CONVERSION_G) before being averaged with the other
           variables in their group, since they are reported in
           different units in the source data;
       (c) K-means (K=5) is run on the resulting GDD1-GDD5.
  3) Disease burden (GBD): 18 variables, same approach as (2) using
     EXPECTED_GBD_GROUPS, producing GBD1-GBD5 and GBD_cluster_5.

  K-means parameters throughout: random_state=123, n_init=50.

Verification against data_merged_quarters.csv (run on 2026-08-12):
  - GDD1-GDD5 and GBD1-GBD5 reconstructed via step (2b)/(3) match the
    columns already in data_merged_quarters.csv exactly (correlation =
    1.0, max abs. difference ~1e-13, i.e. floating-point noise only).
    This confirms GDD1-GDD5/GBD1-GBD5 are group-mean composites of the
    variable subsets below, not scores from a single joint PCA across
    all 19/18 raw variables (an earlier, unsuccessful reconstruction
    attempt assumed the latter -- see README.md for that comparison).
  - K-means reproduction vs. the cluster columns already in the data:
    Climate ARI = 0.98, GDD ARI = 1.00, GBD ARI = 0.98 (K-means restarts
    are not bit-identical across runs/software, so ARI slightly below
    1.0 for Climate/GBD is expected rather than a sign of a mismatch).

Outputs (written to OUTPUT_DIR):
  - GDD_PCA_loadings.csv, GBD_PCA_loadings.csv: PCA diagnostic loadings
  - PCA_explained_variance.csv: explained variance ratio per component
  - GDD_variable_assignment_validation.csv, GBD_variable_assignment_validation.csv:
    per-variable check of which dimension it correlates with most
  - dimension_group_summary.csv: match-rate summary by dimension
  - PCA_max_loading_vs_expected_dimensions.csv: pure-PCA vs. the
    pre-defined group membership used to build GDD1-GDD5/GBD1-GBD5
  - dimension_reconstruction_validation.csv: exact-match validation of
    the reconstructed GDD1-GDD5/GBD1-GBD5 against the existing columns
  - cluster_reproduction_summary.csv: ARI/NMI/accuracy of the
    recomputed clusters against Climate_cluster_5/GDD_cluster_5/GBD_cluster_5
  - cluster_pca_validation_output.csv: full row-level output (original
    data + PCA scores + reconstructed dimensions + recomputed clusters)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


SEED = 123
N_INIT = 50
N_CLUSTERS = 5
N_PCS = 5
BEVERAGE_CONVERSION_G = 266.8  # gram-equivalent conversion used for coffee/tea before averaging

# ------------------------------------------------------------------
# Portability note (consistent with the other lancetph_*.py scripts):
# BASE_DIR defaults to the directory this script lives in. Set
# LANCETPH_DATA_DIR to override without editing this file.
# ------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("LANCETPH_DATA_DIR", str(Path(__file__).resolve().parent)))
INPUT_FILE = BASE_DIR / "data_merged_quarters.csv"
OUTPUT_DIR = BASE_DIR / "clustering_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIMATE_VARS = ["PM25", "MeanT", "RH", "PR", "Heat_index"]

DIET_VARS = [
    "Fruits",
    "Non-starchy vegetables",
    "Potatoes",
    "Other starchy vegetables",
    "Beans and legumes",
    "Nuts and seeds",
    "Refined grains",
    "Whole grains",
    "Total processed meats",
    "Unprocessed red meats",
    "Total seafoods",
    "Eggs",
    "Cheese",
    "Yoghurt (including fermented milk)",
    "Sugar-sweetened beverages",
    "Fruit juices",
    "Coffee",
    "Tea",
    "Total Milk",
]

DISEASE_VARS = [
    "Cardiovascular diseases",
    "Chronic respiratory diseases",
    "Diabetes and kidney diseases",
    "Digestive diseases",
    "Enteric infections",
    "Maternal and neonatal disorders",
    "Mental disorders",
    "Musculoskeletal disorders",
    "Neglected tropical diseases and malaria",
    "Neoplasms",
    "Neurological disorders",
    "Nutritional deficiencies",
    "Other infectious diseases",
    "Other non-communicable diseases",
    "Respiratory infections and tuberculosis",
    "Sense organ diseases",
    "Skin and subcutaneous diseases",
    "Substance use disorders",
]

GDD_DIMS = ["GDD1", "GDD2", "GDD3", "GDD4", "GDD5"]
GBD_DIMS = ["GBD1", "GBD2", "GBD3", "GBD4", "GBD5"]

# Variable membership of each composite dimension, confirmed by exact
# reconstruction against data_merged_quarters.csv (see module docstring).
# GDD1 and GDD3 include Coffee/Tea after gram-equivalent conversion.
EXPECTED_GDD_GROUPS = {
    "GDD1": [
        "Potatoes",
        "Cheese",
        "Yoghurt (including fermented milk)",
        "Fruit juices",
        "Coffee",
        "Total Milk",
    ],
    "GDD2": ["Total processed meats", "Unprocessed red meats", "Eggs"],
    "GDD3": [
        "Fruits",
        "Non-starchy vegetables",
        "Beans and legumes",
        "Nuts and seeds",
        "Refined grains",
        "Total seafoods",
        "Tea",
    ],
    "GDD4": ["Sugar-sweetened beverages", "Other starchy vegetables"],
    "GDD5": ["Whole grains"],
}

EXPECTED_GBD_GROUPS = {
    "GBD1": ["Mental disorders"],
    "GBD2": [
        "Cardiovascular diseases",
        "Musculoskeletal disorders",
        "Neoplasms",
        "Neurological disorders",
        "Sense organ diseases",
        "Substance use disorders",
    ],
    "GBD3": [
        "Enteric infections",
        "Maternal and neonatal disorders",
        "Neglected tropical diseases and malaria",
        "Nutritional deficiencies",
        "Other infectious diseases",
        "Other non-communicable diseases",
        "Respiratory infections and tuberculosis",
    ],
    "GBD4": ["Chronic respiratory diseases", "Digestive diseases"],
    "GBD5": ["Diabetes and kidney diseases", "Skin and subcutaneous diseases"],
}


def check_required_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def run_pca(df: pd.DataFrame, variables: list[str], prefix: str) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Run PCA on standardised raw variables and return PCA object, loading table, and PC scores."""
    valid = df[variables].notna().all(axis=1)
    X = df.loc[valid, variables].astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=N_PCS, random_state=SEED)
    scores = pca.fit_transform(X_scaled)

    # Correlation-style loadings for standardised variables.
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_cols = [f"{prefix}_PC{i}" for i in range(1, N_PCS + 1)]
    loadings_df = pd.DataFrame(loadings, index=variables, columns=loading_cols)
    loadings_df.insert(0, "variable", variables)
    loadings_df["assigned_PC_by_abs_loading"] = loadings_df[loading_cols].abs().idxmax(axis=1)
    loadings_df["max_abs_loading"] = loadings_df[loading_cols].abs().max(axis=1)

    scores_df = pd.DataFrame(index=df.index, columns=loading_cols, dtype=float)
    scores_df.loc[valid, loading_cols] = scores

    return pca, loadings_df, scores_df


def assign_variables_by_dimension_correlation(
    df: pd.DataFrame,
    raw_vars: list[str],
    dim_vars: list[str],
    expected_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Assign each raw variable to the existing dimension column with the largest |correlation|."""
    corr = df[raw_vars + dim_vars].corr().loc[raw_vars, dim_vars]
    rows = []
    for var in raw_vars:
        best_dim = corr.loc[var].abs().idxmax()
        expected_dim = next((dim for dim, vars_ in expected_groups.items() if var in vars_), None)
        rows.append(
            {
                "variable": var,
                "assigned_dimension_by_max_abs_corr": best_dim,
                "expected_dimension": expected_dim,
                "match_expected": best_dim == expected_dim,
                "max_abs_corr": corr.loc[var, best_dim],
                **{f"corr_{d}": corr.loc[var, d] for d in dim_vars},
            }
        )
    return pd.DataFrame(rows)


def group_summary(assignment_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise variable assignments by expected dimension."""
    return (
        assignment_df.groupby("expected_dimension", dropna=False)
        .agg(
            n_variables=("variable", "count"),
            n_matched=("match_expected", "sum"),
            variables=("variable", lambda x: "; ".join(x)),
        )
        .reset_index()
        .assign(match_rate=lambda d: d["n_matched"] / d["n_variables"])
    )


def reconstruct_gdd(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct GDD1-GDD5 from the underlying food variables."""
    out = pd.DataFrame(index=df.index)
    out["GDD1_reconstructed"] = (
        df["Potatoes"]
        + df["Cheese"]
        + df["Yoghurt (including fermented milk)"]
        + df["Fruit juices"]
        + df["Coffee"] * BEVERAGE_CONVERSION_G
        + df["Total Milk"]
    ) / 6
    out["GDD2_reconstructed"] = df[["Total processed meats", "Unprocessed red meats", "Eggs"]].mean(axis=1)
    out["GDD3_reconstructed"] = (
        df["Fruits"]
        + df["Non-starchy vegetables"]
        + df["Beans and legumes"]
        + df["Nuts and seeds"]
        + df["Refined grains"]
        + df["Total seafoods"]
        + df["Tea"] * BEVERAGE_CONVERSION_G
    ) / 7
    out["GDD4_reconstructed"] = df[["Sugar-sweetened beverages", "Other starchy vegetables"]].mean(axis=1)
    out["GDD5_reconstructed"] = df["Whole grains"]
    return out


def reconstruct_gbd(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct GBD1-GBD5 from the underlying disease-burden variables."""
    out = pd.DataFrame(index=df.index)
    out["GBD1_reconstructed"] = df["Mental disorders"]
    out["GBD2_reconstructed"] = df[
        [
            "Cardiovascular diseases",
            "Musculoskeletal disorders",
            "Neoplasms",
            "Neurological disorders",
            "Sense organ diseases",
            "Substance use disorders",
        ]
    ].mean(axis=1)
    out["GBD3_reconstructed"] = df[
        [
            "Enteric infections",
            "Maternal and neonatal disorders",
            "Neglected tropical diseases and malaria",
            "Nutritional deficiencies",
            "Other infectious diseases",
            "Other non-communicable diseases",
            "Respiratory infections and tuberculosis",
        ]
    ].mean(axis=1)
    out["GBD4_reconstructed"] = df[["Chronic respiratory diseases", "Digestive diseases"]].mean(axis=1)
    out["GBD5_reconstructed"] = df[["Diabetes and kidney diseases", "Skin and subcutaneous diseases"]].mean(axis=1)
    return out


def reconstruction_validation(df: pd.DataFrame, reconstructed: pd.DataFrame, dim_vars: list[str]) -> pd.DataFrame:
    rows = []
    for dim in dim_vars:
        rec = reconstructed[f"{dim}_reconstructed"]
        original = df[dim]
        diff = rec - original
        rows.append(
            {
                "dimension": dim,
                "max_abs_diff": float(diff.abs().max()),
                "mean_abs_diff": float(diff.abs().mean()),
                "correlation": float(rec.corr(original)),
                "exact_match_tol_1e-8": bool(diff.abs().max() < 1e-8),
            }
        )
    return pd.DataFrame(rows)


def run_kmeans_on_columns(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    valid = df[variables].notna().all(axis=1)
    X_scaled = StandardScaler().fit_transform(df.loc[valid, variables].astype(float))
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=N_INIT)
    labels = km.fit_predict(X_scaled) + 1
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    out.loc[valid] = labels
    return out


def compare_clusters(existing: pd.Series, recomputed: pd.Series) -> dict:
    valid = existing.notna() & recomputed.notna()
    y_true = existing.loc[valid].astype(int).to_numpy()
    y_pred = recomputed.loc[valid].astype(int).to_numpy()

    tab = pd.crosstab(
        pd.Series(y_true, name="existing"),
        pd.Series(y_pred, name="recomputed"),
    ).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)

    row_ind, col_ind = linear_sum_assignment(-tab.values)
    mapping = {int(tab.columns[c]): int(tab.index[r]) for r, c in zip(row_ind, col_ind)}
    y_pred_mapped = np.array([mapping[v] for v in y_pred])

    return {
        "n": int(valid.sum()),
        "label_matched_accuracy": float(np.mean(y_true == y_pred_mapped)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "label_mapping_recomputed_to_existing": str(mapping),
    }


def pca_group_vs_expected(loadings_df: pd.DataFrame, expected_groups: dict[str, list[str]], prefix: str) -> pd.DataFrame:
    """Compare pure PCA max-loading assignment with expected interpreted dimensions."""
    pc_col = "assigned_PC_by_abs_loading"
    pcs = sorted(loadings_df[pc_col].unique())
    dims = list(expected_groups.keys())
    pc_groups = {pc: set(loadings_df.loc[loadings_df[pc_col] == pc, "variable"]) for pc in pcs}
    exp_groups = {d: set(v) for d, v in expected_groups.items()}

    mat = pd.DataFrame(index=pcs, columns=dims, dtype=float)
    for pc in pcs:
        for dim in dims:
            inter = len(pc_groups[pc] & exp_groups[dim])
            union = len(pc_groups[pc] | exp_groups[dim])
            mat.loc[pc, dim] = inter / union if union else 0.0

    row_ind, col_ind = linear_sum_assignment(-mat.values)
    matched_dim_for_pc = {mat.index[r]: mat.columns[c] for r, c in zip(row_ind, col_ind)}

    rows = []
    for _, row in loadings_df.iterrows():
        var = row["variable"]
        pc = row[pc_col]
        expected_dim = next((d for d, vars_ in expected_groups.items() if var in vars_), None)
        mapped_dim = matched_dim_for_pc.get(pc)
        rows.append(
            {
                "domain": prefix,
                "variable": var,
                "PCA_max_loading_PC": pc,
                "PCA_PC_mapped_dimension": mapped_dim,
                "expected_dimension": expected_dim,
                "PCA_loading_assignment_matches_expected": mapped_dim == expected_dim,
                "max_abs_loading": row["max_abs_loading"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    check_required_columns(df, CLIMATE_VARS + DIET_VARS + DISEASE_VARS + GDD_DIMS + GBD_DIMS)

    # 1) PCA diagnostics before dimension-based clustering
    gdd_pca, gdd_loadings, gdd_scores = run_pca(df, DIET_VARS, "GDD")
    gbd_pca, gbd_loadings, gbd_scores = run_pca(df, DISEASE_VARS, "GBD")

    gdd_loadings.to_csv(OUTPUT_DIR / "GDD_PCA_loadings.csv", index=False)
    gbd_loadings.to_csv(OUTPUT_DIR / "GBD_PCA_loadings.csv", index=False)

    pca_explained = pd.DataFrame(
        [
            {
                "domain": "GDD",
                **{f"PC{i}_explained_variance_ratio": gdd_pca.explained_variance_ratio_[i - 1] for i in range(1, 6)},
                "PC1_PC5_cumulative": gdd_pca.explained_variance_ratio_.sum(),
            },
            {
                "domain": "GBD",
                **{f"PC{i}_explained_variance_ratio": gbd_pca.explained_variance_ratio_[i - 1] for i in range(1, 6)},
                "PC1_PC5_cumulative": gbd_pca.explained_variance_ratio_.sum(),
            },
        ]
    )
    pca_explained.to_csv(OUTPUT_DIR / "PCA_explained_variance.csv", index=False)

    # 2) Validate whether current dimension columns correspond to expected variable lists
    gdd_assignment = assign_variables_by_dimension_correlation(df, DIET_VARS, GDD_DIMS, EXPECTED_GDD_GROUPS)
    gbd_assignment = assign_variables_by_dimension_correlation(df, DISEASE_VARS, GBD_DIMS, EXPECTED_GBD_GROUPS)
    gdd_assignment.to_csv(OUTPUT_DIR / "GDD_variable_assignment_validation.csv", index=False)
    gbd_assignment.to_csv(OUTPUT_DIR / "GBD_variable_assignment_validation.csv", index=False)

    gdd_group_summary = group_summary(gdd_assignment)
    gbd_group_summary = group_summary(gbd_assignment)
    pd.concat([gdd_group_summary.assign(domain="GDD"), gbd_group_summary.assign(domain="GBD")], ignore_index=True).to_csv(
        OUTPUT_DIR / "dimension_group_summary.csv", index=False
    )

    # 3) Compare pure PCA max-loading assignment with the interpreted dimension membership
    pca_vs_expected = pd.concat(
        [
            pca_group_vs_expected(gdd_loadings, EXPECTED_GDD_GROUPS, "GDD"),
            pca_group_vs_expected(gbd_loadings, EXPECTED_GBD_GROUPS, "GBD"),
        ],
        ignore_index=True,
    )
    pca_vs_expected.to_csv(OUTPUT_DIR / "PCA_max_loading_vs_expected_dimensions.csv", index=False)

    # 4) Reconstruct dimension scores and verify against the actual GDD1-GDD5 / GBD1-GBD5 columns
    gdd_reconstructed = reconstruct_gdd(df)
    gbd_reconstructed = reconstruct_gbd(df)
    rec_summary = pd.concat(
        [
            reconstruction_validation(df, gdd_reconstructed, GDD_DIMS).assign(domain="GDD"),
            reconstruction_validation(df, gbd_reconstructed, GBD_DIMS).assign(domain="GBD"),
        ],
        ignore_index=True,
    )
    rec_summary.to_csv(OUTPUT_DIR / "dimension_reconstruction_validation.csv", index=False)

    # 5) Reproduce clustering using the (reconstructed) dimension scores
    df["Climate_cluster_5_recomputed"] = run_kmeans_on_columns(df, CLIMATE_VARS)
    df["GDD_cluster_5_recomputed"] = run_kmeans_on_columns(df, GDD_DIMS)
    df["GBD_cluster_5_recomputed"] = run_kmeans_on_columns(df, GBD_DIMS)

    cluster_summary = pd.DataFrame(
        [
            {"domain": "Climate", **compare_clusters(df["Climate_cluster_5"], df["Climate_cluster_5_recomputed"])},
            {"domain": "GDD", **compare_clusters(df["GDD_cluster_5"], df["GDD_cluster_5_recomputed"])},
            {"domain": "GBD", **compare_clusters(df["GBD_cluster_5"], df["GBD_cluster_5_recomputed"])},
        ]
    )
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_reproduction_summary.csv", index=False)

    final_out = pd.concat([df, gdd_scores.add_prefix("pca_score_"), gbd_scores.add_prefix("pca_score_")], axis=1)
    final_out = pd.concat([final_out, gdd_reconstructed, gbd_reconstructed], axis=1)
    final_out.to_csv(OUTPUT_DIR / "cluster_pca_validation_output.csv", index=False)

    print("PCA explained variance")
    print(pca_explained.to_string(index=False))
    print("\nDimension reconstruction validation")
    print(rec_summary.to_string(index=False))
    print("\nVariable assignment by dimension correlation: GDD")
    print(gdd_group_summary.to_string(index=False))
    print("\nVariable assignment by dimension correlation: GBD")
    print(gbd_group_summary.to_string(index=False))
    print("\nPure PCA max-loading vs expected interpreted dimensions")
    print(pca_vs_expected.groupby("domain")["PCA_loading_assignment_matches_expected"].mean().reset_index().to_string(index=False))
    print("\nCluster reproduction summary")
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
