import pandas as pd
from helper_functions import PROCESSED_DIR

dataset_centroid = pd.read_parquet(
    PROCESSED_DIR
    / "04_centroid_drift"
    / "centroid_drift_wpz_neighbors.parquet"
)

dataset_gaussian = pd.read_parquet(
    PROCESSED_DIR
    / "06_gaussian_distribution_drift"
    / "gaussian_w2_drift_wpz_neighbors.parquet"
)

merge_columns = [
    "modality",
    "genre",
    "window_start",
    "next_window_start",
]

dataset_combined = dataset_centroid.merge(
    dataset_gaussian,
    on=merge_columns,
    how="inner",
    validate="one_to_one",
    suffixes=("_centroid", "_gaussian"),
)

summary_rows = []

for modality, modality_data in dataset_combined.groupby("modality"):
    valid_cosine_w2 = modality_data[
        ["cosine_distance", "w2_distance"]
    ].dropna()

    valid_cosine_mean = modality_data[
        ["cosine_distance", "mean_component"]
    ].dropna()

    valid_cosine_std = modality_data[
        ["cosine_distance", "std_component"]
    ].dropna()

    row = {
        "modality": modality,
        "n_valid_transitions": len(valid_cosine_w2),
        "mean_cosine_drift": valid_cosine_w2["cosine_distance"].mean(),
        "median_cosine_drift": valid_cosine_w2["cosine_distance"].median(),
        "mean_w2_drift": valid_cosine_w2["w2_distance"].mean(),
        "median_w2_drift": valid_cosine_w2["w2_distance"].median(),
        "spearman_cosine_w2": valid_cosine_w2[
            "cosine_distance"
        ].corr(
            valid_cosine_w2["w2_distance"],
            method="spearman",
        ),
        "pearson_cosine_w2": valid_cosine_w2[
            "cosine_distance"
        ].corr(
            valid_cosine_w2["w2_distance"],
            method="pearson",
        ),
        "spearman_cosine_mean_component": valid_cosine_mean[
            "cosine_distance"
        ].corr(
            valid_cosine_mean["mean_component"],
            method="spearman",
        ),
        "spearman_cosine_std_component": valid_cosine_std[
            "cosine_distance"
        ].corr(
            valid_cosine_std["std_component"],
            method="spearman",
        ),
    }

    summary_rows.append(row)

modality_summary = pd.DataFrame(summary_rows)

print(dataset_combined.head())
print(dataset_combined.shape)
print(modality_summary)

output_path = (
    PROCESSED_DIR
    / "07_centroid_gaussian_comparison"
)

output_path.mkdir(parents=True, exist_ok=True)

dataset_combined.to_parquet(
    output_path / "centroid_gaussian_drift_comparison.parquet",
    index=False,
)

modality_summary.to_csv(
    output_path / "modality_drift_summary.csv",
    index=False,
)