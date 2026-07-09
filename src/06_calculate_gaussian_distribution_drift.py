import numpy as np
import pandas as pd
from utils.helper_functions import load_dataset_parquet, save_dataset_parquet, PROCESSED_DIR


GAUSSIAN_PATH = PROCESSED_DIR / "05_gaussian_representations"
DRIFT_PATH = PROCESSED_DIR / "06_gaussian_distribution_drift"

dataset_paths = {
    "essentia": "essentia_gaussian_wpz_windowed.parquet",
    "lyrics_tf_idf": "lyrics_tf_idf_gaussian_wpz_windowed.parquet",
    "word2vec": "word2vec_gaussian_wpz_windowed.parquet",
    "mfcc": "mfcc_gaussian_wpz_windowed.parquet",
    "musicnn": "musicnn_gaussian_wpz_windowed.parquet",
    "vgg19": "vgg19_gaussian_wpz_windowed.parquet",
}

metadata_columns = [
    "genre",
    "modality",
    "window_start",
    "window_end",
    "window_label",
    "n_tracks",
]

rows = []


def diagonal_gaussian_w2(mean_a, std_a, mean_b, std_b):
    mean_component_squared = np.sum((mean_a - mean_b) ** 2)
    std_component_squared = np.sum((std_a - std_b) ** 2)

    w2_squared = mean_component_squared + std_component_squared
    w2_distance = np.sqrt(w2_squared)

    mean_component = np.sqrt(mean_component_squared)
    std_component = np.sqrt(std_component_squared)

    return w2_distance, mean_component, std_component, w2_squared


for modality, dataset_path in dataset_paths.items():
    print(modality)

    gaussian_data = pd.read_parquet(GAUSSIAN_PATH / dataset_path)

    mean_columns = [
        col
        for col in gaussian_data.columns
        if col.startswith("mean_")
        and pd.api.types.is_numeric_dtype(gaussian_data[col])
    ]

    std_columns = [
        col
        for col in gaussian_data.columns
        if col.startswith("std_")
        and pd.api.types.is_numeric_dtype(gaussian_data[col])
    ]

    mean_columns = sorted(mean_columns)
    std_columns = sorted(std_columns)

    if len(mean_columns) != len(std_columns):
        raise ValueError(f"Mean and std column count differs for {modality}")

    gaussian_data = gaussian_data.sort_values(["genre", "window_start"])

    for genre, genre_data in gaussian_data.groupby("genre"):
        genre_data = genre_data.sort_values("window_start").reset_index(drop=True)

        for idx in range(len(genre_data) - 1):
            current_row = genre_data.iloc[idx]
            next_row = genre_data.iloc[idx + 1]

            mean_current = current_row[mean_columns].to_numpy(dtype=float)
            std_current = current_row[std_columns].to_numpy(dtype=float)

            mean_next = next_row[mean_columns].to_numpy(dtype=float)
            std_next = next_row[std_columns].to_numpy(dtype=float)

            w2_distance, mean_component, std_component, w2_squared = diagonal_gaussian_w2(
                mean_current,
                std_current,
                mean_next,
                std_next
            )

            row = {
                "modality": modality,
                "genre": genre,
                "window_start": current_row["window_start"],
                "window_end": current_row["window_end"],
                "next_window_start": next_row["window_start"],
                "next_window_end": next_row["window_end"],
                "window_gap": next_row["window_start"] - current_row["window_start"],
                "n_tracks_t": current_row["n_tracks"],
                "n_tracks_t1": next_row["n_tracks"],
                "w2_distance": w2_distance,
                "w2_squared": w2_squared,
                "mean_component": mean_component,
                "std_component": std_component,
            }

            rows.append(row)


distribution_drift = pd.DataFrame(rows)

print(distribution_drift.head())
print(distribution_drift.shape)

save_dataset_parquet(
    distribution_drift,
    "gaussian_w2_drift_wpz",
    save_dir=DRIFT_PATH
)

distribution_drift_neighbors = distribution_drift[
    distribution_drift["window_gap"] == 1
]

save_dataset_parquet(
    distribution_drift_neighbors,
    "gaussian_w2_drift_wpz_neighbors",
    save_dir=DRIFT_PATH
)

print("saved gaussian distribution drift")