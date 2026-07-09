import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.helper_functions import load_dataset_parquet, save_dataset_parquet, PROCESSED_DIR


DRIFT_PATH = PROCESSED_DIR / "04_centroid_drift"

dataset_paths = {
    "essentia": "essentia_centroids_wpz_windowed.parquet",
    "lyrics_tf_idf": "lyrics_tf_idf_centroids_wpz_windowed.parquet",
    "word2vec": "word2vec_centroids_wpz_windowed.parquet",
    "mfcc": "mfcc_centroids_wpz_windowed.parquet",
    "musicnn": "musicnn_centroids_wpz_windowed.parquet",
    "vgg19": "vgg19_centroids_wpz_windowed.parquet",
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

for modality, dataset_path in dataset_paths.items():
    print(modality)

    centroids = load_dataset_parquet(dataset_path)

    feature_columns = [
        col
        for col in centroids.columns
        if col not in metadata_columns
        and pd.api.types.is_numeric_dtype(centroids[col])
    ]

    centroids = centroids.sort_values(["genre", "window_start"])

    for genre, genre_data in centroids.groupby("genre"):
        genre_data = genre_data.sort_values("window_start").reset_index(drop=True)

        for idx in range(len(genre_data) - 1):
            current_row = genre_data.iloc[idx]
            next_row = genre_data.iloc[idx + 1]

            current_vector = current_row[feature_columns].to_numpy(dtype=float).reshape(1, -1)
            next_vector = next_row[feature_columns].to_numpy(dtype=float).reshape(1, -1)

            similarity = cosine_similarity(current_vector, next_vector)[0, 0]
            distance = 1 - similarity if not np.isnan(similarity) else np.nan

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
                "cosine_similarity": similarity,
                "cosine_distance": distance,
            }

            rows.append(row)

drift = pd.DataFrame(rows)

save_dataset_parquet(
    drift,
    "centroid_drift_wpz",
    save_dir=DRIFT_PATH
)

drift_neighboring = drift[drift["window_gap"] == 1]
save_dataset_parquet(
    drift_neighboring,
    "centroid_drift_wpz_neighbors",
    save_dir=DRIFT_PATH
)
print("saved centroid drift")

print("describe")
print(drift.groupby("modality")["cosine_distance"].describe())
print("window_gap")
print(drift["window_gap"].value_counts().sort_index())
print("describe neighbors")
print(drift_neighboring.groupby("modality")["cosine_distance"].describe())