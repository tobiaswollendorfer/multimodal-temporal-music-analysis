from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.helper_functions import (
    load_dataset_parquet,
    save_dataset_parquet,
    PROCESSED_DIR,
)


PAIRWISE_PATH = PROCESSED_DIR / "08_pairwise_centroid_similarity"

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
        column
        for column in centroids.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(centroids[column])
    ]

    for window_start, window_data in centroids.groupby("window_start"):
        window_data = (
            window_data
            .sort_values("genre")
            .reset_index(drop=True)
        )

        if len(window_data) < 2:
            continue

        for index_a, index_b in combinations(window_data.index, 2):
            row_a = window_data.loc[index_a]
            row_b = window_data.loc[index_b]

            vector_a = row_a[feature_columns].to_numpy(
                dtype=float
            ).reshape(1, -1)

            vector_b = row_b[feature_columns].to_numpy(
                dtype=float
            ).reshape(1, -1)

            similarity = cosine_similarity(
                vector_a,
                vector_b
            )[0, 0]

            distance = (
                1 - similarity
                if not np.isnan(similarity)
                else np.nan
            )

            rows.append({
                "modality": modality,
                "window_start": window_start,
                "window_end": row_a["window_end"],
                "window_label": row_a["window_label"],
                "genre_a": row_a["genre"],
                "genre_b": row_b["genre"],
                "n_tracks_a": row_a["n_tracks"],
                "n_tracks_b": row_b["n_tracks"],
                "cosine_similarity": similarity,
                "cosine_distance": distance,
            })

pairwise_similarity = pd.DataFrame(rows)

print(pairwise_similarity.head())
print(pairwise_similarity.shape)

save_dataset_parquet(
    pairwise_similarity,
    "pairwise_centroid_similarity_wpz",
    save_dir=PAIRWISE_PATH,
)

temporal_summary = (
    pairwise_similarity
    .groupby(["modality", "window_start", "window_end"])
    .agg(
        n_valid_pairs=("cosine_similarity", "count"),
        n_valid_genres=(
            "genre_a",
            lambda values: len(
                set(values).union(
                    set(
                        pairwise_similarity.loc[
                            values.index,
                            "genre_b"
                        ]
                    )
                )
            )
        ),
        mean_cosine_similarity=("cosine_similarity", "mean"),
        median_cosine_similarity=("cosine_similarity", "median"),
        std_cosine_similarity=("cosine_similarity", "std"),
        q25_cosine_similarity=(
            "cosine_similarity",
            lambda values: values.quantile(0.25)
        ),
        q75_cosine_similarity=(
            "cosine_similarity",
            lambda values: values.quantile(0.75)
        ),
        mean_cosine_distance=("cosine_distance", "mean"),
        median_cosine_distance=("cosine_distance", "median"),
    )
    .reset_index()
)

print(temporal_summary.head())
print(temporal_summary.shape)

save_dataset_parquet(
    temporal_summary,
    "pairwise_centroid_similarity_temporal_summary",
    save_dir=PAIRWISE_PATH,
)

print("saved pairwise centroid similarity")