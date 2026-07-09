import pandas as pd
from utils.helper_functions import load_dataset_parquet, save_dataset_parquet


window_size = 5
hop_size = 1
min_tracks = 20

dataset_paths = {
    "essentia": "essentia.parquet",
    "lyrics_tf_idf": "lyrics_tf_idf.parquet",
    "word2vec": "word2vec.parquet",
    "mfcc": "mfcc.parquet",
    "musicnn": "musicnn.parquet",
    "vgg19": "vgg19.parquet",
}

metadata_columns = ["id", "release", "genre"]

"""
This section computes raw sliding-window centroids.

For each modality, it creates five-year temporal windows with a one-year step.
Within each window, it groups tracks by genre and averages all feature columns.
The result is one raw centroid per genre, modality, and temporal window.
Normalization is not applied in this section.
"""

for modality, dataset_path in dataset_paths.items():
    print(modality)

    dataset = load_dataset_parquet(dataset_path)

    dataset["release"] = pd.to_numeric(dataset["release"], errors="coerce")
    dataset = dataset.dropna(subset=["release", "genre"])
    dataset["release"] = dataset["release"].astype(int)

    feature_columns = [
        col for col in dataset.columns
        if col not in metadata_columns
        and pd.api.types.is_numeric_dtype(dataset[col])
]

    min_year = dataset["release"].min()
    print(min_year)
    max_year = dataset["release"].max()

    centroid_rows = []

    for window_start in range(min_year - window_size + 1, max_year + 1, hop_size):
        window_end = window_start + window_size - 1

        window_data = dataset[
            (dataset["release"] >= window_start)
            & (dataset["release"] <= window_end)
        ]

        if window_data.empty:
            continue

        for genre, genre_data in window_data.groupby("genre"):
            if len(genre_data) < min_tracks:
                continue

            centroid = genre_data[feature_columns].mean()

            row = {
                "genre": genre,
                "modality": modality,
                "window_start": window_start,
                "window_end": window_end,
                "window_label": f"{window_start}_{window_end}",
                "n_tracks": len(genre_data),
            }

            row.update(centroid.to_dict())
            centroid_rows.append(row)

    centroids_windowed = pd.DataFrame(centroid_rows)

    print(centroids_windowed.head())
    print(centroids_windowed.shape)

    save_dataset_parquet(centroids_windowed, f"{modality}_centroids_raw_windowed")


"""
This section computes within-period z-normalized sliding-window centroids.

For each modality, it creates five-year temporal windows with a one-year step.
Within each window, all feature columns are z-normalized using only the tracks
inside that same modality-window. Then tracks are grouped by genre and averaged.
The result is one normalized centroid per genre, modality, and temporal window.

These centroids describe the relative position of a genre compared with the
contemporary feature distribution of the same modality and time window.
"""


eps = 1e-8

for modality, dataset_path in dataset_paths.items():
    print(modality)

    dataset = load_dataset_parquet(dataset_path)

    dataset["release"] = pd.to_numeric(dataset["release"], errors="coerce")
    dataset = dataset.dropna(subset=["release", "genre"])
    dataset["release"] = dataset["release"].astype(int)

    feature_columns = [
        col for col in dataset.columns
        if col not in metadata_columns
        and pd.api.types.is_numeric_dtype(dataset[col])
    ]

    min_year = dataset["release"].min()
    max_year = dataset["release"].max()

    centroid_rows = []

    for window_start in range(min_year - window_size + 1, max_year + 1, hop_size):
        window_end = window_start + window_size - 1

        window_data = dataset[
            (dataset["release"] >= window_start)
            & (dataset["release"] <= window_end)
        ].copy()

        if window_data.empty:
            continue

        means = window_data[feature_columns].mean()
        stds = window_data[feature_columns].std(ddof=0).replace(0, eps)

        window_data[feature_columns] = (
            window_data[feature_columns] - means
        ) / stds

        for genre, genre_data in window_data.groupby("genre"):
            if len(genre_data) < min_tracks:
                continue

            centroid = genre_data[feature_columns].mean()

            row = {
                "genre": genre,
                "modality": modality,
                "window_start": window_start,
                "window_end": window_end,
                "window_label": f"{window_start}_{window_end}",
                "n_tracks": len(genre_data),
            }

            row.update(centroid.to_dict())
            centroid_rows.append(row)

    centroids_windowed_z = pd.DataFrame(centroid_rows)

    print(centroids_windowed_z.head())
    print(centroids_windowed_z.shape)

    save_dataset_parquet(centroids_windowed_z, f"{modality}_centroids_wpz_windowed")