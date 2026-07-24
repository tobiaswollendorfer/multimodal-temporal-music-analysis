from math import comb
import pandas as pd
from utils.helper_functions import load_dataset_parquet, save_dataset_parquet, PROCESSED_DIR


DIAG_PATH = PROCESSED_DIR / "02_coverage_diagnostics"

dataset_paths = {
    "essentia": "essentia.parquet",
    "lyrics_tf_idf": "lyrics_tf_idf.parquet",
    "word2vec": "word2vec.parquet",
    "mfcc": "mfcc.parquet",
    "musicnn": "musicnn.parquet",
    "vgg19": "vgg19.parquet",
}

window_size = 5
hop_size = 1

centroid_min_tracks = 20
distribution_min_tracks = 30
gmm_min_tracks = 50

rows = []


def pair_count(n):
    if n < 2:
        return 0
    return comb(n, 2)


for modality, dataset_path in dataset_paths.items():
    print(modality)

    dataset = load_dataset_parquet(dataset_path)

    dataset["release"] = pd.to_numeric(dataset["release"], errors="coerce")
    dataset = dataset.dropna(subset=["release", "genre"])
    dataset["release"] = dataset["release"].astype(int)

    min_year = dataset["release"].min()
    max_year = dataset["release"].max()

    for window_start in range(min_year - window_size + 1, max_year + 1, hop_size):
        window_end = window_start + window_size - 1

        window_data = dataset[
            (dataset["release"] >= window_start)
            & (dataset["release"] <= window_end)
        ]

        if window_data.empty:
            continue

        genre_counts = (
            window_data
            .groupby("genre")
            .size()
            .reset_index(name="n_tracks")
        )

        n_tracks = len(window_data)
        n_genres = genre_counts["genre"].nunique()

        valid_centroid = genre_counts[genre_counts["n_tracks"] >= centroid_min_tracks]
        valid_distribution = genre_counts[genre_counts["n_tracks"] >= distribution_min_tracks]
        valid_gmm = genre_counts[genre_counts["n_tracks"] >= gmm_min_tracks]

        row = {
            "modality": modality,
            "window_start": window_start,
            "window_end": window_end,
            "window_label": f"{window_start}_{window_end}",
            "n_tracks": n_tracks,
            "n_genres": n_genres,
            "n_valid_genres_centroid": len(valid_centroid),
            "n_valid_genres_distribution": len(valid_distribution),
            "n_valid_genres_gmm": len(valid_gmm),
            "n_valid_pairs_centroid": pair_count(len(valid_centroid)),
            "n_valid_pairs_distribution": pair_count(len(valid_distribution)),
            "n_valid_pairs_gmm": pair_count(len(valid_gmm)),
            "min_tracks_per_genre": genre_counts["n_tracks"].min(),
            "median_tracks_per_genre": genre_counts["n_tracks"].median(),
            "mean_tracks_per_genre": genre_counts["n_tracks"].mean(),
            "max_tracks_per_genre": genre_counts["n_tracks"].max(),
        }

        rows.append(row)


coverage = pd.DataFrame(rows)

print(coverage.head())
print(coverage.shape)

save_dataset_parquet(
    coverage,
    "coverage_diagnostics",
    save_dir=DIAG_PATH
)
print("saved coverage diagnostics")