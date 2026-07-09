import pandas as pd
from utils.helper_functions import load_dataset_parquet, save_dataset_parquet, PROCESSED_DIR


GAUSSIAN_PATH = PROCESSED_DIR / "05_gaussian_representations"

window_size = 5
hop_size = 1
min_tracks = 30
eps = 1e-8

dataset_paths = {
    "essentia": "essentia.parquet",
    "lyrics_tf_idf": "lyrics_tf_idf.parquet",
    "word2vec": "word2vec.parquet",
    "mfcc": "mfcc.parquet",
    "musicnn": "musicnn.parquet",
    "vgg19": "vgg19.parquet",
}

metadata_columns = ["id", "release", "genre"]


for modality, dataset_path in dataset_paths.items():
    print(modality)

    dataset = load_dataset_parquet(dataset_path)

    dataset["release"] = pd.to_numeric(dataset["release"], errors="coerce")
    dataset = dataset.dropna(subset=["release", "genre"])
    dataset["release"] = dataset["release"].astype(int)

    feature_columns = [
        col
        for col in dataset.columns
        if col not in metadata_columns
        and pd.api.types.is_numeric_dtype(dataset[col])
    ]

    min_year = dataset["release"].min()
    max_year = dataset["release"].max()

    rows = []

    for window_start in range(min_year - window_size + 1, max_year + 1, hop_size):
        window_end = window_start + window_size - 1

        window_data = dataset[
            (dataset["release"] >= window_start)
            & (dataset["release"] <= window_end)
        ].copy()

        if window_data.empty:
            continue

        window_means = window_data[feature_columns].mean()
        window_stds = window_data[feature_columns].std(ddof=0).replace(0, eps)

        window_data[feature_columns] = (
            window_data[feature_columns] - window_means
        ) / window_stds

        for genre, genre_data in window_data.groupby("genre"):
            if len(genre_data) < min_tracks:
                continue

            genre_features = genre_data[feature_columns]

            mean_vector = genre_features.mean()
            var_vector = genre_features.var(ddof=0).replace(0, eps)
            std_vector = genre_features.std(ddof=0).replace(0, eps)

            row = {
                "genre": genre,
                "modality": modality,
                "window_start": window_start,
                "window_end": window_end,
                "window_label": f"{window_start}_{window_end}",
                "n_tracks": len(genre_data),
            }

            for col in feature_columns:
                row[f"mean_{col}"] = mean_vector[col]
                row[f"var_{col}"] = var_vector[col]
                row[f"std_{col}"] = std_vector[col]

            rows.append(row)

    gaussian_representations = pd.DataFrame(rows)

    print(gaussian_representations.head())
    print(gaussian_representations.shape)

    save_dataset_parquet(
        gaussian_representations,
        f"{modality}_gaussian_wpz_windowed",
        save_dir=GAUSSIAN_PATH
    )

print("saved gaussian representations")