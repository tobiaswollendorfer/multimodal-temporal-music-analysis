import pandas as pd
from utils.helper_functions import load_dataset_parquet, save_dataset_parquet


dataset_paths = {
    "essentia": "essentia.parquet",
    "lyrics_tf_idf": "lyrics_tf_idf.parquet",
    "word2vec": "word2vec.parquet",
    "mfcc": "mfcc.parquet",
    "musicnn": "musicnn.parquet",
    "vgg19": "vgg19.parquet",
}

metadata_columns = ["id", "release", "genre"]

global_centroids = []

for modality, dataset_path in dataset_paths.items():
    print(modality)

    dataset = load_dataset_parquet(dataset_path)
    dataset = dataset.dropna(subset=["genre"])
    feature_columns = [
        col for col in dataset.columns
        if col not in metadata_columns
        and pd.api.types.is_numeric_dtype(dataset[col])
    ]

    centroids = (
        dataset
        .groupby("genre")[feature_columns]
        .mean()
        .reset_index()
    )

    centroids["modality"] = modality
    global_centroids.append(centroids)

global_centroids = pd.concat(global_centroids, ignore_index=True)

print(global_centroids.head())
print(global_centroids.shape)

save_dataset_parquet(global_centroids, "global_genre_centroids")