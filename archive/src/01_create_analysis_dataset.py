from pathlib import Path
import pandas as pd
from utils.helper_functions import load_dataset_csv, save_dataset_parquet


dataset_paths = {
    "essentia": "id_essentia.tsv",
    "lyrics_tf_idf": "id_lyrics_tf-idf.tsv",
    "word2vec": "id_lyrics_word2vec.tsv",
    "mfcc": "id_mfcc_stats.tsv",
    "musicnn": "id_musicnn.tsv",
    "vgg19": "id_vgg19.tsv",
}

genres_path = "id_genres.csv"
genres = load_dataset_csv(genres_path)

genres["genre"] = genres["genres"].astype(str).str.split(",").str[0].str.strip()
genres = genres.drop('genres', axis=1)
metadata_path = "id_metadata.csv"
metadata = load_dataset_csv(metadata_path)

for modality, dataset_path in dataset_paths.items():
    print(modality)

    features = load_dataset_csv(dataset_path)

    merged_genres = pd.merge(genres, features, on="id")
    merged_metadata = pd.merge(
        metadata[["id", "release"]],
        merged_genres,
        on="id"
    )

    save_dataset_parquet(merged_metadata, f"{modality}")