"""
Calculate sliding-window centroids using the final OR-Tools genre assignment.

Outputs are saved to:
    data/processed/03_ortools_centroids/

This script:
    1. Loads the final OR-Tools song-to-genre assignments.
    2. Loads each processed modality dataset.
    3. Replaces the old genre column with the OR-Tools genre assignment.
    4. Keeps only the selected OR-Tools genres.
    5. Computes raw five-year sliding-window centroids.
    6. Computes within-period z-normalized five-year sliding-window centroids.

Example:
    python src/03_calculate_centroids_ortools.py \
        --prefix ortools_top50_mintracks20_topcandidates5_genrepool180 \
        --min-tracks 20
"""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from utils.helper_functions import (
    PROCESSED_DIR,
    load_dataset_parquet,
    save_dataset_parquet,
)


window_size = 5
hop_size = 1
analysis_min_year = 1955
analysis_max_year = 2019

dataset_paths = {
    "essentia": "essentia.parquet",
    "lyrics_tf_idf": "lyrics_tf_idf.parquet",
    "word2vec": "word2vec.parquet",
    "mfcc": "mfcc.parquet",
    "musicnn": "musicnn.parquet",
    "vgg19": "vgg19.parquet",
}

metadata_columns = [
    "id",
    "release",
    "genre",
    "genre_score",
]


def load_ortools_assignment(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_dir = PROCESSED_DIR / "09_selected_genres_ortools"

    assignments = pd.read_parquet(
        assignment_dir / f"{prefix}_assignments.parquet"
    )

    selected_genres = pd.read_csv(
        assignment_dir / f"{prefix}_selected_genres.csv"
    )

    assignments = assignments[["id", "genre", "genre_score"]].copy()

    selected_genres = selected_genres[["genre"]].drop_duplicates().copy()
    selected_names = set(selected_genres["genre"])

    assignments = assignments[
        assignments["genre"].isin(selected_names)
    ].copy()

    return assignments, selected_genres


def prepare_modality_dataset(
    modality: str,
    dataset_path: str,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    dataset = load_dataset_parquet(dataset_path)

    dataset["release"] = pd.to_numeric(dataset["release"], errors="coerce")
    dataset = dataset.dropna(subset=["release"]).copy()
    dataset["release"] = dataset["release"].astype(int)

    dataset = dataset[
        (dataset["release"] >= analysis_min_year)
        & (dataset["release"] <= analysis_max_year)
    ].copy()

    if "genre" in dataset.columns:
        dataset = dataset.drop(columns=["genre"])

    if "genre_score" in dataset.columns:
        dataset = dataset.drop(columns=["genre_score"])

    dataset = dataset.merge(
        assignments,
        on="id",
        how="inner",
    )

    dataset = dataset.dropna(subset=["genre"]).copy()

    feature_columns = [
        column
        for column in dataset.columns
        if column not in metadata_columns
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]

    print()
    print(f"Prepared {modality}")
    print(f"Rows: {len(dataset):,}")
    print(f"Genres: {dataset['genre'].nunique():,}")
    print(f"Feature columns: {len(feature_columns):,}")
    print(f"Release range: {dataset['release'].min()}-{dataset['release'].max()}")

    return dataset, feature_columns


def prepare_normalization_dataset(
    dataset_path: str,
    assignments: pd.DataFrame,
    normalization_scope: str,
) -> pd.DataFrame:
    dataset = load_dataset_parquet(dataset_path)

    dataset["release"] = pd.to_numeric(dataset["release"], errors="coerce")
    dataset = dataset.dropna(subset=["release"]).copy()
    dataset["release"] = dataset["release"].astype(int)

    dataset = dataset[
        (dataset["release"] >= analysis_min_year)
        & (dataset["release"] <= analysis_max_year)
    ].copy()

    if normalization_scope == "selected_assigned_tracks":
        dataset = dataset.merge(
            assignments[["id"]].drop_duplicates(),
            on="id",
            how="inner",
        )

    return dataset


def calculate_raw_centroids(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    modality: str,
    min_tracks: int,
) -> pd.DataFrame:
    centroid_rows = []

    for window_start in range(
        analysis_min_year,
        analysis_max_year - window_size + 2,
        hop_size,
    ):
        window_end = window_start + window_size - 1

        window_data = dataset[
            (dataset["release"] >= window_start)
            & (dataset["release"] <= window_end)
        ]

        if window_data.empty:
            continue

        for genre, genre_data in window_data.groupby("genre"):
            n_tracks = genre_data["id"].nunique()

            if n_tracks < min_tracks:
                continue

            centroid = genre_data[feature_columns].mean()

            row = {
                "genre": genre,
                "modality": modality,
                "window_start": window_start,
                "window_end": window_end,
                "window_label": f"{window_start}_{window_end}",
                "n_tracks": n_tracks,
                "mean_genre_score": genre_data["genre_score"].mean(),
                "median_genre_score": genre_data["genre_score"].median(),
            }

            row.update(centroid.to_dict())
            centroid_rows.append(row)

    return pd.DataFrame(centroid_rows)


def calculate_within_period_z_centroids(
    centroid_dataset: pd.DataFrame,
    normalization_dataset: pd.DataFrame,
    feature_columns: list[str],
    modality: str,
    min_tracks: int,
) -> pd.DataFrame:
    eps = 1e-8
    centroid_rows = []

    for window_start in range(
        analysis_min_year,
        analysis_max_year - window_size + 2,
        hop_size,
    ):
        window_end = window_start + window_size - 1

        normalization_window = normalization_dataset[
            (normalization_dataset["release"] >= window_start)
            & (normalization_dataset["release"] <= window_end)
        ].copy()

        centroid_window = centroid_dataset[
            (centroid_dataset["release"] >= window_start)
            & (centroid_dataset["release"] <= window_end)
        ].copy()

        if normalization_window.empty or centroid_window.empty:
            continue

        means = normalization_window[feature_columns].mean()
        stds = normalization_window[feature_columns].std(ddof=0).replace(0, eps)

        centroid_window[feature_columns] = (
            centroid_window[feature_columns] - means
        ) / stds

        for genre, genre_data in centroid_window.groupby("genre"):
            n_tracks = genre_data["id"].nunique()

            if n_tracks < min_tracks:
                continue

            centroid = genre_data[feature_columns].mean()

            row = {
                "genre": genre,
                "modality": modality,
                "window_start": window_start,
                "window_end": window_end,
                "window_label": f"{window_start}_{window_end}",
                "n_tracks": n_tracks,
                "mean_genre_score": genre_data["genre_score"].mean(),
                "median_genre_score": genre_data["genre_score"].median(),
            }

            row.update(centroid.to_dict())
            centroid_rows.append(row)

    return pd.DataFrame(centroid_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate centroids using OR-Tools genre assignments."
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="ortools_top50_mintracks20_topcandidates5_genrepool180",
        help="Prefix of the OR-Tools output files.",
    )

    parser.add_argument(
        "--min-tracks",
        type=int,
        default=20,
        help="Minimum tracks required for a genre-window centroid.",
    )

    parser.add_argument(
        "--normalization-scope",
        choices=["all_modality_tracks", "selected_assigned_tracks"],
        default="all_modality_tracks",
        help=(
            "Tracks used to compute within-period z-normalization statistics. "
            "'all_modality_tracks' normalizes relative to all tracks available "
            "in the same modality and time window."
        ),
    )

    args = parser.parse_args()

    output_dir = PROCESSED_DIR / "03_ortools_centroids"
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments, selected_genres = load_ortools_assignment(args.prefix)

    print("Loaded OR-Tools assignment")
    print(f"Assigned songs: {assignments['id'].nunique():,}")
    print(f"Selected genres: {selected_genres['genre'].nunique():,}")
    print(f"Minimum tracks per centroid: {args.min_tracks}")
    print(f"Normalization scope: {args.normalization_scope}")
    print(f"Output directory: {output_dir}")

    for modality, dataset_path in dataset_paths.items():
        dataset, feature_columns = prepare_modality_dataset(
            modality=modality,
            dataset_path=dataset_path,
            assignments=assignments,
        )

        normalization_dataset = prepare_normalization_dataset(
            dataset_path=dataset_path,
            assignments=assignments,
            normalization_scope=args.normalization_scope,
        )

        raw_centroids = calculate_raw_centroids(
            dataset=dataset,
            feature_columns=feature_columns,
            modality=modality,
            min_tracks=args.min_tracks,
        )

        z_centroids = calculate_within_period_z_centroids(
            centroid_dataset=dataset,
            normalization_dataset=normalization_dataset,
            feature_columns=feature_columns,
            modality=modality,
            min_tracks=args.min_tracks,
        )

        print()
        print(f"{modality} raw centroids: {raw_centroids.shape}")
        print(f"{modality} within-period z centroids: {z_centroids.shape}")

        save_dataset_parquet(
            raw_centroids,
            f"{modality}_ortools_centroids_raw_windowed",
            save_dir=output_dir,
        )

        save_dataset_parquet(
            z_centroids,
            f"{modality}_ortools_centroids_wpz_windowed",
            save_dir=output_dir,
        )

    print()
    print("Finished OR-Tools centroid calculation.")


if __name__ == "__main__":
    main()