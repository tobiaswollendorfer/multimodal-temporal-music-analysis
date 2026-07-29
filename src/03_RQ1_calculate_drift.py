"""
Calculate RQ1 centroid drift tables from the OR-Tools centroid dataset.

RQ1:
    How much do music genres change over time across acoustic, visual, and
    lyrical feature modalities?

This script performs the calculation steps only. The notebook should load the
saved outputs for inspection, plotting, and interpretation.

Inputs:
    data/processed/03_ortools_centroids/
        {modality}_ortools_centroids_raw_windowed.parquet
        {modality}_ortools_centroids_wpz_windowed.parquet

Outputs:
    data/processed/04_ortools_centroid_drift/
        rq1_ortools_centroid_drift_wpz.parquet
        rq1_ortools_centroid_drift_wpz_neighbors.parquet
        rq1_ortools_centroid_drift_raw.parquet
        rq1_ortools_centroid_drift_raw_neighbors.parquet
        rq1_modality_drift_summary_wpz_neighbors.csv
        rq1_genre_drift_summary_wpz_neighbors.csv
        rq1_selected_genre_landscape_drift_wpz_neighbors.parquet
        rq1_raw_vs_wpz_drift_summary.csv

Example:
    python src/04_answer_RQ1_calculate_drift.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.helper_functions import PROCESSED_DIR


CENTROID_DIR = PROCESSED_DIR / "03_ortools_centroids"
DRIFT_DIR = PROCESSED_DIR / "04_ortools_centroid_drift"

MODALITIES = [
    "essentia",
    "lyrics_tf_idf",
    "word2vec",
    "mfcc",
    "musicnn",
    "vgg19",
]

METADATA_COLUMNS = {
    "genre",
    "modality",
    "window_start",
    "window_end",
    "window_label",
    "n_tracks",
    "mean_genre_score",
    "median_genre_score",
}


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    return [
        column
        for column in data.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(data[column])
    ]


def load_centroids(representation: str) -> dict[str, pd.DataFrame]:
    centroids_by_modality = {}
    missing_paths = []

    for modality in MODALITIES:
        path = (
            CENTROID_DIR
            / f"{modality}_ortools_centroids_{representation}_windowed.parquet"
        )

        if not path.exists():
            missing_paths.append(path)
            continue

        centroids_by_modality[modality] = pd.read_parquet(path)

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing centroid files. Run the OR-Tools centroid script first:\n"
            f"{missing_text}"
        )

    return centroids_by_modality


def calculate_centroid_drift(
    centroids_by_modality: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for modality, centroids in centroids_by_modality.items():
        print(f"Calculating drift: {modality}")

        feature_columns = get_feature_columns(centroids)
        centroids = centroids.sort_values(["genre", "window_start"])

        for genre, genre_data in centroids.groupby("genre"):
            genre_data = genre_data.sort_values("window_start").reset_index(drop=True)

            for idx in range(len(genre_data) - 1):
                current_row = genre_data.iloc[idx]
                next_row = genre_data.iloc[idx + 1]

                current_vector = current_row[feature_columns].to_numpy(dtype=float)
                next_vector = next_row[feature_columns].to_numpy(dtype=float)

                valid_values = np.isfinite(current_vector) & np.isfinite(next_vector)

                if valid_values.sum() == 0:
                    cosine_distance = np.nan
                    euclidean_distance = np.nan
                else:
                    current_valid = current_vector[valid_values].reshape(1, -1)
                    next_valid = next_vector[valid_values].reshape(1, -1)

                    similarity = cosine_similarity(current_valid, next_valid)[0, 0]
                    cosine_distance = 1 - similarity
                    euclidean_distance = np.linalg.norm(current_valid - next_valid)

                rows.append({
                    "modality": modality,
                    "genre": genre,
                    "window_start": current_row["window_start"],
                    "window_end": current_row["window_end"],
                    "next_window_start": next_row["window_start"],
                    "next_window_end": next_row["window_end"],
                    "window_gap": (
                        next_row["window_start"] - current_row["window_start"]
                    ),
                    "n_tracks_t": current_row["n_tracks"],
                    "n_tracks_t1": next_row["n_tracks"],
                    "cosine_distance": cosine_distance,
                    "euclidean_distance": euclidean_distance,
                    "n_features_used": int(valid_values.sum()),
                })

    return pd.DataFrame(rows)


def summarize_modality_drift(drift_neighbors: pd.DataFrame) -> pd.DataFrame:
    return (
        drift_neighbors
        .groupby("modality")
        .agg(
            n_transitions=("cosine_distance", "size"),
            n_genres=("genre", "nunique"),
            mean_cosine_distance=("cosine_distance", "mean"),
            median_cosine_distance=("cosine_distance", "median"),
            std_cosine_distance=("cosine_distance", "std"),
            mean_euclidean_distance=("euclidean_distance", "mean"),
            median_euclidean_distance=("euclidean_distance", "median"),
            std_euclidean_distance=("euclidean_distance", "std"),
            median_tracks_t=("n_tracks_t", "median"),
            median_tracks_t1=("n_tracks_t1", "median"),
        )
        .reset_index()
        .sort_values("mean_cosine_distance", ascending=False)
    )


def summarize_genre_drift(drift_neighbors: pd.DataFrame) -> pd.DataFrame:
    return (
        drift_neighbors
        .groupby(["modality", "genre"])
        .agg(
            n_transitions=("cosine_distance", "size"),
            mean_cosine_distance=("cosine_distance", "mean"),
            median_cosine_distance=("cosine_distance", "median"),
            mean_euclidean_distance=("euclidean_distance", "mean"),
            median_euclidean_distance=("euclidean_distance", "median"),
            median_tracks_t=("n_tracks_t", "median"),
            median_tracks_t1=("n_tracks_t1", "median"),
        )
        .reset_index()
    )


def calculate_weighted_landscape_centroids(
    centroids_by_modality: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for modality, centroids in centroids_by_modality.items():
        print(f"Calculating selected-genre landscape centroid: {modality}")

        feature_columns = get_feature_columns(centroids)

        for window_start, window_data in centroids.groupby("window_start"):
            weights = window_data["n_tracks"].to_numpy(dtype=float)
            feature_matrix = window_data[feature_columns].to_numpy(dtype=float)

            valid_feature_columns = np.isfinite(feature_matrix).all(axis=0)
            feature_matrix = feature_matrix[:, valid_feature_columns]
            kept_features = [
                feature
                for feature, keep in zip(feature_columns, valid_feature_columns)
                if keep
            ]

            if len(window_data) == 0 or weights.sum() == 0 or not kept_features:
                continue

            weighted_centroid = np.average(feature_matrix, axis=0, weights=weights)

            row = {
                "modality": modality,
                "genre": "__selected_genre_landscape__",
                "window_start": window_start,
                "window_end": window_start + 4,
                "n_genres": window_data["genre"].nunique(),
                "n_tracks": window_data["n_tracks"].sum(),
            }

            row.update(dict(zip(kept_features, weighted_centroid)))
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_raw_vs_wpz(
    drift_raw_neighbors: pd.DataFrame,
    modality_summary_wpz: pd.DataFrame,
) -> pd.DataFrame:
    raw_summary = (
        drift_raw_neighbors
        .groupby("modality")
        .agg(
            raw_mean_cosine_distance=("cosine_distance", "mean"),
            raw_median_cosine_distance=("cosine_distance", "median"),
            raw_mean_euclidean_distance=("euclidean_distance", "mean"),
            raw_median_euclidean_distance=("euclidean_distance", "median"),
        )
        .reset_index()
    )

    wpz_summary = modality_summary_wpz[
        [
            "modality",
            "mean_cosine_distance",
            "median_cosine_distance",
            "mean_euclidean_distance",
            "median_euclidean_distance",
        ]
    ].rename(columns={
        "mean_cosine_distance": "wpz_mean_cosine_distance",
        "median_cosine_distance": "wpz_median_cosine_distance",
        "mean_euclidean_distance": "wpz_mean_euclidean_distance",
        "median_euclidean_distance": "wpz_median_euclidean_distance",
    })

    return wpz_summary.merge(raw_summary, on="modality", how="outer")


def save_outputs(
    drift_wpz: pd.DataFrame,
    drift_wpz_neighbors: pd.DataFrame,
    drift_raw: pd.DataFrame,
    drift_raw_neighbors: pd.DataFrame,
    modality_summary_wpz: pd.DataFrame,
    genre_summary_wpz: pd.DataFrame,
    landscape_drift_wpz_neighbors: pd.DataFrame,
    raw_vs_wpz_summary: pd.DataFrame,
) -> None:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    drift_wpz.to_parquet(
        DRIFT_DIR / "rq1_ortools_centroid_drift_wpz.parquet",
        index=False,
    )
    drift_wpz_neighbors.to_parquet(
        DRIFT_DIR / "rq1_ortools_centroid_drift_wpz_neighbors.parquet",
        index=False,
    )
    drift_raw.to_parquet(
        DRIFT_DIR / "rq1_ortools_centroid_drift_raw.parquet",
        index=False,
    )
    drift_raw_neighbors.to_parquet(
        DRIFT_DIR / "rq1_ortools_centroid_drift_raw_neighbors.parquet",
        index=False,
    )

    modality_summary_wpz.to_csv(
        DRIFT_DIR / "rq1_modality_drift_summary_wpz_neighbors.csv",
        index=False,
    )
    genre_summary_wpz.to_csv(
        DRIFT_DIR / "rq1_genre_drift_summary_wpz_neighbors.csv",
        index=False,
    )
    landscape_drift_wpz_neighbors.to_parquet(
        DRIFT_DIR / "rq1_selected_genre_landscape_drift_wpz_neighbors.parquet",
        index=False,
    )
    raw_vs_wpz_summary.to_csv(
        DRIFT_DIR / "rq1_raw_vs_wpz_drift_summary.csv",
        index=False,
    )


def main() -> None:
    print("Loading OR-Tools raw centroids")
    raw_centroids = load_centroids("raw")

    print("Loading OR-Tools within-period z centroids")
    wpz_centroids = load_centroids("wpz")

    print("Calculating within-period z centroid drift")
    drift_wpz = calculate_centroid_drift(wpz_centroids)
    drift_wpz_neighbors = drift_wpz[drift_wpz["window_gap"] == 1].copy()

    print("Calculating raw centroid drift")
    drift_raw = calculate_centroid_drift(raw_centroids)
    drift_raw_neighbors = drift_raw[drift_raw["window_gap"] == 1].copy()

    print("Summarizing within-period z drift by modality")
    modality_summary_wpz = summarize_modality_drift(drift_wpz_neighbors)

    print("Summarizing within-period z drift by genre")
    genre_summary_wpz = summarize_genre_drift(drift_wpz_neighbors)

    print("Calculating selected-genre landscape baseline")
    landscape_centroids_wpz = calculate_weighted_landscape_centroids(wpz_centroids)
    landscape_by_modality = {
        modality: data
        for modality, data in landscape_centroids_wpz.groupby("modality")
    }
    landscape_drift_wpz = calculate_centroid_drift(landscape_by_modality)
    landscape_drift_wpz_neighbors = (
        landscape_drift_wpz[landscape_drift_wpz["window_gap"] == 1].copy()
    )

    print("Comparing raw and within-period z drift")
    raw_vs_wpz_summary = summarize_raw_vs_wpz(
        drift_raw_neighbors=drift_raw_neighbors,
        modality_summary_wpz=modality_summary_wpz,
    )

    print("Saving outputs")
    save_outputs(
        drift_wpz=drift_wpz,
        drift_wpz_neighbors=drift_wpz_neighbors,
        drift_raw=drift_raw,
        drift_raw_neighbors=drift_raw_neighbors,
        modality_summary_wpz=modality_summary_wpz,
        genre_summary_wpz=genre_summary_wpz,
        landscape_drift_wpz_neighbors=landscape_drift_wpz_neighbors,
        raw_vs_wpz_summary=raw_vs_wpz_summary,
    )

    print()
    print("RQ1 drift calculation finished.")
    print(f"Saved outputs to: {DRIFT_DIR}")
    print()
    print("Modality summary:")
    print(modality_summary_wpz)


if __name__ == "__main__":
    main()
