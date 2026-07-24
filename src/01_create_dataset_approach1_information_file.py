"""
Approach 1: metadata-based genre assignment.

Goal:
    1. Assign each song to one metadata genre from id_genres.csv.
    2. Count genre-window coverage for each modality.
    3. Select exactly N genres with the best temporal coverage.
    4. Save outputs in the same style as the OR-Tools approach.

Example:
    python src/01a_genre_assignment_metadata.py \
        --total-genres 50 \
        --min-tracks 20 \
        --genre-position first
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import argparse
import time

import pandas as pd

from utils.helper_functions import RAW_DIR, PROCESSED_DIR


dataset_paths = {
    "essentia": "id_essentia.tsv",
    "lyrics_tf_idf": "id_lyrics_tf-idf.tsv",
    "word2vec": "id_lyrics_word2vec.tsv",
    "mfcc": "id_mfcc_stats.tsv",
    "musicnn": "id_musicnn.tsv",
    "vgg19": "id_vgg19.tsv",
}

genre_metadata_path = "id_genres.csv"
metadata_path = "id_metadata.csv"

analysis_min_year = 1955
analysis_max_year = 2019
window_size = 5
hop_size = 1


def load_tsv(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=usecols)


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def load_release_metadata() -> pd.DataFrame:
    metadata = load_tsv(RAW_DIR / metadata_path)
    metadata["release"] = pd.to_numeric(metadata["release"], errors="coerce")
    metadata = metadata.dropna(subset=["release"]).copy()
    metadata["release"] = metadata["release"].astype(int)

    metadata = metadata[
        (metadata["release"] >= analysis_min_year)
        & (metadata["release"] <= analysis_max_year)
    ].copy()

    return metadata[["id", "release"]]


def load_song_modalities() -> dict[str, list[str]]:
    song_modalities = defaultdict(list)

    for modality, dataset_path in dataset_paths.items():
        print(f"Loading modality IDs: {modality}")

        ids = load_tsv(
            RAW_DIR / dataset_path,
            usecols=["id"],
        )["id"].drop_duplicates()

        for song_id in ids:
            song_modalities[song_id].append(modality)

    return dict(song_modalities)


def choose_metadata_genre(genres_value: str, genre_position: str) -> str | None:
    if pd.isna(genres_value):
        return None

    genres = [
        genre.strip()
        for genre in str(genres_value).split(",")
        if genre.strip()
    ]

    if not genres:
        return None

    if genre_position == "first":
        return genres[0]

    if genre_position == "last":
        return genres[-1]

    raise ValueError(f"Unknown genre_position: {genre_position}")


def create_metadata_assignments(
    metadata: pd.DataFrame,
    song_modalities: dict[str, list[str]],
    genre_position: str,
) -> pd.DataFrame:
    print("Loading metadata genre file")

    genres = load_tsv(RAW_DIR / genre_metadata_path)

    genres["genre"] = genres["genres"].apply(
        lambda value: choose_metadata_genre(value, genre_position)
    )

    genres = genres.dropna(subset=["genre"]).copy()
    genres = genres[["id", "genre"]].drop_duplicates(subset=["id"])

    valid_ids = set(metadata["id"]) & set(song_modalities)

    assignments = genres[genres["id"].isin(valid_ids)].copy()

    # Metadata assignment has no TF-IDF score, but keeping this column makes the
    # output compatible with the coverage code used for the other approaches.
    assignments["genre_score"] = 1.0

    print(f"Assigned songs: {len(assignments):,}")
    print(f"Assigned genres before selection: {assignments['genre'].nunique():,}")

    return assignments


def create_genre_window_coverage(
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
    song_modalities: dict[str, list[str]],
    min_tracks: int,
) -> pd.DataFrame:
    release_by_song = dict(zip(metadata["id"], metadata["release"]))

    assigned = assignments.copy()
    assigned["release"] = assigned["id"].map(release_by_song)
    assigned = assigned.dropna(subset=["release"]).copy()
    assigned["release"] = assigned["release"].astype(int)

    expanded_rows = []
    start_time = time.time()

    for idx, (song_id, genre, score, release) in enumerate(
        assigned[["id", "genre", "genre_score", "release"]]
        .itertuples(index=False, name=None),
        start=1,
    ):
        for modality in song_modalities.get(song_id, []):
            expanded_rows.append({
                "id": song_id,
                "genre": genre,
                "genre_score": score,
                "release": release,
                "modality": modality,
            })

        if idx % 20000 == 0:
            print(
                f"Expanded assigned songs: {idx:,}/{len(assigned):,} | "
                f"elapsed {format_duration(time.time() - start_time)}"
            )

    expanded = pd.DataFrame(expanded_rows)

    rows = []

    windows = list(
        range(
            analysis_min_year,
            analysis_max_year - window_size + 2,
            hop_size,
        )
    )

    for modality, modality_data in expanded.groupby("modality"):
        print(f"Creating coverage for {modality}")

        for window_start in windows:
            window_end = window_start + window_size - 1

            window_data = modality_data[
                (modality_data["release"] >= window_start)
                & (modality_data["release"] <= window_end)
            ]

            if window_data.empty:
                continue

            genre_counts = (
                window_data
                .groupby("genre")
                .agg(
                    n_tracks=("id", "nunique"),
                    mean_genre_score=("genre_score", "mean"),
                    median_genre_score=("genre_score", "median"),
                )
                .reset_index()
            )

            for _, row in genre_counts.iterrows():
                n_tracks = int(row["n_tracks"])

                rows.append({
                    "modality": modality,
                    "genre": row["genre"],
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_label": f"{window_start}_{window_end}",
                    "n_tracks": n_tracks,
                    "mean_genre_score": row["mean_genre_score"],
                    "median_genre_score": row["median_genre_score"],
                    "eligible_window": n_tracks >= min_tracks,
                })

    return pd.DataFrame(rows)


def select_genres(
    coverage: pd.DataFrame,
    total_genres: int,
) -> pd.DataFrame:
    genre_modality_summary = (
        coverage
        .groupby(["genre", "modality"])
        .agg(
            n_valid_windows=("eligible_window", "sum"),
            n_observed_windows=("window_label", "nunique"),
            n_tracks_total=("n_tracks", "sum"),
            mean_genre_score=("mean_genre_score", "mean"),
        )
        .reset_index()
    )

    genre_summary = (
        genre_modality_summary
        .groupby("genre")
        .agg(
            total_valid_windows=("n_valid_windows", "sum"),
            mean_valid_windows=("n_valid_windows", "mean"),
            min_valid_windows_across_modalities=("n_valid_windows", "min"),
            n_modalities=("modality", "nunique"),
            total_tracks=("n_tracks_total", "sum"),
            mean_genre_score=("mean_genre_score", "mean"),
        )
        .reset_index()
    )

    selected = (
        genre_summary
        .sort_values(
            [
                "total_valid_windows",
                "min_valid_windows_across_modalities",
                "mean_valid_windows",
                "n_modalities",
                "total_tracks",
                "mean_genre_score",
            ],
            ascending=[False, False, False, False, False, False],
        )
        .head(total_genres)
        .reset_index(drop=True)
    )

    selected["selection_rank"] = selected.index + 1

    return selected


def summarize_selected_coverage(
    coverage: pd.DataFrame,
    selected_genres: pd.DataFrame,
) -> pd.DataFrame:
    selected_names = set(selected_genres["genre"])

    selected_coverage = coverage[
        coverage["genre"].isin(selected_names)
    ].copy()

    genre_modality_counts = (
        selected_coverage
        .groupby(["modality", "genre"])
        .agg(
            n_valid_windows=("eligible_window", "sum"),
            n_observed_windows=("window_label", "nunique"),
            total_tracks=("n_tracks", "sum"),
            mean_genre_score=("mean_genre_score", "mean"),
        )
        .reset_index()
    )

    return (
        genre_modality_counts
        .groupby("modality")
        .agg(
            n_selected_genres=("genre", "nunique"),
            total_valid_genre_windows=("n_valid_windows", "sum"),
            mean_valid_windows_per_genre=("n_valid_windows", "mean"),
            median_valid_windows_per_genre=("n_valid_windows", "median"),
            min_valid_windows_per_genre=("n_valid_windows", "min"),
            max_valid_windows_per_genre=("n_valid_windows", "max"),
            total_tracks=("total_tracks", "sum"),
            mean_genre_score=("mean_genre_score", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metadata-based genre assignment baseline."
    )

    parser.add_argument("--total-genres", type=int, default=50)
    parser.add_argument("--min-tracks", type=int, default=20)

    parser.add_argument(
        "--genre-position",
        choices=["first", "last"],
        default="first",
        help=(
            "Which genre to use from the comma-separated metadata genre list. "
            "Use 'first' if the first tag is treated as the most specific/main tag."
        ),
    )

    args = parser.parse_args()
    script_start = time.time()

    output_dir = PROCESSED_DIR / "09_selected_genres_metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading release metadata")
    metadata = load_release_metadata()
    print(f"Metadata: {metadata.shape}")

    print("Loading modality availability")
    song_modalities = load_song_modalities()
    print(f"Songs with modality info: {len(song_modalities):,}")

    print("Creating metadata assignments")
    assignments = create_metadata_assignments(
        metadata=metadata,
        song_modalities=song_modalities,
        genre_position=args.genre_position,
    )

    print("Creating full genre-window coverage")
    full_coverage = create_genre_window_coverage(
        assignments=assignments,
        metadata=metadata,
        song_modalities=song_modalities,
        min_tracks=args.min_tracks,
    )

    print("Selecting final genres")
    selected_genres = select_genres(
        coverage=full_coverage,
        total_genres=args.total_genres,
    )

    selected_names = set(selected_genres["genre"])
    selected_assignments = assignments[
        assignments["genre"].isin(selected_names)
    ].copy()

    selected_coverage = full_coverage[
        full_coverage["genre"].isin(selected_names)
    ].copy()

    summary = summarize_selected_coverage(
        coverage=selected_coverage,
        selected_genres=selected_genres,
    )

    prefix = (
        f"metadata_top{args.total_genres}"
        f"_mintracks{args.min_tracks}"
        f"_genreposition{args.genre_position}"
    )

    print("Saving outputs")

    selected_assignments.to_parquet(
        output_dir / f"{prefix}_assignments.parquet",
        index=False,
    )

    selected_genres.to_csv(
        output_dir / f"{prefix}_selected_genres.csv",
        index=False,
    )

    selected_coverage.to_parquet(
        output_dir / f"{prefix}_genre_window_coverage.parquet",
        index=False,
    )

    summary.to_csv(
        output_dir / f"{prefix}_coverage_summary.csv",
        index=False,
    )

    full_coverage.to_parquet(
        output_dir / f"{prefix}_all_genre_window_coverage.parquet",
        index=False,
    )

    print()
    print("Selected genres:")
    print(selected_genres)
    print()
    print("Coverage summary:")
    print(summary)
    print()
    print(f"Saved outputs to: {output_dir}")
    print(f"Total runtime: {format_duration(time.time() - script_start)}")


if __name__ == "__main__":
    main()