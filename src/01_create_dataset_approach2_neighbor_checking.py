"""
Approach 2: local coverage optimization genre assignment.

Objective:
    1. Load genre TF-IDF scores.
    2. Assign each song to its highest-scoring candidate genre.
    3. Iteratively try moving one song to another candidate genre.
    4. Keep the move if it improves temporal genre-window coverage.
    5. Select exactly N genres after optimization.
    6. Save outputs in the same style as the OR-Tools approach.

Example:
    python src/01b_genre_assignment_local_optimization.py \
        --total-genres 50 \
        --min-tracks 20 \
        --top-candidates-per-song 5 \
        --candidate-genre-pool 180 \
        --max-passes 30
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
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

genre_score_candidates = [
    "id_genres_tf-idf.tsv",
    "id_genres_tf-idf.tsv.bz2",
]

metadata_path = "id_metadata.csv"

analysis_min_year = 1955
analysis_max_year = 2019
window_size = 5
hop_size = 1


def load_tsv(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=usecols)


def find_genre_score_path() -> Path:
    for filename in genre_score_candidates:
        path = RAW_DIR / filename
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find genre score file. Expected one of: "
        + ", ".join(str(RAW_DIR / name) for name in genre_score_candidates)
    )


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def progress_message(label: str, completed: int, total: int, start_time: float) -> str:
    elapsed = time.time() - start_time
    progress = completed / total if total else 1

    if completed > 0 and elapsed > 0:
        remaining = (total - completed) / (completed / elapsed)
    else:
        remaining = 0

    finish_time = datetime.now() + timedelta(seconds=remaining)

    return (
        f"{label}: {completed:,}/{total:,} "
        f"({progress * 100:5.1f}%) | "
        f"elapsed {format_duration(elapsed)} | "
        f"remaining {format_duration(remaining)} | "
        f"ETA {finish_time:%H:%M:%S}"
    )


def windows_for_release(release: int) -> list[int]:
    first_start = max(analysis_min_year, release - window_size + 1)
    last_start = min(release, analysis_max_year - window_size + 1)

    if first_start > last_start:
        return []

    return list(range(first_start, last_start + 1, hop_size))


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


def select_candidate_genre_pool(
    genre_scores: pd.DataFrame,
    candidate_genre_pool: int,
) -> list[str]:
    genre_columns = [column for column in genre_scores.columns if column != "id"]

    rows = []

    for genre in genre_columns:
        values = genre_scores[genre]
        nonzero = values[values > 0]

        rows.append({
            "genre": genre,
            "n_nonzero": int(len(nonzero)),
            "score_sum": float(nonzero.sum()),
            "score_mean": float(nonzero.mean()) if len(nonzero) else 0.0,
        })

    genre_strength = pd.DataFrame(rows)

    return (
        genre_strength
        .sort_values(
            ["n_nonzero", "score_sum", "score_mean"],
            ascending=[False, False, False],
        )
        .head(candidate_genre_pool)["genre"]
        .tolist()
    )


def build_candidate_assignments(
    genre_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    song_modalities: dict[str, list[str]],
    top_candidates_per_song: int,
    candidate_genre_pool: int,
    max_songs: int | None,
) -> pd.DataFrame:
    genre_columns = [column for column in genre_scores.columns if column != "id"]
    candidate_genres = select_candidate_genre_pool(
        genre_scores=genre_scores,
        candidate_genre_pool=candidate_genre_pool,
    )
    candidate_set = set(candidate_genres)

    metadata_ids = set(metadata["id"])
    modality_ids = set(song_modalities)

    rows = []
    n_candidate_songs = 0
    start_time = time.time()

    print("Building song candidate assignments")

    for row in genre_scores.itertuples(index=False):
        song_id = row[0]

        if song_id not in metadata_ids or song_id not in modality_ids:
            continue

        scores = []

        for genre, score in zip(genre_columns, row[1:]):
            if genre not in candidate_set:
                continue
            if score > 0:
                scores.append((genre, float(score)))

        if not scores:
            continue

        scores = sorted(scores, key=lambda item: item[1], reverse=True)
        scores = scores[:top_candidates_per_song]

        for rank, (genre, score) in enumerate(scores, start=1):
            rows.append({
                "id": song_id,
                "candidate_genre": genre,
                "candidate_rank": rank,
                "genre_score": score,
            })

        n_candidate_songs += 1

        if n_candidate_songs % 10000 == 0:
            print(
                f"Candidate songs processed: {n_candidate_songs:,} | "
                f"elapsed {format_duration(time.time() - start_time)}"
            )

        if max_songs is not None and n_candidate_songs >= max_songs:
            break

    candidates = pd.DataFrame(rows)

    print(f"Candidate songs: {candidates['id'].nunique():,}")
    print(f"Candidate assignment rows: {len(candidates):,}")
    print(f"Candidate genres: {candidates['candidate_genre'].nunique():,}")

    return candidates


def create_initial_assignments(candidates: pd.DataFrame) -> pd.DataFrame:
    assignments = (
        candidates
        .sort_values(["id", "candidate_rank", "genre_score"], ascending=[True, True, False])
        .groupby("id", as_index=False)
        .first()
        .rename(columns={"candidate_genre": "genre"})
    )

    return assignments[["id", "genre", "genre_score"]]


def build_song_cells(
    candidates: pd.DataFrame,
    metadata: pd.DataFrame,
    song_modalities: dict[str, list[str]],
) -> dict[tuple[str, str], list[tuple[str, str, int]]]:
    release_by_song = dict(zip(metadata["id"], metadata["release"]))

    song_genre_cells = {}

    for song_id, genre in (
        candidates[["id", "candidate_genre"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        release = release_by_song.get(song_id)
        modalities = song_modalities.get(song_id, [])

        cells = []

        if release is not None:
            for modality in modalities:
                for window_start in windows_for_release(int(release)):
                    cells.append((genre, modality, window_start))

        song_genre_cells[(song_id, genre)] = cells

    return song_genre_cells


def initialize_cell_counts(
    assignments: pd.DataFrame,
    song_genre_cells: dict[tuple[str, str], list[tuple[str, str, int]]],
) -> dict[tuple[str, str, int], int]:
    counts = defaultdict(int)

    for song_id, genre in assignments[["id", "genre"]].itertuples(index=False):
        for cell in song_genre_cells.get((song_id, genre), []):
            counts[cell] += 1

    return counts


def valid_cell_count(
    cell_counts: dict[tuple[str, str, int], int],
    min_tracks: int,
) -> int:
    return sum(count >= min_tracks for count in cell_counts.values())


def optimize_assignments(
    candidates: pd.DataFrame,
    assignments: pd.DataFrame,
    metadata: pd.DataFrame,
    song_modalities: dict[str, list[str]],
    min_tracks: int,
    max_passes: int,
    score_weight: float,
    coverage_weight: float,
    report_every_passes: int,
) -> pd.DataFrame:
    song_genre_cells = build_song_cells(
        candidates=candidates,
        metadata=metadata,
        song_modalities=song_modalities,
    )

    cell_counts = initialize_cell_counts(
        assignments=assignments,
        song_genre_cells=song_genre_cells,
    )

    current_assignment = {
        row.id: row.genre
        for row in assignments.itertuples(index=False)
    }

    current_score = {
        row.id: float(row.genre_score)
        for row in assignments.itertuples(index=False)
    }

    candidate_lookup = {
        song_id: group.sort_values("candidate_rank")[
            ["candidate_genre", "genre_score"]
        ].itertuples(index=False, name=None)
        for song_id, group in candidates.groupby("id")
    }

    candidate_lookup = {
        song_id: list(values)
        for song_id, values in candidate_lookup.items()
    }

    song_ids = list(current_assignment.keys())

    start_time = time.time()
    best_valid_cells = valid_cell_count(cell_counts, min_tracks)
    best_score_sum = sum(current_score.values())

    print()
    print("Starting local optimization")
    print(f"Initial valid cells: {best_valid_cells:,}")
    print(f"Initial assignment score: {best_score_sum:,.3f}")
    print()

    for current_pass in range(1, max_passes + 1):
        moves_accepted = 0
        pass_start_valid_cells = best_valid_cells
        pass_start_score = best_score_sum

        for song_id in song_ids:
            old_genre = current_assignment[song_id]
            old_score = current_score[song_id]

            best_move = None
            best_delta_objective = 0

            for new_genre, new_score in candidate_lookup.get(song_id, []):
                if new_genre == old_genre:
                    continue

                old_cells = song_genre_cells.get((song_id, old_genre), [])
                new_cells = song_genre_cells.get((song_id, new_genre), [])

                delta_valid_cells = 0

                for cell in old_cells:
                    before = cell_counts[cell]
                    after = before - 1

                    if before >= min_tracks and after < min_tracks:
                        delta_valid_cells -= 1

                for cell in new_cells:
                    before = cell_counts[cell]
                    after = before + 1

                    if before < min_tracks and after >= min_tracks:
                        delta_valid_cells += 1

                delta_score = float(new_score) - old_score

                delta_objective = (
                    coverage_weight * delta_valid_cells
                    + score_weight * delta_score
                )

                if delta_objective > best_delta_objective:
                    best_delta_objective = delta_objective
                    best_move = (
                        new_genre,
                        float(new_score),
                        old_cells,
                        new_cells,
                        delta_valid_cells,
                        delta_score,
                    )

            if best_move is None:
                continue

            (
                new_genre,
                new_score,
                old_cells,
                new_cells,
                delta_valid_cells,
                delta_score,
            ) = best_move

            for cell in old_cells:
                cell_counts[cell] -= 1

            for cell in new_cells:
                cell_counts[cell] += 1

            current_assignment[song_id] = new_genre
            current_score[song_id] = new_score
            best_valid_cells += delta_valid_cells
            best_score_sum += delta_score
            moves_accepted += 1

        print(
            f"Pass {current_pass:,}/{max_passes:,} | "
            f"accepted moves {moves_accepted:,} | "
            f"valid cells {best_valid_cells:,} "
            f"({best_valid_cells - pass_start_valid_cells:+,}) | "
            f"score {best_score_sum:,.3f} "
            f"({best_score_sum - pass_start_score:+,.3f})"
        )

        if current_pass % report_every_passes == 0 or current_pass == max_passes:
            elapsed = time.time() - start_time
            avg_seconds_per_pass = elapsed / current_pass
            remaining_passes = max_passes - current_pass
            remaining = avg_seconds_per_pass * remaining_passes
            eta = datetime.now() + timedelta(seconds=remaining)

            print(
                f"Progress after pass {current_pass:,}: "
                f"elapsed {format_duration(elapsed)} | "
                f"remaining {format_duration(remaining)} | "
                f"ETA {eta:%H:%M:%S}"
            )

        if moves_accepted == 0:
            print("No accepted moves in this pass. Stopping early.")
            break

    optimized = pd.DataFrame({
        "id": list(current_assignment.keys()),
        "genre": list(current_assignment.values()),
        "genre_score": [
            current_score[song_id]
            for song_id in current_assignment
        ],
    })

    return optimized


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

    for song_id, genre, score, release in assigned[
        ["id", "genre", "genre_score", "release"]
    ].itertuples(index=False, name=None):
        for modality in song_modalities.get(song_id, []):
            expanded_rows.append({
                "id": song_id,
                "genre": genre,
                "genre_score": score,
                "release": release,
                "modality": modality,
            })

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


def select_genres(coverage: pd.DataFrame, total_genres: int) -> pd.DataFrame:
    genre_modality_summary = (
        coverage
        .groupby(["genre", "modality"])
        .agg(
            n_valid_windows=("eligible_window", "sum"),
            n_observed_windows=("window_label", "nunique"),
            n_tracks_total=("n_tracks", "sum"),
            mean_genre_score=("mean_genre_score", "mean"),
            median_genre_score=("median_genre_score", "median"),
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
            median_genre_score=("median_genre_score", "median"),
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
                "mean_genre_score",
                "median_genre_score",
                "total_tracks",
            ],
            ascending=[False, False, False, False, False, False, False],
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
        description="Approach 2: local coverage optimization genre assignment."
    )

    parser.add_argument("--total-genres", type=int, default=50)
    parser.add_argument("--min-tracks", type=int, default=20)
    parser.add_argument("--top-candidates-per-song", type=int, default=5)
    parser.add_argument("--candidate-genre-pool", type=int, default=180)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--max-passes", type=int, default=30)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=1_000_000.0)
    parser.add_argument("--report-every-passes", type=int, default=5)

    args = parser.parse_args()
    script_start = time.time()

    output_dir = PROCESSED_DIR / "09_selected_genres_local_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    genre_score_path = find_genre_score_path()

    print("Loading genre scores")
    print(genre_score_path)
    genre_scores = load_tsv(genre_score_path)
    print(f"Genre scores: {genre_scores.shape}")

    print("Loading release metadata")
    metadata = load_release_metadata()
    print(f"Metadata: {metadata.shape}")

    print("Loading modality availability")
    song_modalities = load_song_modalities()
    print(f"Songs with modality info: {len(song_modalities):,}")

    candidates = build_candidate_assignments(
        genre_scores=genre_scores,
        metadata=metadata,
        song_modalities=song_modalities,
        top_candidates_per_song=args.top_candidates_per_song,
        candidate_genre_pool=args.candidate_genre_pool,
        max_songs=args.max_songs,
    )

    print("Creating initial best-score assignments")
    initial_assignments = create_initial_assignments(candidates)
    print(f"Initial assigned songs: {len(initial_assignments):,}")

    optimized_assignments = optimize_assignments(
        candidates=candidates,
        assignments=initial_assignments,
        metadata=metadata,
        song_modalities=song_modalities,
        min_tracks=args.min_tracks,
        max_passes=args.max_passes,
        score_weight=args.score_weight,
        coverage_weight=args.coverage_weight,
        report_every_passes=args.report_every_passes,
    )

    print("Creating initial coverage table")
    initial_coverage = create_genre_window_coverage(
        assignments=initial_assignments,
        metadata=metadata,
        song_modalities=song_modalities,
        min_tracks=args.min_tracks,
    )

    print("Creating optimized coverage table")
    optimized_coverage = create_genre_window_coverage(
        assignments=optimized_assignments,
        metadata=metadata,
        song_modalities=song_modalities,
        min_tracks=args.min_tracks,
    )

    print("Selecting final genres from optimized coverage")
    selected_genres = select_genres(
        coverage=optimized_coverage,
        total_genres=args.total_genres,
    )

    selected_names = set(selected_genres["genre"])

    selected_assignments = optimized_assignments[
        optimized_assignments["genre"].isin(selected_names)
    ].copy()

    selected_coverage = optimized_coverage[
        optimized_coverage["genre"].isin(selected_names)
    ].copy()

    summary = summarize_selected_coverage(
        coverage=selected_coverage,
        selected_genres=selected_genres,
    )

    initial_selected_genres = select_genres(
        coverage=initial_coverage,
        total_genres=args.total_genres,
    )

    initial_summary = summarize_selected_coverage(
        coverage=initial_coverage[
            initial_coverage["genre"].isin(set(initial_selected_genres["genre"]))
        ],
        selected_genres=initial_selected_genres,
    )

    comparison = initial_summary.merge(
        summary,
        on="modality",
        suffixes=("_initial", "_optimized"),
    )

    comparison["valid_window_gain"] = (
        comparison["total_valid_genre_windows_optimized"]
        - comparison["total_valid_genre_windows_initial"]
    )

    prefix = (
        f"localopt_top{args.total_genres}"
        f"_mintracks{args.min_tracks}"
        f"_topcandidates{args.top_candidates_per_song}"
        f"_genrepool{args.candidate_genre_pool}"
    )

    print("Saving outputs")

    initial_assignments.to_parquet(
        output_dir / f"{prefix}_initial_assignments.parquet",
        index=False,
    )

    optimized_assignments.to_parquet(
        output_dir / f"{prefix}_all_optimized_assignments.parquet",
        index=False,
    )

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

    comparison.to_csv(
        output_dir / f"{prefix}_initial_vs_optimized_summary.csv",
        index=False,
    )

    print()
    print("Selected genres:")
    print(selected_genres)
    print()
    print("Optimized coverage summary:")
    print(summary)
    print()
    print("Initial vs optimized comparison:")
    print(comparison)
    print()
    print(f"Saved outputs to: {output_dir}")
    print(f"Total runtime: {format_duration(time.time() - script_start)}")


if __name__ == "__main__":
    main()