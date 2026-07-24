"""
OR-Tools genre assignment optimizer without solution callback.

Objective:
    1. Select exactly N genres.
    2. Assign songs to candidate genres.
    3. Maximize temporal coverage:
       number of selected genre/modality/window cells with at least min_tracks songs.
    4. Use genre-score quality as a secondary objective.

Example:
    python src/genre_assignment_ortools.py \
        --total-genres 100 \
        --min-tracks 20 \
        --top-candidates-per-song 5 \
        --candidate-genre-pool 180 \
        --time-limit-seconds 1800
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
import argparse
import time

import pandas as pd
from ortools.sat.python import cp_model

from utils.helper_functions import RAW_DIR, PROCESSED_DIR


dataset_paths = {
    "essentia": "id_essentia.tsv",
    "lyrics_tf_idf": "id_lyrics_tf-idf.tsv",
    "word2vec": "id_lyrics_word2vec.tsv",
    "mfcc": "id_mfcc_stats.tsv",
    "musicnn": "id_musicnn.tsv",
    "vgg19": "id_vgg19.tsv",
}

genre_score_path = "id_genres_tf-idf.tsv"
metadata_path = "id_metadata.csv"

analysis_min_year = 1955
analysis_max_year = 2019
window_size = 5
hop_size = 1


def load_tsv(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=usecols)


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def solver_value(solver, var):
    if hasattr(solver, "Value"):
        return solver.Value(var)
    return solver.value(var)


def solver_status_name(solver, status):
    if hasattr(solver, "StatusName"):
        return solver.StatusName(status)
    return solver.status_name(status)


def solver_objective_value(solver):
    if hasattr(solver, "ObjectiveValue"):
        return solver.ObjectiveValue()
    return solver.objective_value


def solver_best_bound(solver):
    if hasattr(solver, "BestObjectiveBound"):
        return solver.BestObjectiveBound()
    return solver.best_objective_bound


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


def build_candidate_assignments(
    genre_scores: pd.DataFrame,
    metadata: pd.DataFrame,
    song_modalities: dict[str, list[str]],
    top_candidates_per_song: int,
    candidate_genre_pool: int,
    max_songs: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    genre_columns = [
        column
        for column in genre_scores.columns
        if column != "id"
    ]

    metadata_ids = set(metadata["id"])
    modality_ids = set(song_modalities)

    print("Estimating candidate genre pool")

    genre_strength = []

    for genre in genre_columns:
        values = genre_scores[genre]
        nonzero = values[values > 0]

        genre_strength.append({
            "genre": genre,
            "n_nonzero": int(len(nonzero)),
            "score_sum": float(nonzero.sum()),
            "score_mean": float(nonzero.mean()) if len(nonzero) > 0 else 0.0,
        })

    genre_strength = pd.DataFrame(genre_strength)

    candidate_genres = (
        genre_strength
        .sort_values(
            ["n_nonzero", "score_sum", "score_mean"],
            ascending=[False, False, False],
        )
        .head(candidate_genre_pool)["genre"]
        .tolist()
    )

    candidate_set = set(candidate_genres)

    rows = []
    n_seen_songs = 0
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

        scores = sorted(
            scores,
            key=lambda item: item[1],
            reverse=True,
        )[:top_candidates_per_song]

        for rank, (genre, score) in enumerate(scores, start=1):
            rows.append({
                "id": song_id,
                "candidate_genre": genre,
                "candidate_rank": rank,
                "genre_score": score,
            })

        n_seen_songs += 1

        if n_seen_songs % 10000 == 0:
            elapsed = time.time() - start_time
            print(
                f"Candidate songs processed: {n_seen_songs:,} | "
                f"elapsed {format_duration(elapsed)}"
            )

        if max_songs is not None and n_seen_songs >= max_songs:
            break

    candidates = pd.DataFrame(rows)

    print(f"Candidate songs: {candidates['id'].nunique():,}")
    print(f"Candidate assignment rows: {len(candidates):,}")
    print(f"Candidate genres: {len(candidate_genres):,}")

    return candidates, candidate_genres


def build_cell_memberships(
    candidates: pd.DataFrame,
    metadata: pd.DataFrame,
    song_modalities: dict[str, list[str]],
) -> dict[tuple[str, str, int], list[tuple[str, str]]]:
    release_by_song = dict(zip(metadata["id"], metadata["release"]))

    cell_memberships = defaultdict(list)

    for song_id, genre in candidates[
        ["id", "candidate_genre"]
    ].drop_duplicates().itertuples(index=False, name=None):
        release = release_by_song.get(song_id)
        modalities = song_modalities.get(song_id, [])

        if release is None or not modalities:
            continue

        for modality in modalities:
            for window_start in windows_for_release(int(release)):
                cell_memberships[(genre, modality, window_start)].append(
                    (song_id, genre)
                )

    return dict(cell_memberships)


def solve_assignment(
    candidates: pd.DataFrame,
    candidate_genres: list[str],
    metadata: pd.DataFrame,
    song_modalities: dict[str, list[str]],
    total_genres: int,
    min_tracks: int,
    time_limit_seconds: int,
    score_scale: int,
    score_weight: int,
    coverage_weight: int,
    num_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = cp_model.CpModel()

    candidate_pairs = list(
        candidates[["id", "candidate_genre"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    score_by_pair = {
        (row.id, row.candidate_genre): int(round(row.genre_score * score_scale))
        for row in candidates.itertuples(index=False)
    }

    print("Creating assignment variables")
    assignment_vars = {
        pair: model.NewBoolVar(f"x_{pair[0]}__{pair[1]}")
        for pair in candidate_pairs
    }

    print("Creating selected genre variables")
    selected_vars = {
        genre: model.NewBoolVar(f"selected_{genre}")
        for genre in candidate_genres
    }

    print("Adding song assignment constraints")
    for song_id, group in candidates.groupby("id"):
        vars_for_song = [
            assignment_vars[(song_id, genre)]
            for genre in group["candidate_genre"].unique()
        ]

        # At most one assignment. A song may remain unassigned if none of its
        # candidate genres is selected.
        model.Add(sum(vars_for_song) <= 1)

    print("Adding selected genre constraints")
    model.Add(sum(selected_vars.values()) == total_genres)

    for song_id, genre in candidate_pairs:
        model.Add(assignment_vars[(song_id, genre)] <= selected_vars[genre])

    print("Building coverage cell variables")
    cell_memberships = build_cell_memberships(
        candidates=candidates,
        metadata=metadata,
        song_modalities=song_modalities,
    )

    valid_cell_vars = {}

    for cell, members in cell_memberships.items():
        genre, modality, window_start = cell

        if genre not in selected_vars:
            continue

        y = model.NewBoolVar(f"valid_{genre}__{modality}__{window_start}")
        valid_cell_vars[cell] = y

        member_vars = [
            assignment_vars[pair]
            for pair in members
            if pair in assignment_vars
        ]

        if not member_vars:
            model.Add(y == 0)
            continue

        # If y == 1, the cell must contain at least min_tracks assigned songs.
        model.Add(sum(member_vars) >= min_tracks * y)

        # A valid cell can only count for a selected genre.
        model.Add(y <= selected_vars[genre])

    print(f"Assignment variables: {len(assignment_vars):,}")
    print(f"Selected genre variables: {len(selected_vars):,}")
    print(f"Valid cell variables: {len(valid_cell_vars):,}")

    print("Adding objective")
    coverage_objective = coverage_weight * sum(valid_cell_vars.values())

    score_objective = score_weight * sum(
        assignment_vars[pair] * score_by_pair[pair]
        for pair in candidate_pairs
    )

    model.Maximize(coverage_objective + score_objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = True

    print("Solving without solution callback")
    start_time = time.time()

    if hasattr(solver, "Solve"):
        status = solver.Solve(model)
    else:
        status = solver.solve(model)

    runtime = time.time() - start_time

    print()
    print(f"Solver finished with status: {solver_status_name(solver, status)}")
    print(f"Runtime: {format_duration(runtime)}")
    print(f"Objective: {solver_objective_value(solver):,.0f}")
    print(f"Best bound: {solver_best_bound(solver):,.0f}")

    selected_rows = []

    for genre, var in selected_vars.items():
        if solver_value(solver, var) == 1:
            selected_rows.append({"genre": genre})

    selected_genres = pd.DataFrame(selected_rows)

    assignment_rows = []

    for (song_id, genre), var in assignment_vars.items():
        if solver_value(solver, var) == 1:
            assignment_rows.append({
                "id": song_id,
                "genre": genre,
                "genre_score": score_by_pair[(song_id, genre)] / score_scale,
            })

    assignments = pd.DataFrame(assignment_rows)

    valid_rows = []

    for (genre, modality, window_start), var in valid_cell_vars.items():
        if solver_value(solver, var) == 1:
            valid_rows.append({
                "genre": genre,
                "modality": modality,
                "window_start": window_start,
                "window_end": window_start + window_size - 1,
                "window_label": f"{window_start}_{window_start + window_size - 1}",
            })

    valid_cells = pd.DataFrame(valid_rows)

    return assignments, selected_genres, valid_cells


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
        description="Optimize genre assignment with OR-Tools CP-SAT."
    )

    parser.add_argument("--total-genres", type=int, default=50)
    parser.add_argument("--min-tracks", type=int, default=20)
    parser.add_argument("--top-candidates-per-song", type=int, default=5)
    parser.add_argument("--candidate-genre-pool", type=int, default=180)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--time-limit-seconds", type=int, default=3600)
    parser.add_argument("--score-scale", type=int, default=1000)
    parser.add_argument("--score-weight", type=int, default=1)
    parser.add_argument("--coverage-weight", type=int, default=1_000_000)
    parser.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args()

    output_dir = PROCESSED_DIR / "09_selected_genres_ortools"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading genre scores")
    genre_scores = load_tsv(RAW_DIR / genre_score_path)
    print(f"Genre scores: {genre_scores.shape}")

    print("Loading metadata")
    metadata = load_release_metadata()
    print(f"Metadata: {metadata.shape}")

    print("Loading modality availability")
    song_modalities = load_song_modalities()
    print(f"Songs with modality info: {len(song_modalities):,}")

    candidates, candidate_genres = build_candidate_assignments(
        genre_scores=genre_scores,
        metadata=metadata,
        song_modalities=song_modalities,
        top_candidates_per_song=args.top_candidates_per_song,
        candidate_genre_pool=args.candidate_genre_pool,
        max_songs=args.max_songs,
    )

    assignments, selected_genres, valid_cells = solve_assignment(
        candidates=candidates,
        candidate_genres=candidate_genres,
        metadata=metadata,
        song_modalities=song_modalities,
        total_genres=args.total_genres,
        min_tracks=args.min_tracks,
        time_limit_seconds=args.time_limit_seconds,
        score_scale=args.score_scale,
        score_weight=args.score_weight,
        coverage_weight=args.coverage_weight,
        num_workers=args.num_workers,
    )

    print("Creating final coverage table")
    coverage = create_genre_window_coverage(
        assignments=assignments,
        metadata=metadata,
        song_modalities=song_modalities,
        min_tracks=args.min_tracks,
    )

    summary = summarize_selected_coverage(
        coverage=coverage,
        selected_genres=selected_genres,
    )

    prefix = (
        f"ortools_top{args.total_genres}"
        f"_mintracks{args.min_tracks}"
        f"_topcandidates{args.top_candidates_per_song}"
        f"_genrepool{args.candidate_genre_pool}"
    )

    assignments.to_parquet(
        output_dir / f"{prefix}_assignments.parquet",
        index=False,
    )
    selected_genres.to_csv(
        output_dir / f"{prefix}_selected_genres.csv",
        index=False,
    )
    valid_cells.to_csv(
        output_dir / f"{prefix}_valid_cells.csv",
        index=False,
    )
    coverage.to_parquet(
        output_dir / f"{prefix}_genre_window_coverage.parquet",
        index=False,
    )
    summary.to_csv(
        output_dir / f"{prefix}_coverage_summary.csv",
        index=False,
    )

    print("Selected genres:")
    print(selected_genres)
    print()
    print("Coverage summary:")
    print(summary)
    print()
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()