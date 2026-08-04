from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from utils.helper_functions import PROCESSED_DIR, RAW_DIR
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


DEFAULT_PREFIX = "ortools_top50_mintracks20_topcandidates5_genrepool180"
METADATA_COLUMNS = [
    "id",
    "release",
    "genre",
    "genre_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate RQ2 lyrics TF-IDF feature summaries from the final "
            "OR-Tools genre assignment. The notebook should only inspect "
            "the saved outputs from this script."
        )
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Filename prefix of the OR-Tools output files.",
    )
    parser.add_argument(
        "--lyrics-file",
        type=Path,
        default=PROCESSED_DIR / "lyrics_tf_idf.parquet",
        help="Processed lyrics TF-IDF parquet file.",
    )
    parser.add_argument(
        "--analysis-min-year",
        type=int,
        default=1955,
        help="First release year included in the analysis.",
    )
    parser.add_argument(
        "--analysis-max-year",
        type=int,
        default=2019,
        help="Last release year included in the analysis.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Sliding window size in years.",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=1,
        help="Sliding window hop size in years.",
    )
    parser.add_argument(
        "--min-tracks",
        type=int,
        default=20,
        help="Minimum tracks required for an eligible genre-window.",
    )
    parser.add_argument(
        "--min-nonzero-share",
        type=float,
        default=0.01,
        help=(
            "Minimum share of selected assigned tracks with non-zero TF-IDF "
            "for a term to be considered during automatic feature selection."
        ),
    )
    parser.add_argument(
        "--min-nonzero-tracks",
        type=int,
        default=100,
        help=(
            "Minimum number of selected assigned tracks with non-zero TF-IDF "
            "for a term to be considered during automatic feature selection."
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=40,
        help="Maximum number of automatically selected lyrics TF-IDF terms.",
    )
    parser.add_argument(
        "--manual-features-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one TF-IDF term per line. If provided, "
            "these terms are used instead of automatic feature selection."
        ),
    )
    parser.add_argument(
        "--ortools-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing OR-Tools assignment outputs. If omitted, "
            "the script tries known project folders."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR / "RQ2_lyrics_tfidf_interpretable_features",
        help="Directory where calculated RQ2 lyrics outputs are saved.",
    )
    return parser.parse_args()


def resolve_ortools_dir(ortools_dir: Path | None, prefix: str) -> Path:
    candidates = []

    if ortools_dir is not None:
        candidates.append(ortools_dir)

    candidates.extend(
        [
            PROCESSED_DIR / "01_selected_genres_ortools",
            PROCESSED_DIR / "09_selected_genres_ortools",
        ]
    )

    for candidate in candidates:
        if (candidate / f"{prefix}_assignments.parquet").exists():
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find OR-Tools assignment file. Checked:\n" + checked
    )


def read_metadata() -> pd.DataFrame:
    metadata_path = RAW_DIR / "id_metadata.csv"
    try:
        metadata = pd.read_csv(metadata_path, sep="\t")
    except UnicodeDecodeError:
        metadata = pd.read_csv(metadata_path, sep="\t", encoding="latin1")

    if len(metadata.columns) == 1:
        metadata = pd.read_csv(metadata_path)

    return metadata


def windows_for_release(
    release: int,
    analysis_min_year: int,
    analysis_max_year: int,
    window_size: int,
    hop_size: int,
) -> list[int]:
    first_start = max(analysis_min_year, int(release) - window_size + 1)
    last_start = min(int(release), analysis_max_year - window_size + 1)

    if first_start > last_start:
        return []

    return list(range(first_start, last_start + 1, hop_size))


def load_assignments(
    ortools_dir: Path,
    prefix: str,
    analysis_min_year: int,
    analysis_max_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    assignments = pd.read_parquet(ortools_dir / f"{prefix}_assignments.parquet")
    selected_genres = pd.read_csv(ortools_dir / f"{prefix}_selected_genres.csv")
    selected_genre_list = selected_genres["genre"].drop_duplicates().tolist()

    required_columns = ["id", "genre", "genre_score", "release"]
    missing_columns = [
        column for column in required_columns
        if column not in assignments.columns
    ]

    if "release" in missing_columns:
        print("Assignment file is missing release. Merging raw metadata.")
        metadata = read_metadata()
        metadata = metadata[["id", "release"]].copy()
        metadata["release"] = pd.to_numeric(metadata["release"], errors="coerce")
        assignments = assignments.merge(metadata, on="id", how="left")
        missing_columns = [
            column for column in required_columns
            if column not in assignments.columns
        ]

    if missing_columns:
        raise ValueError(
            "Assignment file is missing required columns: "
            + ", ".join(missing_columns)
        )

    assignments["release"] = pd.to_numeric(
        assignments["release"],
        errors="coerce",
    )
    assignments = assignments.dropna(subset=["release", "genre"]).copy()
    assignments["release"] = assignments["release"].astype(int)

    assignments = assignments[
        (assignments["release"] >= analysis_min_year)
        & (assignments["release"] <= analysis_max_year)
        & (assignments["genre"].isin(selected_genre_list))
    ].copy()

    return assignments[required_columns], selected_genres, selected_genre_list


def load_lyrics_dataset(
    lyrics_file: Path,
    assignments: pd.DataFrame,
    analysis_min_year: int,
    analysis_max_year: int,
) -> tuple[pd.DataFrame, list[str]]:
    lyrics = pd.read_parquet(lyrics_file)

    if "id" not in lyrics.columns:
        raise ValueError(f"Lyrics TF-IDF file has no id column: {lyrics_file}")

    drop_columns = [
        column
        for column in ["genre", "genre_score", "release"]
        if column in lyrics.columns
    ]

    lyrics = lyrics.drop(columns=drop_columns, errors="ignore")

    dataset = assignments.merge(
        lyrics,
        on="id",
        how="inner",
    )

    dataset = dataset[
        (dataset["release"] >= analysis_min_year)
        & (dataset["release"] <= analysis_max_year)
    ].copy()

    feature_columns = [
        column
        for column in dataset.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric lyrics TF-IDF feature columns found.")

    dataset[feature_columns] = dataset[feature_columns].fillna(0)

    print("Prepared lyrics TF-IDF dataset")
    print(f"Rows: {len(dataset):,}")
    print(f"Tracks: {dataset['id'].nunique():,}")
    print(f"Genres: {dataset['genre'].nunique():,}")
    print(f"Feature columns: {len(feature_columns):,}")

    return dataset, feature_columns


def calculate_tfidf_feature_coverage(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    n_tracks = dataset["id"].nunique()
    rows = []

    for feature in feature_columns:
        values = dataset[feature]
        nonzero_mask = values > 0
        n_nonzero = int(nonzero_mask.sum())

        rows.append(
            {
                "feature": feature,
                "n_tracks": n_tracks,
                "n_nonzero_tracks": n_nonzero,
                "n_zero_tracks": n_tracks - n_nonzero,
                "nonzero_share": n_nonzero / n_tracks if n_tracks else 0,
                "mean_tfidf": values.mean(),
                "std_tfidf": values.std(ddof=0),
                "mean_nonzero_tfidf": (
                    values[nonzero_mask].mean() if n_nonzero else 0
                ),
                "median_nonzero_tfidf": (
                    values[nonzero_mask].median() if n_nonzero else 0
                ),
                "max_tfidf": values.max(),
            }
        )

    return pd.DataFrame(rows)


def expand_tracks_to_windows(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    analysis_min_year: int,
    analysis_max_year: int,
    window_size: int,
    hop_size: int,
) -> pd.DataFrame:
    expanded_rows = []
    columns = METADATA_COLUMNS + feature_columns

    for row in dataset[columns].to_dict("records"):
        for window_start in windows_for_release(
            release=row["release"],
            analysis_min_year=analysis_min_year,
            analysis_max_year=analysis_max_year,
            window_size=window_size,
            hop_size=hop_size,
        ):
            expanded_row = row.copy()
            expanded_row["window_start"] = window_start
            expanded_row["window_end"] = window_start + window_size - 1
            expanded_row["window_label"] = (
                f"{window_start}_{window_start + window_size - 1}"
            )
            expanded_rows.append(expanded_row)

    return pd.DataFrame(expanded_rows)


def calculate_all_genres_window_summary(
    windowed_tracks: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    group_columns = ["window_start", "window_end", "window_label"]
    grouped = windowed_tracks.groupby(group_columns)

    base_summary = grouped.agg(
        n_tracks=("id", "nunique"),
        n_genres=("genre", "nunique"),
    )
    feature_summary = grouped[feature_columns].mean()

    return base_summary.join(feature_summary).reset_index()


def calculate_feature_selection_scores(
    feature_coverage: pd.DataFrame,
    all_genres_window_summary: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    sorted_summary = all_genres_window_summary.sort_values("window_start")

    for feature in feature_columns:
        values = sorted_summary[feature]

        if values.notna().sum() < 2:
            temporal_slope = np.nan
        else:
            temporal_slope = np.polyfit(
                sorted_summary["window_start"],
                values,
                deg=1,
            )[0]

        rows.append(
            {
                "feature": feature,
                "temporal_std": values.std(ddof=0),
                "first_window_value": values.iloc[0],
                "last_window_value": values.iloc[-1],
                "last_minus_first": values.iloc[-1] - values.iloc[0],
                "absolute_last_minus_first": abs(values.iloc[-1] - values.iloc[0]),
                "linear_trend": temporal_slope,
                "absolute_linear_trend": abs(temporal_slope),
            }
        )

    selection_scores = pd.DataFrame(rows)
    return feature_coverage.merge(selection_scores, on="feature", how="left")


def read_manual_features(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def select_lyrics_features(
    selection_scores: pd.DataFrame,
    feature_columns: list[str],
    manual_features_file: Path | None,
    min_nonzero_share: float,
    min_nonzero_tracks: int,
    max_features: int,
) -> list[str]:
    available_features = set(feature_columns)

    if manual_features_file is not None:
        manual_features = read_manual_features(manual_features_file)
        selected_features = [
            feature
            for feature in manual_features
            if feature in available_features
        ]
        missing_features = sorted(set(manual_features) - available_features)

        if missing_features:
            print("Manual features missing from TF-IDF data:")
            print(missing_features)

        if not selected_features:
            raise ValueError("No manual features were found in the TF-IDF data.")

        return selected_features

    candidates = selection_scores[
        (selection_scores["nonzero_share"] >= min_nonzero_share)
        & (selection_scores["n_nonzero_tracks"] >= min_nonzero_tracks)
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No lyrics features passed the automatic coverage thresholds. "
            "Lower --min-nonzero-share or --min-nonzero-tracks."
        )

    candidates = candidates.sort_values(
        [
            "absolute_linear_trend",
            "temporal_std",
            "n_nonzero_tracks",
            "mean_tfidf",
        ],
        ascending=[False, False, False, False],
    )

    return candidates.head(max_features)["feature"].tolist()


def calculate_window_summaries(
    windowed_tracks: pd.DataFrame,
    feature_columns: list[str],
    min_tracks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_columns = ["genre", "window_start", "window_end", "window_label"]
    grouped = windowed_tracks.groupby(group_columns)

    base_summary = grouped.agg(n_tracks=("id", "nunique"))
    feature_summary = grouped[feature_columns].mean()

    genre_window_summary = base_summary.join(feature_summary).reset_index()

    genre_window_summary["eligible_window"] = (
        genre_window_summary["n_tracks"] >= min_tracks
    )

    eligible_genre_window_summary = genre_window_summary[
        genre_window_summary["eligible_window"]
    ].copy()

    all_genres_window_summary = calculate_all_genres_window_summary(
        windowed_tracks=windowed_tracks,
        feature_columns=feature_columns,
    )

    return (
        genre_window_summary,
        eligible_genre_window_summary,
        all_genres_window_summary,
    )


def calculate_within_period_z_summaries(
    windowed_tracks: pd.DataFrame,
    feature_columns: list[str],
    min_tracks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = 1e-8
    z_rows = []

    for _, window_data in windowed_tracks.groupby("window_start"):
        means = window_data[feature_columns].mean()
        stds = window_data[feature_columns].std(ddof=0).replace(0, eps)

        normalized_window = window_data.copy()
        normalized_window[feature_columns] = (
            normalized_window[feature_columns] - means
        ) / stds

        genre_summary = (
            normalized_window
            .groupby(["genre", "window_start", "window_end", "window_label"])
        )

        base_summary = genre_summary.agg(n_tracks=("id", "nunique"))
        feature_summary = genre_summary[feature_columns].mean()
        genre_summary = base_summary.join(feature_summary).reset_index()

        z_rows.append(genre_summary)

    genre_window_z_summary = pd.concat(z_rows, ignore_index=True)
    genre_window_z_summary["eligible_window"] = (
        genre_window_z_summary["n_tracks"] >= min_tracks
    )
    eligible_genre_window_z_summary = genre_window_z_summary[
        genre_window_z_summary["eligible_window"]
    ].copy()

    return genre_window_z_summary, eligible_genre_window_z_summary


def calculate_feature_deviation_summary(
    eligible_genre_window_z_summary: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    deviation_rows = []

    for feature in feature_columns:
        summary = (
            eligible_genre_window_z_summary
            .groupby("genre")
            .agg(
                n_valid_windows=("window_start", "nunique"),
                mean_relative_position=(feature, "mean"),
                median_relative_position=(feature, "median"),
                mean_absolute_relative_position=(
                    feature,
                    lambda values: values.abs().mean(),
                ),
            )
            .reset_index()
        )
        summary["feature"] = feature
        deviation_rows.append(summary)

    return pd.concat(deviation_rows, ignore_index=True)


def calculate_all_genres_feature_change(
    all_genres_window_summary: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    sorted_reference = all_genres_window_summary.sort_values("window_start")
    rows = []

    for feature in feature_columns:
        values = sorted_reference[feature]
        rows.append(
            {
                "feature": feature,
                "first_window_value": values.iloc[0],
                "last_window_value": values.iloc[-1],
                "last_minus_first": values.iloc[-1] - values.iloc[0],
                "linear_trend": np.polyfit(
                    sorted_reference["window_start"],
                    values,
                    deg=1,
                )[0],
                "mean_value": values.mean(),
                "median_value": values.median(),
            }
        )

    return pd.DataFrame(rows)


def calculate_genre_feature_trends(
    eligible_genre_window_summary: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []

    for feature in feature_columns:
        for genre, genre_data in eligible_genre_window_summary.groupby("genre"):
            genre_data = genre_data.sort_values("window_start")

            if len(genre_data) < 2:
                continue

            values = genre_data[feature]

            rows.append(
                {
                    "feature": feature,
                    "genre": genre,
                    "n_valid_windows": genre_data["window_start"].nunique(),
                    "first_window": genre_data["window_start"].iloc[0],
                    "last_window": genre_data["window_start"].iloc[-1],
                    "first_value": values.iloc[0],
                    "last_value": values.iloc[-1],
                    "last_minus_first": values.iloc[-1] - values.iloc[0],
                    "linear_trend": np.polyfit(
                        genre_data["window_start"],
                        values,
                        deg=1,
                    )[0],
                    "mean_value": values.mean(),
                    "median_value": values.median(),
                }
            )

    return pd.DataFrame(rows)


def save_outputs(
    output_dir: Path,
    outputs: dict[str, pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in outputs.items():
        parquet_path = output_dir / f"{name}.parquet"
        csv_path = output_dir / f"{name}.csv"

        data.to_parquet(parquet_path, index=False)
        data.to_csv(csv_path, index=False)

        print(f"Saved {name}: {data.shape} -> {parquet_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ortools_dir = resolve_ortools_dir(args.ortools_dir, args.prefix)

    print("OR-Tools directory:", ortools_dir)
    print("Lyrics file:", args.lyrics_file)
    print("Output directory:", args.output_dir)
    print("Prefix:", args.prefix)

    assignments, selected_genres, selected_genre_list = load_assignments(
        ortools_dir=ortools_dir,
        prefix=args.prefix,
        analysis_min_year=args.analysis_min_year,
        analysis_max_year=args.analysis_max_year,
    )

    print("Prepared assignments:", assignments.shape)
    print("Selected genres:", len(selected_genre_list))

    lyrics_dataset, all_tfidf_features = load_lyrics_dataset(
        lyrics_file=args.lyrics_file,
        assignments=assignments,
        analysis_min_year=args.analysis_min_year,
        analysis_max_year=args.analysis_max_year,
    )

    print("Calculating TF-IDF feature coverage")
    feature_coverage = calculate_tfidf_feature_coverage(
        dataset=lyrics_dataset,
        feature_columns=all_tfidf_features,
    )

    print("Expanding all selected lyrics features into sliding windows")
    full_windowed_tracks = expand_tracks_to_windows(
        dataset=lyrics_dataset,
        feature_columns=all_tfidf_features,
        analysis_min_year=args.analysis_min_year,
        analysis_max_year=args.analysis_max_year,
        window_size=args.window_size,
        hop_size=args.hop_size,
    )
    print("Full windowed lyrics tracks:", full_windowed_tracks.shape)

    full_all_genres_window_summary = calculate_all_genres_window_summary(
        windowed_tracks=full_windowed_tracks,
        feature_columns=all_tfidf_features,
    )

    feature_selection_scores = calculate_feature_selection_scores(
        feature_coverage=feature_coverage,
        all_genres_window_summary=full_all_genres_window_summary,
        feature_columns=all_tfidf_features,
    )

    selected_lyrics_features = select_lyrics_features(
        selection_scores=feature_selection_scores,
        feature_columns=all_tfidf_features,
        manual_features_file=args.manual_features_file,
        min_nonzero_share=args.min_nonzero_share,
        min_nonzero_tracks=args.min_nonzero_tracks,
        max_features=args.max_features,
    )

    selected_feature_table = feature_selection_scores[
        feature_selection_scores["feature"].isin(selected_lyrics_features)
    ].copy()
    selected_feature_table["selection_rank"] = selected_feature_table[
        "feature"
    ].map(
        {
            feature: rank
            for rank, feature in enumerate(selected_lyrics_features, start=1)
        }
    )
    selected_feature_table = selected_feature_table.sort_values("selection_rank")

    print("Selected lyrics TF-IDF features:")
    print(selected_feature_table[["selection_rank", "feature", "nonzero_share"]])

    print("Creating selected-feature dataset")
    selected_lyrics_dataset = lyrics_dataset[
        METADATA_COLUMNS + selected_lyrics_features
    ].copy()

    windowed_tracks = full_windowed_tracks[
        METADATA_COLUMNS
        + ["window_start", "window_end", "window_label"]
        + selected_lyrics_features
    ].copy()

    (
        genre_window_summary,
        eligible_genre_window_summary,
        all_genres_window_summary,
    ) = calculate_window_summaries(
        windowed_tracks=windowed_tracks,
        feature_columns=selected_lyrics_features,
        min_tracks=args.min_tracks,
    )

    (
        genre_window_z_summary,
        eligible_genre_window_z_summary,
    ) = calculate_within_period_z_summaries(
        windowed_tracks=windowed_tracks,
        feature_columns=selected_lyrics_features,
        min_tracks=args.min_tracks,
    )

    feature_deviation_summary = calculate_feature_deviation_summary(
        eligible_genre_window_z_summary=eligible_genre_window_z_summary,
        feature_columns=selected_lyrics_features,
    )

    all_genres_feature_change = calculate_all_genres_feature_change(
        all_genres_window_summary=all_genres_window_summary,
        feature_columns=selected_lyrics_features,
    )

    genre_feature_trends = calculate_genre_feature_trends(
        eligible_genre_window_summary=eligible_genre_window_summary,
        feature_columns=selected_lyrics_features,
    )

    selected_genres.to_csv(
        args.output_dir / "lyrics_tfidf_selected_genres.csv",
        index=False,
    )

    save_outputs(
        output_dir=args.output_dir,
        outputs={
            "lyrics_tfidf_assignments_prepared": selected_lyrics_dataset,
            "lyrics_tfidf_feature_coverage": feature_coverage,
            "lyrics_tfidf_feature_selection_scores": feature_selection_scores,
            "lyrics_tfidf_selected_features": selected_feature_table,
            "lyrics_tfidf_windowed_tracks": windowed_tracks,
            "lyrics_tfidf_genre_window_summary": genre_window_summary,
            "lyrics_tfidf_eligible_genre_window_summary": (
                eligible_genre_window_summary
            ),
            "lyrics_tfidf_all_genres_window_summary": all_genres_window_summary,
            "lyrics_tfidf_genre_window_within_period_z_summary": (
                genre_window_z_summary
            ),
            "lyrics_tfidf_eligible_genre_window_within_period_z_summary": (
                eligible_genre_window_z_summary
            ),
            "lyrics_tfidf_feature_deviation_summary": feature_deviation_summary,
            "lyrics_tfidf_all_genres_first_last_change": (
                all_genres_feature_change
            ),
            "lyrics_tfidf_genre_feature_trends": genre_feature_trends,
        },
    )

    print()
    print("Done. Main inspection tables:")
    print("- lyrics_tfidf_selected_features")
    print("- lyrics_tfidf_all_genres_window_summary")
    print("- lyrics_tfidf_eligible_genre_window_summary")
    print("- lyrics_tfidf_eligible_genre_window_within_period_z_summary")
    print("- lyrics_tfidf_feature_deviation_summary")
    print("- lyrics_tfidf_genre_feature_trends")


if __name__ == "__main__":
    main()
