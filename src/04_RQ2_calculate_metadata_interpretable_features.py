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
DEFAULT_METADATA_FEATURES = [
    "danceability",
    "energy",
    "key",
    "mode",
    "valence",
    "tempo",
    "duration_ms",
]
DEFAULT_CONTINUOUS_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "duration_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate RQ2 metadata feature summaries from the final "
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
        "--early-start",
        type=int,
        default=1955,
        help="Start year for early-period comparison.",
    )
    parser.add_argument(
        "--early-end",
        type=int,
        default=1974,
        help="End year for early-period comparison.",
    )
    parser.add_argument(
        "--late-start",
        type=int,
        default=2000,
        help="Start year for late-period comparison.",
    )
    parser.add_argument(
        "--late-end",
        type=int,
        default=2019,
        help="End year for late-period comparison.",
    )
    parser.add_argument(
        "--ortools-dir",
        type=Path,
        default=PROCESSED_DIR / "01_selected_genres_ortools",
        help="Directory containing OR-Tools assignment outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR / "RQ2_metadata_interpretable_features",
        help="Directory where calculated RQ2 metadata outputs are saved.",
    )
    return parser.parse_args()


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
    metadata_features: list[str],
    analysis_min_year: int,
    analysis_max_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    assignments = pd.read_parquet(ortools_dir / f"{prefix}_assignments.parquet")
    selected_genres = pd.read_csv(ortools_dir / f"{prefix}_selected_genres.csv")
    selected_genre_list = selected_genres["genre"].drop_duplicates().tolist()

    required_columns = ["release"] + metadata_features
    missing_metadata_features = [
        feature for feature in required_columns
        if feature not in assignments.columns
    ]

    if missing_metadata_features:
        print("Assignment file is missing metadata columns. Merging raw metadata.")
        metadata = read_metadata()
        keep_columns = ["id"] + required_columns
        missing_raw_columns = [
            column for column in keep_columns
            if column not in metadata.columns
        ]

        if missing_raw_columns:
            raise ValueError(
                "Raw metadata is missing required columns: "
                + ", ".join(missing_raw_columns)
            )

        metadata = metadata[keep_columns].copy()

        for column in required_columns:
            metadata[column] = pd.to_numeric(metadata[column], errors="coerce")

        assignments = assignments.drop(
            columns=[
                column
                for column in keep_columns
                if column in assignments.columns and column != "id"
            ],
            errors="ignore",
        )
        assignments = assignments.merge(metadata, on="id", how="left")
    else:
        print("Assignment file already contains metadata columns.")

    for column in required_columns:
        assignments[column] = pd.to_numeric(assignments[column], errors="coerce")

    assignments = assignments.dropna(subset=["release", "genre"]).copy()
    assignments["release"] = assignments["release"].astype(int)

    assignments = assignments[
        (assignments["release"] >= analysis_min_year)
        & (assignments["release"] <= analysis_max_year)
        & (assignments["genre"].isin(selected_genre_list))
    ].copy()

    return assignments, selected_genres, selected_genre_list


def calculate_metadata_coverage(
    assignments: pd.DataFrame,
    metadata_features: list[str],
) -> pd.DataFrame:
    rows = []
    n_tracks = assignments["id"].nunique()

    for feature in metadata_features:
        n_non_missing = assignments.dropna(subset=[feature])["id"].nunique()
        rows.append(
            {
                "feature": feature,
                "n_tracks": n_tracks,
                "n_non_missing": n_non_missing,
                "n_missing": n_tracks - n_non_missing,
                "missing_share": 1 - n_non_missing / n_tracks,
            }
        )

    return pd.DataFrame(rows)


def expand_tracks_to_windows(
    assignments: pd.DataFrame,
    metadata_features: list[str],
    analysis_min_year: int,
    analysis_max_year: int,
    window_size: int,
    hop_size: int,
) -> pd.DataFrame:
    expanded_rows = []
    columns = ["id", "genre", "release"] + metadata_features

    for row in assignments[columns].to_dict("records"):
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


def calculate_window_summaries(
    windowed_tracks: pd.DataFrame,
    metadata_features: list[str],
    min_tracks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genre_window_summary = (
        windowed_tracks
        .groupby(["genre", "window_start", "window_end", "window_label"])
        .agg(
            n_tracks=("id", "nunique"),
            **{feature: (feature, "mean") for feature in metadata_features},
        )
        .reset_index()
    )

    genre_window_summary["eligible_window"] = (
        genre_window_summary["n_tracks"] >= min_tracks
    )

    eligible_genre_window_summary = genre_window_summary[
        genre_window_summary["eligible_window"]
    ].copy()

    all_genres_window_summary = (
        windowed_tracks
        .groupby(["window_start", "window_end", "window_label"])
        .agg(
            n_tracks=("id", "nunique"),
            n_genres=("genre", "nunique"),
            **{feature: (feature, "mean") for feature in metadata_features},
        )
        .reset_index()
    )

    return (
        genre_window_summary,
        eligible_genre_window_summary,
        all_genres_window_summary,
    )


def calculate_within_period_z_summaries(
    windowed_tracks: pd.DataFrame,
    metadata_features: list[str],
    min_tracks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = 1e-8
    z_rows = []

    for _, window_data in windowed_tracks.groupby("window_start"):
        means = window_data[metadata_features].mean()
        stds = window_data[metadata_features].std(ddof=0).replace(0, eps)

        normalized_window = window_data.copy()
        normalized_window[metadata_features] = (
            normalized_window[metadata_features] - means
        ) / stds

        genre_summary = (
            normalized_window
            .groupby(["genre", "window_start", "window_end", "window_label"])
            .agg(
                n_tracks=("id", "nunique"),
                **{feature: (feature, "mean") for feature in metadata_features},
            )
            .reset_index()
        )

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
    continuous_features: list[str],
) -> pd.DataFrame:
    deviation_rows = []

    for feature in continuous_features:
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


def calculate_mode_summaries(
    windowed_tracks: pd.DataFrame,
    min_tracks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode_window_summary = (
        windowed_tracks
        .dropna(subset=["mode"])
        .assign(is_major=lambda data: data["mode"] == 1)
        .groupby(["genre", "window_start", "window_end", "window_label"])
        .agg(
            n_tracks=("id", "nunique"),
            major_share=("is_major", "mean"),
        )
        .reset_index()
    )
    mode_window_summary = mode_window_summary[
        mode_window_summary["n_tracks"] >= min_tracks
    ].copy()

    all_genres_mode_summary = (
        windowed_tracks
        .dropna(subset=["mode"])
        .assign(is_major=lambda data: data["mode"] == 1)
        .groupby(["window_start", "window_end", "window_label"])
        .agg(
            n_tracks=("id", "nunique"),
            major_share=("is_major", "mean"),
        )
        .reset_index()
    )

    return mode_window_summary, all_genres_mode_summary


def calculate_key_summary(windowed_tracks: pd.DataFrame) -> pd.DataFrame:
    key_window_counts = (
        windowed_tracks
        .dropna(subset=["key"])
        .assign(key=lambda data: data["key"].round().astype(int))
        .groupby(["window_start", "key"])
        .agg(n_tracks=("id", "nunique"))
        .reset_index()
    )

    key_window_counts["share"] = (
        key_window_counts["n_tracks"]
        / key_window_counts.groupby("window_start")["n_tracks"].transform("sum")
    )

    return key_window_counts


def calculate_early_late_changes(
    assignments: pd.DataFrame,
    continuous_features: list[str],
    early_period: tuple[int, int],
    late_period: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_data = assignments.copy()
    period_data["period"] = np.select(
        [
            period_data["release"].between(*early_period),
            period_data["release"].between(*late_period),
        ],
        ["early", "late"],
        default="other",
    )

    early_late_summary = (
        period_data[period_data["period"].isin(["early", "late"])]
        .groupby(["genre", "period"])
        .agg(
            n_tracks=("id", "nunique"),
            **{feature: (feature, "mean") for feature in continuous_features},
        )
        .reset_index()
    )

    early_late_wide = early_late_summary.pivot_table(
        index="genre",
        columns="period",
        values=continuous_features,
    )

    change_rows = []

    for feature in continuous_features:
        early_column = (feature, "early")
        late_column = (feature, "late")

        if early_column in early_late_wide.columns and late_column in early_late_wide.columns:
            change = early_late_wide[late_column] - early_late_wide[early_column]
            feature_change = change.reset_index(name="late_minus_early")
            feature_change["feature"] = feature
            change_rows.append(feature_change)

    early_late_changes = pd.concat(change_rows, ignore_index=True)
    return early_late_summary, early_late_changes


def calculate_all_genres_feature_change(
    all_genres_window_summary: pd.DataFrame,
    continuous_features: list[str],
) -> pd.DataFrame:
    sorted_reference = all_genres_window_summary.sort_values("window_start")
    rows = []

    for feature in continuous_features:
        first_value = sorted_reference[feature].iloc[0]
        last_value = sorted_reference[feature].iloc[-1]

        rows.append(
            {
                "feature": feature,
                "first_window_value": first_value,
                "last_window_value": last_value,
                "last_minus_first": last_value - first_value,
            }
        )

    return pd.DataFrame(rows)


def calculate_genre_feature_trends(
    eligible_genre_window_summary: pd.DataFrame,
    continuous_features: list[str],
) -> pd.DataFrame:
    rows = []

    for feature in continuous_features:
        for genre, genre_data in eligible_genre_window_summary.groupby("genre"):
            genre_data = genre_data.sort_values("window_start")

            if len(genre_data) < 2:
                continue

            rows.append(
                {
                    "feature": feature,
                    "genre": genre,
                    "n_valid_windows": genre_data["window_start"].nunique(),
                    "first_window": genre_data["window_start"].iloc[0],
                    "last_window": genre_data["window_start"].iloc[-1],
                    "first_value": genre_data[feature].iloc[0],
                    "last_value": genre_data[feature].iloc[-1],
                    "last_minus_first": (
                        genre_data[feature].iloc[-1]
                        - genre_data[feature].iloc[0]
                    ),
                    "mean_value": genre_data[feature].mean(),
                    "median_value": genre_data[feature].median(),
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

    print("OR-Tools directory:", args.ortools_dir)
    print("Output directory:", args.output_dir)
    print("Prefix:", args.prefix)

    assignments, selected_genres, selected_genre_list = load_assignments(
        ortools_dir=args.ortools_dir,
        prefix=args.prefix,
        metadata_features=DEFAULT_METADATA_FEATURES,
        analysis_min_year=args.analysis_min_year,
        analysis_max_year=args.analysis_max_year,
    )

    print("Prepared assignments:", assignments.shape)
    print("Selected genres:", len(selected_genre_list))

    metadata_coverage = calculate_metadata_coverage(
        assignments=assignments,
        metadata_features=DEFAULT_METADATA_FEATURES,
    )

    print("Expanding tracks into sliding windows")
    windowed_tracks = expand_tracks_to_windows(
        assignments=assignments,
        metadata_features=DEFAULT_METADATA_FEATURES,
        analysis_min_year=args.analysis_min_year,
        analysis_max_year=args.analysis_max_year,
        window_size=args.window_size,
        hop_size=args.hop_size,
    )
    print("Windowed tracks:", windowed_tracks.shape)

    (
        genre_window_summary,
        eligible_genre_window_summary,
        all_genres_window_summary,
    ) = calculate_window_summaries(
        windowed_tracks=windowed_tracks,
        metadata_features=DEFAULT_METADATA_FEATURES,
        min_tracks=args.min_tracks,
    )

    (
        genre_window_z_summary,
        eligible_genre_window_z_summary,
    ) = calculate_within_period_z_summaries(
        windowed_tracks=windowed_tracks,
        metadata_features=DEFAULT_METADATA_FEATURES,
        min_tracks=args.min_tracks,
    )

    feature_deviation_summary = calculate_feature_deviation_summary(
        eligible_genre_window_z_summary=eligible_genre_window_z_summary,
        continuous_features=DEFAULT_CONTINUOUS_FEATURES,
    )

    mode_window_summary, all_genres_mode_summary = calculate_mode_summaries(
        windowed_tracks=windowed_tracks,
        min_tracks=args.min_tracks,
    )

    key_window_counts = calculate_key_summary(windowed_tracks=windowed_tracks)

    early_late_summary, early_late_changes = calculate_early_late_changes(
        assignments=assignments,
        continuous_features=DEFAULT_CONTINUOUS_FEATURES,
        early_period=(args.early_start, args.early_end),
        late_period=(args.late_start, args.late_end),
    )

    all_genres_feature_change = calculate_all_genres_feature_change(
        all_genres_window_summary=all_genres_window_summary,
        continuous_features=DEFAULT_CONTINUOUS_FEATURES,
    )

    genre_feature_trends = calculate_genre_feature_trends(
        eligible_genre_window_summary=eligible_genre_window_summary,
        continuous_features=DEFAULT_CONTINUOUS_FEATURES,
    )

    selected_genres.to_csv(
        args.output_dir / "metadata_selected_genres.csv",
        index=False,
    )

    save_outputs(
        output_dir=args.output_dir,
        outputs={
            "metadata_assignments_prepared": assignments,
            "metadata_feature_coverage": metadata_coverage,
            "metadata_windowed_tracks": windowed_tracks,
            "metadata_genre_window_summary": genre_window_summary,
            "metadata_eligible_genre_window_summary": eligible_genre_window_summary,
            "metadata_all_genres_window_summary": all_genres_window_summary,
            "metadata_genre_window_within_period_z_summary": genre_window_z_summary,
            "metadata_eligible_genre_window_within_period_z_summary": (
                eligible_genre_window_z_summary
            ),
            "metadata_feature_deviation_summary": feature_deviation_summary,
            "metadata_mode_window_summary": mode_window_summary,
            "metadata_all_genres_mode_summary": all_genres_mode_summary,
            "metadata_key_window_counts": key_window_counts,
            "metadata_early_late_summary": early_late_summary,
            "metadata_early_late_feature_changes": early_late_changes,
            "metadata_all_genres_first_last_change": all_genres_feature_change,
            "metadata_genre_feature_trends": genre_feature_trends,
        },
    )

    print()
    print("Done. Main inspection tables:")
    print("- metadata_all_genres_window_summary")
    print("- metadata_eligible_genre_window_summary")
    print("- metadata_eligible_genre_window_within_period_z_summary")
    print("- metadata_feature_deviation_summary")
    print("- metadata_genre_feature_trends")


if __name__ == "__main__":
    main()
