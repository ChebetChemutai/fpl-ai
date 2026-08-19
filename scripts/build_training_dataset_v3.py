import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FPL AI — V3 TRAINING DATASET BUILDER
# ============================================================
#
# V3 adds historical team/opponent strength features.
#
# DESIGN RULES
# ------------------------------------------------------------
# 1. No current-GW player performance is used as a feature.
#
# 2. Player rolling features use shift(1).
#
# 3. Team rolling features use shift(1).
#
# 4. Team history is built BEFORE player/GW aggregation.
#
# 5. Team features are merged using:
#
#       fixture + team
#
# 6. Opponent history is matched using:
#
#       fixture + opponent_team_id
#
# 7. The target is:
#
#       player's TOTAL FPL points in GW+1
#
# 8. Multiple fixture rows for a player in one GW are
#    aggregated into one player/GW observation.
#
# 9. Managers are removed using position == "AM".
#
# 10. Team IDs are inferred from the two sides of each fixture.
#
# 11. Current fixture is excluded from all team rolling
#     features through shift(1).
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "historical"
    / "2024-25_merged_gw.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "historical"
    / "training_dataset_v3.csv"
)

ROLLING_WINDOWS = [3, 5]


# ============================================================
# HELPERS
# ============================================================

def rolling_mean_previous(group, column, window):
    """
    Mean of previous N observations.

    Current observation is excluded using shift(1).
    """

    return (
        group[column]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=window,
                min_periods=1,
            )
            .mean()
        )
    )


def rolling_sum_previous(group, column, window):
    """
    Sum of previous N observations.

    Current observation is excluded using shift(1).
    """

    return (
        group[column]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=window,
                min_periods=1,
            )
            .sum()
        )
    )


def safe_divide(numerator, denominator):
    """
    Safe division.

    Invalid / zero denominators become zero.
    """

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    denominator = denominator.replace(
        0,
        np.nan,
    )

    result = numerator / denominator

    return (
        result
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FPL AI — V3 HISTORICAL TRAINING DATASET")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_FILE)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Historical file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# SOURCE DIAGNOSTICS
# ============================================================

print("\n" + "=" * 80)
print("SOURCE DIAGNOSTICS")
print("=" * 80)

print("\nSource column types BEFORE conversion:")

print(
    df.dtypes.to_string()
)


if "team" in df.columns:

    print("\nTEAM SAMPLE:")

    print(
        df["team"]
        .head(20)
        .to_string(index=False)
    )

    print("\nUNIQUE TEAM VALUES:")

    print(
        df["team"]
        .dropna()
        .unique()
    )


if "opponent_team" in df.columns:

    print("\nOPPONENT TEAM SAMPLE:")

    print(
        df["opponent_team"]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "name",
    "position",
    "team",
    "element",
    "GW",
    "value",
    "was_home",
    "opponent_team",
    "fixture",
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "goals_conceded",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================
#
# IMPORTANT:
#
# `team` is intentionally NOT converted to numeric.
#
# In this dataset:
#
#     team           = team name
#     opponent_team  = FPL team ID
#
# We preserve that distinction.
#
# ============================================================

numeric_columns = [
    "element",
    "GW",
    "value",
    "opponent_team",
    "fixture",
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "goals_conceded",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# TEAM CLEANING
# ============================================================

df["team"] = (
    df["team"]
    .astype("string")
    .str.strip()
)


# ============================================================
# SOURCE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("SOURCE VALIDATION")
print("=" * 80)


core_columns = [
    "element",
    "GW",
    "position",
    "total_points",
]


print("\nNull counts in core columns:")

print(
    df[core_columns]
    .isna()
    .sum()
    .to_string()
)


team_nulls = int(
    df["team"].isna().sum()
)


print(
    f"\nSource `team` nulls: {team_nulls}"
)


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

df = df.dropna(
    subset=[
        "element",
        "GW",
        "team",
        "position",
        "total_points",
        "fixture",
        "opponent_team",
    ]
).copy()


print(
    f"\nRows after invalid-record removal: "
    f"{len(df):,}"
)


if len(df) == 0:

    raise RuntimeError(
        "\nERROR: All rows were removed during validation."
    )


# ============================================================
# REMOVE MANAGERS
# ============================================================

non_player_mask = (
    df["position"]
    .astype(str)
    .str.upper()
    .eq("AM")
)


non_player_rows = int(
    non_player_mask.sum()
)


df = df[
    ~non_player_mask
].copy()


print(
    f"\nNon-player rows removed: "
    f"{non_player_rows:,}"
)


if len(df) == 0:

    raise RuntimeError(
        "ERROR: No player rows remain after removing managers."
    )


# ============================================================
# SORT SOURCE
# ============================================================

df = df.sort_values(
    [
        "GW",
        "fixture",
        "team",
        "element",
    ]
).reset_index(
    drop=True
)


# ============================================================
# PART 1
# BUILD TEAM MATCH HISTORY
# ============================================================

print("\n" + "=" * 80)
print("BUILDING TEAM MATCH HISTORY")
print("=" * 80)


# ============================================================
# CREATE RAW FIXTURE/TEAM TABLE
# ============================================================

fixture_source = df[
    [
        "fixture",
        "GW",
        "team",
        "opponent_team",
        "was_home",
        "goals_scored",
        "goals_conceded",
        "expected_goals",
        "expected_goals_conceded",
    ]
].copy()


# ============================================================
# RAW DUPLICATE DIAGNOSTIC
# ============================================================

raw_fixture_team_counts = (
    fixture_source
    .groupby(
        [
            "fixture",
            "team",
        ]
    )
    .size()
)


raw_duplicate_fixture_team_groups = int(
    (raw_fixture_team_counts > 1).sum()
)


print(
    f"\nRaw fixture/team groups with multiple player rows: "
    f"{raw_duplicate_fixture_team_groups:,}"
)

print(
    "These are expected because multiple players belong "
    "to the same team in a fixture."
)


# ============================================================
# AGGREGATE PLAYER DATA → TEAM/FIXTURE
# ============================================================

team_fixture = (
    fixture_source
    .groupby(
        [
            "fixture",
            "GW",
            "team",
            "opponent_team",
            "was_home",
        ],
        as_index=False,
    )
    .agg(
        goals_scored=(
            "goals_scored",
            "sum",
        ),

        goals_conceded=(
            "goals_conceded",
            "sum",
        ),

        expected_goals=(
            "expected_goals",
            "sum",
        ),

        expected_goals_conceded=(
            "expected_goals_conceded",
            "sum",
        ),
    )
)


print(
    f"\nUnique fixture-side observations: "
    f"{len(team_fixture):,}"
)

print(
    f"Unique fixtures: "
    f"{team_fixture['fixture'].nunique():,}"
)

print(
    f"Unique teams: "
    f"{team_fixture['team'].nunique():,}"
)


# ============================================================
# FIXTURE SIDE VALIDATION
# ============================================================

fixture_side_counts = (
    team_fixture
    .groupby("fixture")
    .size()
)


fixtures_not_two_sides = int(
    (fixture_side_counts != 2).sum()
)


print(
    f"Fixtures without exactly two sides: "
    f"{fixtures_not_two_sides:,}"
)


if fixtures_not_two_sides != 0:

    print(
        "\nWARNING:"
    )

    print(
        team_fixture[
            team_fixture["fixture"].isin(
                fixture_side_counts[
                    fixture_side_counts != 2
                ].index
            )
        ]
        .sort_values(
            [
                "fixture",
                "team",
            ]
        )
        .to_string(
            index=False
        )
    )

    raise RuntimeError(
        "\nERROR: Some fixtures do not contain exactly "
        "two team sides."
    )


# ============================================================
# VALIDATE FIXTURE/TEAM UNIQUENESS
# ============================================================

team_fixture_duplicates = int(
    team_fixture.duplicated(
        [
            "fixture",
            "team",
        ]
    ).sum()
)


print(
    f"\nDuplicate fixture/team rows after aggregation: "
    f"{team_fixture_duplicates:,}"
)


if team_fixture_duplicates != 0:

    raise RuntimeError(
        "ERROR: Team fixture table is not unique on "
        "(fixture, team)."
    )


# ============================================================
# DERIVE TEAM ID
# ============================================================
#
# IMPORTANT FIX
# ------------------------------------------------------------
#
# Source structure:
#
#     team           = "Arsenal"
#     opponent_team  = 13
#
# The current team's numeric ID is found on the OTHER side
# of the fixture.
#
# Example:
#
# Arsenal row:
#     team = Arsenal
#     opponent_team = 13   # Chelsea ID
#
# Chelsea row:
#     team = Chelsea
#     opponent_team = 1    # Arsenal ID
#
# Therefore:
#
#     Arsenal team_id = 1
#     Chelsea team_id = 13
#
# Because every fixture has exactly two sides, we can infer
# the team ID by taking the opponent_team value from the
# opposite side.
#
# ============================================================

team_fixture = team_fixture.sort_values(
    [
        "fixture",
        "team",
    ]
).reset_index(
    drop=True
)


team_fixture["team_id"] = (
    team_fixture
    .groupby("fixture")["opponent_team"]
    .transform(
        lambda s: s.iloc[::-1].to_numpy()
    )
)


team_fixture["team_id"] = pd.to_numeric(
    team_fixture["team_id"],
    errors="coerce",
)


if team_fixture["team_id"].isna().any():

    raise RuntimeError(
        "ERROR: Could not derive numeric team IDs."
    )


# ============================================================
# VALIDATE TEAM ID MAPPING
# ============================================================

team_id_mapping = (
    team_fixture[
        [
            "team",
            "team_id",
        ]
    ]
    .drop_duplicates()
)


team_mapping_counts = (
    team_id_mapping
    .groupby("team")["team_id"]
    .nunique()
)


invalid_team_mappings = (
    team_mapping_counts[
        team_mapping_counts != 1
    ]
)


print(
    f"\nUnique team name → team ID mappings: "
    f"{len(team_id_mapping):,}"
)


if len(invalid_team_mappings) != 0:

    print(
        "\nInvalid team mappings:"
    )

    print(
        invalid_team_mappings.to_string()
    )

    raise RuntimeError(
        "ERROR: A team name maps to multiple team IDs."
    )


print(
    "\nTeam ID mapping validation: PASS"
)


# ============================================================
# SORT TEAM HISTORY
# ============================================================

team_fixture = team_fixture.sort_values(
    [
        "team_id",
        "GW",
        "fixture",
    ]
).reset_index(
    drop=True
)


# ============================================================
# TEAM HISTORY GROUP
# ============================================================

team_group = (
    team_fixture
    .groupby(
        "team_id",
        sort=False,
    )
)


# ============================================================
# TEAM ROLLING FEATURES
# ============================================================

for window in ROLLING_WINDOWS:

    team_fixture[
        f"team_goals_last_{window}"
    ] = rolling_mean_previous(
        team_group,
        "goals_scored",
        window,
    )


    team_fixture[
        f"team_xg_last_{window}"
    ] = rolling_mean_previous(
        team_group,
        "expected_goals",
        window,
    )


    team_fixture[
        f"team_goals_conceded_last_{window}"
    ] = rolling_mean_previous(
        team_group,
        "goals_conceded",
        window,
    )


    team_fixture[
        f"team_xgc_last_{window}"
    ] = rolling_mean_previous(
        team_group,
        "expected_goals_conceded",
        window,
    )


# ============================================================
# CREATE OPPONENT HISTORY
# ============================================================
#
# We DO NOT merge string `team` with numeric `opponent_team`.
#
# Instead:
#
#     team_fixture.team_id
#
# is numeric and corresponds to:
#
#     opponent_team
#
# from the player's/team's fixture record.
#
# ============================================================

opponent_history = team_fixture[
    [
        "fixture",
        "team_id",

        "team_goals_last_3",
        "team_goals_last_5",

        "team_xg_last_3",
        "team_xg_last_5",

        "team_goals_conceded_last_3",
        "team_goals_conceded_last_5",

        "team_xgc_last_3",
        "team_xgc_last_5",
    ]
].copy()


opponent_history = opponent_history.rename(
    columns={
        "team_id":
            "opponent_team_id",

        "team_goals_last_3":
            "opponent_goals_last_3",

        "team_goals_last_5":
            "opponent_goals_last_5",

        "team_xg_last_3":
            "opponent_xg_last_3",

        "team_xg_last_5":
            "opponent_xg_last_5",

        "team_goals_conceded_last_3":
            "opponent_goals_conceded_last_3",

        "team_goals_conceded_last_5":
            "opponent_goals_conceded_last_5",

        "team_xgc_last_3":
            "opponent_xgc_last_3",

        "team_xgc_last_5":
            "opponent_xgc_last_5",
    }
)


# ============================================================
# OPPONENT HISTORY MERGE
# ============================================================
#
# Correct key:
#
#     fixture
#     opponent_team_id
#
# ============================================================

team_fixture = team_fixture.merge(
    opponent_history,
    left_on=[
        "fixture",
        "opponent_team",
    ],
    right_on=[
        "fixture",
        "opponent_team_id",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# REMOVE TEMPORARY OPPONENT ID
# ============================================================

team_fixture = team_fixture.drop(
    columns=[
        "opponent_team_id",
    ]
)


# ============================================================
# VALIDATE OPPONENT HISTORY
# ============================================================

opponent_history_nulls = (
    team_fixture[
        [
            "opponent_goals_last_3",
            "opponent_goals_last_5",
            "opponent_xg_last_3",
            "opponent_xg_last_5",
        ]
    ]
    .isna()
    .sum()
    .sum()
)


print(
    f"\nOpponent historical values requiring "
    f"initial-history fill: "
    f"{opponent_history_nulls:,}"
)


# ============================================================
# ATTACK / DEFENCE STRENGTH
# ============================================================

team_fixture["attack_strength"] = safe_divide(
    team_fixture["team_xg_last_5"],
    team_fixture["opponent_xgc_last_5"] + 0.1,
)


team_fixture["defensive_strength"] = safe_divide(
    team_fixture["opponent_xg_last_5"],
    team_fixture["team_xgc_last_5"] + 0.1,
)


team_fixture["opponent_attack_strength"] = safe_divide(
    team_fixture["opponent_xg_last_5"],
    team_fixture["team_xgc_last_5"] + 0.1,
)


team_fixture["opponent_defensive_strength"] = safe_divide(
    team_fixture["team_xg_last_5"],
    team_fixture["opponent_xgc_last_5"] + 0.1,
)


# ============================================================
# TEAM FEATURE COLUMNS
# ============================================================

team_feature_columns = [
    "fixture",
    "team",

    "team_goals_last_3",
    "team_goals_last_5",

    "team_xg_last_3",
    "team_xg_last_5",

    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",

    "team_xgc_last_3",
    "team_xgc_last_5",

    "opponent_goals_last_3",
    "opponent_goals_last_5",

    "opponent_xg_last_3",
    "opponent_xg_last_5",

    "opponent_goals_conceded_last_3",
    "opponent_goals_conceded_last_5",

    "opponent_xgc_last_3",
    "opponent_xgc_last_5",

    "attack_strength",
    "defensive_strength",

    "opponent_attack_strength",
    "opponent_defensive_strength",
]


team_features = team_fixture[
    team_feature_columns
].copy()


# ============================================================
# TEAM FEATURE UNIQUENESS
# ============================================================

team_feature_duplicates = int(
    team_features.duplicated(
        [
            "fixture",
            "team",
        ]
    ).sum()
)


print(
    f"\nDuplicate fixture/team feature rows: "
    f"{team_feature_duplicates:,}"
)


if team_feature_duplicates != 0:

    raise RuntimeError(
        "ERROR: Team feature table is not unique on "
        "(fixture, team)."
    )


# ============================================================
# PART 2
# BUILD PLAYER/GAMEWEEK DATA
# ============================================================

print("\n" + "=" * 80)
print("BUILDING PLAYER/GAMEWEEK DATA")
print("=" * 80)


# ============================================================
# PLAYER/GW DUPLICATE DETECTION
# ============================================================

player_gw_counts = (
    df.groupby(
        [
            "element",
            "GW",
        ]
    )
    .size()
)


duplicate_groups = int(
    (player_gw_counts > 1).sum()
)


duplicate_rows = int(
    player_gw_counts[
        player_gw_counts > 1
    ].sum()
)


print(
    f"Player/GW groups with multiple fixture rows: "
    f"{duplicate_groups:,}"
)


print(
    f"Rows belonging to those groups: "
    f"{duplicate_rows:,}"
)


# ============================================================
# NEXT GW FIXTURE COUNT
# ============================================================
#
# This is calculated BEFORE aggregation so DGWs are preserved.
#
# ============================================================

player_gw_fixture_counts = (
    df.groupby(
        [
            "element",
            "GW",
        ]
    )["fixture"]
    .nunique()
    .rename(
        "fixture_count"
    )
    .reset_index()
)


next_fixture_counts = (
    player_gw_fixture_counts
    .assign(
        GW=lambda x: x["GW"] - 1
    )
    [
        [
            "element",
            "GW",
            "fixture_count",
        ]
    ]
    .rename(
        columns={
            "fixture_count":
                "next_fixture_count",
        }
    )
)


# ============================================================
# PLAYER/GW AGGREGATION
# ============================================================

aggregation = {
    "name": "first",
    "position": "first",
    "team": "first",
    "value": "last",

    "was_home": "last",
    "opponent_team": "first",
    "fixture": "first",

    "minutes": "sum",
    "starts": "sum",
    "total_points": "sum",

    "goals_scored": "sum",
    "assists": "sum",

    "clean_sheets": "sum",
    "saves": "sum",
    "goals_conceded": "sum",

    "bonus": "sum",
    "bps": "sum",

    "expected_goals": "sum",
    "expected_assists": "sum",
    "expected_goal_involvements": "sum",
    "expected_goals_conceded": "sum",
}


aggregation = {
    key: value
    for key, value in aggregation.items()
    if key in df.columns
}


df = (
    df.groupby(
        [
            "element",
            "GW",
        ],
        as_index=False,
    )
    .agg(
        aggregation
    )
)


print(
    f"Rows after player/GW aggregation: "
    f"{len(df):,}"
)


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

remaining_duplicates = int(
    df.duplicated(
        [
            "element",
            "GW",
        ]
    ).sum()
)


if remaining_duplicates != 0:

    raise RuntimeError(
        "ERROR: Duplicate player/GW rows remain after aggregation."
    )


print(
    "Duplicate validation: PASS"
)


# ============================================================
# MERGE NEXT FIXTURE COUNT
# ============================================================

df = df.merge(
    next_fixture_counts,
    on=[
        "element",
        "GW",
    ],
    how="left",
    validate="one_to_one",
)


df["next_fixture_count"] = (
    df["next_fixture_count"]
    .fillna(0)
    .clip(lower=0)
)


# ============================================================
# SORT PLAYER DATA
# ============================================================

df = df.sort_values(
    [
        "element",
        "GW",
    ]
).reset_index(
    drop=True
)


# ============================================================
# BASIC FEATURES
# ============================================================

df["price"] = (
    df["value"] / 10.0
)


# ============================================================
# PLAYER HISTORY GROUP
# ============================================================

group = df.groupby(
    "element",
    sort=False,
)


# ============================================================
# PREVIOUS POINTS
# ============================================================

df["previous_points"] = (
    group["total_points"]
    .shift(1)
)


# ============================================================
# POINTS
# ============================================================

for window in ROLLING_WINDOWS:

    df[
        f"points_last_{window}"
    ] = rolling_mean_previous(
        group,
        "total_points",
        window,
    )


# ============================================================
# MINUTES
# ============================================================

for window in ROLLING_WINDOWS:

    df[
        f"minutes_last_{window}"
    ] = rolling_mean_previous(
        group,
        "minutes",
        window,
    )


# ============================================================
# STARTS
# ============================================================

for window in ROLLING_WINDOWS:

    df[
        f"starts_last_{window}"
    ] = rolling_sum_previous(
        group,
        "starts",
        window,
    )


# ============================================================
# START RATE
# ============================================================

for window in ROLLING_WINDOWS:

    denominator = pd.Series(
        window,
        index=df.index,
        dtype=float,
    )

    df[
        f"start_rate_last_{window}"
    ] = safe_divide(
        df[
            f"starts_last_{window}"
        ],
        denominator,
    )


# ============================================================
# MINUTES PER START
# ============================================================

df["minutes_per_start"] = safe_divide(
    df["minutes_last_5"],
    df["starts_last_5"],
)


# ============================================================
# ATTACKING FEATURES
# ============================================================

for column, prefix in [
    (
        "goals_scored",
        "goals",
    ),
    (
        "assists",
        "assists",
    ),
]:

    for window in ROLLING_WINDOWS:

        df[
            f"{prefix}_last_{window}"
        ] = rolling_sum_previous(
            group,
            column,
            window,
        )


# ============================================================
# EXPECTED PERFORMANCE
# ============================================================

for column, prefix in [
    (
        "expected_goals",
        "xg",
    ),
    (
        "expected_assists",
        "xa",
    ),
    (
        "expected_goal_involvements",
        "xgi",
    ),
]:

    for window in ROLLING_WINDOWS:

        df[
            f"{prefix}_last_{window}"
        ] = rolling_mean_previous(
            group,
            column,
            window,
        )


# ============================================================
# DEFENSIVE FEATURES
# ============================================================

for column, prefix in [
    (
        "clean_sheets",
        "clean_sheets",
    ),
    (
        "saves",
        "saves",
    ),
    (
        "goals_conceded",
        "goals_conceded",
    ),
]:

    for window in ROLLING_WINDOWS:

        df[
            f"{prefix}_last_{window}"
        ] = rolling_sum_previous(
            group,
            column,
            window,
        )


# ============================================================
# EXPECTED GOALS CONCEDED
# ============================================================

for window in ROLLING_WINDOWS:

    df[
        f"xgc_last_{window}"
    ] = rolling_mean_previous(
        group,
        "expected_goals_conceded",
        window,
    )


# ============================================================
# BONUS / BPS
# ============================================================

for column, prefix in [
    (
        "bonus",
        "bonus",
    ),
    (
        "bps",
        "bps",
    ),
]:

    for window in ROLLING_WINDOWS:

        df[
            f"{prefix}_last_{window}"
        ] = rolling_mean_previous(
            group,
            column,
            window,
        )


# ============================================================
# BPS PER 90
# ============================================================

df["bps_per_90_last_3"] = safe_divide(
    df["bps_last_3"] * 90,
    df["minutes_last_3"],
)


df["bps_per_90_last_5"] = safe_divide(
    df["bps_last_5"] * 90,
    df["minutes_last_5"],
)


# ============================================================
# POINTS PER 90
# ============================================================

df["points_per_90_last_3"] = safe_divide(
    df["points_last_3"] * 90,
    df["minutes_last_3"],
)


df["points_per_90_last_5"] = safe_divide(
    df["points_last_5"] * 90,
    df["minutes_last_5"],
)


# ============================================================
# VALUE FEATURES
# ============================================================

df["points_per_million"] = safe_divide(
    df["previous_points"],
    df["price"],
)


df["recent_points_per_million"] = safe_divide(
    df["points_last_5"],
    df["price"],
)


# ============================================================
# PART 3
# BUILD ACTUAL NEXT-GW FIXTURE INFORMATION
# ============================================================
#
# IMPORTANT:
#
# Do NOT use shift(-1) here.
#
# shift(-1) gives the next row for the player, which can be
# GW+2 if the player did not play GW+1.
#
# We explicitly join GW+1.
#
# ============================================================

next_gw_info = df[
    [
        "element",
        "GW",
        "was_home",
        "opponent_team",
    ]
].copy()


next_gw_info["GW"] = (
    next_gw_info["GW"] - 1
)


next_gw_info = next_gw_info.rename(
    columns={
        "was_home":
            "next_home",

        "opponent_team":
            "next_opponent",
    }
)


df = df.merge(
    next_gw_info,
    on=[
        "element",
        "GW",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# NORMALIZE NEXT FIXTURE FIELDS
# ============================================================

df["next_home"] = (
    df["next_home"]
    .fillna(False)
    .astype(int)
)


df["next_opponent"] = pd.to_numeric(
    df["next_opponent"],
    errors="coerce",
)


# ============================================================
# PART 4
# MERGE TEAM FEATURES
# ============================================================

print("\n" + "=" * 80)
print("MERGING TEAM FEATURES")
print("=" * 80)


before_team_merge = len(df)


# ============================================================
# IMPORTANT:
#
# PLAYER `team` IS A STRING.
#
# TEAM FEATURES ALSO CONTAIN THE ORIGINAL STRING `team`.
#
# Therefore this merge is:
#
#     fixture + team
#
# ============================================================

df = df.merge(
    team_features,
    on=[
        "fixture",
        "team",
    ],
    how="left",
    validate="many_to_one",
)


after_team_merge = len(df)


print(
    f"Rows before team merge: "
    f"{before_team_merge:,}"
)


print(
    f"Rows after team merge:  "
    f"{after_team_merge:,}"
)


if after_team_merge != before_team_merge:

    raise RuntimeError(
        "\nERROR: Team feature merge changed the number "
        "of player/GW rows.\n"
        f"Before: {before_team_merge:,}\n"
        f"After:  {after_team_merge:,}"
    )


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

duplicate_after_merge = int(
    df.duplicated(
        [
            "element",
            "GW",
        ]
    ).sum()
)


print(
    f"Duplicate player/GW rows after team merge: "
    f"{duplicate_after_merge:,}"
)


if duplicate_after_merge != 0:

    raise RuntimeError(
        "\nERROR: Team feature merge created "
        "duplicate player/GW observations."
    )


print(
    "Team merge validation: PASS"
)


# ============================================================
# TEAM FEATURE NULL DIAGNOSTIC
# ============================================================

team_feature_check_columns = [
    "team_goals_last_3",
    "team_goals_last_5",
    "team_xg_last_3",
    "team_xg_last_5",
    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",
    "team_xgc_last_3",
    "team_xgc_last_5",
    "opponent_goals_last_3",
    "opponent_goals_last_5",
    "opponent_xg_last_3",
    "opponent_xg_last_5",
    "opponent_goals_conceded_last_3",
    "opponent_goals_conceded_last_5",
    "opponent_xgc_last_3",
    "opponent_xgc_last_5",
]


print("\nTeam feature merge null diagnostics:")

for column in team_feature_check_columns:

    nulls = int(
        df[column].isna().sum()
    )

    print(
        f"{column:<40}"
        f"nulls={nulls:,}"
    )


# ============================================================
# FILL HISTORICAL FEATURES
# ============================================================

feature_columns = [
    "previous_points",

    "points_last_3",
    "points_last_5",

    "points_per_90_last_3",
    "points_per_90_last_5",

    "minutes_last_3",
    "minutes_last_5",

    "starts_last_3",
    "starts_last_5",

    "start_rate_last_3",
    "start_rate_last_5",

    "minutes_per_start",

    "goals_last_3",
    "goals_last_5",

    "assists_last_3",
    "assists_last_5",

    "xg_last_3",
    "xg_last_5",

    "xa_last_3",
    "xa_last_5",

    "xgi_last_3",
    "xgi_last_5",

    "clean_sheets_last_3",
    "clean_sheets_last_5",

    "saves_last_3",
    "saves_last_5",

    "goals_conceded_last_3",
    "goals_conceded_last_5",

    "xgc_last_3",
    "xgc_last_5",

    "bonus_last_3",
    "bonus_last_5",

    "bps_last_3",
    "bps_last_5",

    "bps_per_90_last_3",
    "bps_per_90_last_5",

    "points_per_million",
    "recent_points_per_million",

    "team_goals_last_3",
    "team_goals_last_5",

    "team_xg_last_3",
    "team_xg_last_5",

    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",

    "team_xgc_last_3",
    "team_xgc_last_5",

    "opponent_goals_last_3",
    "opponent_goals_last_5",

    "opponent_xg_last_3",
    "opponent_xg_last_5",

    "opponent_goals_conceded_last_3",
    "opponent_goals_conceded_last_5",

    "opponent_xgc_last_3",
    "opponent_xgc_last_5",

    "attack_strength",
    "defensive_strength",

    "opponent_attack_strength",
    "opponent_defensive_strength",
]


for column in feature_columns:

    if column in df.columns:

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )


# ============================================================
# TARGET
# ============================================================
#
# IMPORTANT:
#
# Target is explicitly GW+1.
#
# We do NOT use shift(-1), because that would incorrectly
# treat GW+2 as the target when a player missed GW+1.
#
# ============================================================

target_lookup = df[
    [
        "element",
        "GW",
        "total_points",
    ]
].copy()


target_lookup["GW"] = (
    target_lookup["GW"] - 1
)


target_lookup = target_lookup.rename(
    columns={
        "total_points":
            "target_points",
    }
)


df = df.merge(
    target_lookup,
    on=[
        "element",
        "GW",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# REMOVE ROWS WITHOUT GW+1 TARGET
# ============================================================

rows_before_target_filter = len(df)


df = df[
    df["target_points"].notna()
].copy()


rows_removed_target = (
    rows_before_target_filter
    - len(df)
)


print(
    f"\nRows removed without actual GW+1 target: "
    f"{rows_removed_target:,}"
)


# ============================================================
# FINAL NEXT OPPONENT TYPE
# ============================================================

df["next_opponent"] = pd.to_numeric(
    df["next_opponent"],
    errors="coerce",
)


# ============================================================
# FINAL DATASET COLUMNS
# ============================================================

training_columns = [
    "element",
    "GW",
    "position",
    "price",

    "next_home",
    "next_opponent",
    "next_fixture_count",

    "previous_points",

    "points_last_3",
    "points_last_5",

    "points_per_90_last_3",
    "points_per_90_last_5",

    "minutes_last_3",
    "minutes_last_5",

    "starts_last_3",
    "starts_last_5",

    "start_rate_last_3",
    "start_rate_last_5",

    "minutes_per_start",

    "goals_last_3",
    "goals_last_5",

    "assists_last_3",
    "assists_last_5",

    "xg_last_3",
    "xg_last_5",

    "xa_last_3",
    "xa_last_5",

    "xgi_last_3",
    "xgi_last_5",

    "clean_sheets_last_3",
    "clean_sheets_last_5",

    "saves_last_3",
    "saves_last_5",

    "goals_conceded_last_3",
    "goals_conceded_last_5",

    "xgc_last_3",
    "xgc_last_5",

    "bonus_last_3",
    "bonus_last_5",

    "bps_last_3",
    "bps_last_5",

    "bps_per_90_last_3",
    "bps_per_90_last_5",

    "points_per_million",
    "recent_points_per_million",

    "team_goals_last_3",
    "team_goals_last_5",

    "team_xg_last_3",
    "team_xg_last_5",

    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",

    "team_xgc_last_3",
    "team_xgc_last_5",

    "opponent_goals_last_3",
    "opponent_goals_last_5",

    "opponent_xg_last_3",
    "opponent_xg_last_5",

    "opponent_goals_conceded_last_3",
    "opponent_goals_conceded_last_5",

    "opponent_xgc_last_3",
    "opponent_xgc_last_5",

    "attack_strength",
    "defensive_strength",

    "opponent_attack_strength",
    "opponent_defensive_strength",

    "target_points",
]


# ============================================================
# CHECK FINAL COLUMNS
# ============================================================

missing_training_columns = [
    column
    for column in training_columns
    if column not in df.columns
]


if missing_training_columns:

    raise RuntimeError(
        "\nMissing final training columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_training_columns
        )
    )


df = df[
    training_columns
].copy()


# ============================================================
# FINAL SORT
# ============================================================

df = df.sort_values(
    [
        "GW",
        "element",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FINAL DATASET SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("V3 TRAINING DATASET CREATED")
print("=" * 80)

print(
    f"\nRows:       {len(df):,}"
)

print(
    f"Columns:    {len(df.columns)}"
)

print(
    f"Players:    {df['element'].nunique():,}"
)

print(
    f"GW range:   {df['GW'].min()} → {df['GW'].max()}"
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# TARGET
# ============================================================

print("\n" + "=" * 80)
print("TARGET")
print("=" * 80)

print(
    "target_points = player's TOTAL FPL points "
    "in the actual NEXT Gameweek (GW+1)"
)


print("\nTarget statistics:")

print(
    df["target_points"]
    .describe()
    .round(3)
    .to_string()
)


# ============================================================
# LEAKAGE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("LEAKAGE VALIDATION")
print("=" * 80)


minimum_gw = df["GW"].min()


gw1 = df[
    df["GW"] == minimum_gw
].copy()


leakage_columns = [
    "previous_points",

    "points_last_3",
    "points_last_5",

    "points_per_90_last_3",
    "points_per_90_last_5",

    "minutes_last_3",
    "minutes_last_5",

    "starts_last_3",
    "starts_last_5",

    "start_rate_last_3",
    "start_rate_last_5",

    "goals_last_3",
    "goals_last_5",

    "assists_last_3",
    "assists_last_5",

    "xg_last_3",
    "xg_last_5",

    "xa_last_3",
    "xa_last_5",

    "xgi_last_3",
    "xgi_last_5",

    "clean_sheets_last_3",
    "clean_sheets_last_5",

    "saves_last_3",
    "saves_last_5",

    "goals_conceded_last_3",
    "goals_conceded_last_5",

    "xgc_last_3",
    "xgc_last_5",

    "bonus_last_3",
    "bonus_last_5",

    "bps_last_3",
    "bps_last_5",

    "team_goals_last_3",
    "team_goals_last_5",

    "team_xg_last_3",
    "team_xg_last_5",

    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",

    "team_xgc_last_3",
    "team_xgc_last_5",

    "opponent_goals_last_3",
    "opponent_goals_last_5",

    "opponent_xg_last_3",
    "opponent_xg_last_5",

    "opponent_goals_conceded_last_3",
    "opponent_goals_conceded_last_5",

    "opponent_xgc_last_3",
    "opponent_xgc_last_5",
]


leakage_found = False


for column in leakage_columns:

    if column not in gw1.columns:
        continue

    maximum = (
        pd.to_numeric(
            gw1[column],
            errors="coerce",
        )
        .abs()
        .max()
    )

    if pd.notna(maximum) and maximum > 0:

        print(
            f"WARNING: {column} "
            f"has GW1 maximum = {maximum}"
        )

        leakage_found = True


if leakage_found:

    raise RuntimeError(
        "\nLEAKAGE VALIDATION FAILED."
    )


print(
    "PASS: GW1 contains no previous "
    "player/team performance information."
)


# ============================================================
# TEAM FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("TEAM FEATURE VALIDATION")
print("=" * 80)


team_validation_columns = [
    "team_xg_last_3",
    "team_xg_last_5",

    "team_goals_last_3",
    "team_goals_last_5",

    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",

    "team_xgc_last_3",
    "team_xgc_last_5",

    "opponent_xg_last_3",
    "opponent_xg_last_5",

    "opponent_goals_last_3",
    "opponent_goals_last_5",

    "opponent_goals_conceded_last_3",
    "opponent_goals_conceded_last_5",

    "opponent_xgc_last_3",
    "opponent_xgc_last_5",

    "attack_strength",
    "defensive_strength",

    "opponent_attack_strength",
    "opponent_defensive_strength",
]


team_validation_failed = False


for column in team_validation_columns:

    nulls = int(
        df[column].isna().sum()
    )

    infs = int(
        np.isinf(
            df[column].astype(float)
        ).sum()
    )

    print(
        f"{column:<35}"
        f"nulls={nulls:<6}"
        f"infs={infs}"
    )

    if nulls > 0 or infs > 0:

        team_validation_failed = True


if team_validation_failed:

    raise RuntimeError(
        "\nERROR: Team features contain NaN "
        "or infinite values."
    )


print(
    "\nPASS: Team features contain no NaN "
    "or infinite values."
)


# ============================================================
# POSITION VALIDATION
# ============================================================

print("\nPosition distribution:")

print(
    df["position"]
    .value_counts()
    .to_string()
)


if "AM" in set(
    df["position"]
    .astype(str)
):

    raise RuntimeError(
        "ERROR: Manager rows (AM) remain."
    )


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

duplicate_count = int(
    df.duplicated(
        [
            "element",
            "GW",
        ]
    ).sum()
)


print(
    "\nDuplicate player/GW rows:"
)

print(
    duplicate_count
)


if duplicate_count != 0:

    raise RuntimeError(
        "ERROR: Duplicate player/GW rows remain."
    )


# ============================================================
# FINAL SANITY CHECKS
# ============================================================

if len(df) == 0:

    raise RuntimeError(
        "ERROR: Final V3 dataset contains zero rows."
    )


if df["target_points"].isna().any():

    raise RuntimeError(
        "ERROR: Missing target_points remain."
    )


if df["element"].nunique() == 0:

    raise RuntimeError(
        "ERROR: No players remain."
    )


# ============================================================
# TARGET RANGE VALIDATION
# ============================================================

if not np.isfinite(
    df["target_points"].astype(float)
).all():

    raise RuntimeError(
        "ERROR: target_points contains invalid values."
    )


# ============================================================
# FINAL SHAPE VALIDATION
# ============================================================

print(
    f"\nFinal dataset row integrity: "
    f"{len(df):,} rows"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("V3 BUILD COMPLETE")
print("=" * 80)

print(
    "\nV3 team features use historical information only "
    "and exclude the current fixture."
)

print(
    "\nTeam/opponent matching:"
)

print(
    "  Team features: fixture + team"
)

print(
    "  Opponent history: fixture + opponent_team_id"
)

print(
    "\nTarget:"
)

print(
    "  Actual GW+1 total_points"
)

print(
    "\nNext step:"
)

print(
    "python scripts/train_model_v3.py"
)

print(
    "\nDo NOT compare V3 with V2 until V3 passes "
    "all validation checks above."
)

print("\n" + "=" * 80)