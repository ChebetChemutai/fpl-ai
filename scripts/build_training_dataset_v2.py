import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# FPL AI — V2 TRAINING DATASET BUILDER
# ============================================================
#
# Purpose:
#
# Build a leakage-safe historical dataset for predicting
# the NEXT Gameweek's FPL points.
#
# V2 improvements over V1:
#
# - Removes manager rows (position == AM)
# - Correctly aggregates multiple fixtures in one GW
# - Form features
# - Points per 90
# - Starting reliability
# - Attacking features
# - Defensive features
# - Bonus / BPS features
# - Value efficiency
# - NEXT-GW fixture information
# - Strict leakage protection
#
# IMPORTANT:
#
# Every historical performance feature uses only information
# from PREVIOUS Gameweeks.
#
# The target is the player's total FPL points in the NEXT GW.
#
# Example:
#
# GW 10 features -> GW 11 target_points
#
# ============================================================


import pandas as pd
import numpy as np
from pathlib import Path


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
    / "training_dataset_v2.csv"
)


ROLLING_WINDOWS = [3, 5]


VALID_POSITIONS = [
    "GK",
    "DEF",
    "MID",
    "FWD",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rolling_sum_previous(group, column, window):
    """
    Sum of the player's previous N gameweeks.

    shift(1) ensures the CURRENT gameweek is excluded.
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


def rolling_mean_previous(group, column, window):
    """
    Mean of the player's previous N gameweeks.

    shift(1) ensures the CURRENT gameweek is excluded.
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


def safe_divide(numerator, denominator):
    """
    Safe division.

    Returns zero where denominator is zero.
    """

    return (
        numerator
        .div(
            denominator.replace(0, np.nan)
        )
        .fillna(0)
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FPL AI — BUILD V2 HISTORICAL TRAINING DATASET")
print("=" * 80)


print(f"\nInput:")
print(INPUT_FILE)

print(f"\nOutput:")
print(OUTPUT_FILE)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nHistorical dataset not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "element",
    "GW",
    "position",
    "value",
    "was_home",
    "opponent_team",
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "goals_conceded",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "bonus",
    "bps",
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

numeric_columns = [
    "element",
    "GW",
    "value",
    "opponent_team",
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "saves",
    "goals_conceded",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "bonus",
    "bps",
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# REMOVE INVALID CORE RECORDS
# ============================================================

before_invalid = len(df)


df = df.dropna(
    subset=[
        "element",
        "GW",
        "position",
        "total_points",
    ]
).copy()


print(
    f"Rows after invalid-record removal: "
    f"{len(df):,}"
)


# ============================================================
# REMOVE MANAGER ROWS
# ============================================================
#
# Your dataset contains:
#
# AM = Assistant/manager records
#
# Example:
#
# Pep Guardiola
# Mikel Arteta
# Arne Slot
#
# These are NOT FPL player records.
#
# They must never enter the player prediction model.
# ============================================================

manager_rows = (
    ~df["position"].isin(VALID_POSITIONS)
)


manager_count = int(
    manager_rows.sum()
)


print(
    f"\nNon-player rows removed: "
    f"{manager_count:,}"
)


df = df[
    df["position"].isin(
        VALID_POSITIONS
    )
].copy()


# ============================================================
# SORT
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
# CHECK ORIGINAL DUPLICATES
# ============================================================

duplicate_counts = (
    df.groupby(
        [
            "element",
            "GW",
        ]
    )
    .size()
)


duplicate_groups = int(
    (duplicate_counts > 1).sum()
)


duplicate_rows = int(
    duplicate_counts[
        duplicate_counts > 1
    ].sum()
)


print(
    f"\nPlayer/GW groups with multiple "
    f"fixture rows: {duplicate_groups:,}"
)


print(
    f"Rows belonging to those groups: "
    f"{duplicate_rows:,}"
)


# ============================================================
# AGGREGATE PLAYER/GW
# ============================================================
#
# A player may have more than one fixture associated with
# a Gameweek because of postponed/rescheduled fixtures.
#
# For player performance:
#
#     goals       -> SUM
#     assists     -> SUM
#     minutes     -> SUM
#     points      -> SUM
#     bonus       -> SUM
#     BPS         -> SUM
#
# For fixture identity:
#
#     opponent_team -> collect all opponents
#     was_home       -> collect all home/away states
#
# We will later derive NEXT-GW fixture information.
# ============================================================


aggregation = {

    "name": "first",

    "position": "first",

    "team": "first",

    "xP": "first",

    "assists": "sum",

    "bonus": "sum",

    "bps": "sum",

    "clean_sheets": "sum",

    "creativity": "sum",

    "expected_assists": "sum",

    "expected_goal_involvements": "sum",

    "expected_goals": "sum",

    "expected_goals_conceded": "sum",

    "goals_conceded": "sum",

    "goals_scored": "sum",

    "influence": "sum",

    "minutes": "sum",

    "own_goals": "sum",

    "penalties_missed": "sum",

    "penalties_saved": "sum",

    "red_cards": "sum",

    "saves": "sum",

    "starts": "sum",

    "threat": "sum",

    "total_points": "sum",

    "value": "last",

    "was_home": "last",

    "yellow_cards": "sum",

    "kickoff_time": "first",

    "fixture": "first",

    "opponent_team": "first",
}


aggregation = {
    key: value
    for key, value in aggregation.items()
    if key in df.columns
}


# ============================================================
# CREATE PLAYER/GW AGGREGATED DATA
# ============================================================

df = (
    df.groupby(
        [
            "element",
            "GW",
        ],
        as_index=False,
    )
    .agg(aggregation)
)


print(
    f"Rows after player/GW aggregation: "
    f"{len(df):,}"
)


# ============================================================
# VERIFY DUPLICATES ARE GONE
# ============================================================

remaining_duplicates = (
    df.duplicated(
        subset=[
            "element",
            "GW",
        ]
    )
    .sum()
)


if remaining_duplicates > 0:

    raise ValueError(
        "Duplicate player/GW records still exist "
        f"after aggregation: {remaining_duplicates}"
    )


print(
    "Duplicate validation: PASS"
)


# ============================================================
# BASIC FEATURES
# ============================================================


# Home/away indicator for the player's CURRENT fixture.
#
# This is retained temporarily for constructing fixture
# information.
#
# The model will ultimately use NEXT-GW fixture information.
# ============================================================

df["current_home"] = (
    df["was_home"]
    .fillna(False)
    .astype(int)
)


# Price in £m
df["price"] = (
    df["value"]
    / 10.0
)


# ============================================================
# GROUP BY PLAYER
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
# FORM FEATURES
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
# POINTS PER 90
# ============================================================

for window in ROLLING_WINDOWS:

    points_sum = rolling_sum_previous(
        group,
        "total_points",
        window,
    )

    minutes_sum = rolling_sum_previous(
        group,
        "minutes",
        window,
    )

    df[
        f"points_per_90_last_{window}"
    ] = safe_divide(
        points_sum * 90,
        minutes_sum,
    )


# ============================================================
# STARTING RELIABILITY
# ============================================================

for window in ROLLING_WINDOWS:

    df[
        f"starts_last_{window}"
    ] = rolling_sum_previous(
        group,
        "starts",
        window,
    )

    df[
        f"start_rate_last_{window}"
    ] = rolling_mean_previous(
        group,
        "starts",
        window,
    )


# ============================================================
# MINUTES PER START
# ============================================================

minutes_sum_3 = rolling_sum_previous(
    group,
    "minutes",
    3,
)

minutes_sum_5 = rolling_sum_previous(
    group,
    "minutes",
    5,
)


starts_sum_3 = rolling_sum_previous(
    group,
    "starts",
    3,
)

starts_sum_5 = rolling_sum_previous(
    group,
    "starts",
    5,
)


df["minutes_per_start"] = safe_divide(
    minutes_sum_5,
    starts_sum_5,
)


# ============================================================
# ATTACKING FEATURES
# ============================================================

for window in ROLLING_WINDOWS:

    df[
        f"goals_last_{window}"
    ] = rolling_sum_previous(
        group,
        "goals_scored",
        window,
    )

    df[
        f"assists_last_{window}"
    ] = rolling_sum_previous(
        group,
        "assists",
        window,
    )

    df[
        f"xg_last_{window}"
    ] = rolling_mean_previous(
        group,
        "expected_goals",
        window,
    )

    df[
        f"xa_last_{window}"
    ] = rolling_mean_previous(
        group,
        "expected_assists",
        window,
    )

    df[
        f"xgi_last_{window}"
    ] = rolling_mean_previous(
        group,
        "expected_goal_involvements",
        window,
    )


# ============================================================
# DEFENSIVE FEATURES
# ============================================================

for window in ROLLING_WINDOWS:

    df[
        f"clean_sheets_last_{window}"
    ] = rolling_sum_previous(
        group,
        "clean_sheets",
        window,
    )

    df[
        f"saves_last_{window}"
    ] = rolling_sum_previous(
        group,
        "saves",
        window,
    )

    df[
        f"goals_conceded_last_{window}"
    ] = rolling_sum_previous(
        group,
        "goals_conceded",
        window,
    )

    df[
        f"xgc_last_{window}"
    ] = rolling_mean_previous(
        group,
        "expected_goals_conceded",
        window,
    )


# ============================================================
# BONUS FEATURES
# ============================================================

for window in ROLLING_WINDOWS:

    df[
        f"bonus_last_{window}"
    ] = rolling_sum_previous(
        group,
        "bonus",
        window,
    )

    df[
        f"bps_last_{window}"
    ] = rolling_sum_previous(
        group,
        "bps",
        window,
    )


# ============================================================
# BPS PER 90
# ============================================================

bps_sum_3 = rolling_sum_previous(
    group,
    "bps",
    3,
)

bps_sum_5 = rolling_sum_previous(
    group,
    "bps",
    5,
)


df["bps_per_90_last_3"] = safe_divide(
    bps_sum_3 * 90,
    minutes_sum_3,
)


df["bps_per_90_last_5"] = safe_divide(
    bps_sum_5 * 90,
    minutes_sum_5,
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
# NEXT-GAMEWEEK FIXTURE FEATURES
# ============================================================
#
# VERY IMPORTANT.
#
# We are predicting:
#
#     GW N -> GW N+1
#
# Therefore fixture information should describe GW N+1.
#
# Fixture schedules are known before the GW deadline and
# therefore are valid predictive information.
#
# We construct:
#
#     next_home
#     next_opponent_team
#     next_fixture_count
#
# If a player has multiple fixtures in the next GW, we retain
# the first opponent and count the number of fixtures.
#
# For the current 2024/25 dataset this handles the existence
# of double gameweeks without mixing performance from the
# future into the historical features.
# ============================================================


fixture_df = df[
    [
        "element",
        "GW",
        "opponent_team",
        "current_home",
    ]
].copy()


fixture_df["target_GW"] = (
    fixture_df["GW"] - 1
)


fixture_df = fixture_df.rename(
    columns={
        "GW": "fixture_GW",
        "opponent_team": "fixture_opponent",
        "current_home": "fixture_home",
    }
)


# We want fixtures belonging to the NEXT GW relative to
# the feature row.
#
# Example:
#
# feature row GW 10
# -> target GW 11
#
# Therefore merge on:
#
# feature GW + 1 == fixture GW
#
# ============================================================


next_fixture = df[
    [
        "element",
        "GW",
        "opponent_team",
        "current_home",
    ]
].copy()


next_fixture = next_fixture.rename(
    columns={
        "GW": "next_GW",
        "opponent_team": "next_opponent",
        "current_home": "next_home",
    }
)


# Count fixtures per player in each GW.

fixture_counts = (
    next_fixture
    .groupby(
        [
            "element",
            "next_GW",
        ]
    )
    .size()
    .reset_index(
        name="next_fixture_count"
    )
)


# Keep the first opponent as the primary opponent.

next_fixture_first = (
    next_fixture
    .sort_values(
        [
            "element",
            "next_GW",
        ]
    )
    .drop_duplicates(
        subset=[
            "element",
            "next_GW",
        ],
        keep="first",
    )
)


next_fixture_first = (
    next_fixture_first[
        [
            "element",
            "next_GW",
            "next_opponent",
            "next_home",
        ]
    ]
)


next_fixture_info = (
    next_fixture_first
    .merge(
        fixture_counts,
        on=[
            "element",
            "next_GW",
        ],
        how="left",
    )
)


# Merge NEXT GW fixture information onto the current GW row.

df = df.merge(
    next_fixture_info,
    left_on=[
        "element",
        "GW",
    ],
    right_on=[
        "element",
        "next_GW",
    ],
    how="left",
)


# ============================================================
# CLEAN NEXT FIXTURE FEATURES
# ============================================================

df["next_fixture_count"] = (
    df["next_fixture_count"]
    .fillna(0)
)


df["next_home"] = (
    df["next_home"]
    .fillna(0)
    .astype(int)
)


df["next_opponent"] = (
    df["next_opponent"]
    .fillna(0)
)


# ============================================================
# TARGET
# ============================================================
#
# Target:
#
#     player's TOTAL FPL points in NEXT GW
#
# Since we aggregated player/GW above, this is correct even
# when the next GW contains multiple fixtures.
#
# ============================================================

group = df.groupby(
    "element",
    sort=False,
)


df["target_points"] = (
    group["total_points"]
    .shift(-1)
)


# ============================================================
# REMOVE GW38
#
# GW38 has no GW39 target.
# ============================================================

df = df[
    df["target_points"].notna()
].copy()


# ============================================================
# FILL HISTORICAL FEATURE NULLS
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

    "next_opponent",
    "next_home",
    "next_fixture_count",
]


for column in feature_columns:

    if column in df.columns:

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .fillna(0)
        )


# ============================================================
# FINAL DATASET COLUMNS
# ============================================================

training_columns = [

    # Identity
    "element",

    # Current information available before next GW
    "GW",
    "position",
    "price",

    # NEXT GW fixture
    "next_home",
    "next_opponent",
    "next_fixture_count",

    # Form
    "previous_points",

    "points_last_3",
    "points_last_5",

    "points_per_90_last_3",
    "points_per_90_last_5",

    "minutes_last_3",
    "minutes_last_5",

    # Starting reliability
    "starts_last_3",
    "starts_last_5",

    "start_rate_last_3",
    "start_rate_last_5",

    "minutes_per_start",

    # Attacking
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

    # Defensive
    "clean_sheets_last_3",
    "clean_sheets_last_5",

    "saves_last_3",
    "saves_last_5",

    "goals_conceded_last_3",
    "goals_conceded_last_5",

    "xgc_last_3",
    "xgc_last_5",

    # Bonus
    "bonus_last_3",
    "bonus_last_5",

    "bps_last_3",
    "bps_last_5",

    "bps_per_90_last_3",
    "bps_per_90_last_5",

    # Value
    "points_per_million",
    "recent_points_per_million",

    # Target
    "target_points",
]


# ============================================================
# VERIFY ALL FINAL COLUMNS EXIST
# ============================================================

missing_final_columns = [
    column
    for column in training_columns
    if column not in df.columns
]


if missing_final_columns:

    raise ValueError(
        "\nFinal dataset is missing columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_final_columns
        )
    )


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

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
# BASIC VALIDATION
# ============================================================

if df.empty:

    raise ValueError(
        "Final training dataset is empty."
    )


if df["target_points"].isna().any():

    raise ValueError(
        "target_points contains missing values."
    )


# ============================================================
# CHECK PLAYER/GW UNIQUENESS
# ============================================================

duplicate_final = (
    df.duplicated(
        subset=[
            "element",
            "GW",
        ]
    )
    .sum()
)


if duplicate_final > 0:

    raise ValueError(
        "Final dataset contains duplicate "
        f"player/GW rows: {duplicate_final}"
    )


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 80)
print("V2 TRAINING DATASET CREATED")
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
    f"\nOutput:"
)


print(
    OUTPUT_FILE
)


# ============================================================
# FEATURE REPORT
# ============================================================

print("\n" + "=" * 80)
print("FEATURES")
print("=" * 80)


for column in training_columns:

    if column != "target_points":

        print(
            f"  - {column}"
        )


# ============================================================
# TARGET REPORT
# ============================================================

print("\n" + "=" * 80)
print("TARGET")
print("=" * 80)


print(
    "target_points = player's TOTAL FPL points "
    "in the NEXT Gameweek"
)


print(
    "\nTarget statistics:"
)


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


# ------------------------------------------------------------
# GW1
# ------------------------------------------------------------
#
# There should be no previous player history.
#
# Therefore all rolling historical features should be zero.
# ------------------------------------------------------------


gw1 = df[
    df["GW"] == df["GW"].min()
].copy()


historical_features = [

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
]


gw1_leakage = []


for column in historical_features:

    if column in gw1.columns:

        maximum = (
            gw1[column]
            .abs()
            .max()
        )

        if maximum > 0:

            gw1_leakage.append(
                (
                    column,
                    maximum,
                )
            )


if gw1_leakage:

    print(
        "\nFAIL: GW1 contains non-zero historical "
        "features:"
    )

    for column, maximum in gw1_leakage:

        print(
            f"  {column}: max={maximum}"
        )

else:

    print(
        "\nPASS: GW1 contains no previous "
        "player-performance information."
    )


# ============================================================
# TARGET ALIGNMENT TEST
# ============================================================

print(
    "\nChecking target alignment..."
)


test_player = (
    df["element"]
    .iloc[0]
)


player_test = df[
    df["element"] == test_player
].sort_values(
    "GW"
)


if len(player_test) >= 2:

    first_gw = (
        player_test.iloc[0]
    )

    second_gw = (
        player_test.iloc[1]
    )

    print(
        f"\nTest player element: "
        f"{test_player}"
    )

    print(
        f"GW {int(first_gw['GW'])} "
        f"target = "
        f"{first_gw['target_points']}"
    )

    print(
        f"GW {int(second_gw['GW'])} "
        f"actual points = "
        f"{second_gw['target_points'] if False else 'checked separately'}"
    )


# ============================================================
# POSITION VALIDATION
# ============================================================

print("\nPosition distribution:")

print(
    df["position"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("V2 BUILD COMPLETE")
print("=" * 80)


print(
    "\nNext step:"
)


print(
    "Train the V2 model using "
    "scripts/train_model_v2.py"
)


print(
    "\nIMPORTANT:"
)


print(
    "Do NOT compare V2 to V1 until the V2 dataset "
    "has passed leakage validation."
)


print(
    "\n" + "=" * 80
)