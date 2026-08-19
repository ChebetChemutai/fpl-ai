import pandas as pd
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
    / "training_dataset.csv"
)


# ============================================================
# CONFIG
# ============================================================

ROLLING_WINDOWS = [3, 5]


# ============================================================
# HELPERS
# ============================================================

def rolling_mean_previous(group, column, window):
    """
    Calculate the mean of the player's previous N gameweeks.

    IMPORTANT:
    shift(1) ensures the current gameweek is never included.
    """

    return (
        group[column]
        .transform(
            lambda s: s.shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
    )


def rolling_sum_previous(group, column, window):
    """
    Calculate the sum of the player's previous N gameweeks.

    IMPORTANT:
    shift(1) ensures the current gameweek is never included.
    """

    return (
        group[column]
        .transform(
            lambda s: s.shift(1)
            .rolling(window=window, min_periods=1)
            .sum()
        )
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FPL AI — BUILD HISTORICAL TRAINING DATASET")
print("=" * 70)

print(f"\nInput: {INPUT_FILE}")


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Historical file not found: {INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


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
    "total_points",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "element",
    "GW",
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "value",
    "opponent_team",
    "starts",
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "element",
        "GW",
        "total_points",
    ]
).copy()


print(
    f"Rows after removing invalid records: "
    f"{len(df):,}"
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "element",
        "GW",
    ]
).reset_index(drop=True)


# ============================================================
# DETECT DUPLICATE PLAYER/GAMEWEEK RECORDS
# ============================================================

duplicate_mask = df.duplicated(
    subset=[
        "element",
        "GW",
    ],
    keep=False,
)

duplicate_rows = int(duplicate_mask.sum())


print(
    f"Duplicate player/GW rows detected: "
    f"{duplicate_rows:,}"
)


# ============================================================
# REMOVE DUPLICATES
#
# FPL data can contain duplicate rows around postponed/
# rescheduled fixtures.
#
# We keep one record per player/gameweek.
#
# If duplicates have different minutes/points, we aggregate
# the performance statistics instead of simply dropping data.
# ============================================================

if duplicate_rows > 0:

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
        "opponent_team": "first",
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
        .agg(aggregation)
    )


print(
    f"Rows after duplicate handling: "
    f"{len(df):,}"
)


# ============================================================
# BASIC FEATURES
# ============================================================

df["home"] = (
    df["was_home"]
    .fillna(False)
    .astype(int)
)


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
# POINTS ROLLING FEATURES
# ============================================================

for window in ROLLING_WINDOWS:

    df[f"points_last_{window}"] = (
        rolling_mean_previous(
            group,
            "total_points",
            window,
        )
    )


# ============================================================
# MINUTES
# ============================================================

for window in ROLLING_WINDOWS:

    df[f"minutes_last_{window}"] = (
        rolling_mean_previous(
            group,
            "minutes",
            window,
        )
    )


# ============================================================
# GOALS
# ============================================================

for window in ROLLING_WINDOWS:

    df[f"goals_last_{window}"] = (
        rolling_sum_previous(
            group,
            "goals_scored",
            window,
        )
    )


# ============================================================
# ASSISTS
# ============================================================

for window in ROLLING_WINDOWS:

    df[f"assists_last_{window}"] = (
        rolling_sum_previous(
            group,
            "assists",
            window,
        )
    )


# ============================================================
# EXPECTED GOALS
# ============================================================

for window in ROLLING_WINDOWS:

    df[f"xg_last_{window}"] = (
        rolling_mean_previous(
            group,
            "expected_goals",
            window,
        )
    )


# ============================================================
# EXPECTED ASSISTS
# ============================================================

for window in ROLLING_WINDOWS:

    df[f"xa_last_{window}"] = (
        rolling_mean_previous(
            group,
            "expected_assists",
            window,
        )
    )


# ============================================================
# EXPECTED GOAL INVOLVEMENT
# ============================================================

for window in ROLLING_WINDOWS:

    df[f"xgi_last_{window}"] = (
        rolling_mean_previous(
            group,
            "expected_goal_involvements",
            window,
        )
    )


# ============================================================
# TARGET
#
# The target is the player's NEXT gameweek points.
#
# Therefore:
#
# GW 1 features → GW 2 points
# GW 2 features → GW 3 points
# ...
#
# We deliberately DO NOT use current GW points as features.
# ============================================================

df["target_points"] = (
    group["total_points"]
    .shift(-1)
)


# ============================================================
# REMOVE LAST GAMEWEEK
#
# GW38 has no GW39 target.
# ============================================================

df = df[
    df["target_points"].notna()
].copy()


# ============================================================
# FILL HISTORICAL FEATURES
#
# Players with no previous history receive zero-history
# features.
#
# This is intentional.
#
# Later, the live model will add cold-start features for
# new signings.
# ============================================================

feature_columns = [
    "previous_points",
    "points_last_3",
    "points_last_5",
    "minutes_last_3",
    "minutes_last_5",
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
]


for column in feature_columns:

    df[column] = (
        df[column]
        .fillna(0)
    )


# ============================================================
# FINAL DATASET
# ============================================================

training_columns = [
    "element",
    "GW",
    "position",
    "price",
    "home",
    "opponent_team",

    "previous_points",

    "points_last_3",
    "points_last_5",

    "minutes_last_3",
    "minutes_last_5",

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

    "target_points",
]


df = df[
    training_columns
].copy()


# ============================================================
# SORT FINAL DATASET
# ============================================================

df = df.sort_values(
    [
        "GW",
        "element",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DATASET CREATED")
print("=" * 70)

print(
    f"Rows:       {len(df):,}"
)

print(
    f"Columns:    {len(df.columns)}"
)

print(
    f"Output:     {OUTPUT_FILE}"
)


print("\nFeatures:")

for column in training_columns:

    if column != "target_points":
        print(f"  - {column}")


print("\nTarget:")
print(
    "  target_points = next gameweek total points"
)


print(
    "\nGameweeks represented:"
)

print(
    f"GW {df.GW.min()} → GW {df.GW.max()}"
)


# ============================================================
# LEAKAGE TEST
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE VALIDATION")
print("=" * 70)


gw1 = df[
    df["GW"] == df["GW"].min()
]


leakage_columns = [
    "previous_points",
    "points_last_3",
    "points_last_5",
    "minutes_last_3",
    "minutes_last_5",
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
]


leakage_found = False


for column in leakage_columns:

    maximum = gw1[column].abs().max()

    if maximum > 0:

        print(
            f"WARNING: {column} "
            f"has GW1 value {maximum}"
        )

        leakage_found = True


if not leakage_found:

    print(
        "PASS: GW1 contains no historical "
        "player-performance information."
    )


# ============================================================
# TARGET VALIDATION
# ============================================================

print("\nTarget statistics:")

print(
    df["target_points"]
    .describe()
    .round(3)
    .to_string()
)


print("\n" + "=" * 70)
print("BUILD COMPLETE")
print("=" * 70)