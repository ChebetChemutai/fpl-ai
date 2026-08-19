import os
import sys
from pathlib import Path

import django
import joblib
import numpy as np
import pandas as pd


# ============================================================
# DJANGO SETUP
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.insert(
    0,
    str(BASE_DIR),
)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

django.setup()


# ============================================================
# IMPORTS
# ============================================================

from fpl.features import (
    get_gw_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

GAMEWEEK = 1

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "v3"
)


MODEL_FILES = {
    "GK":
        MODEL_DIR
        / "model_v3_gk.joblib",

    "DEF":
        MODEL_DIR
        / "model_v3_def.joblib",

    "MID":
        MODEL_DIR
        / "model_v3_mid.joblib",

    "FWD":
        MODEL_DIR
        / "model_v3_fwd.joblib",
}


OUTPUT_FILE = (
    BASE_DIR
    / "predictions_gw1_v3.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("FPL AI — V3 GAMEWEEK PREDICTIONS")
print("=" * 80)

print(
    f"\nGameweek: {GAMEWEEK}"
)

print(
    f"Model directory: {MODEL_DIR}"
)


# ============================================================
# VERIFY MODELS
# ============================================================

for position, model_file in MODEL_FILES.items():

    if not model_file.exists():

        raise FileNotFoundError(
            f"\nMissing V3 {position} model:\n"
            f"{model_file}"
        )


# ============================================================
# LOAD CURRENT FEATURES
# ============================================================

print("\n" + "=" * 80)
print("BUILDING V3 LIVE FEATURES")
print("=" * 80)


features = get_gw_features(
    GAMEWEEK
)


if not features:

    raise RuntimeError(
        "\nNo players found for this gameweek.\n"
        "Check that current FPL data and fixtures "
        "have been loaded into Django."
    )


print(
    f"\nPlayers loaded: {len(features):,}"
)


df = pd.DataFrame(
    features
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_display_columns = [
    "player_id",
    "web_name",
    "position",
    "price",
    "availability_probability",
]


missing_display_columns = [
    column
    for column in required_display_columns
    if column not in df.columns
]


if missing_display_columns:

    raise RuntimeError(
        "\nMissing display columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_display_columns
        )
    )


# ============================================================
# POSITION VALIDATION
# ============================================================

valid_positions = {
    "GK",
    "DEF",
    "MID",
    "FWD",
}


unexpected_positions = (
    set(
        df["position"]
        .astype(str)
        .str.upper()
    )
    - valid_positions
)


if unexpected_positions:

    raise RuntimeError(
        "\nUnexpected positions found:\n"
        + "\n".join(
            sorted(
                unexpected_positions
            )
        )
    )


print("\nPosition distribution:")

print(
    df["position"]
    .value_counts()
    .to_string()
)


# ============================================================
# LOAD MODELS
# ============================================================

models = {}

feature_schemas = {}


print("\n" + "=" * 80)
print("LOADING V3 MODELS")
print("=" * 80)


for position, model_file in MODEL_FILES.items():

    print(
        f"\n{position}: {model_file}"
    )

    model = joblib.load(
        model_file
    )

    models[position] = model

    feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if feature_names is None:

        raise RuntimeError(
            f"\nV3 {position} model does not "
            "contain feature_names_in_."
        )

    feature_schemas[position] = list(
        feature_names
    )

    print(
        f"Features expected: "
        f"{len(feature_names)}"
    )

    print(
        "Feature schema validated."
    )


# ============================================================
# VERIFY ALL V3 MODELS USE SAME SCHEMA
# ============================================================

schema_reference = (
    feature_schemas["GK"]
)


schema_mismatch = False


for position, schema in (
    feature_schemas.items()
):

    if schema != schema_reference:

        print(
            f"\nWARNING: {position} feature "
            "schema differs from GK."
        )

        schema_mismatch = True


if schema_mismatch:

    raise RuntimeError(
        "\nERROR: V3 position models do not "
        "use the same feature schema."
    )


MODEL_FEATURES = schema_reference


print(
    f"\nFinal V3 feature count: "
    f"{len(MODEL_FEATURES)}"
)


# ============================================================
# BUILD MODEL MATRIX
# ============================================================

missing_features = [
    feature
    for feature in MODEL_FEATURES
    if feature not in df.columns
]


if missing_features:

    raise RuntimeError(
        "\nV3 feature generation is incomplete.\n"
        "\nMissing model features:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_features
        )
    )


X = df[
    MODEL_FEATURES
].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in MODEL_FEATURES:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce",
    )


# ============================================================
# INVALID VALUE CHECK
# ============================================================

X = X.replace(
    [
        np.inf,
        -np.inf,
    ],
    np.nan,
)


remaining_nulls = (
    X.isna()
    .sum()
)


problem_features = (
    remaining_nulls[
        remaining_nulls > 0
    ]
)


if len(problem_features) > 0:

    print(
        "\nMissing V3 model inputs:"
    )

    print(
        problem_features
        .to_string()
    )

    raise RuntimeError(
        "\nERROR: Live V3 features contain "
        "missing values."
    )


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING V3 PREDICTIONS")
print("=" * 80)


df["predicted_points"] = np.nan


for position in [
    "GK",
    "DEF",
    "MID",
    "FWD",
]:

    position_mask = (
        df["position"]
        .astype(str)
        .str.upper()
        .eq(position)
    )

    count = int(
        position_mask.sum()
    )

    if count == 0:

        print(
            f"\n{position}: no players"
        )

        continue

    position_model = (
        models[position]
    )

    position_X = X.loc[
        position_mask
    ]

    predictions = (
        position_model.predict(
            position_X
        )
    )

    df.loc[
        position_mask,
        "predicted_points",
    ] = predictions

    print(
        f"\n{position}: "
        f"{count:,} players predicted"
    )


# ============================================================
# PREDICTION VALIDATION
# ============================================================

if df["predicted_points"].isna().any():

    missing_prediction_count = int(
        df["predicted_points"]
        .isna()
        .sum()
    )

    raise RuntimeError(
        "\nERROR: "
        f"{missing_prediction_count:,} "
        "players have no prediction."
    )


if not np.isfinite(
    df["predicted_points"]
    .astype(float)
).all():

    raise RuntimeError(
        "\nERROR: Predictions contain "
        "NaN or infinite values."
    )


# ============================================================
# AVAILABILITY ADJUSTMENT
# ============================================================

df[
    "availability_adjusted_points"
] = (
    df["predicted_points"]
    *
    df[
        "availability_probability"
    ]
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "availability_adjusted_points",
        "predicted_points",
    ],
    ascending=False,
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "player_id",
    "web_name",
    "team_id",
    "position",
    "price",

    "next_home",
    "next_opponent",
    "next_fixture_count",

    "predicted_points",
    "availability_probability",
    "availability_adjusted_points",

    "fixture_difficulty",
    "fixture_id",
    "gameweek_id",
]


available_output_columns = [
    column
    for column in output_columns
    if column in df.columns
]


output = df[
    available_output_columns
].copy()


output.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("V3 PREDICTIONS COMPLETE")
print("=" * 80)

print(
    f"\nPlayers predicted: "
    f"{len(output):,}"
)

print(
    f"\nOutput:"
)

print(
    OUTPUT_FILE
)


print(
    "\nTop 20 V3 predictions:"
)

print(
    output[
        [
            "web_name",
            "position",
            "price",
            "predicted_points",
            "availability_adjusted_points",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 80)