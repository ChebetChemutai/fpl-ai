import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_FILE = (
    BASE_DIR
    / "data"
    / "historical"
    / "training_dataset_v2.csv"
)


MODEL_DIR = (
    BASE_DIR
    / "models"
)


MODEL_FILE = (
    MODEL_DIR
    / "fpl_v2_model.joblib"
)


TARGET = "target_points"


# ============================================================
# V2 FEATURES
#
# These must match the dataset created by:
#
# scripts/build_training_dataset_v2.py
# ============================================================

FEATURES = [
    "GW",

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
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FPL AI — TRAIN V2 MODEL")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nTraining dataset not found:\n{DATA_FILE}\n"
        "\nRun first:\n"
        "python scripts/build_training_dataset_v2.py"
    )


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print(f"\nDataset:")
print(DATA_FILE)


df = pd.read_csv(
    DATA_FILE
)


print(
    f"\nRows loaded: {len(df):,}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURES
    + [
        TARGET,
        "element",
        "position",
    ]
)


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
# VALIDATE TARGET
# ============================================================

if df[TARGET].isna().any():

    raise ValueError(
        f"\nTarget column '{TARGET}' "
        "contains missing values."
    )


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_values = (
    df[FEATURES]
    .isna()
    .sum()
)


missing_values = (
    missing_values[
        missing_values > 0
    ]
)


if len(missing_values) > 0:

    print(
        "\nMissing feature values:"
    )

    print(
        missing_values
        .to_string()
    )

    raise ValueError(
        "\nV2 dataset contains missing feature values."
    )


# ============================================================
# POSITION ENCODING
#
# Random Forest cannot directly consume:
#
# GK
# DEF
# MID
# FWD
#
# We therefore one-hot encode position.
# ============================================================

df = pd.get_dummies(
    df,
    columns=[
        "position",
    ],
    dtype=int,
)


POSITION_COLUMNS = sorted(
    [
        column
        for column in df.columns
        if column.startswith(
            "position_"
        )
    ]
)


FEATURES = [
    feature
    for feature in FEATURES
    if feature != "position"
]


FEATURES += POSITION_COLUMNS


# ============================================================
# VALIDATE FINAL FEATURE MATRIX
# ============================================================

X_all = df[
    FEATURES
]


if not all(
    pd.api.types.is_numeric_dtype(
        X_all[column]
    )
    for column in FEATURES
):

    non_numeric = [
        column
        for column in FEATURES
        if not pd.api.types.is_numeric_dtype(
            X_all[column]
        )
    ]

    raise TypeError(
        "\nNon-numeric features detected:\n"
        + "\n".join(
            f"  - {column}"
            for column in non_numeric
        )
    )


# ============================================================
# TIME-BASED TRAIN / VALIDATION SPLIT
#
# Training:
#
# GW 1 → GW 30
#
# Validation:
#
# GW 31 → GW 37
#
# This simulates:
#
# "Use historical information to predict
# the future."
#
# We deliberately DO NOT randomly split.
# ============================================================

train = df[
    df["GW"] <= 30
].copy()


validation = df[
    df["GW"] >= 31
].copy()


if train.empty:

    raise ValueError(
        "Training dataset is empty."
    )


if validation.empty:

    raise ValueError(
        "Validation dataset is empty."
    )


# ============================================================
# TRAINING DATA
# ============================================================

X_train = train[
    FEATURES
]


y_train = train[
    TARGET
]


# ============================================================
# VALIDATION DATA
# ============================================================

X_validation = validation[
    FEATURES
]


y_validation = validation[
    TARGET
]


# ============================================================
# INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)


print(
    f"\nTraining rows:"
    f"      {len(train):,}"
)


print(
    f"Training GW:"
    f"       {train.GW.min()} → {train.GW.max()}"
)


print(
    f"\nValidation rows:"
    f"    {len(validation):,}"
)


print(
    f"Validation GW:"
    f"     {validation.GW.min()} → {validation.GW.max()}"
)


print(
    f"\nFeatures:"
    f"             {len(FEATURES)}"
)


print(
    f"Players:"
    f"              {df.element.nunique()}"
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(

    n_estimators=500,

    max_depth=14,

    min_samples_leaf=5,

    max_features="sqrt",

    random_state=42,

    n_jobs=-1,
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING V2 RANDOM FOREST")
print("=" * 70)


print(
    "\nTraining model..."
)


model.fit(
    X_train,
    y_train,
)


print(
    "Training complete."
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_validation
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_validation,
    predictions,
)


rmse = np.sqrt(
    mean_squared_error(
        y_validation,
        predictions,
    )
)


# ============================================================
# RANK CORRELATION
#
# FPL prediction is fundamentally a ranking problem.
#
# We calculate Spearman correlation between:
#
# predicted points
#
# and
#
# actual points.
# ============================================================

validation_results = validation[
    [
        "element",
        "GW",
        TARGET,
    ]
].copy()


validation_results[
    "predicted_points"
] = predictions


rank_correlation = (
    validation_results[
        [
            TARGET,
            "predicted_points",
        ]
    ]
    .corr(
        method="spearman"
    )
    .iloc[0, 1]
)


# ============================================================
# TOP-K HIT RATE
#
# For each validation GW:
#
# 1. Rank players by prediction.
# 2. Select top K.
# 3. Check how many actual top-K players
#    were successfully predicted.
#
# This is more useful for FPL than MAE alone.
# ============================================================

def top_k_hit_rate(
    results,
    k,
):
    """
    Calculate average precision@K across
    validation gameweeks.
    """

    scores = []

    for gw, group in results.groupby(
        "GW"
    ):

        if len(group) < k:

            continue

        predicted_top = set(
            group
            .nlargest(
                k,
                "predicted_points",
            )
            .element
        )

        actual_top = set(
            group
            .nlargest(
                k,
                TARGET,
            )
            .element
        )

        hits = len(
            predicted_top
            & actual_top
        )

        scores.append(
            hits / k
        )

    if not scores:

        return np.nan

    return float(
        np.mean(scores)
    )


top10 = top_k_hit_rate(
    validation_results,
    10,
)


top20 = top_k_hit_rate(
    validation_results,
    20,
)


top50 = top_k_hit_rate(
    validation_results,
    50,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        model.feature_importances_,
})


importance = importance.sort_values(
    "importance",
    ascending=False,
)


# ============================================================
# MODEL PACKAGE
# ============================================================

model_package = {

    "model":
        model,

    "features":
        FEATURES,

    "version":
        "V2",

    "target":
        TARGET,

    "train_gw":
        [1, 30],

    "validation_gw":
        [31, 37],

    "metrics":
        {
            "mae":
                float(mae),

            "rmse":
                float(rmse),

            "spearman":
                float(rank_correlation),

            "top10_hit_rate":
                float(top10),

            "top20_hit_rate":
                float(top20),

            "top50_hit_rate":
                float(top50),
        },
}


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model_package,
    MODEL_FILE,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("V2 MODEL PERFORMANCE")
print("=" * 70)


print(
    f"\nMAE:"
    f"              {mae:.4f}"
)


print(
    f"RMSE:"
    f"             {rmse:.4f}"
)


print(
    f"Spearman:"
    f"         {rank_correlation:.4f}"
)


print(
    f"Top-10 hit rate:"
    f"  {top10:.4f}"
)


print(
    f"Top-20 hit rate:"
    f"  {top20:.4f}"
)


print(
    f"Top-50 hit rate:"
    f"  {top50:.4f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 V2 FEATURES")
print("=" * 70)


print(
    importance
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
#
# This will allow us to compare V1 and V2 directly.
# ============================================================

PREDICTION_FILE = (
    BASE_DIR
    / "data"
    / "historical"
    / "v2_validation_predictions.csv"
)


validation_results.to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("V2 MODEL SAVED")
print("=" * 70)


print(
    f"\nModel:"
    f"\n{MODEL_FILE}"
)


print(
    f"\nValidation predictions:"
    f"\n{PREDICTION_FILE}"
)


print(
    f"\nFeatures saved:"
    f" {len(FEATURES)}"
)


print(
    "\n" + "=" * 70
)

print(
    "V2 TRAINING COMPLETE"
)

print(
    "=" * 70
)