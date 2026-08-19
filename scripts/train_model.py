import pandas as pd
from pathlib import Path

import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "historical"
    / "training_dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = (
    MODEL_DIR
    / "fpl_baseline_model.joblib"
)

TARGET = "target_points"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FPL AI — TRAIN BASELINE MODEL")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Training dataset not found:\n{DATA_FILE}"
    )

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print(f"\nDataset: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print(f"Rows: {len(df):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )


# ============================================================
# VALIDATE DATA
# ============================================================

if df[TARGET].isna().any():
    raise ValueError(
        f"Target column '{TARGET}' contains missing values."
    )

if df[FEATURES].isna().any().any():
    missing_features = (
        df[FEATURES]
        .isna()
        .sum()
    )

    missing_features = (
        missing_features[
            missing_features > 0
        ]
        .to_dict()
    )

    raise ValueError(
        f"Features contain missing values:\n"
        f"{missing_features}"
    )


# ============================================================
# ENCODE POSITION
#
# Random Forest cannot directly consume strings such as:
#
# MID
# DEF
# FWD
# GK
#
# Therefore we convert position into one-hot columns.
# ============================================================

df = pd.get_dummies(
    df,
    columns=["position"],
    dtype=int,
)


POSITION_COLUMNS = [
    column
    for column in df.columns
    if column.startswith("position_")
]


# Remove the original categorical feature.

FEATURES = [
    feature
    for feature in FEATURES
    if feature != "position"
]


# Add encoded position features.

FEATURES += POSITION_COLUMNS


# ============================================================
# TIME-BASED TRAIN / VALIDATION SPLIT
#
# IMPORTANT:
#
# We do NOT randomly split this dataset.
#
# FPL is time-dependent.
#
# Training:
#     GW 1 → GW 30
#
# Validation:
#     GW 31 → GW 37
#
# This simulates the real situation:
#
# "Use the past to predict the future."
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

X_train = train[FEATURES]

y_train = train[TARGET]


# ============================================================
# VALIDATION DATA
# ============================================================

X_validation = validation[FEATURES]

y_validation = validation[TARGET]


print("\nTraining:")
print(
    f"Rows: {len(train):,}"
)

print(
    f"GW: {train.GW.min()} → {train.GW.max()}"
)


print("\nValidation:")
print(
    f"Rows: {len(validation):,}"
)

print(
    f"GW: {validation.GW.min()} → {validation.GW.max()}"
)


print(
    f"\nFeatures used: {len(FEATURES)}"
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train,
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
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
})


importance = importance.sort_values(
    "importance",
    ascending=False,
)


# ============================================================
# SAVE MODEL
#
# We save:
#
# 1. The trained Random Forest
# 2. The exact feature list
#
# The feature list is critical because the future
# prediction script must construct the data in exactly
# the same structure used during training.
# ============================================================

model_package = {
    "model": model,
    "features": FEATURES,
}


joblib.dump(
    model_package,
    MODEL_FILE,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)


print(
    f"\nMAE:  {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)


print("\nTop features:")

print(
    importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"\nModel: {MODEL_FILE}"
)

print(
    f"Features saved: {len(FEATURES)}"
)

print(
    f"Training rows: {len(train):,}"
)

print(
    f"Validation rows: {len(validation):,}"
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)