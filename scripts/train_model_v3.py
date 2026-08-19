import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


# ============================================================
# FPL AI — V3 MODEL TRAINING
# ============================================================
#
# V3 TRAINING DESIGN
# ------------------------------------------------------------
# 1. Uses training_dataset_v3.csv
#
# 2. Chronological train/validation/test split.
#
# 3. NO random train/test split.
#
# 4. target_points is NEVER used as a feature.
#
# 5. Player identity is NOT used as a predictive feature.
#
# 6. GW is retained because season/gameweek progression can
#    contain useful temporal information.
#
# 7. Models are trained separately for each position.
#
# 8. Evaluation includes:
#       MAE
#       RMSE
#       R2
#       Spearman rank correlation
#
# 9. Saves:
#       models
#       feature lists
#       metrics
#       validation predictions
#       feature importance
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "historical"
    / "training_dataset_v3.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "v3"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "predictions"
    / "v3"
)


# ------------------------------------------------------------
# Chronological split
# ------------------------------------------------------------
#
# GW 1-25:
#     training
#
# GW 26-31:
#     validation
#
# GW 32-37:
#     final test
#
# This is intentionally chronological.
#
# ------------------------------------------------------------

TRAIN_MAX_GW = 25
VALIDATION_MIN_GW = 26
VALIDATION_MAX_GW = 31
TEST_MIN_GW = 32
TEST_MAX_GW = 37


RANDOM_STATE = 42


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "max_features": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FPL AI — V3 MODEL TRAINING")
print("=" * 80)

print("\nDataset:")
print(DATA_FILE)

print("\nModel directory:")
print(MODEL_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nTraining dataset not found:\n{DATA_FILE}"
    )


df = pd.read_csv(DATA_FILE)

print(
    f"\nRows loaded: {len(df):,}"
)

print(
    f"Columns loaded: {len(df.columns)}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "element",
    "GW",
    "position",
    "price",
    "target_points",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise RuntimeError(
        "\nMissing required columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing
        )
    )


# ============================================================
# TARGET VALIDATION
# ============================================================

if df["target_points"].isna().any():

    raise RuntimeError(
        "ERROR: target_points contains NaN values."
    )


if np.isinf(
    df["target_points"].astype(float)
).any():

    raise RuntimeError(
        "ERROR: target_points contains infinite values."
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
    f"\nDuplicate player/GW rows: "
    f"{duplicate_count}"
)


if duplicate_count != 0:

    raise RuntimeError(
        "ERROR: Duplicate player/GW rows detected."
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


positions = set(
    df["position"]
    .astype(str)
    .str.upper()
)


unexpected_positions = (
    positions - valid_positions
)


if unexpected_positions:

    raise RuntimeError(
        "\nUnexpected positions found:\n"
        + "\n".join(
            str(x)
            for x in sorted(
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
# GAMEWEEK VALIDATION
# ============================================================

minimum_gw = int(
    df["GW"].min()
)

maximum_gw = int(
    df["GW"].max()
)


print(
    f"\nGW range: "
    f"{minimum_gw} → {maximum_gw}"
)


# ============================================================
# SORT
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
# FEATURE SELECTION
# ============================================================
#
# IMPORTANT:
#
# element:
#     Player ID.
#
# We intentionally exclude it because allowing the model to
# memorize player identity is undesirable for this first V3
# baseline.
#
# target_points:
#     Target and therefore excluded.
#
# ============================================================

excluded_columns = {
    "element",
    "target_points",
}


feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]


# ============================================================
# REMOVE NON-NUMERIC FEATURES
# ============================================================
#
# position is categorical.
#
# Instead of one-hot encoding it globally, we train separate
# models for each position.
#
# Therefore position is also removed from X.
#
# ============================================================

if "position" in feature_columns:

    feature_columns.remove(
        "position"
    )


# ============================================================
# FEATURE VALIDATION
# ============================================================

non_numeric_features = [
    column
    for column in feature_columns
    if not pd.api.types.is_numeric_dtype(
        df[column]
    )
]


if non_numeric_features:

    raise RuntimeError(
        "\nNon-numeric features remain:\n"
        + "\n".join(
            f"  - {column}"
            for column in non_numeric_features
        )
    )


print(
    f"\nNumber of model features: "
    f"{len(feature_columns)}"
)


print("\nFeatures:")

for index, column in enumerate(
    feature_columns,
    start=1,
):

    print(
        f"{index:>3}. {column}"
    )


# ============================================================
# FEATURE MATRIX VALIDATION
# ============================================================

X_all = df[
    feature_columns
].copy()


# Convert everything to numeric.

for column in feature_columns:

    X_all[column] = pd.to_numeric(
        X_all[column],
        errors="coerce",
    )


# Replace infinite values.

X_all = X_all.replace(
    [np.inf, -np.inf],
    np.nan,
)


feature_nulls = (
    X_all.isna()
    .sum()
)


problem_features = (
    feature_nulls[
        feature_nulls > 0
    ]
)


if len(problem_features) > 0:

    print(
        "\nFeatures containing missing values:"
    )

    print(
        problem_features
        .to_string()
    )

    # V3 builder already fills historical features.
    # We still use median imputation here as a final safety
    # measure for the model input.

    for column in problem_features.index:

        median = X_all[column].median()

        if pd.isna(median):

            median = 0.0

        X_all[column] = (
            X_all[column]
            .fillna(median)
        )


# ============================================================
# TARGET
# ============================================================

y_all = pd.to_numeric(
    df["target_points"],
    errors="coerce",
)


if y_all.isna().any():

    raise RuntimeError(
        "ERROR: Invalid target values detected."
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_mask = (
    df["GW"]
    <= TRAIN_MAX_GW
)


validation_mask = (
    (df["GW"] >= VALIDATION_MIN_GW)
    &
    (df["GW"] <= VALIDATION_MAX_GW)
)


test_mask = (
    (df["GW"] >= TEST_MIN_GW)
    &
    (df["GW"] <= TEST_MAX_GW)
)


train_df = df[
    train_mask
].copy()


validation_df = df[
    validation_mask
].copy()


test_df = df[
    test_mask
].copy()


print("\n" + "=" * 80)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 80)


print(
    f"\nTraining:"
    f"\n  GW {TRAIN_MAX_GW}"
    f"\n  Rows: {len(train_df):,}"
)


print(
    f"\nValidation:"
    f"\n  GW {VALIDATION_MIN_GW} → "
    f"{VALIDATION_MAX_GW}"
    f"\n  Rows: {len(validation_df):,}"
)


print(
    f"\nTest:"
    f"\n  GW {TEST_MIN_GW} → "
    f"{TEST_MAX_GW}"
    f"\n  Rows: {len(test_df):,}"
)


if len(train_df) == 0:

    raise RuntimeError(
        "ERROR: Training set is empty."
    )


if len(validation_df) == 0:

    raise RuntimeError(
        "ERROR: Validation set is empty."
    )


if len(test_df) == 0:

    raise RuntimeError(
        "ERROR: Test set is empty."
    )


# ============================================================
# SPLIT MATRICES
# ============================================================

train_indices = train_df.index

validation_indices = (
    validation_df.index
)

test_indices = test_df.index


X_train = X_all.loc[
    train_indices
]

X_validation = X_all.loc[
    validation_indices
]

X_test = X_all.loc[
    test_indices
]


y_train = y_all.loc[
    train_indices
]

y_validation = y_all.loc[
    validation_indices
]

y_test = y_all.loc[
    test_indices
]


# ============================================================
# POSITION-SPECIFIC TRAINING
# ============================================================

positions_to_train = [
    "GK",
    "DEF",
    "MID",
    "FWD",
]


metrics = []

all_predictions = []

feature_importance_records = []


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    correlation = spearmanr(
        y_true,
        predictions,
    )

    spearman = correlation.statistic

    if pd.isna(spearman):

        spearman = 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Spearman": float(spearman),
    }


# ============================================================
# TRAIN EACH POSITION
# ============================================================

for position in positions_to_train:

    print("\n" + "=" * 80)

    print(
        f"TRAINING POSITION: {position}"
    )

    print("=" * 80)


    train_position_mask = (
        train_df["position"]
        .astype(str)
        .str.upper()
        .eq(position)
    )


    validation_position_mask = (
        validation_df["position"]
        .astype(str)
        .str.upper()
        .eq(position)
    )


    test_position_mask = (
        test_df["position"]
        .astype(str)
        .str.upper()
        .eq(position)
    )


    X_position_train = X_train.loc[
        train_position_mask
    ]

    y_position_train = y_train.loc[
        train_position_mask
    ]


    X_position_validation = (
        X_validation.loc[
            validation_position_mask
        ]
    )

    y_position_validation = (
        y_validation.loc[
            validation_position_mask
        ]
    )


    X_position_test = (
        X_test.loc[
            test_position_mask
        ]
    )

    y_position_test = (
        y_test.loc[
            test_position_mask
        ]
    )


    print(
        f"\nTraining rows: "
        f"{len(X_position_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_position_validation):,}"
    )

    print(
        f"Test rows: "
        f"{len(X_position_test):,}"
    )


    if len(X_position_train) < 50:

        warnings.warn(
            f"Too few training rows for {position}."
        )

        continue


    # ========================================================
    # MODEL
    # ========================================================

    model = RandomForestRegressor(
        **MODEL_PARAMS
    )


    model.fit(
        X_position_train,
        y_position_train,
    )


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    validation_predictions = (
        model.predict(
            X_position_validation
        )
    )


    validation_metrics = calculate_metrics(
        y_position_validation,
        validation_predictions,
    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    test_predictions = (
        model.predict(
            X_position_test
        )
    )


    test_metrics = calculate_metrics(
        y_position_test,
        test_predictions,
    )


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print("\nValidation metrics:")

    for metric, value in (
        validation_metrics.items()
    ):

        print(
            f"  {metric:<12}: "
            f"{value:.4f}"
        )


    print("\nTest metrics:")

    for metric, value in (
        test_metrics.items()
    ):

        print(
            f"  {metric:<12}: "
            f"{value:.4f}"
        )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_file = (
        MODEL_DIR
        / f"model_v3_{position.lower()}.joblib"
    )


    joblib.dump(
        model,
        model_file,
    )


    print(
        f"\nModel saved:"
        f"\n{model_file}"
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics.append(
        {
            "position": position,

            "train_rows":
                len(X_position_train),

            "validation_rows":
                len(X_position_validation),

            "test_rows":
                len(X_position_test),

            "validation_mae":
                validation_metrics["MAE"],

            "validation_rmse":
                validation_metrics["RMSE"],

            "validation_r2":
                validation_metrics["R2"],

            "validation_spearman":
                validation_metrics["Spearman"],

            "test_mae":
                test_metrics["MAE"],

            "test_rmse":
                test_metrics["RMSE"],

            "test_r2":
                test_metrics["R2"],

            "test_spearman":
                test_metrics["Spearman"],
        }
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    validation_output = (
        validation_df[
            validation_position_mask
        ][
            [
                "element",
                "GW",
                "position",
                "price",
                "next_home",
                "next_opponent",
                "next_fixture_count",
                "target_points",
            ]
        ]
        .copy()
    )


    validation_output[
        "predicted_points"
    ] = validation_predictions


    validation_output[
        "prediction_error"
    ] = (
        validation_output[
            "target_points"
        ]
        -
        validation_output[
            "predicted_points"
        ]
    )


    validation_output[
        "split"
    ] = "validation"


    test_output = (
        test_df[
            test_position_mask
        ][
            [
                "element",
                "GW",
                "position",
                "price",
                "next_home",
                "next_opponent",
                "next_fixture_count",
                "target_points",
            ]
        ]
        .copy()
    )


    test_output[
        "predicted_points"
    ] = test_predictions


    test_output[
        "prediction_error"
    ] = (
        test_output[
            "target_points"
        ]
        -
        test_output[
            "predicted_points"
        ]
    )


    test_output[
        "split"
    ] = "test"


    all_predictions.append(
        pd.concat(
            [
                validation_output,
                test_output,
            ],
            ignore_index=True,
        )
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importances = (
        model.feature_importances_
    )


    for feature, importance in zip(
        feature_columns,
        importances,
    ):

        feature_importance_records.append(
            {
                "position": position,
                "feature": feature,
                "importance": float(
                    importance
                ),
            }
        )


# ============================================================
# METRICS DATAFRAME
# ============================================================

metrics_df = pd.DataFrame(
    metrics
)


if metrics_df.empty:

    raise RuntimeError(
        "ERROR: No position models were trained."
    )


# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = (
    OUTPUT_DIR
    / "model_metrics_v3.csv"
)


metrics_df.to_csv(
    metrics_file,
    index=False,
)


print("\n" + "=" * 80)
print("MODEL METRICS")
print("=" * 80)

print(
    metrics_df
    .round(4)
    .to_string(
        index=False
    )
)


print(
    f"\nMetrics saved:"
    f"\n{metrics_file}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)


predictions_file = (
    OUTPUT_DIR
    / "validation_predictions_v3.csv"
)


predictions_df.to_csv(
    predictions_file,
    index=False,
)


print(
    f"\nPredictions saved:"
    f"\n{predictions_file}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    feature_importance_records
)


importance_df = importance_df.sort_values(
    [
        "position",
        "importance",
    ],
    ascending=[
        True,
        False,
    ],
)


importance_file = (
    OUTPUT_DIR
    / "feature_importance_v3.csv"
)


importance_df.to_csv(
    importance_file,
    index=False,
)


print(
    f"\nFeature importance saved:"
    f"\n{importance_file}"
)


# ============================================================
# TOP FEATURES
# ============================================================

print("\n" + "=" * 80)
print("TOP FEATURES BY POSITION")
print("=" * 80)


for position in positions_to_train:

    position_importance = (
        importance_df[
            importance_df["position"]
            == position
        ]
        .head(15)
    )


    if position_importance.empty:

        continue


    print(
        f"\n{position}:"
    )


    for _, row in (
        position_importance.iterrows()
    ):

        print(
            f"  "
            f"{row['feature']:<40}"
            f"{row['importance']:.6f}"
        )


# ============================================================
# BASELINE COMPARISON
# ============================================================
#
# A model is only useful if it beats a simple baseline.
#
# Baseline:
#
#     Predict the mean target_points from training data.
#
# ============================================================

baseline_mean = float(
    y_train.mean()
)


baseline_predictions = np.full(
    len(y_test),
    baseline_mean,
)


baseline_metrics = calculate_metrics(
    y_test,
    baseline_predictions,
)


print("\n" + "=" * 80)
print("BASELINE")
print("=" * 80)


print(
    f"\nBaseline prediction: "
    f"{baseline_mean:.4f}"
)


for metric, value in (
    baseline_metrics.items()
):

    print(
        f"  {metric:<12}: "
        f"{value:.4f}"
    )


# ============================================================
# OVERALL TEST METRICS
# ============================================================

overall_test_metrics = calculate_metrics(
    predictions_df.loc[
        predictions_df["split"] == "test",
        "target_points",
    ],
    predictions_df.loc[
        predictions_df["split"] == "test",
        "predicted_points",
    ],
)


print("\n" + "=" * 80)
print("OVERALL TEST PERFORMANCE")
print("=" * 80)


for metric, value in (
    overall_test_metrics.items()
):

    print(
        f"{metric:<15}: "
        f"{value:.4f}"
    )


# ============================================================
# TRAINING METADATA
# ============================================================

metadata = {
    "version": "V3",

    "dataset": str(
        DATA_FILE
    ),

    "rows": int(
        len(df)
    ),

    "columns": int(
        len(df.columns)
    ),

    "features": feature_columns,

    "target": "target_points",

    "target_definition":
        "Player total FPL points in actual GW+1",

    "split": {
        "train": f"GW <= {TRAIN_MAX_GW}",
        "validation":
            f"GW {VALIDATION_MIN_GW}-{VALIDATION_MAX_GW}",
        "test":
            f"GW {TEST_MIN_GW}-{TEST_MAX_GW}",
    },

    "model":
        "RandomForestRegressor",

    "model_parameters":
        MODEL_PARAMS,

    "positions":
        positions_to_train,

    "baseline_mean":
        baseline_mean,

    "overall_test_metrics":
        overall_test_metrics,
}


metadata_file = (
    MODEL_DIR
    / "training_metadata_v3.json"
)


with open(
    metadata_file,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


print(
    f"\nTraining metadata saved:"
    f"\n{metadata_file}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("V3 MODEL TRAINING COMPLETE")
print("=" * 80)


print("\nModels:")

for position in positions_to_train:

    model_file = (
        MODEL_DIR
        / f"model_v3_{position.lower()}.joblib"
    )

    if model_file.exists():

        print(
            f"  {model_file}"
        )


print("\nOutputs:")

print(
    f"  {metrics_file}"
)

print(
    f"  {predictions_file}"
)

print(
    f"  {importance_file}"
)

print(
    f"  {metadata_file}"
)


print("\nNext step:")

print(
    "Inspect the validation/test metrics "
    "before building the prediction pipeline."
)

print("\n" + "=" * 80)
