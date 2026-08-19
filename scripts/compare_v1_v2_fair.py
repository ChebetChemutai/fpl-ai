import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

V1_DATASET = (
    BASE_DIR
    / "data"
    / "historical"
    / "training_dataset.csv"
)

V2_DATASET = (
    BASE_DIR
    / "data"
    / "historical"
    / "training_dataset_v2.csv"
)

V1_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "fpl_baseline_model_v1.joblib"
)

V2_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "fpl_v2_model.joblib"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "historical"
    / "v1_v2_fair_comparison.csv"
)


VALIDATION_START_GW = 31
VALIDATION_END_GW = 37

TOP_N_VALUES = [10, 20, 50]


# ============================================================
# HELPERS
# ============================================================

def calculate_spearman(actual, predicted):
    """
    Calculate Spearman rank correlation.

    Returns NaN if correlation cannot be calculated.
    """

    correlation = spearmanr(
        actual,
        predicted,
    ).statistic

    if correlation is None:
        return np.nan

    return float(correlation)


def calculate_top_n_hit_rate(
    df,
    prediction_column,
    n,
):
    """
    Calculate average Top-N hit rate across gameweeks.

    For each GW:

        predicted_top_n
        actual_top_n

    Hit rate:

        intersection / N

    This measures how well the model identifies the
    highest-scoring players.

    Example:

        Actual Top 10:
            A B C D E F G H I J

        Predicted Top 10:
            A B C X Y Z ...

        If 5 players overlap:

            hit rate = 5 / 10 = 0.50
    """

    gameweek_rates = []

    for gw in sorted(df["GW"].unique()):

        gw_df = df[
            df["GW"] == gw
        ].copy()

        if len(gw_df) < n:
            continue

        actual_top = set(
            gw_df
            .nlargest(
                n,
                "target_points",
            )["element"]
        )

        predicted_top = set(
            gw_df
            .nlargest(
                n,
                prediction_column,
            )["element"]
        )

        hits = len(
            actual_top.intersection(
                predicted_top
            )
        )

        hit_rate = hits / n

        gameweek_rates.append(
            hit_rate
        )

    if not gameweek_rates:
        return np.nan

    return float(
        np.mean(gameweek_rates)
    )


def calculate_calibration(df, prediction_column):
    """
    Compare average predicted points with actual points.
    """

    actual_mean = df[
        "target_points"
    ].mean()

    predicted_mean = df[
        prediction_column
    ].mean()

    return {
        "actual_mean": actual_mean,
        "predicted_mean": predicted_mean,
        "bias": predicted_mean - actual_mean,
    }


# ============================================================
# START
# ============================================================

print("=" * 70)
print("FPL AI — FAIR V1 vs V2 COMPARISON")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    V1_DATASET,
    V2_DATASET,
    V1_MODEL_FILE,
    V2_MODEL_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

v1 = pd.read_csv(
    V1_DATASET
)

v2 = pd.read_csv(
    V2_DATASET
)

print(
    f"V1 rows: {len(v1):,}"
)

print(
    f"V2 rows: {len(v2):,}"
)


# ============================================================
# VALIDATION PERIOD
# ============================================================

v1 = v1[
    (v1["GW"] >= VALIDATION_START_GW)
    & (v1["GW"] <= VALIDATION_END_GW)
].copy()

v2 = v2[
    (v2["GW"] >= VALIDATION_START_GW)
    & (v2["GW"] <= VALIDATION_END_GW)
].copy()


print("\nValidation period:")

print(
    f"GW {VALIDATION_START_GW} → "
    f"GW {VALIDATION_END_GW}"
)

print(
    f"V1 validation rows: {len(v1):,}"
)

print(
    f"V2 validation rows: {len(v2):,}"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

v1_package = joblib.load(
    V1_MODEL_FILE
)

v2_package = joblib.load(
    V2_MODEL_FILE
)


v1_model = v1_package["model"]
v1_features = v1_package["features"]

v2_model = v2_package["model"]
v2_features = v2_package["features"]


print(
    f"V1 features: {len(v1_features)}"
)

print(
    f"V2 features: {len(v2_features)}"
)


# ============================================================
# PREPARE V1 FEATURES
# ============================================================

print("\nPreparing V1 features...")

v1_model_data = v1.copy()


# ------------------------------------------------------------
# One-hot encode position
# ------------------------------------------------------------

v1_model_data = pd.get_dummies(
    v1_model_data,
    columns=["position"],
    dtype=int,
)


# Ensure every feature expected by the model exists.

for feature in v1_features:

    if feature not in v1_model_data.columns:

        v1_model_data[feature] = 0


X_v1 = v1_model_data[
    v1_features
]


# ============================================================
# PREPARE V2 FEATURES
# ============================================================

print("Preparing V2 features...")

v2_model_data = v2.copy()


v2_model_data = pd.get_dummies(
    v2_model_data,
    columns=["position"],
    dtype=int,
)


for feature in v2_features:

    if feature not in v2_model_data.columns:

        v2_model_data[feature] = 0


X_v2 = v2_model_data[
    v2_features
]


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

v1["v1_prediction"] = (
    v1_model.predict(X_v1)
)

v2["v2_prediction"] = (
    v2_model.predict(X_v2)
)


# ============================================================
# KEEP ONLY COMMON PLAYER/GW OBSERVATIONS
#
# This is the critical part.
#
# We compare both models on EXACTLY the same observations.
# ============================================================

print(
    "\nFinding common player/GW observations..."
)


common_keys = pd.merge(
    v1[
        ["element", "GW"]
    ],
    v2[
        ["element", "GW"]
    ],
    on=[
        "element",
        "GW",
    ],
    how="inner",
).drop_duplicates()


print(
    f"Common player/GW rows: "
    f"{len(common_keys):,}"
)


# ============================================================
# MERGE PREDICTIONS
# ============================================================

v1_common = v1.merge(
    common_keys,
    on=[
        "element",
        "GW",
    ],
    how="inner",
)


v2_common = v2.merge(
    common_keys,
    on=[
        "element",
        "GW",
    ],
    how="inner",
)


# ============================================================
# BUILD FINAL COMPARISON DATASET
# ============================================================

comparison = v1_common[
    [
        "element",
        "GW",
        "target_points",
        "v1_prediction",
    ]
].merge(
    v2_common[
        [
            "element",
            "GW",
            "target_points",
            "v2_prediction",
        ]
    ],
    on=[
        "element",
        "GW",
    ],
    how="inner",
    suffixes=(
        "_v1",
        "_v2",
    ),
)


# ============================================================
# TARGET CONSISTENCY CHECK
# ============================================================

target_difference = (
    comparison["target_points_v1"]
    - comparison["target_points_v2"]
).abs()


if target_difference.max() > 0:

    print(
        "\nWARNING:"
    )

    print(
        "V1 and V2 have different target values "
        "for some common player/GW rows."
    )

    print(
        f"Maximum difference: "
        f"{target_difference.max()}"
    )

    raise ValueError(
        "Target mismatch detected. "
        "Cannot perform fair comparison."
    )


comparison["target_points"] = (
    comparison["target_points_v1"]
)


comparison = comparison.drop(
    columns=[
        "target_points_v1",
        "target_points_v2",
    ]
)


# ============================================================
# SORT
# ============================================================

comparison = comparison.sort_values(
    [
        "GW",
        "element",
    ]
).reset_index(
    drop=True
)


# ============================================================
# EVALUATE V1
# ============================================================

print("\n" + "=" * 70)
print("V1 — FAIR EVALUATION")
print("=" * 70)


v1_mae = mean_absolute_error(
    comparison["target_points"],
    comparison["v1_prediction"],
)


v1_rmse = np.sqrt(
    mean_squared_error(
        comparison["target_points"],
        comparison["v1_prediction"],
    )
)


v1_spearman = calculate_spearman(
    comparison["target_points"],
    comparison["v1_prediction"],
)


v1_top_rates = {}

for n in TOP_N_VALUES:

    v1_top_rates[n] = (
        calculate_top_n_hit_rate(
            comparison,
            "v1_prediction",
            n,
        )
    )


v1_calibration = calculate_calibration(
    comparison,
    "v1_prediction",
)


print(
    f"\nMAE:              {v1_mae:.4f}"
)

print(
    f"RMSE:             {v1_rmse:.4f}"
)

print(
    f"Spearman:         {v1_spearman:.4f}"
)

for n in TOP_N_VALUES:

    print(
        f"Top-{n} hit rate:  "
        f"{v1_top_rates[n]:.4f}"
    )

print(
    f"Actual mean:      "
    f"{v1_calibration['actual_mean']:.4f}"
)

print(
    f"Predicted mean:   "
    f"{v1_calibration['predicted_mean']:.4f}"
)

print(
    f"Prediction bias:  "
    f"{v1_calibration['bias']:.4f}"
)


# ============================================================
# EVALUATE V2
# ============================================================

print("\n" + "=" * 70)
print("V2 — FAIR EVALUATION")
print("=" * 70)


v2_mae = mean_absolute_error(
    comparison["target_points"],
    comparison["v2_prediction"],
)


v2_rmse = np.sqrt(
    mean_squared_error(
        comparison["target_points"],
        comparison["v2_prediction"],
    )
)


v2_spearman = calculate_spearman(
    comparison["target_points"],
    comparison["v2_prediction"],
)


v2_top_rates = {}

for n in TOP_N_VALUES:

    v2_top_rates[n] = (
        calculate_top_n_hit_rate(
            comparison,
            "v2_prediction",
            n,
        )
    )


v2_calibration = calculate_calibration(
    comparison,
    "v2_prediction",
)


print(
    f"\nMAE:              {v2_mae:.4f}"
)

print(
    f"RMSE:             {v2_rmse:.4f}"
)

print(
    f"Spearman:         {v2_spearman:.4f}"
)

for n in TOP_N_VALUES:

    print(
        f"Top-{n} hit rate:  "
        f"{v2_top_rates[n]:.4f}"
    )

print(
    f"Actual mean:      "
    f"{v2_calibration['actual_mean']:.4f}"
)

print(
    f"Predicted mean:   "
    f"{v2_calibration['predicted_mean']:.4f}"
)

print(
    f"Prediction bias:  "
    f"{v2_calibration['bias']:.4f}"
)


# ============================================================
# IMPROVEMENTS
# ============================================================

mae_improvement = (
    (v1_mae - v2_mae)
    / v1_mae
    * 100
)


rmse_improvement = (
    (v1_rmse - v2_rmse)
    / v1_rmse
    * 100
)


spearman_change = (
    v2_spearman
    - v1_spearman
)


top10_change = (
    v2_top_rates[10]
    - v1_top_rates[10]
)


top20_change = (
    v2_top_rates[20]
    - v1_top_rates[20]
)


top50_change = (
    v2_top_rates[50]
    - v1_top_rates[50]
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    [
        {
            "model": "V1",
            "mae": v1_mae,
            "rmse": v1_rmse,
            "spearman": v1_spearman,
            "top10": v1_top_rates[10],
            "top20": v1_top_rates[20],
            "top50": v1_top_rates[50],
            "actual_mean": v1_calibration["actual_mean"],
            "predicted_mean": v1_calibration["predicted_mean"],
            "bias": v1_calibration["bias"],
        },
        {
            "model": "V2",
            "mae": v2_mae,
            "rmse": v2_rmse,
            "spearman": v2_spearman,
            "top10": v2_top_rates[10],
            "top20": v2_top_rates[20],
            "top50": v2_top_rates[50],
            "actual_mean": v2_calibration["actual_mean"],
            "predicted_mean": v2_calibration["predicted_mean"],
            "bias": v2_calibration["bias"],
        },
    ]
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FAIR V1 vs V2")
print("=" * 70)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 70)
print("V2 CHANGE")
print("=" * 70)


print(
    f"\nMAE improvement:       "
    f"{mae_improvement:+.2f}%"
)

print(
    f"RMSE improvement:      "
    f"{rmse_improvement:+.2f}%"
)

print(
    f"Spearman change:       "
    f"{spearman_change:+.4f}"
)

print(
    f"Top-10 change:         "
    f"{top10_change:+.4f}"
)

print(
    f"Top-20 change:         "
    f"{top20_change:+.4f}"
)

print(
    f"Top-50 change:         "
    f"{top50_change:+.4f}"
)


# ============================================================
# MODEL DECISION
# ============================================================

print("\n" + "=" * 70)
print("MODEL DECISION")
print("=" * 70)


wins = 0
losses = 0


# MAE
if v2_mae < v1_mae:
    wins += 1
else:
    losses += 1


# RMSE
if v2_rmse < v1_rmse:
    wins += 1
else:
    losses += 1


# Spearman
if v2_spearman > v1_spearman:
    wins += 1
else:
    losses += 1


# Top-10
if v2_top_rates[10] > v1_top_rates[10]:
    wins += 1
else:
    losses += 1


# Top-20
if v2_top_rates[20] > v1_top_rates[20]:
    wins += 1
else:
    losses += 1


# Top-50
if v2_top_rates[50] > v1_top_rates[50]:
    wins += 1
else:
    losses += 1


print(
    f"\nV2 metric wins: {wins}/6"
)

print(
    f"V1 metric wins: {losses}/6"
)


if wins >= 5:

    print(
        "\nRESULT: V2 is a strong candidate for promotion."
    )

elif wins >= 4:

    print(
        "\nRESULT: V2 shows a positive result."
    )

else:

    print(
        "\nRESULT: V2 is NOT ready for promotion."
    )

    print(
        "Further analysis is required."
    )


# ============================================================
# SAVE
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\nComparison saved:")

print(
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("FAIR COMPARISON COMPLETE")
print("=" * 70)