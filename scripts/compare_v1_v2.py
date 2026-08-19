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

V1_MODEL = (
    BASE_DIR
    / "models"
    / "fpl_baseline_model_v1.joblib"
)

V2_MODEL = (
    BASE_DIR
    / "models"
    / "fpl_v2_model.joblib"
)


TARGET = "target_points"


# ============================================================
# HELPERS
# ============================================================

def top_k_hit_rate(
    results,
    k,
):
    """
    Calculate average precision@K across
    validation gameweeks.

    For each GW:

        predicted top K
        vs
        actual top K
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


def evaluate_model(
    name,
    dataset_file,
    model_file,
):
    """
    Load an existing model and evaluate it
    on its matching validation dataset.
    """

    print("\n" + "=" * 70)

    print(
        f"EVALUATING {name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        dataset_file
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    package = joblib.load(
        model_file
    )

    model = package["model"]

    features = package["features"]

    print(
        f"\nDataset: {dataset_file}"
    )

    print(
        f"Model:   {model_file}"
    )

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Features saved in model: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    # --------------------------------------------------------
    # POSITION ENCODING
    # --------------------------------------------------------

    if "position" in df.columns:

        df = pd.get_dummies(
            df,
            columns=[
                "position"
            ],
            dtype=int,
        )

    # --------------------------------------------------------
    # ADD MISSING POSITION COLUMNS
    #
    # This guarantees the validation dataframe has
    # exactly the same columns as the trained model.
    # --------------------------------------------------------

    for feature in features:

        if feature not in df.columns:

            df[feature] = 0

    # --------------------------------------------------------
    # VALIDATION SPLIT
    # --------------------------------------------------------

    validation = df[
        df["GW"] >= 31
    ].copy()

    if validation.empty:

        raise ValueError(
            f"{name}: validation dataset is empty."
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = validation[
        features
    ]

    y = validation[
        TARGET
    ]

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions,
        )
    )

    results = validation[
        [
            "element",
            "GW",
            TARGET,
        ]
    ].copy()

    results[
        "predicted_points"
    ] = predictions

    spearman = (
        results[
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

    top10 = top_k_hit_rate(
        results,
        10,
    )

    top20 = top_k_hit_rate(
        results,
        20,
    )

    top50 = top_k_hit_rate(
        results,
        50,
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        f"\nValidation rows:"
        f" {len(validation):,}"
    )

    print(
        f"Validation GW:"
        f" {validation.GW.min()}"
        f" → "
        f"{validation.GW.max()}"
    )

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
        f"         {spearman:.4f}"
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

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "top10": top10,
        "top20": top20,
        "top50": top50,
    }


# ============================================================
# START
# ============================================================

print("=" * 70)

print(
    "FPL AI — V1 vs V2 MODEL COMPARISON"
)

print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

files = {
    "V1 dataset": V1_DATASET,
    "V2 dataset": V2_DATASET,
    "V1 model": V1_MODEL,
    "V2 model": V2_MODEL,
}


for name, path in files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {name}:\n{path}"
        )


# ============================================================
# EVALUATE
# ============================================================

v1 = evaluate_model(
    "V1",
    V1_DATASET,
    V1_MODEL,
)

v2 = evaluate_model(
    "V2",
    V2_DATASET,
    V2_MODEL,
)


# ============================================================
# COMPARISON
# ============================================================

comparison = pd.DataFrame(
    [
        v1,
        v2,
    ]
)


# ============================================================
# IMPROVEMENT
#
# Lower is better for:
#
# MAE
# RMSE
#
# Higher is better for:
#
# Spearman
# Top-K
# ============================================================

v1_mae = v1["mae"]
v2_mae = v2["mae"]

v1_rmse = v1["rmse"]
v2_rmse = v2["rmse"]

v1_spearman = v1["spearman"]
v2_spearman = v2["spearman"]

v1_top10 = v1["top10"]
v2_top10 = v2["top10"]

v1_top20 = v1["top20"]
v2_top20 = v2["top20"]

v1_top50 = v1["top50"]
v2_top50 = v2["top50"]


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


spearman_improvement = (
    v2_spearman
    - v1_spearman
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("\n" + "=" * 70)

print(
    "V1 vs V2"
)

print("=" * 70)


print(
    comparison[
        [
            "model",
            "mae",
            "rmse",
            "spearman",
            "top10",
            "top20",
            "top50",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 70)

print(
    "V2 IMPROVEMENT"
)

print("=" * 70)


print(
    f"\nMAE improvement:"
    f"       {mae_improvement:+.2f}%"
)


print(
    f"RMSE improvement:"
    f"      {rmse_improvement:+.2f}%"
)


print(
    f"Spearman change:"
    f"        {spearman_improvement:+.4f}"
)


print(
    f"\nTop-10:"
    f" V1={v1_top10:.4f}"
    f" → V2={v2_top10:.4f}"
)


print(
    f"Top-20:"
    f" V1={v1_top20:.4f}"
    f" → V2={v2_top20:.4f}"
)


print(
    f"Top-50:"
    f" V1={v1_top50:.4f}"
    f" → V2={v2_top50:.4f}"
)


# ============================================================
# DECISION
# ============================================================

print("\n" + "=" * 70)

print(
    "MODEL DECISION"
)

print("=" * 70)


wins = 0


if v2_mae < v1_mae:
    wins += 1


if v2_rmse < v1_rmse:
    wins += 1


if v2_spearman > v1_spearman:
    wins += 1


if v2_top10 > v1_top10:
    wins += 1


if v2_top20 > v1_top20:
    wins += 1


if v2_top50 > v1_top50:
    wins += 1


print(
    f"\nV2 metric wins:"
    f" {wins}/6"
)


if wins >= 4:

    print(
        "\nRESULT: V2 shows meaningful improvement."
    )

    print(
        "Continue with V2."
    )

elif wins <= 2:

    print(
        "\nRESULT: V2 does not show meaningful improvement."
    )

    print(
        "Do NOT promote V2 yet."
    )

else:

    print(
        "\nRESULT: Mixed result."
    )

    print(
        "We need deeper analysis before promotion."
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "historical"
    / "v1_v2_comparison.csv"
)


comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    f"\nComparison saved:"
    f"\n{OUTPUT_FILE}"
)


print(
    "\n" + "=" * 70
)

print(
    "COMPARISON COMPLETE"
)

print(
    "=" * 70
)