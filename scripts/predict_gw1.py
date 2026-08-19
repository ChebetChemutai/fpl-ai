import os
import sys

import django


# -------------------------------------------------------------------
# Django setup
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, BASE_DIR)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

django.setup()


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

from fpl.features import get_gw_features


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

GAMEWEEK = 1


# -------------------------------------------------------------------
# Generate features
# -------------------------------------------------------------------

features = get_gw_features(
    GAMEWEEK
)


# -------------------------------------------------------------------
# Baseline prediction
# -------------------------------------------------------------------

for player in features:

    predicted_points = (
    player["historical_score"] * 0.35
    + player["position_score"] * 0.25
    + player["fixture_score"] * 2.0
    + player["minutes_reliability"] * 1.5
    + player["value_score"] * 4.0

    )

    player["predicted_points"] = (
        predicted_points
    )


# -------------------------------------------------------------------
# Rank players
# -------------------------------------------------------------------

features.sort(
    key=lambda x: x["predicted_points"],
    reverse=True,
)


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------

print()
print("=" * 80)
print("FPL AI — GAMEWEEK 1 BASELINE PREDICTIONS")
print("=" * 80)

print(
    f"{'Rank':<6}"
    f"{'Player':<18}"
    f"{'Price':<8}"
    f"{'PPG':<8}"
    f"{'xGI':<8}"
    f"{'Fixture':<10}"
    f"{'Prediction':<12}"
)

print("-" * 80)


for rank, player in enumerate(
    features[:30],
    start=1,
):

    print(
        f"{rank:<6}"
        f"{player['web_name']:<18}"
        f"{player['price']:<8.1f}"
        f"{player['points_per_game']:<8.1f}"
        f"{player['expected_goal_involvements']:<8.2f}"
        f"{player['fixture_difficulty']:<10}"
        f"{player['predicted_points']:<12.2f}"
    )


print("=" * 80)

print(
    f"\nPlayers evaluated: {len(features)}"
)