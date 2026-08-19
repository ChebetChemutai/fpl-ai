import json
import os
import sys
from pathlib import Path

import django


# -------------------------------------------------------------------
# Project setup
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "backend.settings",
)

django.setup()


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

from fpl.models import Fixture, Gameweek, Player, Team


# -------------------------------------------------------------------
# Locate raw data
# -------------------------------------------------------------------

RAW_DIR = BASE_DIR / "data" / "raw"

bootstrap_files = sorted(
    RAW_DIR.glob("bootstrap_*.json")
)

fixture_file = RAW_DIR / "fixtures.json"


if not bootstrap_files:
    raise FileNotFoundError(
        "No bootstrap_*.json file found in data/raw/"
    )


if not fixture_file.exists():
    raise FileNotFoundError(
        "fixtures.json not found in data/raw/"
    )


bootstrap_file = bootstrap_files[-1]


print("=" * 60)
print("FPL DATA LOADER")
print("=" * 60)

print(f"Bootstrap: {bootstrap_file}")
print(f"Fixtures:  {fixture_file}")


# -------------------------------------------------------------------
# Load bootstrap data
# -------------------------------------------------------------------

with bootstrap_file.open("r", encoding="utf-8") as file:
    bootstrap = json.load(file)


# -------------------------------------------------------------------
# Load fixtures
# -------------------------------------------------------------------

with fixture_file.open("r", encoding="utf-8") as file:
    fixtures = json.load(file)


# -------------------------------------------------------------------
# Load teams
# -------------------------------------------------------------------

print("\nLoading teams...")

team_count = 0

for team_data in bootstrap["teams"]:

    Team.objects.update_or_create(
        id=team_data["id"],
        defaults={
            "name": team_data["name"],
            "short_name": team_data.get(
                "short_name",
                "",
            ),
            "code": team_data.get("code"),
        },
    )

    team_count += 1


print(f"Teams processed: {team_count}")


# -------------------------------------------------------------------
# Load gameweeks
# -------------------------------------------------------------------

print("\nLoading gameweeks...")

gameweek_count = 0

for event_data in bootstrap["events"]:

    Gameweek.objects.update_or_create(
        id=event_data["id"],
        defaults={
            "name": event_data["name"],
            "deadline_time": event_data.get(
                "deadline_time"
            ),
            "finished": event_data.get(
                "finished",
                False,
            ),
            "is_current": event_data.get(
                "is_current",
                False,
            ),
            "is_next": event_data.get(
                "is_next",
                False,
            ),
        },
    )

    gameweek_count += 1


print(f"Gameweeks processed: {gameweek_count}")


# -------------------------------------------------------------------
# Load players
# -------------------------------------------------------------------

print("\nLoading players...")

player_count = 0

for player_data in bootstrap["elements"]:

    Player.objects.update_or_create(
        id=player_data["id"],
        defaults={

            # -------------------------------------------------------
            # Basic player information
            # -------------------------------------------------------

            "team_id": player_data["team"],

            "first_name": player_data.get(
                "first_name",
                "",
            ),

            "second_name": player_data.get(
                "second_name",
                "",
            ),

            "web_name": player_data.get(
                "web_name",
                "",
            ),

            "position": player_data["element_type"],

            "price": player_data["now_cost"],

            "status": player_data.get(
                "status",
                "a",
            ),

            # -------------------------------------------------------
            # Ownership
            # -------------------------------------------------------

            "ownership": float(
                player_data.get(
                    "selected_by_percent",
                    0,
                )
                or 0
            ),

            # -------------------------------------------------------
            # Performance statistics
            # -------------------------------------------------------

            "total_points": player_data.get(
                "total_points",
                0,
            ),

            "points_per_game": float(
                player_data.get(
                    "points_per_game",
                    0,
                )
                or 0
            ),

            "form": float(
                player_data.get(
                    "form",
                    0,
                )
                or 0
            ),

            # -------------------------------------------------------
            # Playing time
            # -------------------------------------------------------

            "minutes": player_data.get(
                "minutes",
                0,
            ),

            "starts": player_data.get(
                "starts",
                0,
            ),

            # -------------------------------------------------------
            # Attacking / defensive statistics
            # -------------------------------------------------------

            "goals": player_data.get(
                "goals_scored",
                0,
            ),

            "assists": player_data.get(
                "assists",
                0,
            ),

            "clean_sheets": player_data.get(
                "clean_sheets",
                0,
            ),

            # -------------------------------------------------------
            # FPL performance metrics
            # -------------------------------------------------------

            "bonus": player_data.get(
                "bonus",
                0,
            ),

            "bps": player_data.get(
                "bps",
                0,
            ),

            # -------------------------------------------------------
            # Expected statistics
            # -------------------------------------------------------

            "expected_goals": float(
                player_data.get(
                    "expected_goals",
                    0,
                )
                or 0
            ),

            "expected_assists": float(
                player_data.get(
                    "expected_assists",
                    0,
                )
                or 0
            ),

            "expected_goal_involvements": float(
                player_data.get(
                    "expected_goal_involvements",
                    0,
                )
                or 0
            ),

            "expected_goals_conceded": float(
                player_data.get(
                    "expected_goals_conceded",
                    0,
                )
                or 0
            ),
        },
    )

    player_count += 1


print(f"Players processed: {player_count}")


# -------------------------------------------------------------------
# Load fixtures
# -------------------------------------------------------------------

print("\nLoading fixtures...")

fixture_count = 0
skipped_fixtures = 0


for fixture_data in fixtures:

    gameweek_id = fixture_data.get("event")

    # Fixture does not belong to a gameweek
    if gameweek_id is None:
        skipped_fixtures += 1
        continue

    # Referenced gameweek does not exist
    if not Gameweek.objects.filter(
        id=gameweek_id
    ).exists():

        skipped_fixtures += 1
        continue

    # Make sure referenced teams exist
    home_team_id = fixture_data.get("team_h")
    away_team_id = fixture_data.get("team_a")

    if not Team.objects.filter(
        id=home_team_id
    ).exists():

        skipped_fixtures += 1
        continue

    if not Team.objects.filter(
        id=away_team_id
    ).exists():

        skipped_fixtures += 1
        continue

    Fixture.objects.update_or_create(
        id=fixture_data["id"],
        defaults={
            "gameweek_id": gameweek_id,

            "home_team_id": home_team_id,

            "away_team_id": away_team_id,

            "kickoff_time": fixture_data.get(
                "kickoff_time"
            ),

            "home_difficulty": fixture_data.get(
                "team_h_difficulty"
            ),

            "away_difficulty": fixture_data.get(
                "team_a_difficulty"
            ),

            "finished": fixture_data.get(
                "finished",
                False,
            ),
        },
    )

    fixture_count += 1


print(f"Fixtures processed: {fixture_count}")
print(f"Fixtures skipped:   {skipped_fixtures}")


# -------------------------------------------------------------------
# Final database counts
# -------------------------------------------------------------------

print("\n" + "=" * 60)
print("FPL DATA LOAD COMPLETE")
print("=" * 60)

print(f"Teams:     {Team.objects.count()}")
print(f"Players:   {Player.objects.count()}")
print(f"Gameweeks: {Gameweek.objects.count()}")
print(f"Fixtures:  {Fixture.objects.count()}")

print("=" * 60)