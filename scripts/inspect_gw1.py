import json
from pathlib import Path


# Load FPL bootstrap data
bootstrap_file = sorted(
    Path("data/raw").glob("bootstrap_*.json")
)[-1]

with bootstrap_file.open("r", encoding="utf-8") as f:
    bootstrap = json.load(f)


# Create team ID → team name lookup
teams = {
    team["id"]: team["name"]
    for team in bootstrap["teams"]
}


# Load fixtures
fixture_file = Path("data/raw/fixtures.json")

with fixture_file.open("r", encoding="utf-8") as f:
    fixtures = json.load(f)


# Get GW1
gw1 = [
    fixture
    for fixture in fixtures
    if fixture.get("event") == 1
]


print("FPL 2026/27 — GAMEWEEK 1")
print("=" * 50)

for fixture in gw1:

    home = teams[fixture["team_h"]]
    away = teams[fixture["team_a"]]

    print(
        f"{home} vs {away} "
        f"| Home difficulty: {fixture['team_h_difficulty']} "
        f"| Away difficulty: {fixture['team_a_difficulty']}"
    )