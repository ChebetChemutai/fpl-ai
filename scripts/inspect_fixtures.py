import json
from pathlib import Path


files = sorted(Path("data/raw").glob("bootstrap_*.json"))

if not files:
    raise FileNotFoundError("No FPL data snapshot found.")

latest_file = files[-1]

with latest_file.open("r", encoding="utf-8") as file:
    data = json.load(file)

print("Available top-level data:")

for key in data:
    print("-", key)

print("\nTeams:")

for team in data["teams"]:
    print(
        team["id"],
        team["name"],
        "| strength:",
        team["strength"],
        "| attack:",
        team["strength_attack_home"],
        "| defense:",
        team["strength_defence_home"],
    )