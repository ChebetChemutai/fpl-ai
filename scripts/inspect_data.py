import json
from pathlib import Path


files = sorted(Path("data/raw").glob("bootstrap_*.json"))

latest_file = files[-1]

with latest_file.open("r", encoding="utf-8") as file:
    data = json.load(file)

player = data["elements"][0]

print("Latest file:", latest_file)
print("\nFirst player:\n")

for key, value in player.items():
    print(f"{key}: {value}")
