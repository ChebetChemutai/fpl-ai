import json
from datetime import datetime
from pathlib import Path

import requests


URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

response = requests.get(URL, timeout=30)
response.raise_for_status()

data = response.json()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"bootstrap_{timestamp}.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(data, file, indent=2)

print(f"Status: {response.status_code}")
print(f"Players: {len(data['elements'])}")
print(f"Teams: {len(data['teams'])}")
print(f"Gameweeks: {len(data['events'])}")
print(f"Saved to: {output_file}")