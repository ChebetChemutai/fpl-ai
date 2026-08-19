import json
from pathlib import Path

import requests


URL = "https://fantasy.premierleague.com/api/fixtures/"

response = requests.get(URL, timeout=30)
response.raise_for_status()

fixtures = response.json()

print("Status:", response.status_code)
print("Total fixtures:", len(fixtures))

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "fixtures.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(fixtures, file, indent=2)

print("Saved to:", output_file)