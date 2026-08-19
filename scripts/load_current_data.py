import json
from pathlib import Path

import requests


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

BOOTSTRAP_FILE = RAW_DIR / "current_bootstrap.json"
FIXTURES_FILE = RAW_DIR / "current_fixtures.json"

BOOTSTRAP_URL = (
    "https://fantasy.premierleague.com/api/bootstrap-static/"
)

FIXTURES_URL = (
    "https://fantasy.premierleague.com/api/fixtures/"
)


# ============================================================
# HELPERS
# ============================================================

def fetch_json(url):
    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def save_json(data, path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FPL AI — LOAD CURRENT FPL DATA")
print("=" * 70)


RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BOOTSTRAP
# ============================================================

print("\nDownloading bootstrap-static...")

bootstrap = fetch_json(
    BOOTSTRAP_URL
)

save_json(
    bootstrap,
    BOOTSTRAP_FILE,
)

print(
    f"Saved: {BOOTSTRAP_FILE}"
)


# ============================================================
# FIXTURES
# ============================================================

print("\nDownloading fixtures...")

fixtures = fetch_json(
    FIXTURES_URL
)

save_json(
    fixtures,
    FIXTURES_FILE,
)

print(
    f"Saved: {FIXTURES_FILE}"
)


# ============================================================
# SUMMARY
# ============================================================

players = bootstrap.get(
    "elements",
    [],
)

teams = bootstrap.get(
    "teams",
    [],
)

events = bootstrap.get(
    "events",
    [],
)


current_event = next(
    (
        event
        for event in events
        if event.get("is_current")
    ),
    None,
)

next_event = next(
    (
        event
        for event in events
        if event.get("is_next")
    ),
    None,
)


print("\n" + "=" * 70)
print("CURRENT FPL DATA")
print("=" * 70)

print(
    f"\nPlayers: {len(players)}"
)

print(
    f"Teams: {len(teams)}"
)

print(
    f"Gameweeks: {len(events)}"
)

if current_event:
    print(
        f"Current GW: {current_event['id']}"
    )
else:
    print(
        "Current GW: None"
    )

if next_event:
    print(
        f"Next GW: {next_event['id']}"
    )
else:
    print(
        "Next GW: None"
    )

print(
    f"\nFixtures downloaded: {len(fixtures)}"
)


print("\n" + "=" * 70)
print("LOAD COMPLETE")
print("=" * 70)