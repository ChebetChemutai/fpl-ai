import requests

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

response = requests.get(URL, timeout=30)

print("Status:", response.status_code)

data = response.json()

print("\nTop-level data:")
print(data.keys())

print("\nPlayers:", len(data["elements"]))
print("Teams:", len(data["teams"]))
print("Gameweeks:", len(data["events"]))