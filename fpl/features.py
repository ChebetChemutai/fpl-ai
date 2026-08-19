from django.db.models import Q

from fpl.models import Fixture, Player


# ============================================================
# POSITION MAPPING
# ============================================================

POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


# ============================================================
# V3 HISTORICAL FEATURE COLUMNS
# ============================================================
#
# IMPORTANT:
#
# These names MUST match build_training_dataset_v3.py.
#
# GW1 deliberately starts with zero history because the V3
# training dataset also validates GW1 as a zero-history
# cold-start.
#
# ============================================================

V3_HISTORY_FEATURES = [
    "previous_points",

    "points_last_3",
    "points_last_5",

    "points_per_90_last_3",
    "points_per_90_last_5",

    "minutes_last_3",
    "minutes_last_5",

    "starts_last_3",
    "starts_last_5",

    "start_rate_last_3",
    "start_rate_last_5",

    "minutes_per_start",

    "goals_last_3",
    "goals_last_5",

    "assists_last_3",
    "assists_last_5",

    "xg_last_3",
    "xg_last_5",

    "xa_last_3",
    "xa_last_5",

    "xgi_last_3",
    "xgi_last_5",

    "clean_sheets_last_3",
    "clean_sheets_last_5",

    "saves_last_3",
    "saves_last_5",

    "goals_conceded_last_3",
    "goals_conceded_last_5",

    "xgc_last_3",
    "xgc_last_5",

    "bonus_last_3",
    "bonus_last_5",

    "bps_last_3",
    "bps_last_5",

    "bps_per_90_last_3",
    "bps_per_90_last_5",

    "points_per_million",
    "recent_points_per_million",

    "team_goals_last_3",
    "team_goals_last_5",

    "team_xg_last_3",
    "team_xg_last_5",

    "team_goals_conceded_last_3",
    "team_goals_conceded_last_5",

    "team_xgc_last_3",
    "team_xgc_last_5",

    "opponent_goals_last_3",
    "opponent_goals_last_5",

    "opponent_xg_last_3",
    "opponent_xg_last_5",

    "opponent_goals_conceded_last_3",
    "opponent_goals_conceded_last_5",

    "opponent_xgc_last_3",
    "opponent_xgc_last_5",

    "attack_strength",
    "defensive_strength",

    "opponent_attack_strength",
    "opponent_defensive_strength",
]


# ============================================================
# V3 COLD START
# ============================================================

def get_zero_history():
    """
    Return the exact V3 cold-start historical feature set.

    V3 training validates that the first available GW has no
    previous player/team performance information.

    For GW1 live prediction this therefore returns zeroes.
    """

    return {
        feature: 0.0
        for feature in V3_HISTORY_FEATURES
    }


# ============================================================
# POSITION
# ============================================================

def get_position(player):
    """
    Convert FPL numeric position to the V3 position label.
    """

    return POSITION_MAP.get(
        player.position,
        "MID",
    )


# ============================================================
# FIXTURE CONTEXT
# ============================================================

def get_fixture_context(player, fixture):
    """
    Return the player's home/away context and opponent ID.
    """

    if player.team_id == fixture.home_team_id:

        return {
            "home": 1,
            "opponent_team": fixture.away_team_id,
            "difficulty": fixture.home_difficulty or 3,
        }

    if player.team_id == fixture.away_team_id:

        return {
            "home": 0,
            "opponent_team": fixture.home_team_id,
            "difficulty": fixture.away_difficulty or 3,
        }

    return None


# ============================================================
# NEXT FIXTURE COUNT
# ============================================================

def get_fixture_count(player, gameweek_id):
    """
    Count how many fixtures the player's team has in the
    requested gameweek.

    This is important for double gameweeks.
    """

    return Fixture.objects.filter(
        gameweek_id=gameweek_id,
    ).filter(
        Q(home_team_id=player.team_id)
        |
        Q(away_team_id=player.team_id)
    ).count()


# ============================================================
# AVAILABILITY
# ============================================================

def get_availability_status(player):
    """
    Interpret the current FPL player status.
    """

    status = (
        player.status or ""
    ).lower()

    if status == "a":

        return {
            "available": True,
            "availability_probability": 1.0,
        }

    if status in {
        "d",
        "i",
        "s",
        "u",
    }:

        return {
            "available": False,
            "availability_probability": 0.0,
        }

    return {
        "available": True,
        "availability_probability": 1.0,
    }


# ============================================================
# V3 MODEL FEATURES
# ============================================================

def get_model_features(
    player,
    fixture,
    gameweek_id,
):
    """
    Build the V3 feature vector for one player.

    IMPORTANT:

    This function intentionally produces the V3 schema,
    not the old V1 baseline schema.
    """

    context = get_fixture_context(
        player,
        fixture,
    )

    if context is None:
        return None

    position = get_position(
        player
    )

    price = (
        player.price / 10.0
    )

    next_fixture_count = (
        get_fixture_count(
            player,
            gameweek_id,
        )
    )

    # --------------------------------------------------------
    # V3 GW1 COLD START
    # --------------------------------------------------------

    history = get_zero_history()

    # --------------------------------------------------------
    # Basic V3 features
    # --------------------------------------------------------

    features = {

        "GW": float(gameweek_id),

        "price": float(price),

        "next_home": int(
            context["home"]
        ),

        "next_opponent": int(
            context["opponent_team"]
        ),

        "next_fixture_count": float(
            next_fixture_count
        ),

        **history,
    }

    return features


# ============================================================
# DISPLAY FEATURES
# ============================================================

def get_display_features(
    player,
    fixture,
    gameweek_id,
):
    """
    Human-readable information used by the prediction output.

    These values are NOT part of the V3 model input.
    """

    context = get_fixture_context(
        player,
        fixture,
    )

    if context is None:
        return None

    availability = (
        get_availability_status(
            player
        )
    )

    position = get_position(
        player
    )

    fixture_count = (
        get_fixture_count(
            player,
            gameweek_id,
        )
    )

    return {

        "player_id": player.id,

        "web_name": player.web_name,

        "team_id": player.team_id,

        "position": position,

        "price": (
            player.price / 10.0
        ),

        "points_per_game": (
            player.points_per_game or 0.0
        ),

        "form": (
            player.form or 0.0
        ),

        "expected_goals": (
            player.expected_goals or 0.0
        ),

        "expected_assists": (
            player.expected_assists or 0.0
        ),

        "expected_goal_involvements": (
            player.expected_goal_involvements or 0.0
        ),

        "minutes": player.minutes,

        "starts": player.starts,

        "status": player.status,

        "available": (
            availability[
                "available"
            ]
        ),

        "availability_probability": (
            availability[
                "availability_probability"
            ]
        ),

        "fixture_difficulty": (
            context["difficulty"]
        ),

        "next_home": (
            context["home"]
        ),

        "next_opponent": (
            context["opponent_team"]
        ),

        "next_fixture_count": (
            fixture_count
        ),

        "fixture_id": fixture.id,

        "gameweek_id": gameweek_id,
    }


# ============================================================
# GAMEWEEK FEATURES
# ============================================================

def get_gw_features(gameweek_id):
    """
    Generate V3 prediction features for every player who has
    a fixture in the requested gameweek.

    For double gameweeks, a player can appear in multiple raw
    fixture observations. We aggregate those observations
    into one player/gameweek prediction row, matching the
    player/GW granularity used by the V3 training dataset.
    """

    fixtures = list(
        Fixture.objects
        .filter(
            gameweek_id=gameweek_id
        )
        .select_related(
            "home_team",
            "away_team",
        )
    )

    if not fixtures:
        return []

    players = (
        Player.objects
        .select_related("team")
    )

    results = {}

    for fixture in fixtures:

        fixture_players = (
            players.filter(
                team_id__in=[
                    fixture.home_team_id,
                    fixture.away_team_id,
                ]
            )
        )

        for player in fixture_players:

            model_features = (
                get_model_features(
                    player,
                    fixture,
                    gameweek_id,
                )
            )

            if model_features is None:
                continue

            display_features = (
                get_display_features(
                    player,
                    fixture,
                    gameweek_id,
                )
            )

            if display_features is None:
                continue

            # ------------------------------------------------
            # V3 training data is one row per player/GW.
            #
            # For the first fixture encountered we retain the
            # fixture context, matching the V3 builder's
            # "first" aggregation behavior.
            # ------------------------------------------------

            if player.id not in results:

                results[player.id] = {
                    **model_features,
                    **display_features,
                }

    return list(
        results.values()
    )