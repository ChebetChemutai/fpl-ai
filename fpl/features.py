from fpl.models import Fixture, Player


def get_fixture_context(player, fixture):
    """
    Determine whether the player is playing at home or away
    and return the relevant fixture difficulty.
    """

    if player.team_id == fixture.home_team_id:
        return {
            "difficulty": fixture.home_difficulty or 3,
            "home": 1,
        }

    return {
        "difficulty": fixture.away_difficulty or 3,
        "home": 0,
    }


def calculate_fixture_score(difficulty):
    """
    Convert FPL fixture difficulty (1-5)
    into a normalized score.
    """

    return max(
        0.0,
        min(
            1.0,
            1 - ((difficulty - 1) / 5),
        ),
    )


def calculate_minutes_reliability(player):
    """
    Estimate how reliable a player's playing time is.
    """

    if player.minutes <= 0:
        return 0.0

    return min(
        player.minutes / 3000,
        1.0,
    )


def calculate_attacking_score(player):
    """
    Position-independent attacking potential.
    """

    return (
        (player.goals or 0) * 1.0
        + (player.assists or 0) * 0.7
        + (player.expected_goals or 0) * 0.8
        + (player.expected_assists or 0) * 0.6
    )


def calculate_position_score(player):
    """
    Position-aware player score.

    FPL positions:
        1 = GK
        2 = DEF
        3 = MID
        4 = FWD
    """

    xg = player.expected_goals or 0
    xa = player.expected_assists or 0
    goals = player.goals or 0
    assists = player.assists or 0
    bonus = player.bonus or 0
    bps = player.bps or 0

    # Goalkeeper
    if player.position == 1:

        return (
            bonus * 0.10
            + bps * 0.01
            + player.points_per_game * 0.8
        )

    # Defender
    if player.position == 2:

        return (
            goals * 1.2
            + assists * 0.8
            + xg * 0.9
            + xa * 0.7
            + bonus * 0.08
            + bps * 0.008
        )

    # Midfielder
    if player.position == 3:

        return (
            goals * 1.3
            + assists * 1.0
            + xg * 1.0
            + xa * 0.9
            + bonus * 0.08
            + bps * 0.008
        )

    # Forward
    if player.position == 4:

        return (
            goals * 1.4
            + assists * 0.9
            + xg * 1.1
            + xa * 0.8
            + bonus * 0.08
            + bps * 0.008
        )

    return 0.0


def calculate_value_score(player):
    """
    Historical points per million.

    Price is stored in FPL tenths.
    """

    price = player.price / 10

    if price <= 0:
        return 0.0

    return (
        player.points_per_game / price
    )


def get_player_features(player, fixture):
    """
    Build the complete feature vector for a player
    in a particular fixture.
    """

    fixture_context = get_fixture_context(
        player,
        fixture,
    )

    difficulty = fixture_context["difficulty"]

    fixture_score = calculate_fixture_score(
        difficulty
    )

    minutes_reliability = (
        calculate_minutes_reliability(player)
    )

    attacking_score = (
        calculate_attacking_score(player)
    )

    position_score = (
        calculate_position_score(player)
    )

    value_score = (
        calculate_value_score(player)
    )

    historical_score = (
        (player.points_per_game or 0) * 0.7
        + (player.form or 0) * 0.3
    )

    return {
        "player_id": player.id,
        "web_name": player.web_name,
        "position": player.position,

        "price": player.price / 10,

        "ownership": player.ownership,

        "points_per_game": (
            player.points_per_game or 0
        ),

        "form": player.form or 0,

        "minutes_reliability": (
            minutes_reliability
        ),

        "goals": player.goals or 0,

        "assists": player.assists or 0,

        "expected_goals": (
            player.expected_goals or 0
        ),

        "expected_assists": (
            player.expected_assists or 0
        ),

        "expected_goal_involvements": (
            player.expected_goal_involvements or 0
        ),

        "attacking_score": attacking_score,

        "position_score": position_score,

        "historical_score": historical_score,

        "fixture_difficulty": difficulty,

        "fixture_score": fixture_score,

        "home_advantage": (
            fixture_context["home"]
        ),

        "value_score": value_score,

        "fixture_id": fixture.id,

        "gameweek_id": fixture.gameweek_id,
    }


def get_gw_features(gameweek_id):
    """
    Generate features for every player
    who has a fixture in a gameweek.
    """

    fixtures = (
        Fixture.objects
        .filter(gameweek_id=gameweek_id)
        .select_related(
            "home_team",
            "away_team",
        )
    )

    players = Player.objects.select_related(
        "team"
    )

    features = []

    for fixture in fixtures:

        fixture_players = players.filter(
            team_id__in=[
                fixture.home_team_id,
                fixture.away_team_id,
            ]
        )

        for player in fixture_players:

            feature_vector = get_player_features(
                player,
                fixture,
            )

            features.append(
                feature_vector
            )

    return features