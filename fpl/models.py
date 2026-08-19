from django.db import models


class Team(models.Model):
    """
    Premier League team.
    """

    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)
    short_name = models.CharField(max_length=10, blank=True)
    code = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Player(models.Model):
    """
    FPL player and current season statistics.
    """

    id = models.IntegerField(primary_key=True)

    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="players",
    )

    first_name = models.CharField(max_length=100)
    second_name = models.CharField(max_length=100)
    web_name = models.CharField(max_length=100)

    # FPL position:
    # 1 = Goalkeeper
    # 2 = Defender
    # 3 = Midfielder
    # 4 = Forward
    position = models.IntegerField()

    # FPL price is stored in tenths of a million.
    # Example: 155 = £15.5m
    price = models.IntegerField()

    status = models.CharField(max_length=1)

    ownership = models.FloatField(default=0)

    # ---------------------------------------------------------------
    # Performance statistics
    # ---------------------------------------------------------------

    total_points = models.IntegerField(default=0)

    points_per_game = models.FloatField(default=0)

    form = models.FloatField(default=0)

    minutes = models.IntegerField(default=0)

    starts = models.IntegerField(default=0)

    goals = models.IntegerField(default=0)

    assists = models.IntegerField(default=0)

    clean_sheets = models.IntegerField(default=0)

    bonus = models.IntegerField(default=0)

    bps = models.IntegerField(default=0)

    # ---------------------------------------------------------------
    # Expected statistics
    # ---------------------------------------------------------------

    expected_goals = models.FloatField(default=0)

    expected_assists = models.FloatField(default=0)

    expected_goal_involvements = models.FloatField(default=0)

    expected_goals_conceded = models.FloatField(default=0)

    class Meta:
        ordering = ["web_name"]

    def __str__(self):
        return self.web_name


class Gameweek(models.Model):
    """
    FPL gameweek.
    """

    id = models.IntegerField(primary_key=True)

    name = models.CharField(max_length=50)

    deadline_time = models.DateTimeField(
        null=True,
        blank=True,
    )

    finished = models.BooleanField(default=False)

    is_current = models.BooleanField(default=False)

    is_next = models.BooleanField(default=False)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return self.name


class Fixture(models.Model):
    """
    Premier League fixture.
    """

    id = models.IntegerField(primary_key=True)

    gameweek = models.ForeignKey(
        Gameweek,
        on_delete=models.CASCADE,
        related_name="fixtures",
    )

    home_team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="home_fixtures",
    )

    away_team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="away_fixtures",
    )

    kickoff_time = models.DateTimeField(
        null=True,
        blank=True,
    )

    home_difficulty = models.IntegerField(
        null=True,
        blank=True,
    )

    away_difficulty = models.IntegerField(
        null=True,
        blank=True,
    )

    finished = models.BooleanField(default=False)

    class Meta:
        ordering = ["kickoff_time"]

    def __str__(self):
        return f"{self.home_team} vs {self.away_team}"


class PlayerSeasonStats(models.Model):
    """
    Aggregate player statistics for the season.
    """

    player = models.OneToOneField(
        Player,
        on_delete=models.CASCADE,
        related_name="season_stats",
    )

    minutes = models.IntegerField(default=0)

    starts = models.IntegerField(default=0)

    goals = models.IntegerField(default=0)

    assists = models.IntegerField(default=0)

    clean_sheets = models.IntegerField(default=0)

    bonus = models.IntegerField(default=0)

    bps = models.IntegerField(default=0)

    expected_goals = models.FloatField(default=0)

    expected_assists = models.FloatField(default=0)

    expected_goal_involvements = models.FloatField(default=0)

    expected_goals_conceded = models.FloatField(default=0)

    class Meta:
        ordering = ["-expected_goal_involvements"]

    def __str__(self):
        return f"{self.player.web_name} Season Stats"


class PlayerGameweekStats(models.Model):
    """
    Player performance in an individual gameweek.

    This becomes the historical dataset used for model training
    and backtesting.
    """

    player = models.ForeignKey(
        Player,
        on_delete=models.CASCADE,
        related_name="gameweek_stats",
    )

    gameweek = models.ForeignKey(
        Gameweek,
        on_delete=models.CASCADE,
        related_name="player_stats",
    )

    minutes = models.IntegerField(default=0)

    goals = models.IntegerField(default=0)

    assists = models.IntegerField(default=0)

    clean_sheets = models.IntegerField(default=0)

    bonus = models.IntegerField(default=0)

    bps = models.IntegerField(default=0)

    expected_goals = models.FloatField(default=0)

    expected_assists = models.FloatField(default=0)

    expected_goal_involvements = models.FloatField(default=0)

    total_points = models.IntegerField(default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["player", "gameweek"],
                name="unique_player_gameweek",
            )
        ]

        ordering = ["gameweek", "player"]

    def __str__(self):
        return f"{self.player.web_name} - {self.gameweek.name}"


class Prediction(models.Model):
    """
    AI prediction for a player in a particular gameweek.
    """

    player = models.ForeignKey(
        Player,
        on_delete=models.CASCADE,
        related_name="predictions",
    )

    gameweek = models.ForeignKey(
        Gameweek,
        on_delete=models.CASCADE,
        related_name="predictions",
    )

    predicted_points = models.FloatField()

    confidence = models.FloatField(
        null=True,
        blank=True,
    )

    model_version = models.CharField(
        max_length=50,
        default="baseline-v1",
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["player", "gameweek", "model_version"],
                name="unique_prediction_version",
            )
        ]

        ordering = ["-predicted_points"]

    def __str__(self):
        return (
            f"{self.player.web_name} "
            f"GW{self.gameweek.id}: "
            f"{self.predicted_points}"
        )