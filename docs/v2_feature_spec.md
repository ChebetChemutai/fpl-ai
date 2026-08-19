# FPL AI — V2 Feature Specification

## Objective

Improve the V1 Random Forest FPL player prediction model
without introducing data leakage.

The model must predict future FPL points using only
information that would have been available before the
prediction deadline.

---

# 1. Existing V1 Features

- GW
- price
- home
- opponent_team
- previous_points
- points_last_3
- points_last_5
- minutes_last_3
- minutes_last_5
- goals_last_3
- goals_last_5
- assists_last_3
- assists_last_5
- xg_last_3
- xg_last_5
- xa_last_3
- xa_last_5
- xgi_last_3
- xgi_last_5
- position

---

# 2. V2 Feature Families

## Form

- points_per_90_last_3
- points_per_90_last_5
- minutes_last_3
- minutes_last_5

## Starting Reliability

- starts_last_3
- starts_last_5
- start_rate_last_3
- start_rate_last_5
- minutes_per_start

## Attacking

- goals_last_3
- goals_last_5
- assists_last_3
- assists_last_5
- xg_last_3
- xg_last_5
- xa_last_3
- xa_last_5
- xgi_last_3
- xgi_last_5

## Defensive

- clean_sheets_last_3
- clean_sheets_last_5
- saves_last_3
- saves_last_5
- goals_conceded_last_3
- goals_conceded_last_5
- xgc_last_3
- xgc_last_5

## Bonus

- bonus_last_3
- bonus_last_5
- bps_last_3
- bps_last_5
- bps_per_90_last_3
- bps_per_90_last_5

## Fixture

- home
- opponent_team
- fixture_difficulty
- team_strength
- opponent_strength

## Player Value

- price
- points_per_million
- recent_points_per_million

---

# 3. Future Features

These are NOT part of V2 yet.

## Predicted Lineups

- predicted_start_probability
- expected_minutes
- rotation_risk

## Team Strength

- attacking_strength
- defensive_strength
- expected_goals_for
- expected_goals_against

## Multi-Gameweek

- expected_points_gw1
- expected_points_gw2
- expected_points_gw3
- expected_points_gw4
- expected_points_gw5
- five_gameweek_expected_points

## Transfers

- free_transfers_available
- transfer_gain
- transfer_cost
- banked_transfer_value

## Chips

- wildcard_value
- free_hit_value
- bench_boost_value
- triple_captain_value

## League Strategy

- rival_ownership
- effective_ownership
- differential_value
- rank_gain_probability

---

# 4. Data Leakage Rules

A feature is permitted only if it would have been
available before the prediction decision/deadline.

Never use:

- future gameweek points
- future goals
- future assists
- future minutes
- future bonus
- future confirmed lineup
- post-deadline information

Historical rolling features must use shift(1).

---

# 5. Model Evaluation

V2 must be compared against V1.

Metrics:

- MAE
- RMSE
- rank correlation
- top-10 hit rate
- top-20 hit rate
- top-50 hit rate
- expected points calibration

V2 is only promoted if it provides meaningful improvement
over V1.

---

# 6. Engineering Principle

Do not optimize for complexity.

Optimize for:

1. prediction accuracy
2. reliability
3. explainability
4. no data leakage
5. usefulness for actual FPL decisions