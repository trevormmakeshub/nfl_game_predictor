Run python scripts/refresh_week.py once a week during the season. It does not run on a timer.

nflverse is the recorded-stats source. ESPN is a live score check and is not copied into the training scores. Sleeper supplies the current week only. nfl.com is not requested. Real is not a source.

The training rows use only earlier games: win rate, points per game, rest days, the last four games' win rate and margin when that check improves the 2026 holdout, prior-week injury and snap counts, and the starting quarterback's prior-games passer rating and interception rate. A game does not train on its own score or its own quarterback line. The win model is a ridge classifier. Points use linear regression, plus a half-weight of prior points per game only when that blend lowers the holdout total error.

The line is the recorded schedule line, not a live book price, and this does not price a bet. A lean is not a wager.

Week 4 Green Bay at Tampa Bay, game 2026_04_GB_TB, lists Jordan Love (00-0036264) as the away quarterback only. The home quarterback is the Tampa Bay starter on the latest 2026 depth chart, Baker Mayfield (00-0034855). Jordan Love is not the home quarterback. Week 5 Tampa Bay at Dallas was left as saved. Washington, game 2026_04_IND_WAS, stays as saved. The 2026 injury file has no starter row for Marcus Mariota.

Calibration is a check, not a bet.

Rows that could not be scored: 25

spread_line, total_line, home_moneyline, and div_game are model features when the schedule file has those columns. roof, surface, temp, wind, away_qb_id, and home_qb_id are carried from that file. temp and wind are blank on the current week, so they are not filled and are not model features.

Completed rows: 422
Training rows: 395
Feature count: 18
Minimum date: 2025-09-04
Maximum date: 2026-11-12
Next games: 13
High-confidence next games: 4
In-sample accuracy: 0.6684 on 395 scored rows
Baseline holdout: accuracy 0.6269, total MAE 14.513, prior-points blend MAE 12.540, 134 test games after 261 earlier games, from 2026-09-01
Recent-form holdout: accuracy 0.5746, total MAE 14.010, prior-points blend MAE 12.329, 134 test games after 261 earlier games, from 2026-09-01
Recent form adopted: no
Points blend adopted: yes
Date check: pass

Files saved:
- data/live/espn_scoreboard.json
- data/live/sleeper_state.json
- artifacts/nfl_completed.csv
- artifacts/nfl_next_games.csv
- artifacts/seeds.csv
- artifacts/calibration.csv
- artifacts/high_confidence.csv

Files skipped:
- participation was not requested and was not downloaded
- local rebuild did not download
- nfl.com was not requested

ESPN status: 200
nfl.com status: not requested

FanGraphs, stats.nba.com, nba_api, and football-data.org were not called. Raw XY tracking was not downloaded.
