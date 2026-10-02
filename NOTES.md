Run python scripts/refresh_week.py once a week during the season. It does not run on a timer.

nflverse is the recorded-stats source. ESPN is a live score check and is not copied into the training scores. Sleeper supplies the current week only. nfl.com is one score-strip request. Real is not a source.

The training rows use only earlier games: win rate, points per game, rest days, prior-week injury and snap counts, and the starting quarterback's prior-games passer rating and interception rate. A game does not train on its own score or its own quarterback line. The win model is a ridge classifier. Points use linear regression.

The line is the recorded schedule line, not a live book price, and this does not price a bet. A lean is not a wager.

Week 4 Green Bay at Tampa Bay, game 2026_04_GB_TB, lists Jordan Love (00-0036264) as the away quarterback only. The home quarterback is the Tampa Bay starter on the latest 2026 depth chart, Baker Mayfield (00-0034855). Jordan Love is not the home quarterback. Week 5 Tampa Bay at Dallas was left as saved. Washington, game 2026_04_IND_WAS, stays as saved. The 2026 injury file has no starter row for Marcus Mariota.

Calibration is a check, not a bet.

Rows that could not be scored: 25

spread_line, total_line, home_moneyline, and div_game are model features when the schedule file has those columns. roof, surface, temp, wind, away_qb_id, and home_qb_id are carried from that file. temp and wind are blank on the current week, so they are not filled and are not model features.

Completed rows: 333
Training rows: 307
Feature count: 18
Minimum date: 2025-09-04
Maximum date: 2026-09-28
Date check: pass

Files saved:
- data/schedules/games.csv
- data/stats_player/stats_player_post_2025.csv
- data/stats_player/stats_player_regpost_2025.csv
- data/stats_player/stats_player_regpost_2026.csv
- data/stats_player/stats_player_reg_2025.csv
- data/stats_player/stats_player_reg_2026.csv
- data/stats_player/stats_player_week_2025.csv
- data/stats_player/stats_player_week_2026.csv
- data/stats_team/stats_team_post_2025.csv
- data/stats_team/stats_team_regpost_2025.csv
- data/stats_team/stats_team_regpost_2026.csv
- data/stats_team/stats_team_reg_2025.csv
- data/stats_team/stats_team_reg_2026.csv
- data/stats_team/stats_team_week_2025.csv
- data/stats_team/stats_team_week_2026.csv
- data/injuries/injuries_2025.csv
- data/injuries/injuries_2026.csv
- data/depth_charts/depth_charts_2025.csv
- data/depth_charts/depth_charts_2026.csv
- data/snap_counts/snap_counts_2025.csv
- data/snap_counts/snap_counts_2026.csv
- data/nextgen_stats/ngs_passing.csv.gz
- data/nextgen_stats/ngs_receiving.csv.gz
- data/nextgen_stats/ngs_rushing.csv.gz
- data/live/espn_scoreboard.json
- data/live/sleeper_state.json
- artifacts/nfl_completed.csv
- artifacts/nfl_next_games.csv

Files skipped:
- participation was not requested and was not downloaded
- https://www.nfl.com/ajax/scorestrip?season=2026&seasonType=REG&week=4 HTML

ESPN status: 200
nfl.com status: HTML

FanGraphs, stats.nba.com, nba_api, and football-data.org were not called. Raw XY tracking was not downloaded.

This is the roster cheat sheet. It does not price a bet. The win model still uses the quarterback features only.

roster check pass, 3 passes

roster_2026_only.csv is the prop sheet. The mixed file is not. This does not price a bet.

Blank home_win_prob rows left blank: 2025_01_DAL_PHI, 2025_01_KC_LAC, 2025_01_ARI_NO, 2025_01_BAL_BUF, 2025_01_CAR_JAX, 2025_01_CIN_CLE, 2025_01_DET_GB, 2025_01_HOU_LA, 2025_01_LV_NE, 2025_01_MIA_IND, 2025_01_NYG_WAS, 2025_01_PIT_NYJ, 2025_01_SF_SEA, 2025_01_TB_ATL, 2025_01_TEN_DEN, 2025_01_MIN_CHI, 2025_02_SF_NO, 2025_03_CIN_MIN, 2025_03_LV_WAS, 2025_04_LAC_NYG, 2025_11_GB_NYG, 2025_15_IND_SEA, 2025_21_NE_DEN, 2026_01_CLE_JAX, 2026_03_PHI_CHI.

participation not downloaded

Calibration counts from artifacts/calibration.csv: 50-55 is 60, 55-60 is 56, 60-65 is 38, 65-plus is 27.
the 65-plus bin is 27 games and is not a price.
