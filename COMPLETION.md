# Completion

`scripts/build_roster.py` rebuilt the roster sheets on 2026-11-14 from the week files, snap counts, the depth chart, and the injury file. No counting number was typed in.

`artifacts/roster_2026_only.csv` has 1,790 rows from `stats_player_week_2026.csv` only. Weeks on that file are 1 through 10. Ten blank player ids were skipped. All 32 teams are present. Required positions: QB 66, RB 156, WR 224, TE 126, OL 209, DL 272, LB 282, DB 362, K 38, P 34. Every counting cell matches the week-file sum. Snap totals matched 1,667 rows by exact display name and team. The other 123 rows leave snaps blank. The depth-chart stamp is 2026-11-14T14:04:32Z, 2,329 current rows. Starter is the first player at that chart position who is not Out on the team's latest injury week. Jordan Love is Green Bay, QB, starter yes. Baker Mayfield is Tampa Bay, QB, starter yes. Jayden Daniels is Washington, QB, starter yes. Jalon Daniels is Tampa Bay, QB, starter no. None of those four has an Out row on the latest team injury week.

`artifacts/roster_cheat_sheet_2026.csv` has 2,860 rows from the 2025 and 2026 week files. Stats that do not apply to the position are blank. Applicable stats match the week-file sums.

`artifacts/depth_chart_current_2026.csv` is that latest stamp. `artifacts/player_status_2026.csv` joins it to each team's latest injury week. A blank report status means the injury file has no row.

Roster result: pass. The sheet does not price a bet.

These are not downloads. They fail on purpose:

- live book price: fail
- NBA: fail
- soccer: fail
- raw tracking: fail

No blank `home_win_prob` was filled. `2026_04_PIT_CLE` stays completed because the schedule has both scores, Pittsburgh 24 and Cleveland 27. No counting number was typed in.

The 2026-11-14 refresh retrained the win model from the nflverse files. Completed rows: 422, from 2025-09-04 through 2026-11-12. Training rows: 395. Completed rows with a blank `home_win_prob`: 25. Feature count: 18. Next games: 13, on 2026-11-15 and 2026-11-16. A later local rebuild kept those 18 features. Last-four-game form was tested on the 2026 holdout and was not adopted because accuracy fell from 0.6269 to 0.5746. The points projection uses a half-weight of prior points per game because that blend lowered holdout total error from 14.513 to 12.540. In-sample accuracy is 0.6684. Four next games are high confidence. The model does not price a bet.
