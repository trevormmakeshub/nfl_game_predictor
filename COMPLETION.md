# Completion

The full scan compared all 1,432 rows in `artifacts/roster_2026_only.csv` with `data/stats_player/stats_player_week_2026.csv`, `stats_player_reg_2026.csv`, and `stats_player_regpost_2026.csv`. No 2025 row is in the roster file. All 32 teams are present.

The weekly totals and the two season files agree on every counting stat for those players. The first scan found 20,842 counting cells that did not match the 2026 source. Those cells were replaced from the source file. The full rescan found 0 mismatches.

Named rows from the 2026 source: Jordan Love is Green Bay, QB. Baker Mayfield is the Tampa Bay starter, QB. Jayden Daniels is Washington, QB. Jalon Daniels is Tampa Bay, QB, and is not the starter. Jaylen Warren has 38 carries and 216 rushing yards. Aaron Rodgers has 700 passing yards. DK Metcalf has 11 receptions and 98 receiving yards.

Roster result: pass.

These are not downloads. They fail on purpose:

- live book price: fail
- NBA: fail
- soccer: fail
- raw tracking: fail

No blank `home_win_prob` was filled. `2026_04_PIT_CLE` was not marked final. The win model was not retrained. No counting number was typed in.
