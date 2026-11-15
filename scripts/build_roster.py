"""Rebuild the roster sheets from the nflverse files already on disk.

Week stats, snap counts, the latest depth-chart stamp, and each team's
latest injury week are the only inputs. A missing snap stays blank. An
injury status is blank when that week's file has no row. This does not
download, invent a starter, or price a bet.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEEK_2026 = ROOT / "data" / "stats_player" / "stats_player_week_2026.csv"
WEEK_2025 = ROOT / "data" / "stats_player" / "stats_player_week_2025.csv"
SNAPS = ROOT / "data" / "snap_counts" / "snap_counts_2026.csv"
DEPTH = ROOT / "data" / "depth_charts" / "depth_charts_2026.csv"
INJURIES = ROOT / "data" / "injuries" / "injuries_2026.csv"
ROSTER_2026 = ROOT / "artifacts" / "roster_2026_only.csv"
CHEAT = ROOT / "artifacts" / "roster_cheat_sheet_2026.csv"
DEPTH_CURRENT = ROOT / "artifacts" / "depth_chart_current_2026.csv"
STATUS_PATH = ROOT / "artifacts" / "player_status_2026.csv"

REQUIRED = ("QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "K", "P")
COUNT_COLUMNS = (
    "attempts",
    "completions",
    "passing yards",
    "passing touchdowns",
    "interceptions",
    "carries",
    "rushing yards",
    "rushing touchdowns",
    "targets",
    "receptions",
    "receiving yards",
    "receiving touchdowns",
    "tackles",
    "sacks",
    "field goals",
    "extra points",
    "punts",
    "punt yards",
)
APPLIES = {
    "QB": ("attempts", "completions", "passing yards", "passing touchdowns", "interceptions", "carries", "rushing yards", "rushing touchdowns"),
    "RB": ("carries", "rushing yards", "rushing touchdowns", "targets", "receptions", "receiving yards", "receiving touchdowns"),
    "WR": ("carries", "rushing yards", "rushing touchdowns", "targets", "receptions", "receiving yards", "receiving touchdowns"),
    "TE": ("carries", "rushing yards", "rushing touchdowns", "targets", "receptions", "receiving yards", "receiving touchdowns"),
    "DL": ("tackles", "sacks"),
    "LB": ("tackles", "sacks"),
    "DB": ("tackles", "sacks"),
    "K": ("field goals", "extra points"),
    "P": ("punts", "punt yards"),
}
SOURCE = {
    "attempts": "attempts",
    "completions": "completions",
    "passing yards": "passing_yards",
    "passing touchdowns": "passing_tds",
    "interceptions": "passing_interceptions",
    "carries": "carries",
    "rushing yards": "rushing_yards",
    "rushing touchdowns": "rushing_tds",
    "targets": "targets",
    "receptions": "receptions",
    "receiving yards": "receiving_yards",
    "receiving touchdowns": "receiving_tds",
    "sacks": "def_sacks",
    "field goals": "fg_made",
    "extra points": "pat_made",
    "punts": "pt_att",
    "punt yards": "pt_yards",
}
WEEK_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "position_group",
    "season",
    "week",
    "game_id",
    "team",
    *SOURCE.values(),
    "def_tackles_solo",
    "def_tackle_assists",
]


def cell(value: object) -> str:
    if value is None or value == "":
        return ""
    if pd.isna(value):
        return ""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def normalize_position(position: object, group: object) -> str:
    position_text = "" if pd.isna(position) else str(position).strip()
    group_text = "" if pd.isna(group) else str(group).strip()
    if position_text in {"K", "P"}:
        return position_text
    if position_text in REQUIRED:
        return position_text
    if group_text in REQUIRED:
        return group_text
    return position_text or group_text


def load_weeks(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=WEEK_COLUMNS)
    frame["player_id"] = frame["player_id"].fillna("").astype(str).str.strip()
    frame = frame[frame["player_id"] != ""].copy()
    for column in (*SOURCE.values(), "def_tackles_solo", "def_tackle_assists"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["tackles"] = frame["def_tackles_solo"] + frame["def_tackle_assists"]
    return frame


def aggregate(frame: pd.DataFrame, blank_unused: bool) -> list[dict]:
    rows = []
    for (player_id, team), group in frame.groupby(["player_id", "team"], sort=False):
        latest = group.sort_values(["season", "week"]).iloc[-1]
        position = normalize_position(latest["position"], latest["position_group"])
        record = {
            "name": "" if pd.isna(latest["player_display_name"]) else str(latest["player_display_name"]),
            "team": team,
            "position": position,
            "games": str(group["game_id"].nunique()),
            "player_id": player_id,
        }
        totals = {name: float(group[source].sum()) if name != "tackles" else float(group["tackles"].sum()) for name, source in (*SOURCE.items(), ("tackles", "tackles"))}
        allowed = set(APPLIES.get(position, ()))
        for name in COUNT_COLUMNS:
            if blank_unused and name not in allowed:
                record[name] = ""
            else:
                record[name] = cell(totals[name])
        rows.append(record)
    rows.sort(key=lambda row: (row["team"], row["position"], row["name"], row["player_id"]))
    return rows


def latest_injuries(injuries: pd.DataFrame) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str], dict[str, int]]:
    injuries = injuries.copy()
    injuries["gsis_id"] = injuries["gsis_id"].fillna("").astype(str).str.strip()
    latest_week = injuries.groupby("team")["week"].max().to_dict()
    status: dict[tuple[str, str], str] = {}
    detail: dict[tuple[str, str], str] = {}
    for team, week in latest_week.items():
        group = injuries[(injuries["team"] == team) & (injuries["week"] == week)]
        for gsis_id, rows in group.groupby("gsis_id"):
            if not gsis_id:
                continue
            texts = ["" if pd.isna(value) else str(value).strip() for value in rows["report_status"]]
            chosen = "Out" if any(text.lower() == "out" for text in texts) else next((text for text in texts if text), "")
            status[(team, gsis_id)] = chosen
            primary = rows["report_primary_injury"].dropna().astype(str)
            detail[(team, gsis_id)] = primary.iloc[0] if not primary.empty else ""
    return status, detail, {team: int(week) for team, week in latest_week.items()}


def active_starters(depth: pd.DataFrame, status: dict[tuple[str, str], str]) -> set[tuple[str, str, str]]:
    """First player at each team and chart position who is not Out."""
    starters: set[tuple[str, str, str]] = set()
    chart = depth[depth["gsis_id"].fillna("").astype(str).str.strip() != ""].copy()
    chart["gsis_id"] = chart["gsis_id"].astype(str).str.strip()
    for (team, pos_abb), group in chart.groupby(["team", "pos_abb"], sort=False):
        ordered = group.sort_values(["pos_rank", "pos_slot", "player_name"])
        for _, row in ordered.iterrows():
            if status.get((team, row["gsis_id"]), "").lower() == "out":
                continue
            starters.add((team, pos_abb, row["gsis_id"]))
            break
    return starters


def main() -> None:
    weeks_2026 = load_weeks(WEEK_2026)
    weeks_2025 = load_weeks(WEEK_2025)
    blank_2026 = int((pd.read_csv(WEEK_2026, usecols=["player_id"])["player_id"].fillna("").astype(str).str.strip() == "").sum())
    blank_2025 = int((pd.read_csv(WEEK_2025, usecols=["player_id"])["player_id"].fillna("").astype(str).str.strip() == "").sum())
    roster_rows = aggregate(weeks_2026, blank_unused=False)
    cheat_rows = aggregate(pd.concat([weeks_2025, weeks_2026], ignore_index=True), blank_unused=True)

    snaps = pd.read_csv(SNAPS, usecols=["player", "team", "offense_snaps", "defense_snaps", "st_snaps"])
    for column in ("offense_snaps", "defense_snaps", "st_snaps"):
        snaps[column] = pd.to_numeric(snaps[column], errors="coerce").fillna(0.0)
    snap_totals = snaps.groupby(["player", "team"], as_index=False)[["offense_snaps", "defense_snaps", "st_snaps"]].sum()
    snap_index = {
        (row.player, row.team): row
        for row in snap_totals.itertuples(index=False)
    }

    depth = pd.read_csv(DEPTH)
    stamp = str(depth["dt"].max())
    current = depth[depth["dt"] == stamp].copy()
    current.to_csv(DEPTH_CURRENT, index=False)
    injuries = pd.read_csv(INJURIES, usecols=["team", "week", "gsis_id", "full_name", "report_status", "report_primary_injury"])
    status, detail, latest_week = latest_injuries(injuries)
    starter_slots = active_starters(current, status)
    starter_ids = {(team, gsis) for team, _pos, gsis in starter_slots}

    matched_snaps = 0
    for row in roster_rows:
        found = snap_index.get((row["name"], row["team"]))
        if found is None:
            row["offense_snaps"] = ""
            row["defense_snaps"] = ""
            row["st_snaps"] = ""
        else:
            matched_snaps += 1
            row["offense_snaps"] = cell(found.offense_snaps)
            row["defense_snaps"] = cell(found.defense_snaps)
            row["st_snaps"] = cell(found.st_snaps)
        row["starter"] = "yes" if (row["team"], row["player_id"]) in starter_ids else "no"

    status_rows = []
    for _, row in current.sort_values(["team", "pos_abb", "pos_rank", "pos_slot", "player_name"]).iterrows():
        gsis = "" if pd.isna(row["gsis_id"]) else str(row["gsis_id"]).strip()
        key = (row["team"], gsis)
        status_rows.append(
            {
                "team": row["team"],
                "player_name": row["player_name"],
                "gsis_id": gsis,
                "pos_abb": row["pos_abb"],
                "pos_name": row["pos_name"],
                "pos_rank": cell(row["pos_rank"]),
                "pos_slot": cell(row["pos_slot"]),
                "depth_dt": stamp,
                "injury_week": "" if not gsis else str(latest_week.get(row["team"], "")),
                "report_status": status.get(key, ""),
                "report_primary_injury": detail.get(key, ""),
                "starter": "yes" if (row["team"], row["pos_abb"], gsis) in starter_slots else "no",
            }
        )

    roster_columns = ["name", "team", "position", "games", *COUNT_COLUMNS, "offense_snaps", "defense_snaps", "st_snaps", "starter"]
    cheat_columns = ["name", "team", "position", "games", *COUNT_COLUMNS]
    for row in roster_rows:
        row.pop("player_id", None)
    cheat_public = []
    for row in cheat_rows:
        cheat_public.append({column: row[column] for column in cheat_columns})
    write_csv(ROSTER_2026, roster_rows, roster_columns)
    write_csv(CHEAT, cheat_public, cheat_columns)
    write_csv(
        STATUS_PATH,
        status_rows,
        ["team", "player_name", "gsis_id", "pos_abb", "pos_name", "pos_rank", "pos_slot", "depth_dt", "injury_week", "report_status", "report_primary_injury", "starter"],
    )

    problems = validate(weeks_2026, weeks_2025, roster_rows, cheat_rows, snap_index, starter_ids)
    teams = sorted({row["team"] for row in roster_rows})
    positions = {name: sum(row["position"] == name for row in roster_rows) for name in REQUIRED}
    print(f"depth stamp {stamp}")
    print(f"2026 rows {len(roster_rows)} teams {len(teams)} blank player ids skipped {blank_2026}")
    print(f"cheat rows {len(cheat_rows)} blank player ids skipped {blank_2025 + blank_2026}")
    print("2026 positions " + ", ".join(f"{name} {positions[name]}" for name in REQUIRED))
    print(f"snap matches {matched_snaps} of {len(roster_rows)}")
    print(f"current depth rows {len(current)} status rows {len(status_rows)}")
    show_named(roster_rows, weeks_2026, status)
    if problems:
        raise SystemExit("roster check failed: " + "; ".join(problems))
    print("roster check pass")


def show_named(rows: list[dict], weeks: pd.DataFrame, status: dict[tuple[str, str], str]) -> None:
    wanted = {
        "00-0036264": "Jordan Love",
        "00-0034855": "Baker Mayfield",
        "00-0039910": "Jayden Daniels",
        "00-0041251": "Jalon Daniels",
    }
    by_id = weeks.groupby("player_id")
    public = {(row["name"], row["team"]): row for row in rows}
    for player_id, label in wanted.items():
        if player_id not in by_id.groups:
            print(f"missing source row {label}")
            continue
        group = by_id.get_group(player_id).sort_values(["week"])
        latest = group.iloc[-1]
        key = (str(latest["player_display_name"]), str(latest["team"]))
        row = public.get(key)
        report = status.get((str(latest["team"]), player_id), "")
        starter = "" if row is None else row["starter"]
        print(f"{label} {latest['team']} {latest['position']} starter {starter} injury {report or 'blank'}")


def validate(
    weeks_2026: pd.DataFrame,
    weeks_2025: pd.DataFrame,
    roster_rows: list[dict],
    cheat_rows: list[dict],
    snap_index: dict,
    starters: set[tuple[str, str]],
) -> list[str]:
    problems = []
    teams = {row["team"] for row in roster_rows}
    if len(teams) != 32:
        problems.append(f"2026 teams {len(teams)}")
    cheat_teams = {row["team"] for row in cheat_rows}
    if len(cheat_teams) != 32:
        problems.append(f"cheat teams {len(cheat_teams)}")
    for name in REQUIRED:
        if not any(row["position"] == name for row in roster_rows):
            problems.append(f"2026 missing {name}")
        if not any(row["position"] == name for row in cheat_rows):
            problems.append(f"cheat missing {name}")
    if len(roster_rows) != len({(row["name"], row["team"], row["position"], row["games"]) for row in roster_rows}):
        problems.append("duplicate 2026 name rows")

    source_ids = set(zip(weeks_2026["player_id"], weeks_2026["team"]))
    built_ids = set()
    grouped = weeks_2026.groupby(["player_id", "team"])
    for row_key, group in grouped:
        built_ids.add(row_key)
        latest = group.sort_values(["season", "week"]).iloc[-1]
        position = normalize_position(latest["position"], latest["position_group"])
        name = str(latest["player_display_name"])
        match = next((row for row in roster_rows if row["name"] == name and row["team"] == row_key[1] and row["position"] == position), None)
        if match is None:
            problems.append(f"missing roster row {name} {row_key[1]}")
            if len(problems) > 8:
                return problems
            continue
        if int(match["games"]) != group["game_id"].nunique():
            problems.append(f"games mismatch {name}")
        for column, source in SOURCE.items():
            if match[column] != cell(group[source].sum()):
                problems.append(f"stat mismatch {name} {column}")
                break
        if match["tackles"] != cell(group["tackles"].sum()):
            problems.append(f"tackle mismatch {name}")
        found = snap_index.get((name, row_key[1]))
        if found is None:
            if any(match[column] != "" for column in ("offense_snaps", "defense_snaps", "st_snaps")):
                problems.append(f"snap filled without a match {name}")
        elif match["offense_snaps"] != cell(found.offense_snaps) or match["defense_snaps"] != cell(found.defense_snaps) or match["st_snaps"] != cell(found.st_snaps):
            problems.append(f"snap mismatch {name}")
        expected = "yes" if (row_key[1], row_key[0]) in starters else "no"
        if match["starter"] != expected:
            problems.append(f"starter mismatch {name}")
    if built_ids != source_ids:
        problems.append("player coverage mismatch")

    both = pd.concat([weeks_2025, weeks_2026], ignore_index=True)
    for (player_id, team), group in both.groupby(["player_id", "team"]):
        latest = group.sort_values(["season", "week"]).iloc[-1]
        position = normalize_position(latest["position"], latest["position_group"])
        name = str(latest["player_display_name"])
        match = next((row for row in cheat_rows if row["name"] == name and row["team"] == team and row["position"] == position), None)
        if match is None:
            problems.append(f"missing cheat row {name} {team}")
            if len(problems) > 12:
                return problems
            continue
        allowed = set(APPLIES.get(position, ()))
        for column, source in (*SOURCE.items(), ("tackles", "tackles")):
            expected = "" if column not in allowed else cell(group[source].sum() if column != "tackles" else group["tackles"].sum())
            if match[column] != expected:
                problems.append(f"cheat mismatch {name} {column}")
                break
        if len(problems) > 12:
            return problems
    love = weeks_2026[weeks_2026["player_id"] == "00-0036264"]
    if love.empty or set(love["team"]) != {"GB"}:
        problems.append("Jordan Love is not Green Bay in the 2026 week file")
    return problems


if __name__ == "__main__":
    main()
