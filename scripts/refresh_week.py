"""Manual weekly refresh for the 2025-2026 NFL table.

Run once a week during the season. This file does not start a timer.
Each run downloads the public nflverse release files, makes one ESPN
scoreboard request, one Sleeper state request, and one nfl.com score-strip
request. It retrains only when the latest completed-game date has moved.
"""

from __future__ import annotations

import csv
import gzip
import json
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
NOTES = ROOT / "NOTES.md"
README = ROOT / "README.md"
MAX_PATH = ARTIFACTS / "last_max_date.txt"
RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SLEEPER_URL = "https://api.sleeper.app/v1/state/nfl"
SEASONS = {2025, 2026}
GAME_TYPES = {"REG", "WC", "DIV", "CON", "SB"}
BASE_FEATURES = (
    "home_win_rate",
    "away_win_rate",
    "home_ppg",
    "away_ppg",
    "home_rest_days",
    "away_rest_days",
)
UA = {"User-Agent": "Mozilla/5.0"}


def release_files() -> list[tuple[str, str]]:
    rows = [("schedules/games.csv", f"{RELEASE}/schedules/games.csv")]
    for kind in ("stats_player", "stats_team"):
        for name in (
            f"{kind}_post_2025.csv",
            f"{kind}_regpost_2025.csv",
            f"{kind}_regpost_2026.csv",
            f"{kind}_reg_2025.csv",
            f"{kind}_reg_2026.csv",
            f"{kind}_week_2025.csv",
            f"{kind}_week_2026.csv",
        ):
            rows.append((f"{kind}/{name}", f"{RELEASE}/{kind}/{name}"))
    for kind in ("injuries", "depth_charts", "snap_counts"):
        for season in (2025, 2026):
            name = f"{kind}_{season}.csv"
            rows.append((f"{kind}/{name}", f"{RELEASE}/{kind}/{name}"))
    for stat in ("passing", "receiving", "rushing"):
        name = f"ngs_{stat}.csv.gz"
        rows.append((f"nextgen_stats/{name}", f"{RELEASE}/nextgen_stats/{name}"))
    return rows


def download(url: str, dest: Path, saved: list[str], skipped: list[str]) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                skipped.append(f"{url} HTTP {status}")
                return False
            payload = response.read()
    except urllib.error.HTTPError as exc:
        skipped.append(f"{url} HTTP {exc.code}")
        return False
    except Exception as exc:
        skipped.append(f"{url} {type(exc).__name__}: {exc}")
        return False
    dest.write_bytes(payload)
    saved.append(dest.relative_to(ROOT).as_posix())
    print(f"saved {dest.relative_to(ROOT).as_posix()} bytes {len(payload)}", flush=True)
    return True


def fetch_text(url: str) -> tuple[int, str, bytes]:
    request = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return getattr(response, "status", 200), response.headers.get("Content-Type", ""), body
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, exc.headers.get("Content-Type", "") if exc.headers else "", body
    except Exception as exc:
        return 0, "", f"{type(exc).__name__}: {exc}".encode("utf-8")


def looks_like_html(body: bytes, content_type: str) -> bool:
    if "html" in content_type.lower():
        return True
    head = body[:400].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<html" in head


def blank(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    text = str(value).strip().lower()
    return text in {"", "na", "nan", "null", "none"}


def as_int(value):
    if blank(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != int(number):
        return None
    return int(number)


def as_date(value) -> date | None:
    if blank(value):
        return None
    text = str(value).strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return pd.read_csv(handle, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def load_schedule() -> list[dict]:
    frame = read_csv(DATA / "schedules" / "games.csv")
    games = []
    for row in frame.itertuples(index=False):
        raw = row._asdict()
        season = as_int(raw.get("season"))
        game_type = "" if blank(raw.get("game_type")) else str(raw.get("game_type")).strip()
        if season not in SEASONS or game_type not in GAME_TYPES:
            continue
        game_date = as_date(raw.get("gameday"))
        week = as_int(raw.get("week"))
        away = "" if blank(raw.get("away_team")) else str(raw.get("away_team")).strip()
        home = "" if blank(raw.get("home_team")) else str(raw.get("home_team")).strip()
        if game_date is None or week is None or not away or not home:
            continue
        games.append(
            {
                "game_id": "" if blank(raw.get("game_id")) else str(raw.get("game_id")).strip(),
                "season": season,
                "week": week,
                "game_type": game_type,
                "date": game_date,
                "away": away,
                "home": home,
                "away_score": as_int(raw.get("away_score")),
                "home_score": as_int(raw.get("home_score")),
            }
        )
    games.sort(key=lambda game: (game["date"], game["season"], game["week"], game["game_id"]))
    return games


def count_table(paths: list[Path], value_columns: list[str] | None) -> dict[tuple, float] | None:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = read_csv(path)
        if "season" not in frame.columns or "week" not in frame.columns or "team" not in frame.columns:
            continue
        frames.append(frame)
    if not frames:
        return None
    frame = pd.concat(frames, ignore_index=True)
    frame["season"] = frame["season"].map(as_int)
    frame["week"] = frame["week"].map(as_int)
    frame = frame.dropna(subset=["season", "week"])
    frame["team"] = frame["team"].astype(str).str.strip()
    present = {(int(season), int(week)) for season, week in frame[["season", "week"]].drop_duplicates().itertuples(index=False)}
    if value_columns is None:
        counts = frame.groupby(["season", "week", "team"], dropna=False).size()
        return {"present": present, "values": {key: float(value) for key, value in counts.items()}}
    columns = [column for column in value_columns if column in frame.columns]
    if not columns:
        return None
    numeric = frame[["season", "week", "team"] + columns].copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    grouped = numeric.groupby(["season", "week", "team"], dropna=False)[columns].sum(min_count=1)
    values = {}
    for key, row in grouped.iterrows():
        if row.isna().all():
            continue
        values[key] = float(row.sum(min_count=1))
    return {"present": present, "values": values}


def prior_value(table: dict | None, season: int, week: int, team: str):
    if table is None:
        return None
    key = (season, week, team)
    if (season, week) not in table["present"]:
        return None
    return table["values"].get(key, 0.0)


def features_for(state: dict, game: dict, injuries: dict | None, snaps: dict | None) -> dict:
    home = state.get(game["home"])
    away = state.get(game["away"])
    row = {name: None for name in BASE_FEATURES}
    if injuries is not None:
        row["home_prior_injuries"] = None
        row["away_prior_injuries"] = None
    if snaps is not None:
        row["home_prior_snaps"] = None
        row["away_prior_snaps"] = None
    if home and home["games"] > 0:
        row["home_win_rate"] = home["wins"] / home["games"]
        row["home_ppg"] = home["points"] / home["games"]
        row["home_rest_days"] = (game["date"] - home["date"]).days
        if injuries is not None:
            row["home_prior_injuries"] = prior_value(injuries, home["season"], home["week"], game["home"])
        if snaps is not None:
            row["home_prior_snaps"] = prior_value(snaps, home["season"], home["week"], game["home"])
    if away and away["games"] > 0:
        row["away_win_rate"] = away["wins"] / away["games"]
        row["away_ppg"] = away["points"] / away["games"]
        row["away_rest_days"] = (game["date"] - away["date"]).days
        if injuries is not None:
            row["away_prior_injuries"] = prior_value(injuries, away["season"], away["week"], game["away"])
        if snaps is not None:
            row["away_prior_snaps"] = prior_value(snaps, away["season"], away["week"], game["away"])
    return row


def update_state(state: dict, game: dict) -> None:
    home_score = game["home_score"]
    away_score = game["away_score"]
    for team, scored, allowed in (
        (game["home"], home_score, away_score),
        (game["away"], away_score, home_score),
    ):
        slot = state.setdefault(
            team,
            {"games": 0, "wins": 0, "points": 0, "date": game["date"], "season": game["season"], "week": game["week"]},
        )
        slot["games"] += 1
        slot["points"] += scored
        if scored > allowed:
            slot["wins"] += 1
        slot["date"] = game["date"]
        slot["season"] = game["season"]
        slot["week"] = game["week"]


def completed(game: dict) -> bool:
    return game["away_score"] is not None and game["home_score"] is not None


def sleeper_week(payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    week = as_int(payload.get("week"))
    if week is None:
        week = as_int(payload.get("leg"))
    season = as_int(payload.get("season"))
    if season is None:
        season = as_int(payload.get("league_season"))
    season_type = str(payload.get("season_type") or "").strip().lower()
    mapped = {"regular": "REG", "post": "POST", "pre": "PRE"}.get(season_type)
    if week is None or season is None or mapped is None:
        return None
    return {"week": week, "season": season, "season_type": mapped}


def choose_next(games: list[dict], sleeper: dict | None) -> list[dict]:
    unscored = [game for game in games if not completed(game)]
    if sleeper:
        week_games = [
            game
            for game in unscored
            if game["season"] == sleeper["season"] and game["week"] == sleeper["week"]
        ]
        if week_games:
            return week_games
    dated = [game for game in games if completed(game)]
    if not dated:
        return []
    latest = max(game["date"] for game in dated)
    future = [game for game in unscored if game["date"] > latest]
    if not future:
        return []
    first = min(future, key=lambda game: (game["date"], game["week"], game["game_id"]))
    return [game for game in future if game["season"] == first["season"] and game["week"] == first["week"]]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for column in columns:
                value = row.get(column)
                if isinstance(value, date):
                    value = value.isoformat()
                elif isinstance(value, float):
                    value = f"{value:.6f}".rstrip("0").rstrip(".")
                out[column] = "" if value is None else value
            writer.writerow(out)


def train(completed_rows: list[dict], next_rows: list[dict], features: list[str]) -> tuple[int, list[dict]]:
    trainable = []
    for row in completed_rows:
        if row["home_win"] is None:
            continue
        if any(row.get(feature) is None for feature in features):
            continue
        trainable.append(row)
    if len(trainable) < 2 or len({row["home_win"] for row in trainable}) < 2:
        print("train_rows", len(trainable), flush=True)
        return len(trainable), next_rows
    frame = pd.DataFrame(trainable)
    matrix = frame[features].to_numpy(dtype=float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (matrix - center) / scale
    target = frame["home_win"].to_numpy(dtype=int)
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(scaled, target)
    home_points = LinearRegression().fit(scaled, frame["home_score"].to_numpy(dtype=float))
    away_points = LinearRegression().fit(scaled, frame["away_score"].to_numpy(dtype=float))
    for row in next_rows:
        if any(row.get(feature) is None for feature in features):
            row["predicted_home_win"] = None
            row["predicted_away_points"] = None
            row["predicted_home_points"] = None
            continue
        sample = (np.array([row[feature] for feature in features], dtype=float) - center) / scale
        sample = sample.reshape(1, -1)
        row["predicted_home_win"] = int(ridge.predict(sample)[0])
        row["predicted_away_points"] = float(away_points.predict(sample)[0])
        row["predicted_home_points"] = float(home_points.predict(sample)[0])
    print("train_rows", len(trainable), flush=True)
    return len(trainable), next_rows


def write_readme(max_date: str) -> None:
    README.write_text(
        "# NFL Game Predictor 2026\n\n"
        f"2025 through {max_date}. nflverse is the recorded-stats source. "
        "ESPN is the live score check. Sleeper is the current week only. "
        "nfl.com is a one-request check. Real is not a source. "
        "Weekly refresh is manual. This does not price a bet.\n\n"
        "Run `python scripts/refresh_week.py` once a week during the season.\n",
        encoding="utf-8",
        newline="\n",
    )


def write_notes(lines: list[str]) -> None:
    NOTES.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    saved: list[str] = []
    skipped: list[str] = ["participation was not requested and was not downloaded"]
    for relative, url in release_files():
        download(url, DATA / relative, saved, skipped)

    espn_status, espn_type, espn_body = fetch_text(ESPN_URL)
    espn_path = DATA / "live" / "espn_scoreboard.json"
    if espn_status == 200 and not looks_like_html(espn_body, espn_type):
        espn_path.parent.mkdir(parents=True, exist_ok=True)
        espn_path.write_bytes(espn_body)
        saved.append(espn_path.relative_to(ROOT).as_posix())
        print("espn 200", flush=True)
    else:
        detail = "HTML" if looks_like_html(espn_body, espn_type) else espn_type or "not JSON"
        skipped.append(f"{ESPN_URL} HTTP {espn_status} {detail}".strip())
        print(f"espn {espn_status}", flush=True)

    sleeper_status, sleeper_type, sleeper_body = fetch_text(SLEEPER_URL)
    sleeper_path = DATA / "live" / "sleeper_state.json"
    sleeper = None
    if sleeper_status == 200 and not looks_like_html(sleeper_body, sleeper_type):
        sleeper_path.parent.mkdir(parents=True, exist_ok=True)
        sleeper_path.write_bytes(sleeper_body)
        saved.append(sleeper_path.relative_to(ROOT).as_posix())
        try:
            sleeper = sleeper_week(json.loads(sleeper_body.decode("utf-8")))
        except json.JSONDecodeError:
            sleeper = None
        print("sleeper 200", sleeper, flush=True)
    else:
        detail = "HTML" if looks_like_html(sleeper_body, sleeper_type) else sleeper_type or "not JSON"
        skipped.append(f"{SLEEPER_URL} HTTP {sleeper_status} {detail}".strip())
        print(f"sleeper {sleeper_status}", flush=True)

    nfl_status = "not requested"
    nfl_url = ""
    if sleeper:
        nfl_url = (
            "https://www.nfl.com/ajax/scorestrip"
            f"?season={sleeper['season']}&seasonType={sleeper['season_type']}&week={sleeper['week']}"
        )
        nfl_status_code, nfl_type, nfl_body = fetch_text(nfl_url)
        nfl_path = DATA / "live" / "nfl_scorestrip.json"
        if nfl_status_code in {403, 404} or looks_like_html(nfl_body, nfl_type):
            kind = "HTML" if looks_like_html(nfl_body, nfl_type) else str(nfl_status_code)
            nfl_status = kind
            skipped.append(f"{nfl_url} {kind}")
        elif nfl_status_code == 200:
            text = nfl_body.decode("utf-8", errors="replace").lstrip()
            if text.startswith("{") or text.startswith("["):
                nfl_path.parent.mkdir(parents=True, exist_ok=True)
                nfl_path.write_bytes(nfl_body)
                saved.append(nfl_path.relative_to(ROOT).as_posix())
                nfl_status = "200 JSON"
            else:
                nfl_status = "200 not JSON"
                skipped.append(f"{nfl_url} HTTP 200 but the body was not JSON")
        else:
            nfl_status = str(nfl_status_code)
            skipped.append(f"{nfl_url} HTTP {nfl_status_code}")
        print("nfl.com", nfl_status, flush=True)
    else:
        skipped.append("nfl.com score strip was not requested because the Sleeper state did not include a week")
        print("nfl.com skipped", flush=True)

    schedule_path = DATA / "schedules" / "games.csv"
    if not schedule_path.exists():
        write_notes(
            [
                "Run python scripts/refresh_week.py once a week during the season. It does not run on a timer.",
                "",
                "The schedule file was not saved, so no training table was written.",
                "",
                "Files saved:",
                *([f"- {item}" for item in saved] or ["- none"]),
                "",
                "Files skipped:",
                *([f"- {item}" for item in skipped] or ["- none"]),
                "",
                f"ESPN status: {espn_status}",
                f"nfl.com status: {nfl_status}",
                "",
                "Real is not a source. No odds and no props were used. FanGraphs, stats.nba.com, nba_api, and football-data.org were not called.",
            ]
        )
        print("max_date missing", flush=True)
        return 1

    games = load_schedule()
    done = [game for game in games if completed(game)]
    if not done:
        print("max_date missing", flush=True)
        return 1
    min_date = min(game["date"] for game in done)
    max_date = max(game["date"] for game in done)
    print("min_date", min_date.isoformat(), flush=True)
    print("max_date", max_date.isoformat(), flush=True)
    previous = MAX_PATH.read_text(encoding="utf-8").strip() if MAX_PATH.exists() else ""
    if previous == max_date.isoformat() and (ARTIFACTS / "nfl_completed.csv").exists():
        print("retrain skipped", flush=True)
        print("rows kept", flush=True)
        return 0

    injury_paths = [DATA / "injuries" / "injuries_2025.csv", DATA / "injuries" / "injuries_2026.csv"]
    snap_paths = [DATA / "snap_counts" / "snap_counts_2025.csv", DATA / "snap_counts" / "snap_counts_2026.csv"]
    injuries = count_table([path for path in injury_paths if path.exists()], None)
    snaps = count_table(
        [path for path in snap_paths if path.exists()],
        ["offense_snaps", "defense_snaps", "st_snaps"],
    )
    if injuries is None:
        skipped.append("prior-week injury counts were left out because no injury file with season, week, and team was saved")
    if snaps is None:
        skipped.append("prior-week snap counts were left out because no snap file with offense, defense, or special-teams snaps was saved")

    features = list(BASE_FEATURES)
    if injuries is not None:
        features.extend(["home_prior_injuries", "away_prior_injuries"])
    if snaps is not None:
        features.extend(["home_prior_snaps", "away_prior_snaps"])

    next_ids = {game["game_id"] for game in choose_next(games, sleeper)}
    state: dict = {}
    completed_rows = []
    next_rows = []
    for game in games:
        feature_row = features_for(state, game, injuries, snaps)
        if completed(game):
            home_win = None
            if game["home_score"] > game["away_score"]:
                home_win = 1
            elif game["away_score"] > game["home_score"]:
                home_win = 0
            completed_rows.append({**game, **feature_row, "home_win": home_win})
            update_state(state, game)
        elif game["game_id"] in next_ids:
            next_rows.append({**game, **feature_row})

    train_rows, next_rows = train(completed_rows, next_rows, features)
    id_columns = ["game_id", "date", "season", "week", "game_type", "away", "home", "away_score", "home_score", "home_win"]
    write_csv(ARTIFACTS / "nfl_completed.csv", completed_rows, id_columns + features)
    next_columns = [
        "game_id",
        "date",
        "season",
        "week",
        "game_type",
        "away",
        "home",
        *features,
        "predicted_home_win",
        "predicted_away_points",
        "predicted_home_points",
    ]
    write_csv(ARTIFACTS / "nfl_next_games.csv", next_rows, next_columns)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MAX_PATH.write_text(max_date.isoformat() + "\n", encoding="utf-8", newline="\n")
    saved.extend(["artifacts/nfl_completed.csv", "artifacts/nfl_next_games.csv"])
    write_readme(max_date.isoformat())
    date_ok = date(2025, 9, 1) <= min_date <= date(2026, 2, 28) and max_date >= date(2026, 9, 28)
    write_notes(
        [
            "Run python scripts/refresh_week.py once a week during the season. It does not run on a timer.",
            "",
            "nflverse is the recorded-stats source. ESPN is a live score check and is not copied into the training scores. Sleeper supplies the current week only. nfl.com is one score-strip request. Real is not a source.",
            "",
            "The training rows use only earlier games: win rate, points per game, rest days, and prior-week injury and snap counts when those files saved. A game is not a feature of itself. The win model is a ridge classifier. Points use linear regression. No odds and no props.",
            "",
            f"Completed rows: {len(completed_rows)}",
            f"Training rows: {train_rows}",
            f"Feature count: {len(features)}",
            f"Minimum date: {min_date.isoformat()}",
            f"Maximum date: {max_date.isoformat()}",
            f"Date check: {'pass' if date_ok else 'fail'}",
            "",
            "Files saved:",
            *([f"- {item}" for item in saved] or ["- none"]),
            "",
            "Files skipped:",
            *([f"- {item}" for item in skipped] or ["- none"]),
            "",
            f"ESPN status: {espn_status}",
            f"nfl.com status: {nfl_status}",
            "",
            "FanGraphs, stats.nba.com, nba_api, and football-data.org were not called. Raw XY tracking was not downloaded.",
        ]
    )
    print("rows", len(completed_rows), flush=True)
    print("feature_count", len(features), flush=True)
    print("date_check", "pass" if date_ok else "fail", flush=True)
    return 0 if date_ok and train_rows > 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
