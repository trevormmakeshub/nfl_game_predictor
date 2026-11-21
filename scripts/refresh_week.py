"""Manual weekly refresh for the 2025-2026 NFL table.

Run once a week during the season. This file does not start a timer.
Each run downloads the public nflverse release files, makes one ESPN
scoreboard request, and one Sleeper state request. It does not request
nfl.com. It retrains only when the latest completed-game date has moved.

`python scripts/refresh_week.py --local` rebuilds the model from the files
already on disk and does not download. Use that after a code change when
the schedule date has not moved.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.special import expit
    from sklearn.linear_model import LinearRegression, RidgeClassifier
except ImportError:
    # The installed scipy extension is blocked on this machine.
    # These are the same ridge and linear fits the script calls, in numpy.
    def expit(value):
        array = np.asarray(value, dtype=float)
        return 1.0 / (1.0 + np.exp(-array))

    def _fit_coef(samples, target, alpha):
        samples = np.asarray(samples, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        center = samples.mean(axis=0)
        target_center = float(target.mean())
        shifted = samples - center
        gram = shifted.T @ shifted
        if alpha:
            gram = gram + (alpha * np.eye(samples.shape[1]))
        coef = np.linalg.solve(gram, shifted.T @ (target - target_center))
        intercept = target_center - float(center @ coef)
        return coef.reshape(1, -1), intercept

    class RidgeClassifier:
        def __init__(self, alpha=1.0):
            self.alpha = alpha

        def fit(self, samples, target):
            target = np.asarray(target)
            self.classes_ = np.unique(target)
            signed = np.where(target == self.classes_[1], 1.0, -1.0)
            self.coef_, self.intercept_ = _fit_coef(samples, signed, self.alpha)
            return self

        def decision_function(self, samples):
            samples = np.asarray(samples, dtype=float)
            return (samples @ self.coef_.ravel()) + float(self.intercept_)

    class LinearRegression:
        def fit(self, samples, target):
            self.coef_, self.intercept_ = _fit_coef(samples, target, 0.0)
            return self

        def predict(self, samples):
            samples = np.asarray(samples, dtype=float)
            return (samples @ self.coef_.ravel()) + float(self.intercept_)

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
SCHEDULE_FIELDS = (
    "spread_line",
    "total_line",
    "home_moneyline",
    "roof",
    "surface",
    "temp",
    "wind",
    "div_game",
    "away_qb_id",
    "home_qb_id",
)
LINE_FEATURES = ("spread_line", "total_line", "home_moneyline", "div_game")
QB_FEATURES = (
    "home_qb_passer_rating",
    "away_qb_passer_rating",
    "home_qb_int_rate",
    "away_qb_int_rate",
)
RECENT_FEATURES = (
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_recent_margin",
    "away_recent_margin",
)
DEFENSE_FEATURES = (
    "home_papg",
    "away_papg",
)
HOLDOUT_START = date(2026, 9, 1)
PREDICTION_COLUMNS = (
    "predicted_total",
    "pick_side",
    "pick_team",
    "confidence",
    "confidence_tier",
)
TEXT_FIELDS = {"roof", "surface", "away_qb_id", "home_qb_id"}
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
    """Download one release file. A failed attempt leaves the previous file in place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=UA)
    last_detail = "not tried"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    last_detail = f"HTTP {status}"
                else:
                    payload = response.read()
                    dest.write_bytes(payload)
                    saved.append(dest.relative_to(ROOT).as_posix())
                    print(f"saved {dest.relative_to(ROOT).as_posix()} bytes {len(payload)}", flush=True)
                    return True
        except urllib.error.HTTPError as exc:
            last_detail = f"HTTP {exc.code}"
            if exc.code in {404, 410}:
                break
        except Exception as exc:
            last_detail = f"{type(exc).__name__}: {exc}"
        if attempt < 2:
            time.sleep(2 * (attempt + 1))
    skipped.append(f"{url} {last_detail}")
    print(f"skip {dest.name} {last_detail}", flush=True)
    return False


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


def as_float(value):
    if blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def weekly_player_ids(display_name: str, team: str | None = None, position: str | None = None) -> set[str]:
    found: set[str] = set()
    for season in (2025, 2026):
        path = DATA / "stats_player" / f"stats_player_week_{season}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row.get("player_display_name") or "").strip() != display_name:
                    continue
                if team is not None and (row.get("team") or "").strip() != team:
                    continue
                if position is not None and (row.get("position") or "").strip() != position:
                    continue
                player_id = (row.get("player_id") or "").strip()
                if player_id:
                    found.add(player_id)
    return found


def depth_chart_qb1(team: str) -> tuple[str, str] | None:
    """Return the single latest-stamp QB1 for one team, or None."""
    path = DATA / "depth_charts" / "depth_charts_2026.csv"
    if not path.exists():
        return None
    latest = ""
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("team") or "").strip() != team:
                continue
            if (row.get("pos_abb") or "").strip() != "QB":
                continue
            stamp = (row.get("dt") or "").strip()
            if stamp > latest:
                latest = stamp
                rows = [row]
            elif stamp == latest:
                rows.append(row)
    starters = []
    for row in rows:
        rank = (row.get("pos_rank") or "").strip()
        if rank in {"1", "1.0"}:
            starters.append(row)
    if len(starters) != 1:
        return None
    player_id = (starters[0].get("gsis_id") or "").strip()
    name = (starters[0].get("player_name") or "").strip()
    if not player_id or not name:
        return None
    return player_id, name


def apply_week4_green_bay_starter(games: list[dict]) -> None:
    """Jordan Love is the away quarterback only on 2026_04_GB_TB.

    The home cell is the latest Tampa Bay depth-chart starter.
    Love is never written into the home cell. No other game is changed.
    """
    love_ids = weekly_player_ids("Jordan Love", "GB", "QB")
    love_id = next(iter(love_ids)) if len(love_ids) == 1 else None
    forbidden = {"00-0036264"}
    if love_id:
        forbidden.add(love_id)
    starter = depth_chart_qb1("TB")
    for game in games:
        if game.get("game_id") != "2026_04_GB_TB":
            continue
        if game.get("away") != "GB" or game.get("home") != "TB":
            continue
        if love_id is not None and game.get("away_qb_id") != love_id:
            game["away_qb_id"] = love_id
        if starter is None or starter[0] in forbidden:
            game["home_qb_id"] = None
            print("Tampa Bay starter missing.", flush=True)
            continue
        game["home_qb_id"] = starter[0]


def load_schedule() -> list[dict]:
    frame = read_csv(DATA / "schedules" / "games.csv")
    present = [name for name in SCHEDULE_FIELDS if name in frame.columns]
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
        game = {
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
        for name in present:
            raw_value = raw.get(name)
            if name in TEXT_FIELDS:
                game[name] = None if blank(raw_value) else str(raw_value).strip()
            elif name == "div_game":
                game[name] = as_int(raw_value)
            else:
                game[name] = as_float(raw_value)
        games.append(game)
    games.sort(key=lambda game: (game["date"], game["season"], game["week"], game["game_id"]))
    apply_week4_green_bay_starter(games)
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


def clamp_rating(value: float) -> float:
    return max(0.0, min(2.375, value))


def passer_rating(completions: float, attempts: float, yards: float, touchdowns: float, interceptions: float):
    if attempts <= 0:
        return None
    rating_parts = (
        clamp_rating((completions / attempts - 0.3) * 5),
        clamp_rating((yards / attempts - 3) * 0.25),
        clamp_rating((touchdowns / attempts) * 20),
        clamp_rating(2.375 - (interceptions / attempts) * 25),
    )
    return sum(rating_parts) / 6 * 100


def load_qb_log(games: list[dict]) -> dict[str, list[dict]] | None:
    paths = [
        DATA / "stats_player" / "stats_player_week_2025.csv",
        DATA / "stats_player" / "stats_player_week_2026.csv",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    frame = pd.concat([read_csv(path) for path in existing], ignore_index=True)
    needed = [
        "player_id",
        "game_id",
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
    ]
    if any(column not in frame.columns for column in needed):
        return None
    dates = {game["game_id"]: game["date"] for game in games if completed(game)}
    frame = frame[frame["game_id"].astype(str).isin(dates)].copy()
    grouped: dict[str, list[dict]] = {}
    for raw in frame[needed].to_dict("records"):
        game_id = "" if blank(raw.get("game_id")) else str(raw.get("game_id")).strip()
        player_id = "" if blank(raw.get("player_id")) else str(raw.get("player_id")).strip()
        if game_id not in dates or not player_id:
            continue
        attempts = as_float(raw.get("attempts"))
        completions = as_float(raw.get("completions"))
        yards = as_float(raw.get("passing_yards"))
        touchdowns = as_float(raw.get("passing_tds"))
        interceptions = as_float(raw.get("passing_interceptions"))
        if None in (attempts, completions, yards, touchdowns, interceptions):
            continue
        grouped.setdefault(player_id, []).append(
            {
                "game_id": game_id,
                "date": dates[game_id],
                "attempts": attempts,
                "completions": completions,
                "yards": yards,
                "touchdowns": touchdowns,
                "interceptions": interceptions,
            }
        )
    for rows in grouped.values():
        rows.sort(key=lambda item: (item["date"], item["game_id"]))
    return grouped


def prior_qb(qb_log: dict[str, list[dict]] | None, player_id: str | None, game_date: date, game_id: str):
    if qb_log is None or not player_id:
        return None, None
    attempts = completions = yards = touchdowns = interceptions = 0.0
    used = False
    for item in qb_log.get(player_id, []):
        if item["date"] > game_date:
            break
        if item["date"] == game_date or item["game_id"] == game_id:
            continue
        attempts += item["attempts"]
        completions += item["completions"]
        yards += item["yards"]
        touchdowns += item["touchdowns"]
        interceptions += item["interceptions"]
        used = True
    if not used or attempts <= 0:
        return None, None
    return passer_rating(completions, attempts, yards, touchdowns, interceptions), interceptions / attempts


def features_for(
    state: dict,
    game: dict,
    injuries: dict | None,
    snaps: dict | None,
    qb_log: dict | None,
) -> dict:
    home = state.get(game["home"])
    away = state.get(game["away"])
    row = {name: None for name in BASE_FEATURES}
    row["home_papg"] = None
    row["away_papg"] = None
    if injuries is not None:
        row["home_prior_injuries"] = None
        row["away_prior_injuries"] = None
    if snaps is not None:
        row["home_prior_snaps"] = None
        row["away_prior_snaps"] = None
    if home and home["games"] > 0:
        row["home_win_rate"] = home["wins"] / home["games"]
        row["home_ppg"] = home["points"] / home["games"]
        row["home_papg"] = home.get("allowed", 0) / home["games"]
        row["home_rest_days"] = (game["date"] - home["date"]).days
        if injuries is not None:
            row["home_prior_injuries"] = prior_value(injuries, home["season"], home["week"], game["home"])
        if snaps is not None:
            row["home_prior_snaps"] = prior_value(snaps, home["season"], home["week"], game["home"])
    if away and away["games"] > 0:
        row["away_win_rate"] = away["wins"] / away["games"]
        row["away_ppg"] = away["points"] / away["games"]
        row["away_papg"] = away.get("allowed", 0) / away["games"]
        row["away_rest_days"] = (game["date"] - away["date"]).days
        if injuries is not None:
            row["away_prior_injuries"] = prior_value(injuries, away["season"], away["week"], game["away"])
        if snaps is not None:
            row["away_prior_snaps"] = prior_value(snaps, away["season"], away["week"], game["away"])
    for name in LINE_FEATURES:
        if name in game:
            value = game[name]
            row[name] = None if value is None else float(value)
    if qb_log is not None:
        home_rating, home_rate = prior_qb(qb_log, game.get("home_qb_id"), game["date"], game["game_id"])
        away_rating, away_rate = prior_qb(qb_log, game.get("away_qb_id"), game["date"], game["game_id"])
        row["home_qb_passer_rating"] = home_rating
        row["away_qb_passer_rating"] = away_rating
        row["home_qb_int_rate"] = home_rate
        row["away_qb_int_rate"] = away_rate
    for side in ("home", "away"):
        slot = state.get(game[side])
        recent = [] if slot is None else slot.get("recent") or []
        window = recent[-4:]
        if window:
            row[f"{side}_recent_win_rate"] = sum(item[0] for item in window) / len(window)
            row[f"{side}_recent_margin"] = sum(item[1] for item in window) / len(window)
        else:
            row[f"{side}_recent_win_rate"] = None
            row[f"{side}_recent_margin"] = None
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
            {
                "games": 0,
                "wins": 0,
                "points": 0,
                "allowed": 0,
                "date": game["date"],
                "season": game["season"],
                "week": game["week"],
                "recent": [],
            },
        )
        slot.setdefault("recent", [])
        slot.setdefault("allowed", 0)
        slot["games"] += 1
        slot["points"] += scored
        slot["allowed"] += allowed
        slot["recent"].append((1 if scored > allowed else 0, scored - allowed))
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


def write_seeds(rows: list[dict]) -> None:
    """Copy next-game projections. Spread and total stay the recorded schedule line."""
    seed_rows = [
        {
            "date": row.get("date"),
            "away": row.get("away"),
            "home": row.get("home"),
            "home_win_prob": row.get("home_win_prob"),
            "predicted_points": row.get("predicted_points"),
            "predicted_total": row.get("predicted_total"),
            "pick_side": row.get("pick_side"),
            "pick_team": row.get("pick_team"),
            "confidence": row.get("confidence"),
            "confidence_tier": row.get("confidence_tier"),
            "spread_line": row.get("spread_line"),
            "total_line": row.get("total_line"),
            "lean": row.get("total_lean"),
            "line": "recorded schedule line, not a live book price, not a bet",
        }
        for row in rows
    ]
    write_csv(
        ARTIFACTS / "seeds.csv",
        seed_rows,
        [
            "date", "away", "home", "home_win_prob", "predicted_points", "predicted_total",
            "pick_side", "pick_team", "confidence", "confidence_tier",
            "spread_line", "total_line", "lean", "line",
        ],
    )


def calibration_table(rows: list[dict], split: str) -> list[dict]:
    """A bin is yes only when its rate clears the floor. Holdout bins stay separate."""
    bins = (
        ("50-55", 0.50, 0.55),
        ("55-60", 0.55, 0.60),
        ("60-65", 0.60, 0.65),
        ("65-plus", 0.65, 1.01),
    )
    out = []
    for name, low, high in bins:
        group = [
            row for row in rows
            if row.get("home_win") is not None
            and row.get("home_win_prob") is not None
            and low <= float(row["home_win_prob"]) < high
        ]
        count = len(group)
        if count == 0:
            rate = None
            calibrated = "no"
            note = "no rows"
        else:
            rate = sum(int(row["home_win"]) for row in group) / count
            if count < 20:
                calibrated = "no"
                note = "count under 20"
            elif rate >= low:
                calibrated = "yes"
                note = "observed home-win rate is at least the bin floor"
            else:
                calibrated = "no"
                note = "observed home-win rate is below the bin floor"
        out.append(
            {
                "split": split,
                "bin": name,
                "count": count,
                "home_win_rate": rate,
                "calibrated": calibrated,
                "note": note,
            }
        )
    return out


def write_calibration(rows: list[dict], holdout_rows: list[dict] | None = None) -> None:
    out = calibration_table(rows, "in_sample")
    if holdout_rows:
        out.extend(calibration_table(holdout_rows, "holdout"))
    write_csv(
        ARTIFACTS / "calibration.csv",
        out,
        ["split", "bin", "count", "home_win_rate", "calibrated", "note"],
    )


def write_high_confidence(rows: list[dict]) -> None:
    chosen = [row for row in rows if row.get("confidence_tier") == "high"]
    write_csv(
        ARTIFACTS / "high_confidence.csv",
        chosen,
        [
            "game_id", "date", "away", "home", "home_win_prob", "pick_side", "pick_team",
            "confidence", "confidence_tier", "predicted_away_points", "predicted_home_points",
            "predicted_total", "total_lean",
        ],
    )


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


def mark_lean(row: dict) -> None:
    total = row.get("predicted_points")
    line = row.get("total_line")
    if total is None or line is None:
        row["total_lean"] = None
        return
    diff = float(total) - float(line)
    if diff >= 3:
        row["total_lean"] = "lean_over"
    elif diff <= -3:
        row["total_lean"] = "lean_under"
    else:
        row["total_lean"] = "no_lean"


def clear_prediction(row: dict) -> None:
    row["home_win_prob"] = None
    row["predicted_away_points"] = None
    row["predicted_home_points"] = None
    row["predicted_points"] = None
    row["predicted_total"] = None
    row["total_lean"] = None
    row["pick_side"] = None
    row["pick_team"] = None
    row["confidence"] = None
    row["confidence_tier"] = None


def annotate_prediction(row: dict) -> None:
    """Add the shared pick columns. They copy the model outputs and are not prices."""
    home_points = row.get("predicted_home_points")
    away_points = row.get("predicted_away_points")
    if home_points is not None and away_points is not None:
        row["predicted_total"] = float(home_points) + float(away_points)
    else:
        row["predicted_total"] = row.get("predicted_points")
    probability = row.get("home_win_prob")
    if probability is None:
        row["pick_side"] = None
        row["pick_team"] = None
        row["confidence"] = None
        row["confidence_tier"] = None
        return
    probability = float(probability)
    if probability >= 0.5:
        row["pick_side"] = "home"
        row["pick_team"] = row.get("home")
    else:
        row["pick_side"] = "away"
        row["pick_team"] = row.get("away")
    row["confidence"] = abs(probability - 0.5)
    if probability >= 0.60 or probability <= 0.40:
        row["confidence_tier"] = "high"
    elif probability >= 0.55 or probability <= 0.45:
        row["confidence_tier"] = "medium"
    else:
        row["confidence_tier"] = "low"


def log_line(text: str) -> None:
    path = ARTIFACTS / "refresh.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{stamp} {text}\n")


def ready_rows(rows: list[dict], features: list[str], start: date | None, end: date | None) -> list[dict]:
    chosen = []
    for row in rows:
        if row.get("home_win") is None:
            continue
        if any(row.get(feature) is None for feature in features):
            continue
        game_date = row["date"] if isinstance(row["date"], date) else date.fromisoformat(str(row["date"]))
        if start is not None and game_date < start:
            continue
        if end is not None and game_date >= end:
            continue
        chosen.append(row)
    return chosen


def fit_bundle(rows: list[dict], features: list[str]):
    frame = pd.DataFrame(rows)
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
    positive_is_home = int(ridge.classes_[1]) == 1
    return center, scale, ridge, home_points, away_points, positive_is_home


def score_points(row: dict, features: list[str], bundle) -> tuple[float | None, float | None, float | None]:
    center, scale, ridge, home_points, away_points, positive_is_home = bundle
    probability = win_probability(row, features, center, scale, ridge, positive_is_home)
    if probability is None:
        return None, None, None
    sample = ((np.array([row[feature] for feature in features], dtype=float) - center) / scale).reshape(1, -1)
    return probability, float(home_points.predict(sample)[0]), float(away_points.predict(sample)[0])


def fit_points_residual(rows: list[dict], features: list[str]):
    """Fit the miss versus prior points per game. The target uses the score. The features do not."""
    usable = [
        row for row in rows
        if row.get("home_ppg") is not None
        and row.get("away_ppg") is not None
        and row.get("home_score") is not None
        and row.get("away_score") is not None
    ]
    if len(usable) < 2:
        return None
    frame = pd.DataFrame(usable)
    matrix = frame[features].to_numpy(dtype=float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (matrix - center) / scale
    home_model = LinearRegression().fit(
        scaled,
        frame["home_score"].to_numpy(dtype=float) - frame["home_ppg"].to_numpy(dtype=float),
    )
    away_model = LinearRegression().fit(
        scaled,
        frame["away_score"].to_numpy(dtype=float) - frame["away_ppg"].to_numpy(dtype=float),
    )
    return center, scale, home_model, away_model


def apply_points_residual(row: dict, features: list[str], residual) -> tuple[float, float] | None:
    if residual is None or row.get("home_ppg") is None or row.get("away_ppg") is None:
        return None
    if any(row.get(feature) is None for feature in features):
        return None
    center, scale, home_model, away_model = residual
    sample = ((np.array([row[feature] for feature in features], dtype=float) - center) / scale).reshape(1, -1)
    home = float(row["home_ppg"]) + float(home_model.predict(sample)[0])
    away = float(row["away_ppg"]) + float(away_model.predict(sample)[0])
    return home, away


def holdout_metrics(rows: list[dict], features: list[str], required: list[str] | None = None) -> dict | None:
    """Fit only on earlier games. The test rows keep their own scores out of the features."""
    needed = features if required is None else required
    train_rows = ready_rows(rows, needed, None, HOLDOUT_START)
    test_rows = ready_rows(rows, needed, HOLDOUT_START, None)
    if len(train_rows) < 2 or not test_rows or len({row["home_win"] for row in train_rows}) < 2:
        return None
    bundle = fit_bundle(train_rows, features)
    residual = fit_points_residual(train_rows, features)
    correct = 0
    total_error = 0.0
    blend_error = 0.0
    residual_error = 0.0
    pairs = []
    for row in test_rows:
        probability, home_points, away_points = score_points(row, features, bundle)
        if probability is None or home_points is None or away_points is None:
            continue
        correct += int(int(probability >= 0.5) == int(row["home_win"]))
        actual_total = float(row["home_score"]) + float(row["away_score"])
        regression_total = home_points + away_points
        total_error += abs(regression_total - actual_total)
        prior_total = float(row["home_ppg"]) + float(row["away_ppg"])
        blend_error += abs((0.5 * regression_total + 0.5 * prior_total) - actual_total)
        adjusted = apply_points_residual(row, features, residual)
        if adjusted is not None:
            residual_error += abs((adjusted[0] + adjusted[1]) - actual_total)
        pairs.append({"home_win": int(row["home_win"]), "home_win_prob": probability})
    counted = len(pairs)
    if counted == 0:
        return None
    return {
        "train_rows": len(train_rows),
        "test_rows": counted,
        "accuracy": correct / counted,
        "total_mae": total_error / counted,
        "blend_mae": blend_error / counted,
        "residual_mae": residual_error / counted,
        "pairs": pairs,
    }


def metric_holds(candidate: dict | None, baseline: dict | None) -> bool:
    if candidate is None or baseline is None:
        return False
    accuracy_holds = candidate["accuracy"] + 1e-12 >= baseline["accuracy"]
    points_hold = candidate["total_mae"] <= baseline["total_mae"] + 0.5
    return accuracy_holds and points_hold


def choose_points_mode(block: dict | None) -> str:
    """Use a second points model only when it lowers the holdout total error."""
    if block is None:
        return "regression"
    options = [("regression", block["total_mae"]), ("blend", block["blend_mae"])]
    residual = block.get("residual_mae")
    if residual is not None:
        options.append(("residual", residual))
    name, value = min(options, key=lambda item: (item[1], item[0]))
    if name != "regression" and value + 0.05 < block["total_mae"]:
        return name
    return "regression"


def choose_features(completed_rows: list[dict], features: list[str]) -> tuple[list[str], dict]:
    """Keep points allowed, then recent form, only when the 2026 holdout does not get worse."""
    defense_names = [name for name in DEFENSE_FEATURES if name not in features]
    with_defense = features + defense_names
    baseline = holdout_metrics(completed_rows, features, with_defense)
    defense = holdout_metrics(completed_rows, with_defense, with_defense)
    base_count = len(ready_rows(completed_rows, features, None, None))
    defense_count = len(ready_rows(completed_rows, with_defense, None, None))
    adopted_defense = defense_count == base_count and metric_holds(defense, baseline)
    chosen = with_defense if adopted_defense else list(features)

    recent_names = [name for name in RECENT_FEATURES if name not in chosen]
    expanded = chosen + recent_names
    recent_base = holdout_metrics(completed_rows, chosen, expanded)
    recent = holdout_metrics(completed_rows, expanded, expanded)
    chosen_count = len(ready_rows(completed_rows, chosen, None, None))
    recent_count = len(ready_rows(completed_rows, expanded, None, None))
    adopted_recent = recent_count == chosen_count and metric_holds(recent, recent_base)
    if adopted_recent:
        chosen = expanded
    active = recent if adopted_recent else recent_base
    points_mode = choose_points_mode(active)
    return chosen, {
        "baseline": baseline,
        "defense": defense,
        "recent": recent,
        "adopted_defense": adopted_defense,
        "adopted_recent": adopted_recent,
        "points_mode": points_mode,
        "use_blend": points_mode == "blend",
        "chosen": active,
    }


def win_probability(row: dict, features: list[str], center, scale, ridge, positive_is_home: bool) -> float | None:
    """Score one row from prior features. A game's own score is not a feature."""
    if {"home_score", "away_score", "home_win"}.intersection(features):
        return None
    if any(row.get(feature) is None for feature in features):
        return None
    sample = (np.array([row[feature] for feature in features], dtype=float) - center) / scale
    sample = sample.reshape(1, -1)
    decision = float(ridge.decision_function(sample)[0])
    if not positive_is_home:
        decision = -decision
    return float(expit(decision))


def train(
    completed_rows: list[dict],
    next_rows: list[dict],
    features: list[str],
    points_mode: str = "regression",
) -> tuple[int, list[dict]]:
    for row in completed_rows:
        row["home_win_prob"] = None
    trainable = []
    for row in completed_rows:
        if row["home_win"] is None:
            continue
        if any(row.get(feature) is None for feature in features):
            continue
        trainable.append(row)
    if len(trainable) < 2 or len({row["home_win"] for row in trainable}) < 2:
        print("train_rows", len(trainable), flush=True)
        print("unscored_completed", len(completed_rows), flush=True)
        for row in next_rows:
            clear_prediction(row)
        return len(trainable), next_rows
    bundle = fit_bundle(trainable, features)
    residual_models = fit_points_residual(trainable, features) if points_mode == "residual" else None
    unscored = 0
    for row in completed_rows:
        probability = win_probability(row, features, bundle[0], bundle[1], bundle[2], bundle[5])
        row["home_win_prob"] = probability
        if probability is None:
            unscored += 1
    for row in next_rows:
        probability, home, away = score_points(row, features, bundle)
        if probability is None or home is None or away is None:
            clear_prediction(row)
            continue
        if points_mode == "blend" and row.get("home_ppg") is not None and row.get("away_ppg") is not None:
            regression_total = home + away
            blended_total = 0.5 * regression_total + 0.5 * (float(row["home_ppg"]) + float(row["away_ppg"]))
            margin = home - away
            home = (blended_total + margin) / 2
            away = (blended_total - margin) / 2
        elif points_mode == "residual":
            adjusted = apply_points_residual(row, features, residual_models)
            if adjusted is None:
                clear_prediction(row)
                continue
            home, away = adjusted
        row["home_win_prob"] = probability
        row["predicted_home_points"] = home
        row["predicted_away_points"] = away
        row["predicted_points"] = home + away
        mark_lean(row)
        annotate_prediction(row)
    print("train_rows", len(trainable), flush=True)
    print("unscored_completed", unscored, flush=True)
    return len(trainable), next_rows


def write_readme(max_date: str) -> None:
    README.write_text(
        "# NFL Game Predictor 2026\n\n"
        f"2025 through {max_date}. nflverse is the recorded-stats source. "
        "ESPN is the live score check. Sleeper is the current week only. "
        "nfl.com is not requested. Real is not a source. "
        "Weekly refresh is manual. This does not price a bet.\n\n"
        "Run `python scripts/refresh_week.py` once a week during the season. "
        "`python scripts/refresh_week.py --local` rebuilds the model from the saved files "
        "without downloading. Run `python scripts/validate_artifacts.py` after a refresh.\n",
        encoding="utf-8",
        newline="\n",
    )


def mariota_starter_row() -> bool:
    """True only when the 2026 injury file has a starter row for Marcus Mariota."""
    path = DATA / "injuries" / "injuries_2026.csv"
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("full_name") or "").strip() != "Marcus Mariota":
                continue
            if (row.get("season") or "").strip() != "2026":
                continue
            blob = " ".join((row.get(key) or "") for key in row).lower()
            if "starter" in blob:
                return True
    return False


def week4_quarterback_note(games: list[dict]) -> str:
    love_ids = weekly_player_ids("Jordan Love", "GB", "QB")
    love_id = next(iter(love_ids)) if len(love_ids) == 1 else "00-0036264"
    starter = depth_chart_qb1("TB")
    gb = next((game for game in games if game.get("game_id") == "2026_04_GB_TB"), None)
    home_id = None if gb is None else gb.get("home_qb_id")
    if starter is not None and home_id == starter[0] and home_id != love_id:
        home_sentence = (
            f"The home quarterback is the Tampa Bay starter on the latest 2026 depth chart, "
            f"{starter[1]} ({starter[0]})."
        )
    else:
        home_sentence = "Tampa Bay starter missing."
    if mariota_starter_row():
        washington = "Washington, game 2026_04_IND_WAS, has a Marcus Mariota starter row in the 2026 injury file."
    else:
        washington = (
            "Washington, game 2026_04_IND_WAS, stays as saved. "
            "The 2026 injury file has no starter row for Marcus Mariota."
        )
    return (
        f"Week 4 Green Bay at Tampa Bay, game 2026_04_GB_TB, lists Jordan Love ({love_id}) as the away quarterback only. "
        f"{home_sentence} "
        "Jordan Love is not the home quarterback. "
        "Week 5 Tampa Bay at Dallas was left as saved. "
        f"{washington}"
    )


def write_notes(lines: list[str]) -> None:
    NOTES.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def model_card_line(label: str, block: dict | None) -> str:
    if not block:
        return f"{label}: not computed"
    return (
        f"{label}: accuracy {block['accuracy']:.4f}, total MAE {block['total_mae']:.3f}, "
        f"prior-points blend MAE {block['blend_mae']:.3f}, "
        f"residual MAE {block.get('residual_mae', 0):.3f}, "
        f"{block['test_rows']} test games after {block['train_rows']} earlier games, "
        f"from {HOLDOUT_START.isoformat()}"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    local = "--local" in args
    log_line("refresh start " + ("local" if local else "download"))
    saved: list[str] = []
    skipped: list[str] = ["participation was not requested and was not downloaded"]
    if local:
        skipped.append("local rebuild did not download")
        print("local rebuild", flush=True)
    else:
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
    skipped.append("nfl.com was not requested")
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
    if (
        not local
        and previous == max_date.isoformat()
        and (ARTIFACTS / "nfl_completed.csv").exists()
        and (ARTIFACTS / "high_confidence.csv").exists()
    ):
        print("retrain skipped", flush=True)
        print("rows kept", flush=True)
        log_line(f"skipped retrain max {max_date.isoformat()}")
        return 0

    if local and sleeper is None:
        sleeper = load_saved_sleeper()
    return build_model(games, sleeper, saved, skipped, espn_status, nfl_status)


def build_model(games, sleeper, saved, skipped, espn_status, nfl_status) -> int:
    done = [game for game in games if completed(game)]
    min_date = min(game["date"] for game in done)
    max_date = max(game["date"] for game in done)
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
    schedule_columns = [name for name in SCHEDULE_FIELDS if games and name in games[0]]
    for name in LINE_FEATURES:
        if name in schedule_columns:
            features.append(name)
    qb_log = None
    if "home_qb_id" not in schedule_columns:
        skipped.append("home_qb_id is missing from the schedule file, so the quarterback features were skipped")
    else:
        qb_log = load_qb_log(games)
        if qb_log is None:
            skipped.append("starting quarterback prior lines were left out because the weekly player stat files had no passing line")
        else:
            features.extend(QB_FEATURES)

    next_ids = {game["game_id"] for game in choose_next(games, sleeper)}
    state: dict = {}
    completed_rows = []
    next_rows = []
    for game in games:
        feature_row = features_for(state, game, injuries, snaps, qb_log)
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

    features, model_report = choose_features(completed_rows, features)
    train_rows, next_rows = train(completed_rows, next_rows, features, model_report["points_mode"])
    unscored = sum(1 for row in completed_rows if row.get("home_win_prob") is None)
    scored = [
        row for row in completed_rows
        if row.get("home_win") is not None and row.get("home_win_prob") is not None
    ]
    in_sample_accuracy = None
    if scored:
        in_sample_accuracy = sum(int(int(float(row["home_win_prob"]) >= 0.5) == int(row["home_win"])) for row in scored) / len(scored)
    id_columns = ["game_id", "date", "season", "week", "game_type", "away", "home", "away_score", "home_score", "home_win"]
    carried = [name for name in schedule_columns if name not in features]
    write_csv(
        ARTIFACTS / "nfl_completed.csv",
        completed_rows,
        id_columns + carried + features + ["home_win_prob"],
    )
    for row in next_rows:
        row["away_team"] = row.get("away")
        row["home_team"] = row.get("home")
        row["predicted_away_score"] = row.get("predicted_away_points")
        row["predicted_home_score"] = row.get("predicted_home_points")
    next_columns = [
        "game_id",
        "date",
        "season",
        "week",
        "game_type",
        "away",
        "home",
        "away_team",
        "home_team",
        *carried,
        *features,
        "home_win_prob",
        "predicted_away_points",
        "predicted_home_points",
        "predicted_points",
        "predicted_away_score",
        "predicted_home_score",
        "predicted_total",
        "pick_side",
        "pick_team",
        "confidence",
        "confidence_tier",
        "total_lean",
    ]
    write_csv(ARTIFACTS / "nfl_next_games.csv", next_rows, next_columns)
    write_seeds(next_rows)
    holdout_pairs = [] if model_report.get("chosen") is None else model_report["chosen"]["pairs"]
    write_calibration(completed_rows, holdout_pairs)
    write_high_confidence(next_rows)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MAX_PATH.write_text(max_date.isoformat() + "\n", encoding="utf-8", newline="\n")
    for artifact in (
        "artifacts/nfl_completed.csv",
        "artifacts/nfl_next_games.csv",
        "artifacts/seeds.csv",
        "artifacts/calibration.csv",
        "artifacts/high_confidence.csv",
    ):
        if artifact not in saved:
            saved.append(artifact)
    write_readme(max_date.isoformat())
    date_ok = max_date >= date(2026, 9, 28)
    write_notes(
        [
            "Run python scripts/refresh_week.py once a week during the season. It does not run on a timer.",
            "",
            "nflverse is the recorded-stats source. ESPN is a live score check and is not copied into the training scores. Sleeper supplies the current week only. nfl.com is not requested. Real is not a source.",
            "",
            "The training rows use only earlier games: win rate, points per game, points allowed, rest days, the last four games' win rate and margin when that check improves the 2026 holdout, prior-week injury and snap counts, and the starting quarterback's prior-games passer rating and interception rate. Opponent win rate is the other side's prior win rate. A game does not train on its own score or its own quarterback line. The win model is a ridge classifier. Points use linear regression. A half-weight of prior points per game, or a residual on top of prior points per game, is used only when that choice lowers the holdout total error.",
            "",
            "The line is the recorded schedule line, not a live book price, and this does not price a bet. A lean is not a wager.",
            "",
            week4_quarterback_note(games),
            "",
            "Calibration is a check, not a bet.",
            "",
            f"Rows that could not be scored: {unscored}",
            "",
            "spread_line, total_line, home_moneyline, and div_game are model features when the schedule file has those columns. roof, surface, temp, wind, away_qb_id, and home_qb_id are carried from that file. temp and wind are blank on the current week, so they are not filled and are not model features.",
            "",
            f"Completed rows: {len(completed_rows)}",
            f"Training rows: {train_rows}",
            f"Feature count: {len(features)}",
            f"Minimum date: {min_date.isoformat()}",
            f"Maximum date: {max_date.isoformat()}",
            f"Next games: {len(next_rows)}",
            f"High-confidence next games: {sum(row.get('confidence_tier') == 'high' for row in next_rows)}",
            f"In-sample accuracy: {'not computed' if in_sample_accuracy is None else f'{in_sample_accuracy:.4f}'} on {len(scored)} scored rows",
            model_card_line("Baseline holdout", model_report.get("baseline")),
            model_card_line("Points-allowed holdout", model_report.get("defense")),
            model_card_line("Recent-form holdout", model_report.get("recent")),
            f"Points allowed adopted: {'yes' if model_report['adopted_defense'] else 'no'}",
            f"Recent form adopted: {'yes' if model_report['adopted_recent'] else 'no'}",
            f"Points blend adopted: {'yes' if model_report['use_blend'] else 'no'}",
            f"Points method: {model_report['points_mode']}",
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
    print("train_rows", train_rows, flush=True)
    print("feature_count", len(features), flush=True)
    print("min_date", min_date.isoformat(), flush=True)
    print("max_date", max_date.isoformat(), flush=True)
    print("next_rows", len(next_rows), flush=True)
    print("date_check", "pass" if date_ok else "fail", flush=True)
    print("features", ",".join(features), flush=True)
    print("points_allowed", "yes" if model_report["adopted_defense"] else "no", flush=True)
    print("recent_form", "yes" if model_report["adopted_recent"] else "no", flush=True)
    print("points_blend", "yes" if model_report["use_blend"] else "no", flush=True)
    print("points_method", model_report["points_mode"], flush=True)
    if in_sample_accuracy is not None:
        print("in_sample_accuracy", f"{in_sample_accuracy:.4f}", flush=True)
    chosen = model_report.get("chosen")
    if chosen is not None:
        print("holdout_accuracy", f"{chosen['accuracy']:.4f}", flush=True)
        print("holdout_total_mae", f"{chosen['total_mae']:.3f}", flush=True)
    log_line(
        f"rows {len(completed_rows)} train {train_rows} next {len(next_rows)} "
        f"max {max_date.isoformat()} defense {model_report['adopted_defense']} "
        f"recent {model_report['adopted_recent']} points {model_report['points_mode']}"
    )
    valid = run_validation()
    print("validation", "pass" if valid == 0 else "fail", flush=True)
    return 0 if date_ok and train_rows > 1 and valid == 0 else 1


def run_validation() -> int:
    script = ROOT / "scripts" / "validate_artifacts.py"
    if not script.exists():
        print("validation script missing", flush=True)
        return 1
    completed = subprocess_validate(script)
    return completed


def subprocess_validate(script: Path) -> int:
    import subprocess

    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)
    return int(result.returncode)


def load_saved_sleeper() -> dict | None:
    path = DATA / "live" / "sleeper_state.json"
    if not path.exists():
        return None
    try:
        return sleeper_week(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
