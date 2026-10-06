"""
Match data from API-Sports Handball (https://api-sports.io/sports/handball).

    GET https://v1.handball.api-sports.io/leagues
    GET https://v1.handball.api-sports.io/games?league=ID&season=YYYY
    GET https://v1.handball.api-sports.io/games?date=YYYY-MM-DD
with header  x-apisports-key: <key>.  Free plan: 100 requests/day, reset 00:00 UTC. This module keeps a
daily counter (data/api_usage.json) and stops before the budget so the refresh never gets locked out.

Every response is saved to data/raw/api_<endpoint>_<args>.json so the parser can be checked against the
real payload (field names below follow the API-Sports family: hockey / basketball / volleyball share it).
"""
from __future__ import annotations

import json
import logging
import re
import time

import pandas as pd
import requests

from . import config, store

log = logging.getLogger(__name__)

FINISHED = {"FT", "AOT", "AET", "AP", "AP."}                       # full time, after overtime, after penalties
VOID = {"CANC", "PST", "POST", "ABD", "INTR", "WO", "AWD", "AW"}
LIVE = {"1H", "HT", "2H", "ET", "BT", "PT", "LIVE"}

MATCH_COLS = ["match_id", "date", "start_time", "league_id", "league", "country", "season", "round",
              "home", "away", "home_id", "away_id", "home_goals", "away_goals", "ht_home", "ht_away",
              "status", "finished", "winner"]


# --------------------------------------------------------------------------- budget
def _usage() -> dict:
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    u = json.loads(config.USAGE_JSON.read_text()) if config.USAGE_JSON.exists() else {}
    if u.get("day") != today:
        u = {"day": today, "used": 0}
    return u


def requests_left() -> int:
    return max(0, config.API_DAILY_BUDGET - int(_usage().get("used", 0)))


def _count(n: int = 1) -> None:
    u = _usage()
    u["used"] = int(u.get("used", 0)) + n
    config.ensure_dirs()
    config.USAGE_JSON.write_text(json.dumps(u))


class BudgetExhausted(RuntimeError):
    pass


class PlanLimited(RuntimeError):
    """The free plan refuses some seasons ('try from 2022 to 2024')."""


PLAN_JSON = config.DATA / "plan_limits.json"


def plan_seasons() -> tuple[int | None, int | None]:
    """(min, max) season the plan allows for whole-season queries, learned from the API's own error text."""
    if PLAN_JSON.exists():
        d = json.loads(PLAN_JSON.read_text())
        return d.get("min"), d.get("max")
    return None, None


def _remember_plan(msg: str) -> None:
    m = re.search(r"from\s+(\d{4})\s+to\s+(\d{4})", msg)
    if m:
        config.ensure_dirs()
        PLAN_JSON.write_text(json.dumps({"min": int(m.group(1)), "max": int(m.group(2)), "message": msg}))


# --------------------------------------------------------------------------- HTTP
def get(endpoint: str, retries: int = 3, **params) -> dict:
    key = config.api_key()
    if not key:
        raise RuntimeError("API_SPORTS_KEY is not set (GitHub secret / Streamlit secret / env)")
    if requests_left() <= 0:
        raise BudgetExhausted(f"daily API budget ({config.API_DAILY_BUDGET}) used up; resets 00:00 UTC")
    url = f"{config.API_BASE}/{endpoint.lstrip('/')}"
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"x-apisports-key": key}, params=params, timeout=30)
            _count()
            if r.status_code == 429:
                raise BudgetExhausted("API returned 429 (rate limit)")
            r.raise_for_status()
            payload = r.json()
            errs = payload.get("errors")
            if errs and (isinstance(errs, dict) and errs or isinstance(errs, list) and errs):
                if isinstance(errs, dict) and "plan" in errs:
                    _remember_plan(str(errs["plan"]))
                    raise PlanLimited(str(errs["plan"]))
                raise RuntimeError(f"API error: {errs}")
            tag = re.sub(r"[^A-Za-z0-9]+", "_", f"{endpoint}_{'_'.join(f'{k}{v}' for k, v in params.items())}")
            config.ensure_dirs()
            (config.RAW / f"api_{tag}.json").write_text(json.dumps(payload)[:2_000_000])
            return payload
        except (BudgetExhausted, PlanLimited):
            raise
        except Exception as ex:
            last = ex
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{endpoint} {params}: {last}")


# --------------------------------------------------------------------------- leagues
def fetch_leagues() -> pd.DataFrame:
    """All handball leagues the API knows, one row per (league, season)."""
    rows = []
    for item in get("leagues").get("response", []):
        lg = item.get("league", item)                      # some sports nest under "league", some don't
        country = item.get("country", {}) or {}
        seasons = item.get("seasons") or lg.get("seasons") or []
        for s in seasons:
            rows.append({"league_id": lg.get("id", item.get("id")), "league": lg.get("name", item.get("name")),
                         "type": lg.get("type", item.get("type")), "country": country.get("name"),
                         "season": s.get("season") if isinstance(s, dict) else s,
                         "current": bool(s.get("current")) if isinstance(s, dict) else False,
                         "start": s.get("start") if isinstance(s, dict) else None,
                         "end": s.get("end") if isinstance(s, dict) else None})
    df = pd.DataFrame(rows)
    if not df.empty:
        store.write(df, config.LEAGUES_CSV)
    return df


def selected_leagues(leagues: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows of data/leagues.csv matching config.LEAGUE_PATTERNS (or HB_LEAGUE_IDS)."""
    lg = store.read(config.LEAGUES_CSV) if leagues is None else leagues
    if lg.empty:
        return lg
    ids = config.league_ids_override()
    if ids:
        return lg[lg["league_id"].isin(ids)]
    keep = pd.Series(False, index=lg.index)
    for country_re, name_re in config.LEAGUE_PATTERNS:
        keep |= (lg["country"].astype(str).str.contains(country_re, case=False, regex=True, na=False)
                 & lg["league"].astype(str).str.contains(name_re, case=False, regex=True, na=False))
    return lg[keep]


# --------------------------------------------------------------------------- games
def _parse_game(g: dict) -> dict | None:
    try:
        teams, scores = g.get("teams", {}), g.get("scores", {}) or {}
        home, away = teams.get("home", {}) or {}, teams.get("away", {}) or {}
        status = g.get("status", {}) or {}
        short = status.get("short") if isinstance(status, dict) else str(status)
        hg, ag = scores.get("home"), scores.get("away")
        hg = int(hg) if hg is not None else None
        ag = int(ag) if ag is not None else None
        finished = short in FINISHED and hg is not None and ag is not None
        winner = None
        if finished:
            winner = "home" if hg > ag else "away" if ag > hg else "draw"
        periods = g.get("periods", {}) or {}
        first = periods.get("first", {}) or {}
        lg = g.get("league", {}) or {}
        start = pd.to_datetime(g.get("date"), utc=True, errors="coerce")
        if pd.isna(start) and g.get("timestamp"):
            start = pd.to_datetime(int(g["timestamp"]), unit="s", utc=True)
        return {"match_id": int(g["id"]), "date": start.tz_convert(None).normalize() if pd.notna(start) else pd.NaT,
                "start_time": start.tz_convert(None) if pd.notna(start) else pd.NaT,
                "league_id": lg.get("id"), "league": lg.get("name"), "country": (g.get("country") or {}).get("name"),
                "season": lg.get("season"), "round": g.get("week") or g.get("round"),
                "home": home.get("name"), "away": away.get("name"), "home_id": home.get("id"), "away_id": away.get("id"),
                "home_goals": hg, "away_goals": ag, "ht_home": first.get("home"), "ht_away": first.get("away"),
                "status": short, "finished": finished, "winner": winner}
    except Exception as ex:                                   # one odd game must not kill a whole pull
        log.warning("skipping game %s: %s", g.get("id"), ex)
        return None


def fetch_games(**params) -> pd.DataFrame:
    payload = get("games", timezone="UTC", **params)
    games = list(payload.get("response", []))
    total = int((payload.get("paging") or {}).get("total") or 1)
    for page in range(2, min(total, 20) + 1):
        games += get("games", timezone="UTC", page=page, **params).get("response", [])
    rows = [r for r in (_parse_game(g) for g in games) if r]
    df = pd.DataFrame(rows, columns=MATCH_COLS)
    return df[~df["status"].isin(VOID)]


def _merge(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df.dropna(subset=["home", "away", "start_time"])
    store.upsert(df, config.MATCHES_CSV, "match_id")
    return int(len(df))


# --------------------------------------------------------------------------- jobs
def backfill(seasons_back: int | None = None, progress=None) -> dict:
    """One request per (league, season) for the selected leagues, newest season first, within the daily budget.
    Re-run on later days to continue; already-pulled (league, season) pairs with a finished season are skipped."""
    seasons_back = config.SEASONS_BACK if seasons_back is None else seasons_back
    lg = store.read(config.LEAGUES_CSV)
    if lg.empty:
        lg = fetch_leagues()
    sel = selected_leagues(lg)
    if sel.empty:
        return {"error": "no leagues matched; check data/leagues.csv and config.LEAGUE_PATTERNS / HB_LEAGUE_IDS"}
    have = store.read(config.MATCHES_CSV)
    done = set()
    if not have.empty:
        fin = have[have["finished"].astype(bool)].groupby(["league_id", "season"]).size()
        for (lid, s), n in fin.items():
            # treat a season as complete when its games are all finished and its listed end date is past
            row = lg[(lg["league_id"] == lid) & (lg["season"] == s)]
            end = pd.to_datetime(row["end"].iloc[0]) if len(row) and pd.notna(row["end"].iloc[0]) else None
            if end is not None and end < pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=7):
                done.add((lid, s))
    out = {"pulled": [], "skipped": sorted(map(str, done)), "requests_left": requests_left()}
    todo = []
    lo_s, hi_s = plan_seasons()
    for lid, grp in sel.groupby("league_id"):
        seasons = sorted(grp["season"].dropna().unique(), reverse=True)[:seasons_back]
        for s in seasons:
            if hi_s is not None and (s > hi_s or (lo_s is not None and s < lo_s)):
                continue                                   # whole-season query refused on this plan; the daily
            if (lid, s) not in done:                       # date pulls cover the current season instead
                todo.append((int(lid), int(s), grp["league"].iloc[0]))
    todo.sort(key=lambda t: -t[1])
    for i, (lid, s, name) in enumerate(todo):
        if requests_left() <= 1:
            out["stopped"] = "daily budget reached; run again tomorrow to continue"
            break
        if progress:
            progress(i / max(len(todo), 1), f"{name} {s}")
        try:
            n = _merge(fetch_games(league=lid, season=s))
            out["pulled"].append(f"{name} {s}: {n} games")
        except BudgetExhausted as ex:
            out["stopped"] = str(ex)
            break
        except PlanLimited as ex:
            out["pulled"].append(f"{name} {s}: not on this plan ({ex})")
        except Exception as ex:
            out["pulled"].append(f"{name} {s}: FAILED {ex}")
    out["requests_left"] = requests_left()
    config.ensure_dirs()
    (config.DATA / "backfill_log.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def update(days_back: int = 1, days_ahead: int | None = None) -> dict:
    """Results and fixtures around today: one request per day (yesterday .. today+horizon)."""
    days_ahead = config.HORIZON_DAYS if days_ahead is None else days_ahead
    sel_ids = set(selected_leagues()["league_id"].astype(int)) if config.LEAGUES_CSV.exists() else set()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    out, total = {}, 0
    for d in range(-days_back, days_ahead + 1):
        day = (today + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        try:
            df = fetch_games(date=day)
        except BudgetExhausted as ex:
            out["stopped"] = str(ex)
            break
        except PlanLimited as ex:
            out["stopped"] = f"plan: {ex}"
            break
        if sel_ids:
            df = df[df["league_id"].isin(sel_ids)]
        total += _merge(df)
        out[day] = int(len(df))
    out["merged"] = total
    out["requests_left"] = requests_left()
    return out
