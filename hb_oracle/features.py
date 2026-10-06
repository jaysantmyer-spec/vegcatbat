"""
Leak-free features for handball games.

Games are walked in start-time order, so a game's features only use games that finished before it started.
Per team, before each game: Elo (margin-scaled, with a home-advantage term in the expectation), Elo form,
results over the last 10 games, goals for / against / margin over the last 10, rest days, games in the last
7 and 14 days, games this season, career games, win rate at home (home side) or away (away side), streak,
and the head-to-head net record. Side A is always the HOME team, B the away team; the model is not mirrored
because home advantage is real and large in handball.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import config

DIFF_FEATURES = ["elo", "elo_form", "win10", "gf10", "ga10", "margin10", "rest", "games7", "games14",
                 "season_n", "career_n", "venue_rate", "streak", "h2h_net"]
CONTEXT_FEATURES = ["neutral", "both_n_min", "exp_total"]


def _expected(ra: float, rb: float, home_adv: float = 0.0) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra - home_adv) / 400.0))


class _T:
    __slots__ = ("elo", "hist", "elo_hist", "times", "n", "season", "season_n", "home", "away", "streak")

    def __init__(self):
        self.elo = 1500.0
        self.hist = deque(maxlen=30)      # (time, result 1/0.5/0, gf, ga)
        self.elo_hist = deque(maxlen=11)
        self.times = deque(maxlen=20)     # start times of recent games (for load)
        self.n = 0
        self.season = None
        self.season_n = 0
        self.home = [0.0, 0]              # (points, games) at home
        self.away = [0.0, 0]
        self.streak = 0


def _side(t: _T, now: pd.Timestamp, is_home: bool) -> dict:
    h = list(t.hist)
    last10 = h[-10:]
    rest = (now - t.times[-1]).total_seconds() / 86400 if t.times else 14.0
    g7 = sum(1 for x in t.times if x >= now - pd.Timedelta(days=7))
    g14 = sum(1 for x in t.times if x >= now - pd.Timedelta(days=14))
    v = t.home if is_home else t.away
    return {
        "elo": t.elo,
        "elo_form": (t.elo - t.elo_hist[0]) if len(t.elo_hist) >= 2 else 0.0,
        "win10": np.mean([x[1] for x in last10]) if last10 else 0.5,
        "gf10": np.mean([x[2] for x in last10]) if last10 else 28.0,
        "ga10": np.mean([x[3] for x in last10]) if last10 else 28.0,
        "margin10": np.mean([x[2] - x[3] for x in last10]) if last10 else 0.0,
        "rest": min(rest, 14.0),
        "games7": g7, "games14": g14,
        "season_n": t.season_n, "career_n": t.n,
        "venue_rate": (v[0] + 1) / (v[1] + 2),
        "streak": t.streak,
    }


def build_features(matches: pd.DataFrame, elo_k: float | None = None, home_adv: float | None = None) -> pd.DataFrame:
    k = config.DEFAULTS["elo_k"] if elo_k is None else elo_k
    ha = config.DEFAULTS["home_adv"] if home_adv is None else home_adv
    m = matches.copy()
    m["start_time"] = pd.to_datetime(m["start_time"])
    m = m.sort_values(["start_time", "match_id"]).reset_index(drop=True)
    teams: dict[str, _T] = defaultdict(_T)
    h2h: dict[tuple, list] = defaultdict(lambda: [0.0, 0])   # sorted pair -> [points of first, games]
    rows = []
    for r in m.itertuples(index=False):
        a, b, now = r.home, r.away, r.start_time
        ta, tb = teams[a], teams[b]
        for t in (ta, tb):
            if t.season != r.season:
                t.season, t.season_n = r.season, 0
        neutral = bool(getattr(r, "neutral", False))
        fa, fb = _side(ta, now, not neutral), _side(tb, now, False)
        key = tuple(sorted((a, b)))
        hp, hn = h2h[key]
        a_pts = hp if key[0] == a else hn - hp
        fa["h2h_net"], fb["h2h_net"] = (2 * a_pts - hn), (hn - 2 * a_pts)
        row = {"match_id": r.match_id, "date": r.date, "start_time": now, "league_id": r.league_id, "league": r.league,
               "season": r.season, "home": a, "away": b, "finished": bool(r.finished),
               "home_goals": r.home_goals, "away_goals": r.away_goals,
               "y": (1.0 if r.winner == "home" else 0.0 if r.winner == "away" else np.nan),
               "draw": (1.0 if r.winner == "draw" else 0.0) if r.finished else np.nan,
               "margin": (r.home_goals - r.away_goals) if r.finished else np.nan,
               "total": (r.home_goals + r.away_goals) if r.finished else np.nan,
               "elo_p_a": _expected(ta.elo, tb.elo, 0.0 if neutral else ha), "h2h_n": hn,
               "neutral": 1.0 if neutral else 0.0}
        for f in DIFF_FEATURES:
            row[f"A_{f}"], row[f"B_{f}"] = fa[f], fb[f]
            row[f"d_{f}"] = fa[f] - fb[f]
        row["both_n_min"] = min(ta.n, tb.n)
        row["exp_total"] = 0.5 * (fa["gf10"] + fb["ga10"]) + 0.5 * (fb["gf10"] + fa["ga10"])
        rows.append(row)

        if r.finished and r.winner in ("home", "away", "draw"):
            hg, ag = int(r.home_goals), int(r.away_goals)
            sa = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            ea = _expected(ta.elo, tb.elo, 0.0 if neutral else ha)
            mult = np.log1p(abs(hg - ag)) if hg != ag else 0.5
            delta = k * mult * (sa - ea)
            for t, s, gf, ga, d, is_home in ((ta, sa, hg, ag, delta, not neutral), (tb, 1 - sa, ag, hg, -delta, False)):
                t.elo_hist.append(t.elo)
                t.elo += d
                t.hist.append((now, s, gf, ga))
                t.n += 1
                t.season_n += 1
                v = t.home if is_home else t.away
                v[0] += s; v[1] += 1
                t.streak = (t.streak + 1 if t.streak >= 0 else 1) if s == 1 else (t.streak - 1 if t.streak <= 0 else -1) if s == 0 else 0
            h2h[key][0] += sa if key[0] == a else 1 - sa
            h2h[key][1] += 1
        for t in (ta, tb):
            t.times.append(now)
    return pd.DataFrame(rows)


def team_snapshot(feat: pd.DataFrame) -> pd.DataFrame:
    """Latest rating / form per team with every per-side feature, so the app can score any hypothetical
    matchup without rebuilding features."""
    a = feat[["home", "start_time", "league"] + [f"A_{k}" for k in DIFF_FEATURES]].rename(
        columns={"home": "team", **{f"A_{k}": k for k in DIFF_FEATURES}})
    b = feat[["away", "start_time", "league"] + [f"B_{k}" for k in DIFF_FEATURES]].rename(
        columns={"away": "team", **{f"B_{k}": k for k in DIFF_FEATURES}})
    s = pd.concat([a, b]).sort_values("start_time").groupby("team").last().reset_index()
    return s.sort_values("elo", ascending=False)


def matchup_row(sa: pd.Series, sb: pd.Series, h2h_a_pts: float, h2h_n: int, neutral: bool = False, now=None) -> pd.DataFrame:
    """One feature row for a hypothetical home (sa) vs away (sb) game from two team snapshots."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    ha = 0.0 if neutral else config.DEFAULTS["home_adv"]
    row = {"match_id": "h2h", "date": now.normalize(), "start_time": now, "league_id": None, "league": sa.get("league"),
           "season": None, "home": sa["team"], "away": sb["team"], "finished": False, "y": np.nan,
           "elo_p_a": _expected(float(sa["elo"]), float(sb["elo"]), ha), "h2h_n": h2h_n, "neutral": 1.0 if neutral else 0.0}
    for side, s in (("A", sa), ("B", sb)):
        for k in DIFF_FEATURES:
            if k == "h2h_net":
                v = (2 * h2h_a_pts - h2h_n) if side == "A" else (h2h_n - 2 * h2h_a_pts)
            elif k == "rest":
                v = 7.0
            elif k in ("games7", "games14"):
                v = 1.0 if k == "games7" else 2.0
            else:
                v = float(s[k])
            row[f"{side}_{k}"] = v
    for k in DIFF_FEATURES:
        row[f"d_{k}"] = row[f"A_{k}"] - row[f"B_{k}"]
    row["both_n_min"] = min(float(sa["career_n"]), float(sb["career_n"]))
    row["exp_total"] = 0.5 * (row["A_gf10"] + row["B_ga10"]) + 0.5 * (row["B_gf10"] + row["A_ga10"])
    return pd.DataFrame([row])
