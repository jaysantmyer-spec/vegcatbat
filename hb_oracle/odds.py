"""
Handball lines and the pricing desk.

Primary feed: The Odds API (https://the-odds-api.com), key in ODDS_API_KEY. It returns every handball
competition it prices (sport keys starting "handball_") with moneyline (3-way), spread and total from ~20
books, DraftKings included whenever DraftKings posts the game. The free plan is 500 credits a month and a
request costs regions x markets credits, so the refresh pulls US books (h2h+spreads+totals, 3 credits) and
EU books (h2h, 1 credit) and only every other cycle; the app's button can pull on demand.

Fallback: DraftKings' own site feed (often blocked from data-centre IPs) and a paste box in the app.

Market probability for a game: DraftKings' line when present, otherwise the vig-free average across books
("consensus"). Both are stored so the app can show which one it used.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

from . import config, store

ODDS_API = "https://api.the-odds-api.com/v4"
DK_V5_LIST = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups?format=json"
DK_V5 = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{gid}?format=json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/126.0 Safari/537.36", "Accept": "application/json"}
LAST_FETCH_LOG: list[str] = []
ODDS_COLS = ["event_id", "sport", "commence_time", "home", "away", "bookmaker", "odds_home", "odds_draw", "odds_away",
             "spread_home", "spread_home_price", "spread_away_price", "total", "over_price", "under_price", "fetched_at"]
STOP = {"hc", "sg", "tsv", "thw", "hsg", "tvb", "sc", "vfl", "tbv", "hsv", "fa", "rk", "kc", "mt", "rn", "hbw", "gwd",
        "handball", "handbold", "hb", "club", "team", "the", "and", "und", "de", "es", "cb", "bm", "kif", "gog"}


# --------------------------------------------------------------------------- odds maths
def american_to_decimal(a) -> float:
    a = float(str(a).replace("+", "").replace("−", "-"))
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def decimal_to_american(d) -> str:
    if d is None or pd.isna(d) or float(d) <= 1:
        return "–"
    d = float(d)
    return f"+{(d - 1) * 100:.0f}" if d >= 2 else f"-{100 / (d - 1):.0f}"


def fair_american(p: float) -> str:
    p = float(min(max(p, 1e-6), 1 - 1e-6))
    return f"-{100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def kelly(p: float, d: float, fraction: float = 0.25) -> float:
    b = d - 1
    if b <= 0:
        return 0.0
    return max(0.0, (p * b - (1 - p)) / b * fraction)


def _tokens(name) -> frozenset:
    s = unicodedata.normalize("NFKD", str(name).replace("ß", "ss")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"(?<=[a-z])oe|(?<=[a-z])ue|(?<=[a-z])ae", lambda m: m.group(0)[0], s)   # Loewen ~ Löwen ~ Lowen
    return frozenset(t for t in re.split(r"[^a-z0-9]+", s) if (len(t) >= 3 or t.isdigit()) and t not in STOP)


def same_team(a, b) -> bool:
    """'Rhein-Neckar Löwen' ~ 'Rhein-Neckar Loewen' ~ 'RN Löwen'; needs a shared distinctive token."""
    ka, kb = _tokens(a), _tokens(b)
    if not ka or not kb:
        return False
    return ka <= kb or kb <= ka


def odds_api_key() -> str | None:
    k = os.getenv("ODDS_API_KEY")
    if not k:
        try:
            import streamlit as st
            k = st.secrets.get("ODDS_API_KEY")
        except Exception:
            k = None
    return k or None


# --------------------------------------------------------------------------- The Odds API
def _oa(path: str, **params) -> tuple[list | dict, dict]:
    key = odds_api_key()
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set")
    r = requests.get(f"{ODDS_API}/{path}", params={"apiKey": key, **params}, timeout=30)
    r.raise_for_status()
    used = {"used": r.headers.get("x-requests-used"), "remaining": r.headers.get("x-requests-remaining")}
    return r.json(), used


def handball_sports(active_only: bool = True) -> list[str]:
    data, _ = _oa("sports", all="false" if active_only else "true")
    return [s["key"] for s in data if str(s.get("key", "")).startswith("handball")]


def _rows_from_event(ev: dict, sport: str, now) -> list[dict]:
    rows = []
    home, away = ev.get("home_team"), ev.get("away_team")
    for bk in ev.get("bookmakers", []):
        row = {"event_id": ev.get("id"), "sport": sport, "commence_time": ev.get("commence_time"), "home": home,
               "away": away, "bookmaker": bk.get("key"), "fetched_at": now}
        for mk in bk.get("markets", []):
            outs = {o.get("name"): o for o in mk.get("outcomes", [])}
            if mk.get("key") == "h2h":
                row["odds_home"] = (outs.get(home) or {}).get("price")
                row["odds_away"] = (outs.get(away) or {}).get("price")
                row["odds_draw"] = (outs.get("Draw") or {}).get("price")
            elif mk.get("key") == "spreads":
                h, a = outs.get(home) or {}, outs.get(away) or {}
                row["spread_home"], row["spread_home_price"], row["spread_away_price"] = h.get("point"), h.get("price"), a.get("price")
            elif mk.get("key") == "totals":
                o, u = outs.get("Over") or {}, outs.get("Under") or {}
                row["total"], row["over_price"], row["under_price"] = o.get("point"), o.get("price"), u.get("price")
        if row.get("odds_home") or row.get("total") or row.get("spread_home"):
            rows.append(row)
    return rows


def fetch_odds_api(save_raw: bool = True) -> pd.DataFrame:
    """Every priced handball game from The Odds API, one row per (game, bookmaker), decimal odds."""
    global LAST_FETCH_LOG
    log, rows, now = [], [], pd.Timestamp.now(tz="UTC").tz_localize(None)
    try:
        sports = handball_sports()
    except Exception as ex:
        LAST_FETCH_LOG = [f"sports list: {ex}"]
        return pd.DataFrame(columns=ODDS_COLS)
    log.append(f"active handball sports: {sports or 'none'}")
    for sport in sports:
        for regions, markets in (("us", "h2h,spreads,totals"), ("eu,uk", "h2h")):
            try:
                data, used = _oa(f"sports/{sport}/odds", regions=regions, markets=markets, oddsFormat="decimal")
                if save_raw:
                    config.ensure_dirs()
                    (config.RAW / f"oddsapi_{sport}_{regions.replace(',', '')}.json").write_text(json.dumps(data)[:1_500_000])
                n0 = len(rows)
                for ev in data:
                    rows += _rows_from_event(ev, sport, now)
                log.append(f"{sport} [{regions}]: {len(data)} events, {len(rows) - n0} book lines (credits left {used.get('remaining')})")
            except Exception as ex:
                log.append(f"{sport} [{regions}]: {ex}")
    LAST_FETCH_LOG = log
    config.ensure_dirs()
    (config.RAW / "odds_fetch_log.txt").write_text("\n".join(log))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ODDS_COLS)
    for c in ODDS_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True).dt.tz_convert(None)
    return df[ODDS_COLS]


# --------------------------------------------------------------------------- DraftKings direct (fallback)
def fetch_dk_direct(save_raw: bool = True) -> pd.DataFrame:
    global LAST_FETCH_LOG
    log, rows, now = [], [], pd.Timestamp.now(tz="UTC").tz_localize(None)
    try:
        gid = os.getenv("DK_EVENT_GROUP_ID")
        gids = [gid] if gid else []
        if not gids:
            lst = requests.get(DK_V5_LIST, headers=HEADERS, timeout=20).json()
            for eg in lst.get("eventGroupInfos", lst.get("eventGroups", [])):
                if "handball" in str(eg.get("name", "")).lower() or "handball" in str(eg.get("sportName", "")).lower():
                    gids.append(str(eg.get("eventGroupId")))
            log.append(f"handball event groups on DraftKings: {gids}")
        for g in gids:
            payload = requests.get(DK_V5.format(gid=g), headers=HEADERS, timeout=20).json()
            if save_raw:
                config.ensure_dirs()
                (config.RAW / f"dk_{g}.json").write_text(json.dumps(payload)[:800000])
            eg = payload.get("eventGroup") or payload
            events = {e.get("eventId"): e for e in eg.get("events", [])}
            for cat in eg.get("offerCategories", []):
                for sub in cat.get("offerSubcategoryDescriptors", []):
                    for group in (sub.get("offerSubcategory") or {}).get("offers", []):
                        for offer in group:
                            label = (offer.get("label") or "").lower()
                            if offer.get("isSuspended") or "moneyline" not in label:
                                continue
                            outs = offer.get("outcomes", [])
                            ev = events.get(offer.get("eventId"), {})
                            names = ev.get("name", " vs ").split(" vs ")
                            home, away = (names[1].strip(), names[0].strip()) if len(names) == 2 else (None, None)
                            row = {"event_id": offer.get("eventId"), "sport": eg.get("name"), "commence_time": ev.get("startDate"),
                                   "home": home, "away": away, "bookmaker": "draftkings", "fetched_at": now}
                            for o in outs:
                                lab = (o.get("label") or "").lower()
                                d = o.get("oddsDecimal")
                                if lab == "draw" or lab == "tie":
                                    row["odds_draw"] = d
                                elif home and same_team(lab, home):
                                    row["odds_home"] = d
                                elif away and same_team(lab, away):
                                    row["odds_away"] = d
                            if row.get("odds_home") and row.get("odds_away"):
                                rows.append(row)
            log.append(f"group {g}: {len(rows)} moneylines")
    except Exception as ex:
        log.append(f"DraftKings direct: {ex}")
    LAST_FETCH_LOG = log
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ODDS_COLS)
    for c in ODDS_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce").dt.tz_convert(None)
    return df[ODDS_COLS]


def parse_pasted(text: str) -> pd.DataFrame:
    """'THW Kiel -250 / SG Flensburg +190' per line (home first; optional third number for the draw)."""
    rows, now = [], pd.Timestamp.now(tz="UTC").tz_localize(None)
    for line in (text or "").splitlines():
        parts = re.split(r"\s*(?:/|vs\.?|,)\s*", line.strip())
        if len(parts) < 2:
            continue
        m1 = re.match(r"(.+?)\s+([+-−]?\d+(?:\.\d+)?)\s*$", parts[0])
        m2 = re.match(r"(.+?)\s+([+-−]?\d+(?:\.\d+)?)\s*$", parts[1])
        if not (m1 and m2):
            continue
        def dec(s):
            v = float(s.replace("−", "-").replace("+", ""))
            return v if 1.01 <= v <= 30 and "+" not in s and "-" not in s else american_to_decimal(s)
        draw = None
        if len(parts) >= 3:
            m3 = re.match(r"(?:draw\s+)?([+-−]?\d+(?:\.\d+)?)\s*$", parts[2].strip(), re.I)
            draw = dec(m3.group(1)) if m3 else None
        rows.append({"event_id": None, "sport": "pasted", "commence_time": pd.NaT, "home": m1.group(1).strip(),
                     "away": m2.group(1).strip(), "bookmaker": "draftkings", "odds_home": dec(m1.group(2)),
                     "odds_draw": draw, "odds_away": dec(m2.group(2)), "fetched_at": now})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ODDS_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[ODDS_COLS]


def snapshot_odds(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = fetch_odds_api() if odds_api_key() else fetch_dk_direct()
        if df.empty and odds_api_key():
            dk = fetch_dk_direct()
            df = dk if not dk.empty else df
    if df.empty:
        return {"lines": 0, "log": LAST_FETCH_LOG}
    store.write(df, config.ODDS_CSV)
    hist = store.read(config.ODDS_HISTORY_CSV)
    hist = pd.concat([hist, df], ignore_index=True) if not hist.empty else df
    store.write(hist.tail(200_000), config.ODDS_HISTORY_CSV)
    return {"lines": int(len(df)), "books": int(df["bookmaker"].nunique()), "games": int(df[["home", "away"]].drop_duplicates().shape[0])}


# --------------------------------------------------------------------------- market per game
def market_table(odds: pd.DataFrame) -> pd.DataFrame:
    """One row per game: DraftKings line if present, else vig-free consensus over books; plus DK/consensus
    spread and total."""
    if odds is None or odds.empty:
        return pd.DataFrame()
    rows = []
    for (h, a), g in odds.groupby(["home", "away"]):
        g = g.sort_values("fetched_at")
        dk = g[g["bookmaker"] == "draftkings"]
        ml = g.dropna(subset=["odds_home", "odds_away"])
        src = None
        if not dk.dropna(subset=["odds_home", "odds_away"]).empty:
            r = dk.dropna(subset=["odds_home", "odds_away"]).iloc[-1]
            oh, oa, od, src = float(r["odds_home"]), float(r["odds_away"]), r["odds_draw"], "DraftKings"
        elif not ml.empty:
            oh, oa, od, src = float(ml["odds_home"].mean()), float(ml["odds_away"].mean()), ml["odds_draw"].mean(), f"consensus ({ml['bookmaker'].nunique()} books)"
        else:
            continue
        ih, ia = 1 / oh, 1 / oa
        idr = 1 / float(od) if pd.notna(od) and float(od) > 1 else 0.0
        s = ih + ia + idr
        sp = (dk if not dk.dropna(subset=["spread_home"]).empty else g).dropna(subset=["spread_home"])
        tt = (dk if not dk.dropna(subset=["total"]).empty else g).dropna(subset=["total"])
        rows.append({"home": h, "away": a, "commence_time": g["commence_time"].iloc[-1], "market_source": src,
                     "odds_home": oh, "odds_away": oa, "odds_draw": od,
                     "mkt_home": ih / s, "mkt_away": ia / s, "mkt_draw": idr / s,
                     "market_p_a": ih / (ih + ia),                            # 2-way (draw no bet) share
                     "spread_home": sp["spread_home"].iloc[-1] if not sp.empty else np.nan,
                     "total_line": tt["total"].iloc[-1] if not tt.empty else np.nan,
                     "n_books": int(ml["bookmaker"].nunique())})
    return pd.DataFrame(rows)


def attach_market(P: pd.DataFrame, odds: pd.DataFrame | None = None) -> pd.DataFrame:
    odds = store.read(config.ODDS_CSV) if odds is None else odds
    out = P.copy()
    for c in ("dk_odds_a", "dk_odds_b", "dk_odds_draw", "market_p_a", "mkt_home", "mkt_away", "mkt_draw",
              "spread_home", "total_line"):
        out[c] = np.nan
    out["market_source"] = None
    mk = market_table(odds)
    if mk.empty or out.empty:
        return out
    for i, r in out.iterrows():
        for _, m in mk.iterrows():
            if same_team(r["home"], m["home"]) and same_team(r["away"], m["away"]):
                if pd.notna(m["commence_time"]) and pd.notna(r["start_time"]) and abs((m["commence_time"] - pd.Timestamp(r["start_time"])).total_seconds()) > 36 * 3600:
                    continue
                out.at[i, "dk_odds_a"], out.at[i, "dk_odds_b"], out.at[i, "dk_odds_draw"] = m["odds_home"], m["odds_away"], m["odds_draw"]
                for c in ("market_p_a", "mkt_home", "mkt_away", "mkt_draw", "spread_home", "total_line", "market_source"):
                    out.at[i, c] = m[c]
                break
    out["edge_a"] = out["p_a"] - out["market_p_a"]
    return out


def price(P: pd.DataFrame, bankroll: float = 1000.0, kelly_fraction: float = 0.25, edge_threshold: float = 0.04) -> pd.DataFrame:
    """Price the 3-way moneyline sides (home / away) at the book's odds using the model's 3-way probabilities."""
    rows = []
    for _, r in P.iterrows():
        for side, team, opp, p, mp, d in (("home", r["home"], r["away"], r.get("p_home"), r.get("mkt_home"), r.get("dk_odds_a")),
                                          ("away", r["away"], r["home"], r.get("p_away"), r.get("mkt_away"), r.get("dk_odds_b"))):
            p = float(p) if pd.notna(p) else np.nan
            edge = p - mp if pd.notna(mp) else np.nan
            ev = p * (d - 1) - (1 - p) if pd.notna(d) else np.nan
            k = kelly(p, d, kelly_fraction) if pd.notna(d) else 0.0
            rows.append({"match_id": r["match_id"], "start_time": r["start_time"], "league": r.get("league"), "team": team,
                         "opponent": opp, "side": side, "p_model": p, "fair": fair_american(p) if pd.notna(p) else "–",
                         "p_market": mp, "market": fair_american(mp) if pd.notna(mp) else "–",
                         "book": decimal_to_american(d), "book_dec": d, "source": r.get("market_source"),
                         "edge": edge, "ev": ev, "kelly": k, "stake": round(k * bankroll),
                         "thin": bool(r.get("thin_history", False)),
                         "bet": bool(pd.notna(edge) and edge >= edge_threshold and ev > 0 and not r.get("thin_history", False))})
    return pd.DataFrame(rows)
