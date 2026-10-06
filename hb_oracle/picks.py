"""
Saved picks ("My picks"): the user's own shortlist, saved from the Today cards or a Head-to-head matchup.

Stored as data/my_picks.csv. When a GitHub token is available (Streamlit secret or env GITHUB_TOKEN with
contents:write on the repo), every change is also committed to the repo through the GitHub API, so picks
survive the app's restarts and redeploys. Without a token they live on the app's disk only, which Streamlit
Cloud wipes on every redeploy (about every 6 hours here, when the refresh workflow commits).
"""
from __future__ import annotations

import base64
import json
import os

import pandas as pd
import requests

from . import config, store

PICKS_CSV = config.DATA / "my_picks.csv"
COLS = ["pick_id", "saved_at", "source", "match_id", "start_time", "league", "home", "away",
        "pick", "p_pick", "confidence", "dk_odds", "note"]


def _token() -> str | None:
    t = os.getenv("GITHUB_TOKEN")
    if not t:
        try:
            import streamlit as st
            t = st.secrets.get("GITHUB_TOKEN")
        except Exception:
            t = None
    return t or None


def _repo() -> str:
    r = os.getenv("GITHUB_REPO")
    if not r:
        try:
            import streamlit as st
            r = st.secrets.get("GITHUB_REPO")
        except Exception:
            r = None
    return r or "jaysantmyer-spec/vegcatbat"


def _gh_headers():
    return {"Authorization": f"Bearer {_token()}", "Accept": "application/vnd.github+json"}


def _gh_url():
    return f"https://api.github.com/repos/{_repo()}/contents/data/my_picks.csv"


def pull_from_github() -> pd.DataFrame | None:
    """Latest picks from the repo (None if no token or request failed)."""
    if not _token():
        return None
    try:
        r = requests.get(_gh_url(), headers=_gh_headers(), timeout=15)
        if r.status_code == 404:
            return pd.DataFrame(columns=COLS)
        r.raise_for_status()
        raw = base64.b64decode(r.json()["content"]).decode("utf-8")
        from io import StringIO
        df = pd.read_csv(StringIO(raw)) if raw.strip() else pd.DataFrame(columns=COLS)
        return df
    except Exception:
        return None


def push_to_github(df: pd.DataFrame, message: str) -> tuple[bool, str]:
    if not _token():
        return False, "no token"
    try:
        r = requests.get(_gh_url(), headers=_gh_headers(), timeout=15)
        sha = r.json().get("sha") if r.status_code == 200 else None
        body = {"message": message, "content": base64.b64encode(df.to_csv(index=False).encode()).decode()}
        if sha:
            body["sha"] = sha
        r = requests.put(_gh_url(), headers={**_gh_headers(), "Content-Type": "application/json"}, json=body, timeout=20)
        r.raise_for_status()
        return True, "saved to GitHub"
    except Exception as ex:
        return False, f"GitHub save failed: {ex}"


def load() -> pd.DataFrame:
    df = pull_from_github()
    local = store.read(PICKS_CSV)
    if df is None or (df.empty and not local.empty and not _token()):
        df = local
    if df.empty:
        return pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    return df[COLS]


def _save(df: pd.DataFrame, message: str) -> tuple[bool, str]:
    store.write(df, PICKS_CSV)
    ok, msg = push_to_github(df, message)
    return ok, msg


def add(pick: dict) -> tuple[bool, str]:
    df = load()
    pid = f"{pick.get('match_id') or 'h2h'}|{pick['pick']}|{pd.Timestamp(pick['start_time']).strftime('%Y%m%d%H%M')}"
    if (df["pick_id"] == pid).any():
        return True, "already saved"
    row = {c: pick.get(c) for c in COLS}
    row["pick_id"], row["saved_at"] = pid, pd.Timestamp.now().floor("s")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return _save(df, f"save pick: {pick['pick']}")


def remove(pick_ids: list[str]) -> tuple[bool, str]:
    df = load()
    df = df[~df["pick_id"].isin(pick_ids)]
    return _save(df, f"remove {len(pick_ids)} pick(s)")


def clear() -> tuple[bool, str]:
    return _save(pd.DataFrame(columns=COLS), "clear all picks")


def grade(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Attach the result for each saved pick: by match_id when it came from the Today tab, else by the two
    names and a start time within a day (Head-to-head picks have no match id)."""
    out = df.copy()
    out["result"], out["winner"] = None, None
    if matches is None or matches.empty or out.empty:
        return out
    fin = matches[matches["finished"].astype(bool)]
    by_id = fin.set_index("match_id")
    for i, r in out.iterrows():
        w = None
        if pd.notna(r.get("match_id")) and r["match_id"] in by_id.index:
            m = by_id.loc[r["match_id"]]
            w = "draw" if m["winner"] == "draw" else (m["home"] if m["winner"] == "home" else m["away"])
        elif pd.notna(r["start_time"]):
            t0 = pd.Timestamp(r["start_time"])
            c = fin[(((fin["home"] == r["home"]) & (fin["away"] == r["away"])) |
                     ((fin["home"] == r["away"]) & (fin["away"] == r["home"]))) &
                    (fin["start_time"] >= t0 - pd.Timedelta(hours=12)) & (fin["start_time"] <= t0 + pd.Timedelta(hours=36))]
            if not c.empty:
                m = c.sort_values("start_time").iloc[0]
                w = "draw" if m["winner"] == "draw" else (m["home"] if m["winner"] == "home" else m["away"])
        if w is not None:
            out.at[i, "winner"] = w
            out.at[i, "result"] = "draw" if w == "draw" else ("won" if str(w) == str(r["pick"]) else "lost")
    return out
