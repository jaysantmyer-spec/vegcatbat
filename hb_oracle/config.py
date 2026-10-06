from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(os.environ.get("HB_ORACLE_ROOT", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
MODELS = ROOT / "models"
RAW = DATA / "raw"

MATCHES_CSV = DATA / "matches.csv"            # every game pulled from the API, finished or not
LEAGUES_CSV = DATA / "leagues.csv"            # league ids / seasons resolved from the API
TEAMS_CSV = DATA / "teams.csv"                # latest Elo / form per team (read by the app)
ODDS_CSV = DATA / "odds.csv"                  # latest DraftKings snapshot
ODDS_HISTORY_CSV = DATA / "odds_history.csv"
LEDGER_CSV = DATA / "ledger.csv"              # predictions made before the game, graded after
PREDICTIONS_CSV = DATA / "predictions.csv"    # precomputed predictions for upcoming games (read by the app)
BACKTEST_CSV = DATA / "backtest_latest.csv"
USAGE_JSON = DATA / "api_usage.json"          # requests used today (free plan: 100/day)
MODEL_FILE = MODELS / "predictor.joblib"
STATE_FILE = MODELS / "learning_state.json"

API_BASE = "https://v1.handball.api-sports.io"
API_DAILY_BUDGET = int(os.environ.get("HB_API_BUDGET", "95"))

# Leagues to follow: (country regex, league-name regex). Resolved to API ids by `python -m hb_oracle leagues`.
# Override with HB_LEAGUE_IDS="39,78,..." once data/leagues.csv shows the ids you want.
LEAGUE_PATTERNS = [   # men's top flights that DraftKings / the market price; ids from data/leagues.csv
    ("Germany", r"^Bundesliga$"),                       # 39
    ("France", r"^Starligue$"),                         # 34
    ("Denmark", r"^Herre Handbold Ligaen$"),            # 23
    ("Spain", r"^Liga ASOBAL$"),                        # 103
    ("Poland", r"^Superliga$"),                         # 78
    ("Hungary", r"^NB I$"),                             # 49
    ("Sweden", r"^Handbollsligan$"),                    # 113
    ("Norway", r"^REMA 1000-ligaen$"),                  # 75
    ("Europe", r"^Champions League$"),                  # 131
    ("Europe", r"^EHF European League$"),               # 145
]
SEASONS_BACK = int(os.environ.get("HB_SEASONS_BACK", "4"))   # seasons of history per league in the backfill
HORIZON_DAYS = 3                                              # fixtures predicted this far ahead

DEFAULTS = {
    "elo_k": 20.0,
    "home_adv": 60.0,            # Elo points added to the home side (handball home advantage is large)
    "half_life_days": 540.0,     # recency weighting of training games (about 1.5 seasons)
    "hardness_alpha": 0.5,
    "hedge_eta": 0.05,
    "hedge_window": 1500,
    "min_history": 8,            # games a team needs before a prediction is trusted
}


def league_ids_override() -> list[int]:
    v = os.environ.get("HB_LEAGUE_IDS", "").strip()
    return [int(x) for x in v.split(",") if x.strip().isdigit()] if v else []


def api_key() -> str | None:
    k = os.environ.get("API_SPORTS_KEY") or os.environ.get("APISPORTS_KEY")
    if not k:
        try:
            import streamlit as st
            k = st.secrets.get("API_SPORTS_KEY")
        except Exception:
            k = None
    return (k or "").strip() or None


def ensure_dirs() -> None:
    for p in (DATA, MODELS, RAW):
        p.mkdir(parents=True, exist_ok=True)
