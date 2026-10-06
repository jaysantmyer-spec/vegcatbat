"""Diagnostics for the API connections; output goes to data/probe.txt (never prints the keys)."""
import json, os, sys, traceback, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import requests
from hb_oracle import config, api

print("API_SPORTS_KEY set:", bool(os.getenv("API_SPORTS_KEY")), "len", len(os.getenv("API_SPORTS_KEY") or ""))
print("ODDS_API_KEY set:", bool(os.getenv("ODDS_API_KEY")))
for ep in ("status", "leagues"):
    try:
        r = requests.get(f"{config.API_BASE}/{ep}", headers={"x-apisports-key": os.getenv("API_SPORTS_KEY", "")}, timeout=30)
        print(f"\n== GET /{ep}: HTTP {r.status_code}")
        txt = r.text
        print(txt[:1500])
        if ep == "leagues" and r.ok:
            d = r.json()
            resp = d.get("response", [])
            print("\nresults:", d.get("results"), "errors:", d.get("errors"))
            print("first item keys:", list(resp[0].keys()) if resp else None)
            print(json.dumps(resp[0], indent=1)[:1500] if resp else "")
            config.ensure_dirs()
            (config.DATA / "samples_leagues.json").write_text(json.dumps(d)[:400000])
    except Exception:
        traceback.print_exc()
print("\n== parser on leagues")
try:
    lg = api.fetch_leagues()
    print("rows", len(lg)); print(lg.head(10).to_string())
    sel = api.selected_leagues(lg)
    print("selected:"); print(sel.groupby(["league_id", "country", "league"])["season"].agg(["min", "max", "count"]).to_string() if not sel.empty else "none")
except Exception:
    traceback.print_exc()
print("\n== one games request (first selected league, newest season)")
try:
    lg = api.store.read(config.LEAGUES_CSV); sel = api.selected_leagues(lg)
    if not sel.empty:
        row = sel.sort_values("season", ascending=False).iloc[0]
        r = requests.get(f"{config.API_BASE}/games", headers={"x-apisports-key": os.getenv("API_SPORTS_KEY", "")},
                         params={"league": int(row["league_id"]), "season": int(row["season"]), "timezone": "UTC"}, timeout=30)
        d = r.json(); resp = d.get("response", [])
        print("HTTP", r.status_code, "results", d.get("results"), "errors", d.get("errors"))
        print(json.dumps(resp[0], indent=1)[:2500] if resp else d)
        df = api.fetch_games(league=int(row["league_id"]), season=int(row["season"]))
        print("parsed rows", len(df), "finished", int(df["finished"].sum()) if len(df) else 0)
        print(df.head(5).to_string())
except Exception:
    traceback.print_exc()
print("\n== The Odds API")
try:
    from hb_oracle import odds
    print("handball sports:", odds.handball_sports(active_only=False))
except Exception:
    traceback.print_exc()
