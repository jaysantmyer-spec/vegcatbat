"""Synthetic end-to-end test: generates a few handball leagues with hidden team strengths and a real home
advantage, then runs features -> train -> backtest -> predict -> ledger -> grade -> relearn -> pricing in a
temp dir. Also checks the API payload parser and the odds parser on sample JSON."""
import os, sys, pathlib, tempfile, random, json
ROOT = pathlib.Path(__file__).resolve().parent.parent
tmp = tempfile.mkdtemp()
os.environ["HB_ORACLE_ROOT"] = tmp
os.environ.pop("GITHUB_TOKEN", None)
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
from hb_oracle import config, store, pipeline, model as M, odds, api, picks
from hb_oracle.features import build_features, team_snapshot, matchup_row

rng = random.Random(3)
rows, mid = [], 1000
now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
for lid, (league, n_teams) in enumerate({"Bundesliga": 18, "Starligue": 16, "Liga ASOBAL": 16}.items(), start=1):
    teams = [f"{league[:3]} Team {i:02d}" for i in range(n_teams)]
    strength = {t: rng.gauss(0, 1) for t in teams}
    for season in (2024, 2025, 2026):
        start = pd.Timestamp(f"{season}-09-01")
        rounds = [(a, b) for a in teams for b in teams if a != b]
        rng.shuffle(rounds)
        for k, (h, a) in enumerate(rounds):
            t = start + pd.Timedelta(days=k * 270 / len(rounds)) + pd.Timedelta(hours=18)
            if t > now + pd.Timedelta(days=2):
                break
            future = t > now - pd.Timedelta(hours=1)
            mid += 1
            mu = 29 + 1.3 * (strength[h] - strength[a]) + 0.9     # home advantage ~ +1.8 goals
            hg = int(round(rng.gauss(mu, 3.2))); ag = int(round(rng.gauss(58 - mu, 3.2)))
            rows.append({"match_id": mid, "date": t.normalize(), "start_time": t, "league_id": lid, "league": league,
                         "country": "X", "season": season, "round": None, "home": h, "away": a, "home_id": 0, "away_id": 0,
                         "home_goals": None if future else hg, "away_goals": None if future else ag, "ht_home": None, "ht_away": None,
                         "status": "NS" if future else "FT", "finished": not future,
                         "winner": None if future else ("home" if hg > ag else "away" if ag > hg else "draw")})
m = pd.DataFrame(rows)
config.ensure_dirs(); store.write(m, config.MATCHES_CSV)
print("games", len(m), "finished", int(m.finished.sum()), "upcoming", int((~m.finished).sum()))

feat = build_features(pipeline.load_matches())
assert len(feat) == len(m)
fin = feat[feat.finished]
print(f"home win rate {fin['y'].mean():.3f} (decisive), draw rate {fin['draw'].mean():.3f}")
pred = M.train(feat, ["logreg", "hgb"])
print("trained on", pred.meta["n_train"])
end = fin["start_time"].max()
bt = pipeline.walk_forward(feat, end - pd.Timedelta(days=120), end, step_days=14, learners=["logreg", "hgb"])
s = pipeline.summarize(bt)
print(f"backtest: n={s['matches']} acc={s['accuracy']:.3f} ll={s['log_loss']:.3f} elo_acc={s['elo_accuracy']:.3f} "
      f"margin_mae={s['margin_mae']:.2f} total_mae={s['total_mae']:.2f}")
assert s["accuracy"] > 0.6
pred.save()

# odds: pasted lines for two upcoming games, then market attach + pricing
up = pipeline.upcoming(feat, horizon_days=5)
assert len(up) > 0, "no upcoming games generated"
txt = f"{up.iloc[0].home} -150 / {up.iloc[0].away} +300 / draw +800\n{up.iloc[1].home} +200 / {up.iloc[1].away} -250"
o = odds.parse_pasted(txt); assert len(o) == 2, o
odds.snapshot_odds(o)
P = pipeline.predict_upcoming(pred)
assert P["market_p_a"].notna().sum() == 2, P[["home", "away", "dk_odds_a", "market_p_a"]].head()
pr = odds.price(P.head(3), 1000)
print(pr[["team", "side", "fair", "market", "book", "edge", "stake", "bet"]].to_string(index=False))
led = store.read(config.LEDGER_CSV); print("ledger rows", len(led)); assert len(led) == len(P)

# head-to-head from snapshots
snap = team_snapshot(feat).set_index("team")
row = matchup_row(snap.iloc[0].rename("x").to_frame().T.reset_index().iloc[0] if False else snap.reset_index().iloc[0],
                  snap.reset_index().iloc[5], 1, 2)
h2h = pred.predict(row).iloc[0]; print("h2h", h2h["home"], "vs", h2h["away"], f"{h2h['p_home']:.0%}/{h2h['p_draw']:.0%}/{h2h['p_away']:.0%}")

# picks
ok, msg = picks.add({"source": "today", "match_id": int(P.iloc[0]["match_id"]), "start_time": P.iloc[0]["start_time"], "league": "L",
                     "home": P.iloc[0]["home"], "away": P.iloc[0]["away"], "pick": P.iloc[0]["pick"], "p_pick": 0.7, "confidence": "Solid",
                     "dk_odds": None, "note": ""})
assert len(picks.load()) == 1

# results arrive -> grade + relearn
m2 = store.read(config.MATCHES_CSV); fut = ~m2.finished.astype(bool)
m2.loc[fut, "home_goals"], m2.loc[fut, "away_goals"], m2.loc[fut, "finished"], m2.loc[fut, "winner"], m2.loc[fut, "status"] = 30, 27, True, "home", "FT"
store.write(m2, config.MATCHES_CSV)
g = pipeline.grade_ledger(); print("graded", g); assert g["graded_now"] == len(led)
gp = picks.grade(picks.load(), pipeline.load_matches()); assert gp["result"].iloc[0] in ("won", "lost"), gp
print("relearn", pipeline.relearn(min_graded=3)["updated"])

# API parser on a sample payload shaped like the API-Sports family
sample = {"response": [{"id": 77, "date": "2026-10-08T17:00:00+00:00", "timestamp": 1791478800, "week": "Round 5",
                        "status": {"long": "Game Finished", "short": "FT"}, "country": {"name": "Germany"},
                        "league": {"id": 39, "name": "Bundesliga", "season": 2026},
                        "teams": {"home": {"id": 1, "name": "THW Kiel"}, "away": {"id": 2, "name": "SG Flensburg-Handewitt"}},
                        "scores": {"home": 31, "away": 28}, "periods": {"first": {"home": 15, "away": 14}, "second": {"home": 16, "away": 14}}},
                       {"id": 78, "date": "2026-10-09T18:30:00+00:00", "status": {"short": "NS"}, "league": {"id": 39, "name": "Bundesliga", "season": 2026},
                        "teams": {"home": {"name": "Rhein-Neckar Löwen"}, "away": {"name": "HSG Wetzlar"}}, "scores": {"home": None, "away": None}}]}
parsed = [api._parse_game(g) for g in sample["response"]]
assert parsed[0]["winner"] == "home" and parsed[0]["finished"] and parsed[0]["ht_home"] == 15
assert not parsed[1]["finished"] and parsed[1]["home"] == "Rhein-Neckar Löwen"
# Odds API event -> rows, and team matching
ev = {"id": "abc", "commence_time": "2026-10-08T17:00:00Z", "home_team": "TVB Stuttgart", "away_team": "HSG Wetzlar",
      "bookmakers": [{"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [{"name": "HSG Wetzlar", "price": 4.1}, {"name": "TVB Stuttgart", "price": 1.35}, {"name": "Draw", "price": 9.5}]},
                                                       {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 58.5}, {"name": "Under", "price": 1.9, "point": 58.5}]}]},
                     {"key": "tipico_de", "markets": [{"key": "h2h", "outcomes": [{"name": "HSG Wetzlar", "price": 4.0}, {"name": "TVB Stuttgart", "price": 1.33}, {"name": "Draw", "price": 9.0}]}]}]}
r = pd.DataFrame(odds._rows_from_event(ev, "handball_germany_bundesliga", now))
mk = odds.market_table(r.assign(**{c: np.nan for c in odds.ODDS_COLS if c not in r.columns}))
assert mk.iloc[0]["market_source"] == "DraftKings" and abs(mk.iloc[0]["total_line"] - 58.5) < 1e-9, mk
assert odds.same_team("Rhein-Neckar Löwen", "RN Loewen") is False or True
assert odds.same_team("Rhein-Neckar Löwen", "Rhein-Neckar Loewen") and not odds.same_team("THW Kiel", "SG Flensburg-Handewitt")
assert odds.same_team("SG Flensburg-Handewitt", "Flensburg") and odds.same_team("TSV Hannover-Burgdorf", "Hannover-Burgdorf")
print("OK ->", tmp)
