"""
Handball Oracle - Streamlit front end.   Run:  streamlit run app.py

The app reads what the GitHub Actions refresh commits to data/: games, predictions, odds, ledger, teams and
backtest. It never pulls data or retrains on its own, so it stays fast on Streamlit Cloud.
"""
from __future__ import annotations

import html
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from hb_oracle import config, odds as O, picks, pipeline, store

st.set_page_config(page_title="Handball Oracle", page_icon="🤾", layout="wide")
config.ensure_dirs()
TZ = "America/New_York"

TEAL, NAVY, GOLD = "#0F8B8D", "#1E3A5F", "#E3B341"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Barlow', system-ui, sans-serif; }}
h1, h2, h3 {{ font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; }}
.m {{ border: 1px solid #D5DAE1; border-radius: 10px; padding: 12px 16px 10px; margin: 8px 0; background: #fff; }}
.m.hi {{ border: 2px solid {GOLD}; background: #FFFDF5; }}
.m .meta {{ display:flex; justify-content:space-between; color:#5B6470; font-size:.82rem; }}
.m .names {{ display:flex; justify-content:space-between; gap:12px; font-family:'Barlow Condensed',sans-serif; font-size:1.35rem; font-weight:600; }}
.m .a {{ color:{TEAL}; }} .m .b {{ color:{NAVY}; text-align:right; }}
.m .split {{ display:flex; height:12px; border-radius:6px; overflow:hidden; margin:6px 0 2px; }}
.m .split .ra {{ background:{TEAL}; }} .m .split .rd {{ background:#C9CED6; }} .m .split .rb {{ background:{NAVY}; }}
.m .pcts {{ display:flex; justify-content:space-between; font-family:'Barlow Condensed',sans-serif; font-size:1.2rem; font-weight:700; }}
.m .line {{ font-size:.9rem; margin-top:4px; }}
.badge {{ font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; padding:2px 7px; border-radius:4px; background:{GOLD}; color:#1B1D20; }}
.warn {{ background:#F3E6E6; color:#8B2E2E; }}
</style>""", unsafe_allow_html=True)


MATCH_COLS = ["match_id", "start_time", "league", "season", "home", "away", "home_goals", "away_goals", "finished", "winner"]


@st.cache_data(show_spinner=False)
def load(sig):
    m = pd.read_csv(config.MATCHES_CSV, usecols=MATCH_COLS, parse_dates=["start_time"]) if config.MATCHES_CSV.exists() else pd.DataFrame()
    if not m.empty:
        m["finished"] = m["finished"].astype(bool)
    return {"pred": store.read(config.PREDICTIONS_CSV), "matches": m, "odds": store.read(config.ODDS_CSV),
            "ledger": store.read(config.LEDGER_CSV), "bt": store.read(config.BACKTEST_CSV),
            "teams": store.read(config.TEAMS_CSV), "leagues": store.read(config.LEAGUES_CSV)}


def _sig():
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in
                 (config.PREDICTIONS_CSV, config.MATCHES_CSV, config.ODDS_CSV, config.LEDGER_CSV, config.BACKTEST_CSV, config.TEAMS_CSV))


def local(t) -> pd.Timestamp:
    """UTC on disk -> US Eastern on screen."""
    t = pd.Timestamp(t)
    return t.tz_localize("UTC").tz_convert(TZ) if t.tzinfo is None else t.tz_convert(TZ)


def _fold(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


D = load(_sig())
pred, matches, odds, ledger, bt, teams = D["pred"], D["matches"], D["odds"], D["ledger"], D["bt"], D["teams"]
state = json.loads(config.STATE_FILE.read_text()) if config.STATE_FILE.exists() else {}
e = html.escape


def card_html(r) -> str:
    ph, pdr, pa = float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])
    hi = r["confidence"] in ("Strong", "Solid") and not bool(r.get("thin_history", False))
    badge = f'<span class="badge">{e(str(r["confidence"]))} pick</span>' if hi else ""
    thin = '<span class="badge warn">thin history</span>' if bool(r.get("thin_history", False)) else ""
    t = local(r["start_time"])
    mk = ""
    if pd.notna(r.get("market_p_a")):
        edge = float(r["p_a"]) - float(r["market_p_a"])
        side = r["home"] if edge > 0 else r["away"]
        src = r.get("market_source") or "market"
        extra = []
        if pd.notna(r.get("dk_odds_draw")):
            extra.append(f"draw {O.decimal_to_american(r['dk_odds_draw'])}")
        if pd.notna(r.get("spread_home")):
            extra.append(f"spread {float(r['spread_home']):+.1f}")
        if pd.notna(r.get("total_line")):
            extra.append(f"total {float(r['total_line']):.1f}")
        mk = (f'<div class="line">{e(str(src))}: {e(str(r["home"]))} {O.decimal_to_american(r["dk_odds_a"])} / '
              f'{e(str(r["away"]))} {O.decimal_to_american(r["dk_odds_b"])}' + (" · " + " · ".join(extra) if extra else "") +
              f' · market {float(r["market_p_a"]):.0%} on home (2-way) · model sees {abs(edge):.0%} more on <b>{e(str(side))}</b></div>')
    em = f"{float(r['exp_margin']):+.1f}" if pd.notna(r.get("exp_margin")) else "–"
    et = f"{float(r['exp_total_model']):.1f}" if pd.notna(r.get("exp_total_model")) else "–"
    return f"""<div class="m{' hi' if hi else ''}"><div class="meta"><span>{t:%a %d %b %I:%M %p} ET · {e(str(r.get('league') or ''))}</span><span>{thin} {badge}</span></div>
<div class="names"><div class="a">{e(str(r['home']))} <span style="font-size:.8rem;color:#5B6470">home</span></div><div class="b">{e(str(r['away']))}</div></div>
<div class="split"><div class="ra" style="width:{ph*100:.1f}%"></div><div class="rd" style="width:{pdr*100:.1f}%"></div><div class="rb" style="width:{pa*100:.1f}%"></div></div>
<div class="pcts"><span style="color:{TEAL}">{ph:.0%}</span><span style="color:#5B6470;font-size:.9rem">draw {pdr:.0%}</span><span style="color:{NAVY}">{pa:.0%}</span></div>
<div class="line"><b>{e(str(r['pick']))}</b> ({str(r['confidence']).lower()}, {float(r['pick_prob']):.0%} draw-no-bet). Model margin {em} (home), model total {et}. Elo alone: {float(r['elo_p_a']):.0%} home. H2H games: {int(r['h2h_n'])}. Rest: {float(r['A_rest']):.0f} / {float(r['B_rest']):.0f} days.</div>{mk}</div>"""


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("Handball Oracle")
    if matches.empty:
        st.warning("No data yet. Run the `backfill` GitHub Actions workflow once (see DEPLOY.md).")
    else:
        fin = matches[matches["finished"]]
        st.write(f"**{len(fin):,}** games since {fin['start_time'].min():%b %Y} across **{matches['league'].nunique()}** competitions, "
                 f"**{pd.concat([matches['home'], matches['away']]).nunique():,}** teams")
        st.write(f"Results through **{local(fin['start_time'].max()):%d %b %Y}**")
        st.write(f"**{int((~matches['finished']).sum())}** fixtures on file")
    if state:
        st.caption(f"Learners: {', '.join(state.get('learners') or [])}")
    if not odds.empty:
        st.caption(f"Lines: {odds[['home', 'away']].drop_duplicates().shape[0]} games, {odds['bookmaker'].nunique()} books "
                   f"({local(pd.to_datetime(odds['fetched_at']).max()):%d %b %I:%M %p} ET)")
    st.divider()
    st.caption("Times shown in US Eastern. Data refreshes every 6 hours via GitHub Actions; predictions are logged before each game and graded after.")

tab_today, tab_price, tab_h2h, tab_mine, tab_track, tab_data = st.tabs(["Upcoming", "Pricing desk", "Head to head", "My picks", "Track record", "Data"])


def save_button(r, source: str, key: str):
    if st.button("Save pick", key=key, help="Adds this pick to the My picks tab"):
        side = "a" if str(r["pick"]) == str(r["home"]) else "b"
        ok, msg = picks.add({"source": source, "match_id": r.get("match_id") if source == "today" else None,
                             "start_time": pd.Timestamp(r["start_time"]), "league": r.get("league"),
                             "home": r["home"], "away": r["away"], "pick": r["pick"],
                             "p_pick": float(r["pick_prob"]), "confidence": r.get("confidence"),
                             "dk_odds": (O.decimal_to_american(r[f"dk_odds_{side}"]) if pd.notna(r.get(f"dk_odds_{side}")) else None),
                             "note": ""})
        st.toast(f"Saved: {r['pick']} ({msg})" if ok else f"Saved locally only — {msg}", icon="✅" if ok else "⚠️")


def search_filter(df, q, cols):
    if not q.strip():
        return df
    words = [_fold(w) for w in q.split()]
    both = df[cols[0]].map(_fold)
    for c in cols[1:]:
        both = both + " " + df[c].map(_fold)
    return df[both.apply(lambda s: all(w in s for w in words))]


# --------------------------------------------------------------------------- upcoming
with tab_today:
    if pred.empty:
        st.info("No predictions yet. Once the refresh workflow has run with data, upcoming games appear here.")
    else:
        P = pred.copy()
        P["start_time"] = pd.to_datetime(P["start_time"])
        P["day"] = P["start_time"].map(lambda t: local(t).date())
        P = P.sort_values("start_time")
        c1, c2, c3 = st.columns(3)
        c1.metric("Upcoming games", len(P))
        ok = ~P["thin_history"].astype(bool)
        c2.metric("Solid / strong picks", int((P["confidence"].isin(["Solid", "Strong"]) & ok).sum()), f"{int(((P['pick_prob'] >= 0.75) & ok).sum())} at 75%+")
        c3.metric("With a market line", int(P["market_p_a"].notna().sum()) if "market_p_a" in P else 0)
        cd1, cd2, cd3 = st.columns([1, 1, 2])
        day = cd1.selectbox("Day", ["All"] + sorted(P["day"].unique()))
        lg = cd2.selectbox("Competition", ["All"] + sorted(P["league"].dropna().unique()))
        min_p = cd3.radio("Minimum model probability (draw no bet)", [0.0, 0.62, 0.70, 0.75], index=0, horizontal=True,
                          format_func=lambda v: {0.0: "All games", 0.62: "Solid+ (62%+)", 0.70: "Strong (70%+)", 0.75: "75%+"}[v])
        q = st.text_input("Search a team or game", placeholder="e.g. Kiel, or Kiel Flensburg", key="today_q")
        view = P if day == "All" else P[P["day"] == day]
        if lg != "All":
            view = view[view["league"] == lg]
        if min_p > 0:
            view = view[(view["pick_prob"] >= min_p) & ~view["thin_history"].astype(bool)]
        view = search_filter(view, q, ["home", "away"])
        st.caption(f"{len(view)} games shown" + (" (teams with thin history excluded)" if min_p else ""))
        for d, vd in view.groupby("day"):
            st.subheader(f"{pd.Timestamp(d):%A %d %B} · {len(vd)} games")
            for _, r in vd.iterrows():
                st.markdown(card_html(r), unsafe_allow_html=True)
                save_button(r, "today", f"save_{r['match_id']}")
        st.download_button("Download predictions (CSV)", P.to_csv(index=False), file_name="handball_predictions.csv")

# --------------------------------------------------------------------------- pricing desk
with tab_price:
    st.write("Model probability as a fair price vs the market (DraftKings when it has the game, otherwise the vig-free "
             "average of every book on the feed). A bet is flagged when the edge beats your threshold, EV is positive, "
             "and both teams have enough history. Sides are priced on the 3-way moneyline (a draw loses).")
    c1, c2, c3 = st.columns(3)
    bankroll = c1.number_input("Bankroll ($)", 50.0, 1_000_000.0, 500.0, step=50.0)
    kf = c2.select_slider("Kelly fraction", options=[0.1, 0.25, 0.5], value=0.25,
                          format_func=lambda v: {0.1: "1/10", 0.25: "Quarter", 0.5: "Half"}[v])
    thr = c3.slider("Edge threshold", 0.0, 0.15, 0.05, 0.01, format="%.2f")
    cfa, cfb = st.columns([1, 3])
    if cfa.button("Fetch lines now"):
        with st.spinner("Asking the odds feed…"):
            try:
                res = O.snapshot_odds()
                if res.get("lines"):
                    st.success(f"Got {res['lines']} book lines on {res.get('games')} games from {res.get('books')} books.")
                    st.cache_data.clear()
                    odds = store.read(config.ODDS_CSV)
                else:
                    st.warning("The feed returned no handball lines right now. Details below; the paste box still works.")
                    st.code("\n".join(O.LAST_FETCH_LOG) or "(no log)")
            except Exception as ex:
                st.error(f"Fetch failed: {ex}")
                st.code("\n".join(O.LAST_FETCH_LOG) or "(no log)")
    cfb.caption("Uses The Odds API (ODDS_API_KEY) and, failing that, DraftKings' site feed. Each pull costs about 4 of the "
                "500 monthly credits on the free plan; the refresh already pulls twice a day.")
    with st.expander("Paste lines (if the feed has nothing)"):
        st.caption("One game per line, home team first: `THW Kiel -250 / SG Flensburg +190 / draw +900`. Odds can be American or decimal.")
        txt = st.text_area("Lines", height=120, key="paste_odds")
        if st.button("Use these lines"):
            o = O.parse_pasted(txt)
            if o.empty:
                st.error("Couldn't read any lines.")
            else:
                O.snapshot_odds(o)
                st.success(f"Saved {len(o)} lines. Re-pricing below.")
                st.cache_data.clear()
                odds = store.read(config.ODDS_CSV)
    if pred.empty:
        st.info("No predictions on file.")
    else:
        P = O.attach_market(pred, odds) if not odds.empty else pred
        priced = O.price(P, bankroll, kf, thr)
        pq = st.text_input("Search a team", placeholder="e.g. Kiel", key="price_q")
        priced = search_filter(priced, pq, ["team", "opponent"])
        flagged = priced[priced["bet"]].sort_values("edge", ascending=False)
        st.subheader(f"Flagged bets: {len(flagged)}")
        fmt = {"p_model": "{:.0%}", "p_market": "{:.0%}", "edge": "{:+.1%}", "ev": "{:+.2f}", "stake": "${:,.0f}"}
        show_cols = ["start_time", "league", "team", "opponent", "side", "p_model", "fair", "book", "source", "p_market", "edge", "ev", "stake"]
        if not flagged.empty:
            f = flagged.copy(); f["start_time"] = f["start_time"].map(lambda t: local(t).strftime("%a %d %b %I:%M %p"))
            st.dataframe(f[show_cols].style.format(fmt), hide_index=True)
        else:
            st.caption("Nothing clears the bar right now" + ("" if not odds.empty else " — no lines loaded."))
        st.subheader("Every side")
        a = priced.copy(); a["start_time"] = a["start_time"].map(lambda t: local(t).strftime("%a %d %b %I:%M %p"))
        st.dataframe(a[show_cols + ["thin", "bet"]].style.format(fmt), hide_index=True)
        if "total_line" in P and P["total_line"].notna().any():
            st.subheader("Totals and spreads: model vs line")
            tl = P[P["total_line"].notna() | P["spread_home"].notna()].copy()
            tl["start_time"] = tl["start_time"].map(lambda t: local(t).strftime("%a %d %b %I:%M %p"))
            tl["model_total"], tl["model_margin"] = tl["exp_total_model"], tl["exp_margin"]
            tl["total_diff"] = tl["model_total"] - tl["total_line"]
            tl["spread_diff"] = tl["model_margin"] + tl["spread_home"]      # book spread is from the home side's view
            st.dataframe(tl[["start_time", "home", "away", "model_total", "total_line", "total_diff", "model_margin", "spread_home", "spread_diff"]]
                         .style.format({c: "{:+.1f}" if "diff" in c or "margin" in c or "spread" in c else "{:.1f}" for c in
                                        ["model_total", "total_line", "total_diff", "model_margin", "spread_home", "spread_diff"]}), hide_index=True)
            st.caption("Positive total_diff: model expects more goals than the line (lean over). Positive spread_diff: model likes the home side against the spread. "
                       "The backtest tab shows how far off the margin/total models typically are; treat leans under ~2 goals as noise.")

# --------------------------------------------------------------------------- head to head
with tab_h2h:
    if matches.empty or teams.empty:
        st.info("Team ratings appear after the first refresh run.")
    else:
        names = teams["team"].tolist()
        folded = {n: _fold(n) for n in names}

        def team_picker(col, label, key, default_index):
            q = col.text_input(f"Search {label}", key=f"{key}_q", placeholder="type part of a name, e.g. kiel")
            toks = _fold(q).split()
            opts = [n for n in names if all(t in folded[n] for t in toks)] if toks else names
            if not opts:
                col.warning(f"No team matching '{q}'.")
                opts = names
            if toks:
                opts = sorted(opts, key=lambda n: 0 if any(w.startswith(toks[0]) for w in folded[n].split()) else 1)
            return col.selectbox(label, opts, index=0 if toks else min(default_index, len(opts) - 1), key=f"{key}_sel")

        c1, c2 = st.columns(2)
        a = team_picker(c1, "Home team", "h2h_a", 0)
        b = team_picker(c2, "Away team", "h2h_b", 1 if len(names) > 1 else 0)
        neutral = st.checkbox("Neutral venue", value=False)
        fin = matches[matches["finished"]]
        h = fin[((fin["home"] == a) & (fin["away"] == b)) | ((fin["home"] == b) & (fin["away"] == a))].sort_values("start_time", ascending=False)
        a_pts = float(((h["home"] == a) & (h["winner"] == "home")).sum() + ((h["away"] == a) & (h["winner"] == "away")).sum() + 0.5 * (h["winner"] == "draw").sum())
        sa, sb = teams.set_index("team").loc[a], teams.set_index("team").loc[b]
        if a != b:
            @st.cache_resource(show_spinner=False)
            def _model():
                from hb_oracle.model import Predictor
                return Predictor.load()
            mdl = _model()
            if mdl is None:
                st.caption("Model file not available on this server yet.")
            else:
                from hb_oracle.features import matchup_row
                row = matchup_row(sa.to_dict() | {"team": a}, sb.to_dict() | {"team": b}, a_pts, len(h), neutral)
                pr = mdl.predict(row).iloc[0]
                pr["elo_p_a"] = row["elo_p_a"].iloc[0]; pr["h2h_n"] = len(h)
                pr["A_rest"], pr["B_rest"] = 7, 7
                pr["league"] = "hypothetical, both sides rested"; pr["start_time"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
                st.markdown(card_html(pr), unsafe_allow_html=True)
                save_button(pr, "h2h", f"save_h2h_{a}_{b}_{neutral}")
                st.caption("Model prediction if this game were played now with both sides on a week's rest. The Upcoming tab uses the real schedule and fatigue.")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{a} Elo", f"{sa['elo']:.0f}", f"last 10: {sa['win10']:.0%} · {sa['gf10']:.0f}-{sa['ga10']:.0f} avg")
        c2.metric("Head to head (pts)", f"{a_pts:g}–{len(h) - a_pts:g}", f"{len(h)} games")
        c3.metric(f"{b} Elo", f"{sb['elo']:.0f}", f"last 10: {sb['win10']:.0%} · {sb['gf10']:.0f}-{sb['ga10']:.0f} avg")
        if not h.empty:
            hh = h.head(20).copy(); hh["start_time"] = hh["start_time"].map(lambda t: local(t).strftime("%Y-%m-%d"))
            st.dataframe(hh[["start_time", "league", "home", "home_goals", "away_goals", "away"]], hide_index=True)
        st.subheader("Ratings leaderboard")
        st.dataframe(teams.head(50)[["team", "league", "elo", "win10", "gf10", "ga10", "career_n"]].rename(columns={"career_n": "games"})
                     .style.format({"elo": "{:.0f}", "win10": "{:.0%}", "gf10": "{:.1f}", "ga10": "{:.1f}"}), hide_index=True)

# --------------------------------------------------------------------------- my picks
with tab_mine:
    mine = picks.grade(picks.load(), matches)
    if picks._token() is None:
        st.warning("Picks are saved on the app's disk only, which resets on every redeploy (about every 6 hours). "
                   "To keep them permanently, add a GitHub token to the app's Secrets (see DEPLOY.md, 'Saved picks').")
    if mine.empty:
        st.info("No saved picks yet. Use the Save pick button under any card on the Upcoming or Head to head tabs.")
    else:
        g = mine[mine["result"].notna()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Saved picks", len(mine))
        c2.metric("Graded", len(g))
        c3.metric("Record", f"{int((g['result'] == 'won').sum())}–{int((g['result'] == 'lost').sum())}–{int((g['result'] == 'draw').sum())}" if len(g) else "–",
                  f"{(g['result'] == 'won').mean():.0%} hit" if len(g) else None)
        show = mine.sort_values("start_time", ascending=False).copy()
        show["game"] = show["home"].astype(str) + " v " + show["away"].astype(str)
        show["result"] = show["result"].fillna("pending")
        show["when"] = show["start_time"].map(lambda t: local(t).strftime("%a %d %b %I:%M %p") if pd.notna(t) else "")
        st.dataframe(show[["when", "league", "game", "pick", "p_pick", "confidence", "dk_odds", "source", "result", "winner"]]
                     .rename(columns={"p_pick": "model"}).style.format({"model": "{:.0%}"}), hide_index=True)
        st.subheader("Manage")
        labels = {r["pick_id"]: f"{r['when']} · {r['pick']} ({r['game']})" for _, r in show.iterrows()}
        sel = st.multiselect("Remove selected picks", options=list(labels), format_func=labels.get)
        cm1, cm2 = st.columns([1, 3])
        if cm1.button("Remove", disabled=not sel):
            ok, msg = picks.remove(sel); st.toast(msg); st.rerun()
        sure = cm2.checkbox("I want to clear all saved picks")
        if cm2.button("Clear all", type="primary", disabled=not sure):
            ok, msg = picks.clear(); st.toast(msg); st.rerun()
        st.download_button("Download my picks (CSV)", show.to_csv(index=False), file_name="my_handball_picks.csv")

# --------------------------------------------------------------------------- track record
with tab_track:
    st.write("Two views: the **ledger** (real predictions logged before each game, then graded) and the "
             "**walk-forward backtest** (retrained every two weeks on earlier games only). Hit rates are on decisive games; draws are pushes.")
    g = ledger[ledger["y"].notna() & (ledger["y"] != 0.5)] if not ledger.empty else pd.DataFrame()
    if g.empty:
        st.caption("Ledger: no graded predictions yet.")
    else:
        g = g.copy(); g["correct"] = ((g["p_a"].astype(float) >= 0.5) == (g["y"].astype(float) == 1)).astype(int)
        c1, c2, c3 = st.columns(3)
        c1.metric("Graded predictions", len(g))
        c2.metric("Winners called", f"{g['correct'].mean():.1%}")
        hi = g[g["confidence"].isin(["Strong", "Solid"])]
        c3.metric("Strong / solid hit rate", f"{hi['correct'].mean():.1%}" if len(hi) else "–", f"{len(hi)} picks")
        st.dataframe(g.groupby("confidence")["correct"].agg(picks="size", hit_rate="mean").reindex(["Strong", "Solid", "Lean", "Coin flip"])
                     .style.format({"hit_rate": "{:.1%}"}))
        if "market_p_a" in g and g["market_p_a"].notna().sum() >= 20:
            mk = g[g["market_p_a"].notna()]
            st.caption(f"On {len(mk)} games with a market line: model {mk['correct'].mean():.1%} vs market favourite "
                       f"{((mk['market_p_a'].astype(float) >= 0.5) == (mk['y'].astype(float) == 1)).mean():.1%}.")
    st.subheader("Walk-forward backtest")
    if bt.empty:
        st.caption("No backtest on file yet (the backfill runs one; the refresh workflow repeats it weekly).")
    else:
        s = pipeline.summarize(bt)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Games", s["matches"], f"{s['decisive']} decisive")
        c2.metric("Accuracy", f"{s['accuracy']:.1%}", f"home wins {s['home_win_rate']:.0%}")
        c3.metric("Log loss", f"{s['log_loss']:.4f}"); c4.metric("Elo-only accuracy", f"{s['elo_accuracy']:.1%}")
        if "margin_mae" in s:
            k1, k2, k3 = st.columns(3)
            k1.metric("Margin error (avg goals)", f"{s['margin_mae']:.2f}")
            k2.metric("Total error (avg goals)", f"{s['total_mae']:.2f}")
            k3.metric("Draw rate", f"{s['draw_rate']:.1%}")
        st.dataframe(pd.DataFrame(s["by_confidence"]).T.reindex(["Strong", "Solid", "Lean", "Coin flip"]).style.format({"hit_rate": "{:.1%}"}))
        st.dataframe(pipeline.calibration_table(bt).style.format({"predicted": "{:.1%}", "actual": "{:.1%}"}), hide_index=True)
        if "league" in bt:
            d = bt[bt["y"].notna()]
            st.dataframe(d.groupby("league")["correct"].agg(games="size", hit_rate="mean").sort_values("games", ascending=False)
                         .style.format({"hit_rate": "{:.1%}"}))

# --------------------------------------------------------------------------- data
with tab_data:
    st.write("Everything here is refreshed by the `refresh` GitHub Actions workflow. See DEPLOY.md to run it by hand.")
    if not matches.empty:
        cov = matches.groupby(["league", "season"]).agg(games=("match_id", "size"), finished=("finished", "sum"),
                                                        first=("start_time", "min"), last=("start_time", "max")).reset_index()
        st.dataframe(cov.sort_values(["league", "season"]), hide_index=True)
    if config.USAGE_JSON.exists():
        st.caption(f"API-Sports usage: {config.USAGE_JSON.read_text()}")
    if state:
        st.json({k: v for k, v in state.items() if k != "history"})
        if state.get("history"):
            st.dataframe(pd.DataFrame(state["history"]).tail(10))
