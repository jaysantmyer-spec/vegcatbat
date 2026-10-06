"""
Headless usage (what the GitHub Actions workflows call):
    python -m hb_oracle leagues                 # list / resolve league ids (1 API request)
    python -m hb_oracle backfill [--seasons 4]  # history for the selected leagues (1 request per league-season)
    python -m hb_oracle update                  # results + fixtures around today
    python -m hb_oracle odds                    # The Odds API / DraftKings snapshot
    python -m hb_oracle train
    python -m hb_oracle predict
    python -m hb_oracle backtest --days 120
    python -m hb_oracle cycle                   # update -> odds -> grade -> learn -> retrain -> predict
"""
import argparse
import json
import logging

import pandas as pd

from . import api, config, pipeline, store


def _p(f, m):
    print(f"[{f:5.0%}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="hb_oracle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("leagues")
    b = sub.add_parser("backfill"); b.add_argument("--seasons", type=int, default=None)
    sub.add_parser("update")
    sub.add_parser("odds")
    sub.add_parser("train")
    pr = sub.add_parser("predict"); pr.add_argument("--no-log", action="store_true")
    bt = sub.add_parser("backtest"); bt.add_argument("--days", type=int, default=120); bt.add_argument("--step", type=int, default=14)
    bt.add_argument("--static", action="store_true")
    c = sub.add_parser("cycle"); c.add_argument("--no-update", action="store_true"); c.add_argument("--no-odds", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config.ensure_dirs()

    if a.cmd == "leagues":
        lg = api.fetch_leagues()
        sel = api.selected_leagues(lg)
        print(f"{lg['league_id'].nunique() if not lg.empty else 0} leagues known; selected:")
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(sel.groupby(["league_id", "country", "league"])["season"].agg(["min", "max", "count"]).to_string() if not sel.empty else "none")
    elif a.cmd == "backfill":
        print(json.dumps(api.backfill(a.seasons, progress=_p), indent=2))
    elif a.cmd == "update":
        print(json.dumps(api.update(), indent=2))
    elif a.cmd == "odds":
        from .odds import snapshot_odds
        print(json.dumps(snapshot_odds(), indent=2, default=str))
    elif a.cmd == "train":
        print(pipeline.retrain().meta)
    elif a.cmd == "predict":
        P = pipeline.predict_upcoming(log_to_ledger=not a.no_log)
        with pd.option_context("display.width", 220, "display.max_rows", 300):
            print(P[["start_time", "league", "home", "away", "p_home", "p_draw", "p_away", "pick", "confidence",
                     "exp_margin", "exp_total_model", "market_p_a"]].round(3).to_string(index=False)
                  if not P.empty else "no upcoming games")
    elif a.cmd == "backtest":
        feat = pipeline.features_now()
        end = feat[feat["finished"].astype(bool)]["start_time"].max()
        bt_df = pipeline.walk_forward(feat, end - pd.Timedelta(days=a.days), end, step_days=a.step,
                                      adaptive=not a.static, progress=_p)
        store.write(bt_df, config.BACKTEST_CSV)
        print(json.dumps(pipeline.summarize(bt_df), indent=2, default=str))
    elif a.cmd == "cycle":
        print(json.dumps(pipeline.run_cycle(_p, scrape=not a.no_update, odds=not a.no_odds), indent=2, default=str))


if __name__ == "__main__":
    main()
