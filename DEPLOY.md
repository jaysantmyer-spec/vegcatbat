# Handball Oracle — setup

Same shape as the TT Elite Oracle: GitHub Actions do the data work and commit to `data/`; the Streamlit app only
reads those files.

## 1. Secrets (GitHub → repo → Settings → Secrets and variables → Actions)

| Secret | What | Where to get it |
|---|---|---|
| `API_SPORTS_KEY` | match history, results, fixtures (API-Sports Handball, free 100 requests/day) | dashboard.api-football.com → your key |
| `ODDS_API_KEY` | moneyline / spread / total from ~20 books incl. DraftKings (The Odds API, free 500 credits/month) | the-odds-api.com → dashboard |

Optional **repository variable** `HB_LEAGUE_IDS` (Settings → Secrets and variables → Actions → Variables): a comma
list of API-Sports league ids to follow instead of the name patterns in `hb_oracle/config.py`. `data/leagues.csv`
(written by the first run) lists every league the API knows with its id.

## 2. First data pull

Actions → **backfill history** → Run workflow. One API request per league-season (default 4 seasons × ~10 leagues),
well inside the free plan. If it stops early with "daily budget reached", run it again the next day; it continues
where it left off. It then trains, backtests and predicts, and commits everything.

The **refresh** workflow then runs every 6 hours: results + fixtures (yesterday → +3 days, 5 requests), odds twice a
day, grade the ledger, relearn ensemble weights, retrain, predict.

## 3. Streamlit Cloud

share.streamlit.io → New app → this repo, branch `main`, file `app.py`. In the app's **Settings → Secrets** add:

```toml
ODDS_API_KEY = "..."        # lets the "Fetch lines now" button work from the app
GITHUB_TOKEN = "github_pat_..."   # optional: makes saved picks permanent (see below)
```

## Saved picks (My picks tab)

Streamlit's disk is wiped on every redeploy, so picks are committed to `data/my_picks.csv` through the GitHub API
when a token is present. Fine-grained personal access token: GitHub → Settings → Developer settings → Personal
access tokens → Fine-grained → repository `vegcatbat`, permission **Contents: read and write**. Put it in the
app's Secrets as `GITHUB_TOKEN`.

## Run by hand

```
export API_SPORTS_KEY=... ODDS_API_KEY=...
python -m hb_oracle leagues          # 1 request: lists leagues, shows which ones are selected
python -m hb_oracle backfill         # history
python -m hb_oracle cycle            # update -> odds -> grade -> learn -> retrain -> predict
python -m hb_oracle backtest --days 180
streamlit run app.py
```

## Budget notes

* API-Sports free plan: 100 requests/day, counted in `data/api_usage.json` (stop at 95). A refresh uses 5.
* The Odds API free plan: 500 credits/month. A pull costs 4 credits per priced competition (US books with
  h2h+spreads+totals = 3, EU/UK books h2h = 1). Two pulls a day ≈ 240–480/month depending on how many
  competitions are live; the app's "Fetch lines now" button spends the same per click.
