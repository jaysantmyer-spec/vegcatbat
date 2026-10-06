"""
Ensemble win model for handball: logistic regression + histogram gradient boosting, plus XGBoost / LightGBM /
CatBoost when installed, blended with Hedge weights and a temperature calibration (same learning engine as
the UFC and TT Elite oracles). Alongside the win model: a draw model (handball draws about 1 game in 12),
and ridge regressions for the goal margin and the total, used for spreads and totals.

p_a is P(home wins | no draw) — the "draw no bet" / 2-way probability the ledger and backtest grade.
p_home / p_draw / p_away are the 3-way probabilities shown on the cards and priced against 3-way lines.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import CONTEXT_FEATURES, DIFF_FEATURES


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


HAS_XGB, HAS_LGBM, HAS_CAT = _importable("xgboost"), _importable("lightgbm"), _importable("catboost")
LEARNER_NAMES = {"logreg": "Logistic regression", "hgb": "Hist. gradient boosting", "xgb": "XGBoost",
                 "lgbm": "LightGBM", "cat": "CatBoost"}


def available_learners() -> list[str]:
    out = ["logreg", "hgb"]
    if HAS_XGB:
        out.append("xgb")
    if HAS_LGBM:
        out.append("lgbm")
    if HAS_CAT:
        out.append("cat")
    return out


def make_learner(name: str, seed: int = 42):
    if name == "logreg":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.1, max_iter=3000))])
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15,
                                              min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.8,
                                 colsample_bytree=0.7, min_child_weight=15, reg_lambda=2.0,
                                 eval_metric="logloss", n_jobs=-1, random_state=seed)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=350, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                                  verbose=-1, random_state=seed)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=5.0, subsample=0.8,
                                  bootstrap_type="Bernoulli", loss_function="Logloss", allow_writing_files=False,
                                  verbose=0, thread_count=-1, random_seed=seed)
    raise ValueError(name)


def _fit(est, X, y, w):
    if w is None:
        return est.fit(X, y)
    if isinstance(est, Pipeline):
        last = est.steps[-1][0]
        return est.fit(X, y, **{f"{last}__sample_weight": w})
    return est.fit(X, y, sample_weight=w)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blend(components: pd.DataFrame, weights: dict | None = None, temperature: float = 1.0) -> np.ndarray:
    names = list(components.columns)
    w = np.array([(weights or {}).get(n, 1.0) for n in names], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1 / len(names))
    p = components.to_numpy(float) @ w
    if temperature != 1.0:
        p = _sigmoid(temperature * _logit(p))
    return np.clip(p, 1e-4, 1 - 1e-4)


def hedge_weights(components: pd.DataFrame, y, eta: float) -> dict:
    y = np.asarray(y, float)
    out = {}
    for c in components.columns:
        p = np.clip(components[c].to_numpy(float), 1e-6, 1 - 1e-6)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        out[c] = float(np.exp(-eta * ll * 100))
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def fit_temperature(p, y) -> float:
    y = np.asarray(y, float)
    z = _logit(np.asarray(p, float))
    best, best_ll = 1.0, np.inf
    for t in np.linspace(0.5, 2.0, 31):
        q = np.clip(_sigmoid(t * z), 1e-6, 1 - 1e-6)
        ll = -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
        if ll < best_ll:
            best, best_ll = float(t), ll
    return best


def _matrix(df: pd.DataFrame) -> np.ndarray:
    D = df[[f"d_{k}" for k in DIFF_FEATURES]].to_numpy(float)
    C = df[CONTEXT_FEATURES].to_numpy(float)
    return np.hstack([D, C])


def _draw_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.abs(df["d_elo"].to_numpy(float)), np.abs(df["d_margin10"].to_numpy(float)),
                            df["exp_total"].to_numpy(float), df["neutral"].to_numpy(float)])


OUT_COLS = ["match_id", "date", "start_time", "league", "home", "away", "finished", "y", "elo_p_a", "h2h_n",
            "A_career_n", "B_career_n", "A_elo", "B_elo", "A_win10", "B_win10", "A_rest", "B_rest",
            "A_games7", "B_games7", "exp_total"]


@dataclass
class Predictor:
    models: dict
    learners: list
    weights: dict = field(default_factory=dict)
    temperature: float = 1.0
    meta: dict = field(default_factory=dict)
    draw_model: object = None
    margin_model: object = None
    total_model: object = None

    def save(self, path=config.MODEL_FILE):
        config.ensure_dirs()
        joblib.dump(self, path)

    @staticmethod
    def load(path=config.MODEL_FILE):
        try:
            return joblib.load(path)
        except Exception:
            return None

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _matrix(df)
        return pd.DataFrame({n: m.predict_proba(X)[:, 1] for n, m in self.models.items()}, index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        comp = self.components(df)
        q = blend(comp, self.weights, self.temperature)                       # P(home | decisive)
        pdraw = (np.clip(self.draw_model.predict_proba(_draw_matrix(df))[:, 1], 0.01, 0.4)
                 if self.draw_model is not None else np.full(len(df), 0.08))
        out = df[[c for c in OUT_COLS if c in df.columns]].copy()
        out["p_a"], out["p_b"] = q, 1 - q
        out["p_draw"] = pdraw
        out["p_home"], out["p_away"] = q * (1 - pdraw), (1 - q) * (1 - pdraw)
        X = _matrix(df)
        out["exp_margin"] = self.margin_model.predict(X) if self.margin_model is not None else np.nan
        out["exp_total_model"] = self.total_model.predict(X) if self.total_model is not None else df["exp_total"]
        for c in comp.columns:
            out[f"comp_{c}"] = comp[c].to_numpy()
        out["pick"] = np.where(q >= 0.5, out["home"], out["away"])
        out["pick_prob"] = np.maximum(q, 1 - q)
        out["confidence"] = pd.cut(out["pick_prob"], [0, 0.55, 0.62, 0.70, 1.01],
                                   labels=["Coin flip", "Lean", "Solid", "Strong"]).astype(str)
        mh = config.DEFAULTS["min_history"]
        out["thin_history"] = (out["A_career_n"] < mh) | (out["B_career_n"] < mh)
        return out


def train(feat: pd.DataFrame, learners: list[str] | None = None, cutoff=None, half_life_days: float | None = None,
          hardness: dict | None = None, hardness_alpha: float = 0.0, weights: dict | None = None,
          temperature: float = 1.0, seed: int = 42) -> Predictor:
    hl = config.DEFAULTS["half_life_days"] if half_life_days is None else half_life_days
    fin = feat[feat["finished"].astype(bool) & feat["draw"].notna()]
    if cutoff is not None:
        fin = fin[fin["start_time"] < pd.Timestamp(cutoff)]
    fin = fin[fin["both_n_min"] >= 1]
    tr = fin[fin["y"].notna()]                                               # decisive games for the win model
    if len(tr) < 200:
        raise ValueError(f"only {len(tr)} finished games with history; need more data")
    ref = pd.Timestamp(cutoff) if cutoff is not None else tr["start_time"].max()
    age = (ref - tr["start_time"]).dt.total_seconds() / 86400.0
    w = (0.5 ** (age / hl)).to_numpy(float)
    if hardness and hardness_alpha > 0:
        h = tr["match_id"].map(hardness).fillna(0.0).to_numpy(float)
        w = w * (1.0 + hardness_alpha * np.clip(h, 0, 1))
    X, y = _matrix(tr), tr["y"].to_numpy(int)
    learners = learners or available_learners()
    models = {n: _fit(make_learner(n, seed), X, y, w) for n in learners}
    # draw, margin, total models on every finished game (draws included)
    wf = (0.5 ** (((ref - fin["start_time"]).dt.total_seconds() / 86400.0) / hl)).to_numpy(float)
    draw = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.5, max_iter=2000))])
    draw.fit(_draw_matrix(fin), fin["draw"].to_numpy(int), clf__sample_weight=wf)
    Xf = _matrix(fin)
    margin = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("r", Ridge(alpha=3.0))])
    margin.fit(Xf, fin["margin"].to_numpy(float), r__sample_weight=wf)
    total = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("r", Ridge(alpha=3.0))])
    total.fit(Xf, fin["total"].to_numpy(float), r__sample_weight=wf)
    return Predictor(models, list(learners), weights or {}, temperature, {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"), "n_train": int(len(tr)),
        "data_through": str(tr["start_time"].max()), "learners": list(learners), "half_life_days": hl,
        "hardness_alpha": hardness_alpha, "version": pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")},
        draw, margin, total)


# --------------------------------------------------------------------------- learning state
def load_state() -> dict:
    if config.STATE_FILE.exists():
        return json.loads(config.STATE_FILE.read_text())
    return {"weights": {}, "temperature": 1.0, "hardness_alpha": config.DEFAULTS["hardness_alpha"],
            "half_life_days": config.DEFAULTS["half_life_days"], "learners": None, "history": []}


def save_state(state: dict) -> None:
    config.ensure_dirs()
    config.STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
