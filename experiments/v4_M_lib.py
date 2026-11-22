"""v4 / исполнитель M: модель и ансамбль на признаках v3 (признаки не трогаем).
Компоненты (лес, ET, HGB, логрегрессия) считаются один раз на (fit_df, pred_df) и кэшируются на диск,
поэтому смеси разных весов / ранговое усреднение не требуют повторного обучения лесов.
Каждая попытка (идея) пишется в v4_M.csv (единые колонки)."""
import os, sys, time, pickle, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import v3core as V
from v3core import (OUT, SEED, N_TREES, LR_C, make_rf, fit_preprocessing, apply_preprocessing)
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

CACHE = os.path.join(OUT, "v4_M_cache.pkl")
LOG = "v4_M.csv"
_C = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}   # свой файл, записан этим же модулем
_F = {}


def _key(df):
    return hashlib.md5(",".join(map(str, df["sample_id"].tolist())).encode()).hexdigest()


def _feats(fit_df, pred_df):
    kf = _key(fit_df)
    if kf not in _F:
        if len(_F) > 30: _F.pop(next(iter(_F)))
        prep = fit_preprocessing(fit_df)
        _F[kf] = (prep, apply_preprocessing(fit_df, prep), fit_df["label"].values)
    prep, Xf, yf = _F[kf]
    return Xf, yf, apply_preprocessing(pred_df, prep)


def f_rf(X, y, Xp, **kw):
    return make_rf(**kw).fit(X, y).predict_proba(Xp)[:, 1]


def f_et(X, y, Xp, **kw):
    kw.setdefault("n_estimators", N_TREES); kw.setdefault("max_features", "sqrt"); kw.setdefault("min_samples_leaf", 2)
    return ExtraTreesClassifier(random_state=SEED, n_jobs=1, **kw).fit(X, y).predict_proba(Xp)[:, 1]


def f_lr(X, y, Xp, C=LR_C, l1_ratio=None):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    if l1_ratio is None:
        m = LogisticRegression(C=C, max_iter=5000)
    else:
        m = LogisticRegression(C=C, penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, max_iter=5000, random_state=SEED)
    return m.fit((X - mu) / sd, y).predict_proba((Xp - mu) / sd)[:, 1]


def f_hgb(X, y, Xp, **kw):
    # ранняя остановка по внутреннему отложенному куску (20 %) ТОЛЬКО внутри fit_df
    p = dict(max_depth=2, learning_rate=0.03, max_iter=400, min_samples_leaf=10, l2_regularization=5.0, max_features=0.1,
             early_stopping=True, validation_fraction=0.2, n_iter_no_change=25, random_state=SEED)
    p.update(kw)
    return HistGradientBoostingClassifier(**p).fit(X, y).predict_proba(Xp)[:, 1]


_FIT = dict(rf=f_rf, et=f_et, lr=f_lr, hgb=f_hgb)


def comp(kind, params, fit_df, pred_df):
    k = (_key(fit_df), _key(pred_df), kind, tuple(sorted(params.items())))
    if k not in _C:
        Xf, yf, Xp = _feats(fit_df, pred_df)
        _C[k] = np.asarray(_FIT[kind](Xf, yf, Xp, **params), dtype=float)
    return _C[k]


def blend(ps, ws, mode):
    ws = np.asarray(ws, float) / np.sum(ws)
    if mode == "rank":
        ps = [(rankdata(p) - 1) / max(len(p) - 1, 1) for p in ps]
    return sum(w * p for w, p in zip(ws, ps))


def make_fp(spec, mode="prob"):
    def fp(fit_df, pred_df):
        return blend([comp(k, p, fit_df, pred_df) for k, p, w in spec], [w for _, _, w in spec], mode)
    return fp


def save_cache():
    tmp = CACHE + ".tmp"
    pickle.dump(_C, open(tmp, "wb")); os.replace(tmp, CACHE)


def wait_baseline():
    p = os.path.join(OUT, "v3_baseline.pkl")
    while True:
        try:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                pickle.load(open(p, "rb")); return
        except Exception:
            pass
        time.sleep(20)


COLS = ["stage", "name", "spec", "mode", "cv", "sd", "loco", "matched", "matched_d", "d_cv", "se", "wins", "d_loco", "d_matched",
        "d_matched_d", "cv_conf", "d_cv_conf", "adopt", "sec"]


def run(name, spec, mode="prob", confirm=False):
    wait_baseline()
    r = V.evaluate(make_fp(spec, mode), name, log=None, confirm=confirm)
    r.update(stage="confirm" if confirm else "screen", spec=repr(spec), mode=mode)
    path = os.path.join(OUT, LOG)
    pd.DataFrame([{c: r.get(c, np.nan) for c in COLS}]).to_csv(path, mode="a", header=not os.path.exists(path), index=False, encoding="utf-8")
    save_cache()
    return r


RF = ("rf", {}, 0.7); LR = ("lr", {"C": LR_C}, 0.3)
V3 = [RF, LR]
