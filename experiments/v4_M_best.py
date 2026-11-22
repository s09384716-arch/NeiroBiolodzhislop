"""v4 / M: лучшая НЕпринятая конфигурация (по правилу v3core ни одна идея не прошла).
Те же признаки v3; единственное отличие — усреднение РАНГОВ вместо вероятностей:
    0.7 * ранг(RandomForest(500, sqrt, leaf=2)) + 0.3 * ранг(L2-логрегрессия C=0.01).
Δ к v3 на стенде: CV +0.0011 (SE 0.0028), LOCO +0.0012, matched +0.0054, matched+глубина +0.0039 — внутри шума."""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from v3core import make_rf, fit_preprocessing, apply_preprocessing, LR_C, W_RF


def fit_predict_best(fit_df, pred_df):
    prep = fit_preprocessing(fit_df)
    X, y, Xp = apply_preprocessing(fit_df, prep), fit_df["label"].values, apply_preprocessing(pred_df, prep)
    p_rf = make_rf().fit(X, y).predict_proba(Xp)[:, 1]
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    p_lr = LogisticRegression(C=LR_C, max_iter=5000).fit((X - mu) / sd, y).predict_proba((Xp - mu) / sd)[:, 1]
    r = lambda p: (rankdata(p) - 1) / max(len(p) - 1, 1)
    return W_RF * r(p_rf) + (1 - W_RF) * r(p_lr)
