"""Лучшая (по числам) конфигурация исполнителя F: v3 + очистка остатками 4 литературных признаков (A_lit_resid).
ВНИМАНИЕ: по заранее заданному правилу НЕ принята (Δloco +0.0003 < +0.005, Δmatched +0.0009 < +0.005) — выигрыш в пределах шума.
Остатки считаются только по fit_df; таксоны и 4 лит.-признака чистятся одной регрессией (возраст, log10 глубины, страна)."""
from v3core import *


def fit_predict_best(fit_df, pred_df):
    Xf, Xp = taxa_matrix(fit_df), taxa_matrix(pred_df)
    keep = np.flatnonzero((Xf > 0).mean(axis=0) >= MIN_PREV)
    st = dict(age_mu=fit_df["age"].mean(), age_sd=fit_df["age"].std() + 1e-9,
              dep_mu=np.log10(fit_df["library_size"]).mean(), dep_sd=np.log10(fit_df["library_size"]).std() + 1e-9)
    block = lambda X: np.column_stack([np.log10(X[:, keep] + PSEUDO), lit_features(X)])   # таксоны + 4 лит. признака
    Ff, Fp = block(Xf), block(Xp)
    reg = LinearRegression().fit(cov_design(fit_df, st, FULL_SPEC), Ff)
    Ff = Ff - reg.predict(cov_design(fit_df, st, FULL_SPEC)); Fp = Fp - reg.predict(cov_design(pred_df, st, FULL_SPEC))
    return ForestLogit().fit(Ff, fit_df["label"].values).predict_proba(Fp)[:, 1]


if __name__ == "__main__":
    evaluate(fit_predict_best, "F_best_A_lit_resid", log=None, confirm=False)
