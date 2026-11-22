"""Общий стенд для v4: код модели v3 как есть (импортируемый) + честная оценка идей.
Трогаем ТОЛЬКО train_df (332 образца). validation (83) и test здесь НЕ загружаются вообще.

    from v3core import *
    def my_fp(fit_df, pred_df): ...      # учится на fit_df, возвращает скоры КРР для pred_df
    r = evaluate(my_fp, "идея", log="v4_agentX.csv")

Протокол (правило задано ДО экспериментов, одно на всех):
  отбор:        повторная 5x5 CV (random_state=42), «новая страна» (LOCO), подвыборка с выровненными страной и возрастом
  подтверждение: ДРУГИЕ фолды (повторная 5x5 CV, random_state=2025) — идея должна не проиграть и там
  идея принимается, если: Δloco >= +0.005, Δmatched >= +0.005, Δcv >= -0.005 против v3 и Δcv_подтверждение >= 0.
Все модели random_state = 42, лес n_jobs = 1.
"""
import os, random, time, pickle, warnings
import numpy as np, pandas as pd
from scipy.stats import t as t_dist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
D = r"D:\hackaton"
OUT = os.path.join(D, "experiments")
META = ["cohort", "age", "gender", "bmi", "library_size"]
PSEUDO, N_TREES = 1e-5, 500
MIN_PREV, W_RF, LR_C = 0.05, 0.7, 0.01
FULL_SPEC = ("age", "depth", "cohort")
LIT_GENERA = ["Fusobacterium", "Parvimonas", "Peptostreptococcus", "Gemella", "Porphyromonas", "Solobacterium"]

_train_all = pd.read_csv(os.path.join(D, "train.csv")).sort_values("sample_id").reset_index(drop=True)
TAXA = sorted(c for c in _train_all.columns if c not in ["sample_id", "label"] + META)


def split_data(train, seed=SEED, val_frac=0.20):
    train = train.sort_values("sample_id").reset_index(drop=True)
    strata = train["cohort"].astype(str) + "|" + train["label"].astype(str)
    tr_ids, va_ids = train_test_split(train["sample_id"], test_size=val_frac, stratify=strata, random_state=seed)
    return (train[train["sample_id"].isin(tr_ids)].reset_index(drop=True),
            train[train["sample_id"].isin(va_ids)].reset_index(drop=True))


train_df, _val_do_not_use = split_data(_train_all)
del _val_do_not_use                                   # validation в экспериментах не используется
y_tr = train_df["label"].values
COHORTS = sorted(train_df["cohort"].unique())


def taxa_matrix(df):
    return np.nan_to_num(df.reindex(columns=TAXA, fill_value=0.0).to_numpy(dtype=float), nan=0.0)


def genus(col):
    w = col.split()
    return w[1] if w[0] == "unknown" else w[0]


MARKER_IDX = [i for i, c in enumerate(TAXA) if genus(c) in LIT_GENERA]


def lit_features(X):
    M = X[:, MARKER_IDX]
    tot = X.sum(axis=1, keepdims=True)
    P = X / np.where(tot > 0, tot, 1.0)
    shannon = -np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0), axis=1)
    return np.column_stack([np.log10(M.sum(axis=1) + PSEUDO), (M > 0).sum(axis=1), (X > 0).sum(axis=1), shannon])


def cov_design(df, st, spec):
    cols = []
    if "age" in spec: cols.append((df["age"].to_numpy(float) - st["age_mu"]) / st["age_sd"])
    if "depth" in spec: cols.append((np.log10(df["library_size"].to_numpy(float)) - st["dep_mu"]) / st["dep_sd"])
    if "cohort" in spec: cols += [(df["cohort"] == c).to_numpy(float) for c in COHORTS]
    return np.column_stack(cols)


def fit_preprocessing(fit_df, spec=FULL_SPEC, min_prev=MIN_PREV):
    X = taxa_matrix(fit_df)
    keep = np.flatnonzero((X > 0).mean(axis=0) >= min_prev)
    prep = {"keep": keep, "spec": tuple(spec)}
    if spec:
        prep["st"] = dict(age_mu=fit_df["age"].mean(), age_sd=fit_df["age"].std() + 1e-9,
                          dep_mu=np.log10(fit_df["library_size"]).mean(), dep_sd=np.log10(fit_df["library_size"]).std() + 1e-9)
        prep["reg"] = LinearRegression().fit(cov_design(fit_df, prep["st"], spec), np.log10(X[:, keep] + PSEUDO))
    return prep


def apply_preprocessing(df, prep, lit=True):
    X = taxa_matrix(df)
    F = np.log10(X[:, prep["keep"]] + PSEUDO)
    if prep["spec"]:
        F = F - prep["reg"].predict(cov_design(df, prep["st"], prep["spec"]))
    return np.column_stack([F, lit_features(X)]) if lit else F


def make_rf(seed=SEED, n_jobs=1, **kw):
    kw.setdefault("n_estimators", N_TREES); kw.setdefault("max_features", "sqrt"); kw.setdefault("min_samples_leaf", 2)
    return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **kw)


class ForestLogit:
    def __init__(self, w_rf=W_RF, C=LR_C, seed=SEED, n_jobs=1):
        self.w_rf, self.C, self.seed, self.n_jobs = w_rf, C, seed, n_jobs

    def fit(self, X, y):
        self.rf = make_rf(self.seed, self.n_jobs).fit(X, y)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.lr = None if self.w_rf >= 1 else LogisticRegression(C=self.C, max_iter=5000).fit((X - self.mu) / self.sd, y)
        return self

    def predict_proba(self, X):
        p = self.rf.predict_proba(X)[:, 1]
        if self.lr is not None:
            p = self.w_rf * p + (1 - self.w_rf) * self.lr.predict_proba((X - self.mu) / self.sd)[:, 1]
        return np.column_stack([1 - p, p])


class GutPipeline:
    def __init__(self, spec=FULL_SPEC, min_prev=MIN_PREV, lit=True, w_rf=W_RF, n_jobs=1):
        self.spec, self.min_prev, self.lit, self.w_rf, self.n_jobs = tuple(spec), min_prev, lit, w_rf, n_jobs

    def fit(self, df):
        self.prep = fit_preprocessing(df, self.spec, self.min_prev)
        self.model = ForestLogit(w_rf=self.w_rf, n_jobs=self.n_jobs).fit(apply_preprocessing(df, self.prep, self.lit), df["label"].values)
        return self

    def predict(self, df):
        return self.model.predict_proba(apply_preprocessing(df, self.prep, self.lit))[:, 1]


def v3_fp(fit_df, pred_df):
    """Текущая модель v3 (базовая линия для всех идей)."""
    return GutPipeline().fit(fit_df).predict(pred_df)


# ---- стенд ----
_strata = (train_df["cohort"].astype(str) + "|" + train_df["label"].astype(str)).values
FOLDS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(train_df, _strata))
FOLDS_CONF = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2025).split(train_df, _strata))


def matched_index(df, age_cal=3.0, dep_cal=None):
    idx, ld = [], np.log10(df["library_size"])
    for co in COHORTS:
        d = df[df["cohort"] == co]
        cases, pool, used = d[d["label"] == 1].sort_values("sample_id"), d[d["label"] == 0].sort_values("sample_id"), set()
        for i, r in cases.iterrows():
            cand = pool[~pool.index.isin(used)]
            if cand.empty: continue
            da = (cand["age"] - r["age"]).abs(); score = da / age_cal; ok = da <= age_cal
            if dep_cal is not None:
                dd = (ld[cand.index] - ld[i]).abs(); score = score + dd / dep_cal; ok &= dd <= dep_cal
            if ok.any():
                j = score[ok].idxmin(); used.add(j); idx += [i, j]
    return np.array(sorted(idx))


MATCH = matched_index(train_df)
MATCH_D = matched_index(train_df, age_cal=5.0, dep_cal=0.08)


def _cv(fp, folds):
    aucs, oof = np.zeros(len(folds)), np.zeros((5, len(train_df)))
    for k, (tr, te) in enumerate(folds):
        p = np.asarray(fp(train_df.iloc[tr], train_df.iloc[te])); oof[k // 5, te] = p; aucs[k] = roc_auc_score(y_tr[te], p)
    return aucs, oof.mean(axis=0)


def _evaluate_raw(fp):
    aucs, om = _cv(fp, FOLDS)
    loco = {}
    for co in COHORTS:
        m = (train_df["cohort"] == co).values
        loco[co] = roc_auc_score(y_tr[m], np.asarray(fp(train_df[~m], train_df[m])))
    return dict(aucs=aucs, loco=loco, loco_mean=float(np.mean(list(loco.values()))), oof=om,
                matched=roc_auc_score(y_tr[MATCH], om[MATCH]), matched_d=roc_auc_score(y_tr[MATCH_D], om[MATCH_D]))


def _baseline():
    # pickle читается только из собственного файла, который этот же модуль записал локально (не из внешних источников)
    p = os.path.join(OUT, "v3_baseline.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    b = _evaluate_raw(v3_fp); b["aucs_conf"] = _cv(v3_fp, FOLDS_CONF)[0]
    pickle.dump(b, open(p, "wb")); return b


def evaluate(fp, name, log=None, confirm=True):
    """Оценка идеи против v3 по заранее заданному правилу. log — свой csv у каждого исполнителя (не общий)."""
    t0 = time.time(); b = _baseline(); r = _evaluate_raw(fp)
    d = r["aucs"] - b["aucs"]; se = d.std(ddof=1) * np.sqrt(1 / len(d) + 0.25) if d.std() > 0 else 0.0
    res = dict(name=name, cv=r["aucs"].mean(), sd=r["aucs"].std(ddof=1), loco=r["loco_mean"], matched=r["matched"], matched_d=r["matched_d"],
               d_cv=d.mean(), se=se, wins=int((d > 0).sum()), d_loco=r["loco_mean"] - b["loco_mean"], d_matched=r["matched"] - b["matched"],
               d_matched_d=r["matched_d"] - b["matched_d"])
    if confirm:
        ac = _cv(fp, FOLDS_CONF)[0]; res["d_cv_conf"] = (ac - b["aucs_conf"]).mean(); res["cv_conf"] = ac.mean()
    res["adopt"] = bool(res["d_loco"] >= 0.005 and res["d_matched"] >= 0.005 and res["d_cv"] >= -0.005 and res.get("d_cv_conf", 0) >= 0)
    res["sec"] = round(time.time() - t0)
    if log:
        pd.DataFrame([res]).to_csv(os.path.join(OUT, log), mode="a", header=not os.path.exists(os.path.join(OUT, log)), index=False, encoding="utf-8")
    print(f"{name:40s} CV {res['cv']:.4f}±{res['sd']:.4f} | LOCO {res['loco']:.4f} | matched {res['matched']:.4f} (+глуб. {res['matched_d']:.4f}) | "
          f"Δcv {res['d_cv']:+.4f} (SE {se:.4f}, {res['wins']}/25) Δloco {res['d_loco']:+.4f} Δmatch {res['d_matched']:+.4f} "
          f"Δconf {res.get('d_cv_conf', float('nan')):+.4f} | {'ПРИНЯТЬ' if res['adopt'] else 'нет'} | {res['sec']} с", flush=True)
    return res


if __name__ == "__main__":
    b = _baseline()
    print("v3 на стенде: CV %.4f | LOCO %.4f | matched %.4f | +глубина %.4f | CV(подтв.) %.4f" %
          (b["aucs"].mean(), b["loco_mean"], b["matched"], b["matched_d"], b["aucs_conf"].mean()))
