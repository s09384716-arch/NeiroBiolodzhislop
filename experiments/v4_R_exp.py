"""v4_R: «безопасные» варианты v3 без чтения глубины. Запуск: python -X utf8 v4_R_exp.py [имена идей через запятую]
Каждая идея -> evaluate(..., log="v4_R.csv", confirm=True) + диагностика глубины по OOF (Spearman среди здоровых, AUC по странам).
Свой конвейер повторяет v3 один-в-один при настройках по умолчанию (проверяется assert'ом на одном фолде)."""
import sys, os, time, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
import v3core as V

OUT_PKL = os.path.join(V.OUT, "v4_R_oof.pkl")


# ---------------- свой конвейер с настройками ----------------
class Cfg:
    def __init__(self, depth=True, interact=False, lit="raw", ipw=False, post=False, rare=None, w_rf=V.W_RF):
        self.depth, self.interact, self.lit, self.ipw, self.post, self.rare, self.w_rf = depth, interact, lit, ipw, post, rare, w_rf


def design(df, st, cfg):
    cols = [(df["age"].to_numpy(float) - st["age_mu"]) / st["age_sd"]]
    ld = (np.log10(df["library_size"].to_numpy(float)) - st["dep_mu"]) / st["dep_sd"]
    if cfg.depth: cols.append(ld)
    oh = [(df["cohort"] == c).to_numpy(float) for c in V.COHORTS]
    cols += oh
    if cfg.depth and cfg.interact: cols += [ld * o for o in oh]
    return np.column_stack(cols)


def fit_prep(fit_df, cfg):
    X = V.taxa_matrix(fit_df); T0 = None
    if cfg.rare is not None:
        T0 = float(np.percentile(counts(X).sum(1), cfg.rare)); X = rarefy(X, fit_df, T0)
    keep = np.flatnonzero((X > 0).mean(axis=0) >= V.MIN_PREV)
    ld = np.log10(fit_df["library_size"])
    st = dict(age_mu=fit_df["age"].mean(), age_sd=fit_df["age"].std() + 1e-9, dep_mu=ld.mean(), dep_sd=ld.std() + 1e-9)
    D = design(fit_df, st, cfg)
    p = dict(keep=keep, st=st, cfg=cfg, T0=T0, reg=LinearRegression().fit(D, np.log10(X[:, keep] + V.PSEUDO)))
    if cfg.lit == "resid": p["lreg"] = LinearRegression().fit(D, V.lit_features(X))
    return p


def features(df, p):
    cfg = p["cfg"]; X = V.taxa_matrix(df)
    if cfg.rare is not None: X = rarefy(X, df, p["T0"])
    D = design(df, p["st"], cfg)
    F = np.log10(X[:, p["keep"]] + V.PSEUDO) - p["reg"].predict(D)
    if cfg.lit == "none": return F
    L = V.lit_features(X)
    if cfg.lit == "resid": L = L - p["lreg"].predict(D)
    if cfg.lit == "norich": L = L[:, [0, 1, 3]]
    return np.column_stack([F, L])


def counts(X):
    """целые счётчики восстанавливаются точно: доля / минимальная ненулевая доля в строке (проверено в v4_R_leak.py)"""
    mn = np.where(X > 0, X, np.inf).min(axis=1, keepdims=True)
    return np.rint(np.where(X > 0, X / mn, 0.0))


def rarefy(X, df, T0):
    """биномиальное прореживание счётчиков образца до общей суммы T0 (если сумма больше). Генератор случайных чисел привязан к номеру образца
    (random_state = 42 + номер), поэтому результат не зависит от состава фолда. Порог обнаружения становится ~ S_i / T0 у всех образцов."""
    C = counts(X); T = C.sum(1); S = X.sum(1); out = X.copy()
    ids = df["sample_id"].astype(str).str.extract(r"(\d+)")[0].astype(int).to_numpy()
    for i in range(len(X)):
        if T[i] > T0:
            c = np.random.RandomState(42 + ids[i]).binomial(C[i].astype(np.int64), T0 / T[i])
            out[i] = c / T0 * S[i]
    return out


def ipw_weights(df):
    ld = np.log10(df["library_size"].to_numpy(float)); y = df["label"].to_numpy(); w = np.ones(len(df))
    for c in V.COHORTS:
        m = (df["cohort"] == c).to_numpy()
        if m.sum() < 10 or len(np.unique(y[m])) < 2: continue
        z = (ld[m] - ld[m].mean()) / (ld[m].std() + 1e-9)
        p = LogisticRegression(C=1.0).fit(z[:, None], y[m]).predict_proba(z[:, None])[:, 1]
        pi = y[m].mean(); w[m] = np.where(y[m] == 1, pi / p, (1 - pi) / (1 - p))
    w = np.clip(w, 0.25, 4.0); return w / w.mean()


class WForestLogit(V.ForestLogit):
    def fit(self, X, y, w=None, oob=False):
        self.rf = V.make_rf(self.seed, self.n_jobs, oob_score=oob, bootstrap=True).fit(X, y, sample_weight=w)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.lr = LogisticRegression(C=self.C, max_iter=5000).fit((X - self.mu) / self.sd, y, sample_weight=w)
        return self


def make_fp(cfg):
    def fp(fit_df, pred_df):
        p = fit_prep(fit_df, cfg); F = features(fit_df, p); y = fit_df["label"].values
        w = ipw_weights(fit_df) if cfg.ipw else None
        m = WForestLogit(w_rf=cfg.w_rf).fit(F, y, w, oob=cfg.post)
        s = m.predict_proba(features(pred_df, p))[:, 1]
        if cfg.post:
            # оценка «чтения глубины» по OOB-скорам здоровых в обучающей выборке: единый наклон по центрированной внутри страны log-глубине
            s_oob = cfg.w_rf * m.rf.oob_decision_function_[:, 1] + (1 - cfg.w_rf) * m.lr.predict_proba((F - m.mu) / m.sd)[:, 1]
            ld = np.log10(fit_df["library_size"].to_numpy(float)); coh = fit_df["cohort"].to_numpy()
            mu_c = {c: ld[coh == c].mean() for c in V.COHORTS}
            ldc = ld - np.array([mu_c[c] for c in coh]); ok = (y == 0) & np.isfinite(s_oob)
            beta = np.polyfit(ldc[ok], s_oob[ok], 1)[0]
            ldp = np.log10(pred_df["library_size"].to_numpy(float)); cp = pred_df["cohort"].to_numpy()
            if len(set(cp)) == 1 and cp[0] not in set(coh): ref = np.full(len(cp), ldp.mean())
            else: ref = np.array([mu_c.get(c, ldp.mean()) for c in cp])
            s = s - beta * (ldp - ref)
        return s
    return fp


IDEAS = {
    "R1_age_cohort_noDepth": Cfg(depth=False),                       # остатки только по возрасту и стране
    "R2_v3_litResid": Cfg(lit="resid"),                              # v3 + lit-признаки очищены от возраста/глубины/страны
    "R3_perCohortSlope_litResid": Cfg(interact=True, lit="resid"),   # свой наклон глубины в каждой стране
    "R4_v3_IPWdepth": Cfg(ipw=True),                                 # веса, разрывающие связь глубина-метка внутри страны
    "R5_v3_postDepthOOB": Cfg(post=True),                            # скор очищается от наклона на глубину, оцененного по OOB здоровых
    "R6_rarefy10_v3": Cfg(rare=10),                                   # прореживание счётчиков до глубины 10-го перцентиля (демпфирование + шум)
}


class Cap:
    """обёртка fp: запоминает предсказания первых 25 вызовов (= FOLDS) для OOF"""
    def __init__(self, fp): self.fp, self.calls = fp, []
    def __call__(self, a, b): s = np.asarray(self.fp(a, b)); self.calls.append(s); return s
    def oof(self):
        om = np.zeros((5, len(V.train_df)))
        for k, (tr, te) in enumerate(V.FOLDS): om[k // 5, te] = self.calls[k]
        return om.mean(0)


def depth_diag(oof, name):
    t = V.train_df; ld = np.log10(t.library_size.values); y = V.y_tr; co = t.cohort.values
    sp_pool = stats.spearmanr(oof[y == 0], ld[y == 0])[0]
    sp_c = {c: stats.spearmanr(oof[(y == 0) & (co == c)], ld[(y == 0) & (co == c)])[0] for c in V.COHORTS}
    sp_case = stats.spearmanr(oof[y == 1], ld[y == 1])[0]
    auc_c = {c: roc_auc_score(y[co == c], oof[co == c]) for c in V.COHORTS}
    print(f"   [{name}] Spearman(OOF, log-глубина) среди здоровых: общий {sp_pool:+.3f}; по странам " + ", ".join(f"{c} {v:+.2f}" for c, v in sp_c.items()) +
          f" (среднее {np.mean(list(sp_c.values())):+.3f}); среди больных {sp_case:+.3f}; AUC по странам " + ", ".join(f"{c} {v:.2f}" for c, v in auc_c.items()), flush=True)
    return dict(sp_h_pool=sp_pool, sp_h_mean=float(np.mean(list(sp_c.values()))), sp_case=sp_case, **{f"sp_{c}": v for c, v in sp_c.items()}, **{f"auc_{c}": v for c, v in auc_c.items()})


def leak_diag(cfg, name):
    from sklearn.linear_model import Ridge
    t = V.train_df; ld = np.log10(t.library_size.values); ldc = ld - t.groupby("cohort")["library_size"].transform(lambda x: np.log10(x).mean()).values
    sp = []
    for tr, te in V.FOLDS[:5]:
        p = fit_prep(t.iloc[tr], cfg); a, b = features(t.iloc[tr], p), features(t.iloc[te], p)
        mu, sd = a.mean(0), a.std(0) + 1e-9
        sp.append(stats.spearmanr(Ridge(alpha=100.0).fit((a - mu) / sd, ldc[tr]).predict((b - mu) / sd), ldc[te])[0])
    print(f"   [{name}] утечка глубины в признаках: Spearman(ridge-предсказание глубины, истина) = {np.mean(sp):+.3f}", flush=True)
    return float(np.mean(sp))


def wait_baseline():
    p = os.path.join(V.OUT, "v3_baseline.pkl"); n = 0
    while True:
        if os.path.exists(p):
            s1 = os.path.getsize(p); time.sleep(5)
            if os.path.getsize(p) == s1 and s1 > 0: return
        time.sleep(20); n += 1
        if n % 6 == 0: print("  ждём v3_baseline.pkl ...", flush=True)


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(IDEAS)
    # проверка: конвейер с настройками по умолчанию == v3_fp (побитово)
    a, b = V.train_df.iloc[V.FOLDS[0][0]], V.train_df.iloc[V.FOLDS[0][1]]
    s0, s1 = np.asarray(V.v3_fp(a, b)), make_fp(Cfg())(a, b)
    print("конвейер v4_R(по умолч.) == v3_fp: побитово", np.array_equal(s0, s1), "| макс.разность", np.abs(s0 - s1).max(), flush=True)
    wait_baseline()
    # pickle читается только из собственного локального файла стенда (не из внешних источников)
    base = pickle.load(open(os.path.join(V.OUT, "v3_baseline.pkl"), "rb"))   # файл записан тем же стендом локально
    print("v3 baseline: CV %.4f LOCO %.4f matched %.4f +глуб %.4f" % (base["aucs"].mean(), base["loco_mean"], base["matched"], base["matched_d"]))
    # свой локальный файл v4_R_oof.pkl, записанный этим же скриптом
    store = pickle.load(open(OUT_PKL, "rb")) if os.path.exists(OUT_PKL) else {}
    store["v3"] = base["oof"]; depth_diag(base["oof"], "v3")
    for nm in names:
        cap = Cap(make_fp(IDEAS[nm]))
        r = V.evaluate(cap, nm, log="v4_R.csv", confirm=True)
        store[nm] = cap.oof(); d = depth_diag(store[nm], nm); d["leak_sp"] = leak_diag(IDEAS[nm], nm)
        pd.DataFrame([dict(name=nm, **d)]).to_csv(os.path.join(V.OUT, "v4_R_depth.csv"), mode="a", header=not os.path.exists(os.path.join(V.OUT, "v4_R_depth.csv")), index=False)
        pickle.dump(store, open(OUT_PKL, "wb"))
