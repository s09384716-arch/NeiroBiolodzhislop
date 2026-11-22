"""Исполнитель F: конструктор признаковых вариантов поверх v3 (только fit_df-статистики, только таблица train.csv).
make_fp(...) возвращает fit_predict(fit_df, pred_df). При настройках по умолчанию воспроизводит v3 бит-в-бит."""
from v3core import *
from scipy.stats import rankdata, spearmanr


def _idx(pred):
    return [i for i, c in enumerate(TAXA) if pred(c)]


# --- группы по названиям (литература; направление проверено на train по странам в v4_F_explore.py) ---
BUT = _idx(lambda c: c.startswith(("Faecalibacterium prausnitzii", "Eubacterium rectale", "Anaerostipes hadrus", "Eubacterium hallii"))
           or genus(c) in ("Roseburia", "Coprococcus", "Butyrivibrio") or "butyrate-producing" in c)
SULF = _idx(lambda c: genus(c) in ("Bilophila", "Desulfovibrio"))
_EXT = ("Streptococcus anginosus", "Streptococcus constellatus", "Streptococcus mutans", "Streptococcus oralis", "Streptococcus parasanguinis",
        "Streptococcus sanguinis", "Haemophilus parainfluenzae", "Granulicatella adiacens", "Atopobium parvulum", "Prevotella intermedia",
        "Prevotella nigrescens", "Mogibacterium timidum")
ORAL_EXT = sorted(set(MARKER_IDX) | set(_idx(lambda c: c.startswith(_EXT) or genus(c) == "Veillonella")))
_g = {}
for _i, _c in enumerate(TAXA):
    _g.setdefault(genus(_c), []).append(_i)
GENERA = {k: v for k, v in _g.items() if len(v) >= 2}      # одиночные роды дублируют сам таксон


def lsum(X, ix):
    return np.log10(X[:, ix].sum(axis=1) + PSEUDO)


def extras_raw(X, names):
    """Групповые признаки (log10 суммы долей / их разности) -> столбцы."""
    out = []
    for n in names:
        if n == "but": out.append(lsum(X, BUT))
        elif n == "sulf": out.append(lsum(X, SULF))
        elif n == "oral_ext": out.append(lsum(X, ORAL_EXT))
        elif n == "ratio_ext": out.append(lsum(X, ORAL_EXT) - lsum(X, BUT))      # log(ротовые_расш / бутират)
        elif n == "ratio_lit": out.append(lsum(X, MARKER_IDX) - lsum(X, BUT))    # log(ротовые_6_родов / бутират)
        else: raise ValueError(n)
    return np.column_stack(out) if out else np.zeros((X.shape[0], 0))


def cov_x(df, st, spec):
    """Расширенный дизайн остатков: age, depth, cohort + depth2 (квадрат), dxc (глубина x страна)."""
    base = [s for s in spec if s in ("age", "depth", "cohort")]
    cols = [cov_design(df, st, tuple(base))] if base else []
    z = (np.log10(df["library_size"].to_numpy(float)) - st["dep_mu"]) / st["dep_sd"]
    if "depth2" in spec: cols.append((z ** 2)[:, None])
    if "dxc" in spec: cols.append(np.column_stack([z * (df["cohort"] == c).to_numpy(float) for c in COHORTS]))
    return np.hstack(cols)


def mw_z(F, y, coh):
    """Объединённый по странам Z Манна-Уитни (взвешенный Стоуффер): отбор только по fit_df."""
    num, W = np.zeros(F.shape[1]), 0.0
    for c in np.unique(coh):
        m = coh == c; yy = y[m]; n1 = int(yy.sum()); n0 = len(yy) - n1
        if n1 < 3 or n0 < 3: continue
        R = rankdata(F[m], axis=0)
        z = (R[yy == 1].sum(axis=0) - n1 * (n1 + 1) / 2 - n1 * n0 / 2) / np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12)
        w = np.sqrt(n1 + n0); num += w * z; W += w * w
    return num / np.sqrt(W)


def make_fp(spec=FULL_SPEC, lit_resid=False, extras=(), extras_resid=True, genus_agg=False, rare_bin=False, topk=None, w_rf=W_RF, lit_cols=None):
    spec = tuple(spec)
    LC = list(range(4)) if lit_cols is None else list(lit_cols)

    def fp(fit_df, pred_df):
        Xf, Xp = taxa_matrix(fit_df), taxa_matrix(pred_df)
        keep = np.flatnonzero((Xf > 0).mean(axis=0) >= MIN_PREV)
        st = dict(age_mu=fit_df["age"].mean(), age_sd=fit_df["age"].std() + 1e-9,
                  dep_mu=np.log10(fit_df["library_size"]).mean(), dep_sd=np.log10(fit_df["library_size"]).std() + 1e-9)
        gl = []
        if genus_agg:
            gl = [k for k, v in GENERA.items() if (Xf[:, v].sum(axis=1) > 0).mean() >= MIN_PREV]

        def resid_block(X):          # всё, что чистится остатками: таксоны [+ роды] [+ групповые] [+ лит.]
            blocks = [np.log10(X[:, keep] + PSEUDO)]
            if genus_agg: blocks.append(np.column_stack([np.log10(X[:, GENERA[k]].sum(axis=1) + PSEUDO) for k in gl]))
            if extras and extras_resid: blocks.append(extras_raw(X, extras))
            if lit_resid: blocks.append(lit_features(X)[:, LC])
            return np.hstack(blocks)

        Rf, Rp = resid_block(Xf), resid_block(Xp)
        if spec:
            reg = LinearRegression().fit(cov_x(fit_df, st, spec), Rf)
            Rf = Rf - reg.predict(cov_x(fit_df, st, spec)); Rp = Rp - reg.predict(cov_x(pred_df, st, spec))
        n_t = len(keep)
        if topk:                    # отбор таксонов по |Z| внутри стран; только fit_df
            z = np.abs(mw_z(Rf[:, :n_t], fit_df["label"].values, fit_df["cohort"].values))
            sel = np.argsort(-z)[:topk]
            cols = np.concatenate([np.sort(sel), np.arange(n_t, Rf.shape[1])])
            Rf, Rp = Rf[:, cols], Rp[:, cols]
        tailf, tailp = [], []
        if extras and not extras_resid: tailf.append(extras_raw(Xf, extras)); tailp.append(extras_raw(Xp, extras))
        if not lit_resid: tailf.append(lit_features(Xf)[:, LC]); tailp.append(lit_features(Xp)[:, LC])
        if rare_bin:
            rare = [i for i in ORAL_EXT if (Xf[:, i] > 0).mean() < MIN_PREV and (Xf[:, i] > 0).sum() >= 3]
            if rare:
                bf, bp = (Xf[:, rare] > 0).astype(float), (Xp[:, rare] > 0).astype(float)
                tailf += [bf, bf.sum(axis=1, keepdims=True)]; tailp += [bp, bp.sum(axis=1, keepdims=True)]
        Ff = np.hstack([Rf] + tailf); Fp = np.hstack([Rp] + tailp)
        return ForestLogit(w_rf=w_rf).fit(Ff, fit_df["label"].values).predict_proba(Fp)[:, 1]
    return fp


class Rec:
    """Обёртка: запоминает предсказания первых 25 вызовов (повторная 5x5 CV) для OOF и анализа зависимости от глубины."""
    def __init__(self, fp): self.fp, self.calls = fp, []

    def __call__(self, fit_df, pred_df):
        p = np.asarray(self.fp(fit_df, pred_df))
        if len(self.calls) < 25: self.calls.append((pred_df.index.to_numpy(), p))
        return p

    def oof(self):
        o = np.zeros((5, len(train_df)))
        for k, (te, p) in enumerate(self.calls): o[k // 5, te] = p
        return o.mean(axis=0)


def depth_rho(oof):
    """Spearman(OOF-скор, log10 глубины) среди здоровых по странам: среднее и по странам."""
    ld = np.log10(train_df["library_size"].values); co = train_df["cohort"].values
    r = {c: spearmanr(oof[(co == c) & (y_tr == 0)], ld[(co == c) & (y_tr == 0)])[0] for c in COHORTS}
    return float(np.mean(list(r.values()))), r


def wait_baseline():
    p = os.path.join(OUT, "v3_baseline.pkl")
    while not os.path.exists(p): time.sleep(20)
    time.sleep(5)
