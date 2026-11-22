"""v4_R итог: по сохранённым OOF (v4_R_oof.pkl, v3 берётся из v3_baseline.pkl) считает сравнимые метрики.
- AUC OOF общий и по странам; matched / matched+глубина; AUC со взвешиванием, разрывающим связь глубина-метка (IPW по стране, все 332 образца);
- парный bootstrap (по образцам) для разностей против v3; глубина как единственный признак (страна-специфичный знак) для масштаба.
Только train_df; test не используется."""
import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import v3core as V

# оба pkl записаны локально нашим же стендом / этим скриптом (не внешние источники)
store = pickle.load(open(os.path.join(V.OUT, "v4_R_oof.pkl"), "rb"))
t = V.train_df; y = V.y_tr; co = t.cohort.values; ld = np.log10(t.library_size.values)
print("MATCH n=%d, MATCH_D n=%d" % (len(V.MATCH), len(V.MATCH_D)))

# веса, разрывающие связь глубина-метка внутри страны (стабилизированные, как в идее R4, но по всем 332 образцам; только для оценки)
w = np.ones(len(t))
for c in V.COHORTS:
    m = co == c; z = (ld[m] - ld[m].mean()) / ld[m].std()
    p = LogisticRegression(C=1.0).fit(z[:, None], y[m]).predict_proba(z[:, None])[:, 1]; pi = y[m].mean()
    w[m] = np.where(y[m] == 1, pi / p, (1 - pi) / (1 - p))
w = np.clip(w, .25, 4.); w /= w.mean()
# проверка: после взвешивания связь глубина-метка внутри страны исчезла?
print("взвешенный AUC глубины по странам (должен быть ~0.5):", {c: round(roc_auc_score(y[co == c], ld[co == c], sample_weight=w[co == c]), 2) for c in V.COHORTS})

# глубина как единственный признак: логрегрессия по стране, 5x5 CV (те же фолды)
oo = np.zeros((5, len(t)))
for k, (tr, te) in enumerate(V.FOLDS):
    out = np.zeros(len(te))
    for c in V.COHORTS:
        a, b = tr[co[tr] == c], np.flatnonzero(co[te] == c)
        z = lambda i: ((ld[i] - ld[a].mean()) / ld[a].std())[:, None]
        out[b] = LogisticRegression(C=1.0).fit(z(a), y[a]).predict_proba(z(te[b]))[:, 1]
    oo[k // 5, te] = out
dep = oo.mean(0)
print("глубина-одна (логрегрессия по стране): общий AUC %.3f; по странам %s" % (roc_auc_score(y, dep), {c: round(roc_auc_score(y[co == c], dep[co == c]), 2) for c in V.COHORTS}))
store["depth_only"] = dep

rng = np.random.RandomState(42)
mi = V.MATCH_D


def metrics(o):
    return dict(cv_pooled=roc_auc_score(y, o), matched=roc_auc_score(y[V.MATCH], o[V.MATCH]), matched_d=roc_auc_score(y[mi], o[mi]),
                depth_balanced=roc_auc_score(y, o, sample_weight=w), **{c: roc_auc_score(y[co == c], o[co == c]) for c in V.COHORTS},
                sp_healthy_mean=np.mean([stats.spearmanr(o[(y == 0) & (co == c)], ld[(y == 0) & (co == c)])[0] for c in V.COHORTS]))


base = store["v3"]; rows = []
for nm, o in store.items():
    if nm == "depth_only": continue
    m = metrics(o); r = dict(name=nm, **m)
    if nm != "v3":
        d_md, d_db = [], []
        for _ in range(1000):   # парный bootstrap: ресэмплинг внутри MATCH_D / внутри всей выборки
            im = rng.choice(mi, len(mi)); ia = rng.randint(0, len(t), len(t))
            if len(set(y[im])) < 2: continue
            d_md.append(roc_auc_score(y[im], o[im]) - roc_auc_score(y[im], base[im]))
            d_db.append(roc_auc_score(y[ia], o[ia], sample_weight=w[ia]) - roc_auc_score(y[ia], base[ia], sample_weight=w[ia]))
        r["d_matched_d"] = m["matched_d"] - roc_auc_score(y[mi], base[mi]); r["se_matched_d"] = np.std(d_md)
        r["d_depth_bal"] = m["depth_balanced"] - roc_auc_score(y, base, sample_weight=w); r["se_depth_bal"] = np.std(d_db)
    rows.append(r)
R = pd.DataFrame(rows).set_index("name")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(R[["cv_pooled", "matched", "matched_d", "depth_balanced", "sp_healthy_mean"]].round(4).to_string())
print(R[["AUS", "CHI", "FRA", "GER", "USA"]].round(3).to_string())
print(R[["d_matched_d", "se_matched_d", "d_depth_bal", "se_depth_bal"]].round(4).to_string())
R.to_csv(os.path.join(V.OUT, "v4_R_summary.csv"), encoding="utf-8")
