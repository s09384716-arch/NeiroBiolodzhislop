# Разведка групп (только train_df через v3core): направление эффекта по странам, зависимость от глубины.
from v3core import *
from scipy.stats import spearmanr
X = taxa_matrix(train_df); y = y_tr; co = train_df["cohort"].values
ld = np.log10(train_df["library_size"].values)
def idx_by(pred): return [i for i, c in enumerate(TAXA) if pred(c)]
BUT = idx_by(lambda c: c.startswith("Faecalibacterium prausnitzii") or genus(c) in ("Roseburia","Coprococcus","Butyrivibrio") or c.startswith(("Eubacterium rectale","Anaerostipes hadrus","Eubacterium hallii")) or "butyrate-producing" in c)
SULF = idx_by(lambda c: genus(c) in ("Bilophila","Desulfovibrio"))
EXTO = ("Streptococcus anginosus","Streptococcus constellatus","Streptococcus mutans","Streptococcus oralis","Streptococcus parasanguinis","Streptococcus sanguinis",
        "Haemophilus parainfluenzae","Granulicatella adiacens","Atopobium parvulum","Prevotella intermedia","Prevotella nigrescens","Mogibacterium timidum")
ORAL_EXT = sorted(set(MARKER_IDX) | set(idx_by(lambda c: c.startswith(EXTO) or genus(c) == "Veillonella")))
print(len(BUT), len(SULF), len(MARKER_IDX), len(ORAL_EXT))
print([TAXA[i] for i in ORAL_EXT if i not in MARKER_IDX])
def lg(ix): return np.log10(X[:, ix].sum(1) + PSEUDO)
feats = {"lit_log_sum": lg(MARKER_IDX), "lit_nmark": (X[:, MARKER_IDX] > 0).sum(1), "richness": (X > 0).sum(1),
         "butyrate": lg(BUT), "sulfidogen": lg(SULF), "oral_ext": lg(ORAL_EXT)}
feats["ratio_oral_but"] = feats["oral_ext"] - feats["butyrate"]
P = X / X.sum(1, keepdims=True); feats["shannon"] = -np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0), 1)
print("%-14s" % "feature", *["%6s" % c for c in COHORTS], " | meanAUC  | rho_depth(healthy, mean over countries)")
for k, v in feats.items():
    aucs = [roc_auc_score(y[co == c], v[co == c]) for c in COHORTS]
    rho = np.mean([spearmanr(v[(co == c) & (y == 0)], ld[(co == c) & (y == 0)])[0] for c in COHORTS])
    print("%-14s" % k, *["%6.3f" % a for a in aucs], " | %.3f | %+.3f" % (np.mean(aucs), rho))
print("rare oral presence prevalence <5%% in fit:", sum(1 for i in ORAL_EXT if (X[:, i] > 0).mean() < MIN_PREV), "of", len(ORAL_EXT))
print("genus count (>=5%% prev sums):", len(set(genus(c) for c in TAXA)))
