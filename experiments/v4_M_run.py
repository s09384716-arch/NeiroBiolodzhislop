"""Запуск идей исполнителя M:  python -X utf8 v4_M_run.py [--confirm] ключ [ключ ...]
Каждая идея = смесь компонентов (вид, параметры, вес) + режим усреднения (prob|rank). Все попытки пишутся в v4_M.csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v4_M_lib import *

IDEAS = {
    # (а) гиперпараметры леса (остальное как в v3: 0.7 лес + 0.3 логрегрессия C=0.01)
    "rf_mf0.1":      ("RF max_features=0.1 + LR", [("rf", {"max_features": 0.1}, .7), LR], "prob"),
    "rf_mf0.02":     ("RF max_features=0.02 + LR", [("rf", {"max_features": 0.02}, .7), LR], "prob"),
    "rf_msl5":       ("RF min_samples_leaf=5 + LR", [("rf", {"min_samples_leaf": 5}, .7), LR], "prob"),
    "rf_balsub":     ("RF class_weight=balanced_subsample + LR", [("rf", {"class_weight": "balanced_subsample"}, .7), LR], "prob"),
    # (б) ExtraTrees
    "et":            ("ExtraTrees(500,sqrt,leaf2) + LR", [("et", {}, .7), LR], "prob"),
    # (в) градиентный бустинг со строгой регуляризацией
    "hgb":           ("HGB(depth2,lr.03,ES) вместо части леса: .4RF+.3HGB+.3LR", [("rf", {}, .4), ("hgb", {}, .3), LR], "prob"),
    # (г) логрегрессия
    "lr_C0.03":      ("LR C=0.03 (лес как v3)", [RF, ("lr", {"C": 0.03}, .3)], "prob"),
    "lr_C0.003":     ("LR C=0.003 (лес как v3)", [RF, ("lr", {"C": 0.003}, .3)], "prob"),
    "lr_en":         ("LR elasticnet C=0.1 l1=.5", [RF, ("lr", {"C": 0.1, "l1_ratio": 0.5}, .3)], "prob"),
    # (д) смесь / ранги / 3 модели
    "w0.5":          ("веса 0.5 RF / 0.5 LR", [("rf", {}, .5), ("lr", {"C": LR_C}, .5)], "prob"),
    "w0.8":          ("веса 0.8 RF / 0.2 LR", [("rf", {}, .8), ("lr", {"C": LR_C}, .2)], "prob"),
    "rank":          ("усреднение рангов 0.7 RF / 0.3 LR", V3, "rank"),
    "rank_w0.8":     ("усреднение рангов 0.8 RF / 0.2 LR", [("rf", {}, .8), ("lr", {"C": LR_C}, .2)], "rank"),
    "rf_et_lr":      ("3 модели RF .35 + ET .35 + LR .3", [("rf", {}, .35), ("et", {}, .35), LR], "prob"),
    "rf_et_lr_rank": ("3 модели ранги RF .35 + ET .35 + LR .3", [("rf", {}, .35), ("et", {}, .35), LR], "rank"),
}

if __name__ == "__main__":
    args = sys.argv[1:]
    conf = "--confirm" in args
    for k in [a for a in args if not a.startswith("--")]:
        desc, spec, mode = IDEAS[k]
        run(f"{k}: {desc}", spec, mode, confirm=conf)
