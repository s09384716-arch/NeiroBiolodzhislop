"""Запуск идей F: python -X utf8 v4_F_run.py screen|confirm имя1 имя2 ...   (один процесс, лог v4_F.csv, глубина -> v4_F_depth.csv)"""
import sys
from v4_F_feat import *

FS = FULL_SPEC
IDEAS = {   # имя -> (kwargs, обоснование)
    "A_lit_resid": dict(lit_resid=True),                                           # 4 лит. признака чистим остатками как таксоны (богатство/Шеннон зависят от глубины)
    "B1_depth2": dict(spec=FS + ("depth2",)),                                      # нелинейность по глубине: квадрат log-глубины
    "B2_depth2_dxc": dict(spec=FS + ("depth2", "dxc")),                            # + глубина x страна (разные протоколы секвенирования)
    "C1_butyrate_sulfidogen": dict(extras=("but", "sulf")),                        # продуценты бутирата (ниже), Bilophila/Desulfovibrio (выше)
    "C2_oral_ext_ratio_ext": dict(extras=("oral_ext", "ratio_ext")),               # расширенный ротовой индекс + log(ротовые/бутират)
    "C3_ratio_lit": dict(extras=("ratio_lit",)),                                   # log(6 родов-маркёров / бутират-продуценты)
    "D_genus_agg": dict(genus_agg=True),                                           # суммы по родам как доп. признаки (устойчивее к дроблению mOTU)
    "E_rare_oral_bin": dict(rare_bin=True),                                        # присутствие редких ротовых таксонов (<5 % не попали в log-признаки)
    "F_topk150_MW": dict(topk=150),                                                # отбор топ-150 таксонов по MW внутри стран (fit_df-only)
}
# комбинации задаются позже через аргумент combo:ИМЯ=a+b (см. COMBOS)
COMBOS = {
    "G1_drop_richness_shannon": dict(lit_cols=(0, 1)),                       # убрать богатство и Шеннон (зависят от глубины, по странам знак эффекта неустойчив)
    "G2_drop_richness_shannon_resid": dict(lit_cols=(0, 1), lit_resid=True),
    "K1_A_C3": dict(lit_resid=True, extras=("ratio_lit",)),
    "K2_A_C1_C3": dict(lit_resid=True, extras=("but", "sulf", "ratio_lit")),
    "K3_A_D": dict(lit_resid=True, genus_agg=True),
    "K4_A_D_F": dict(lit_resid=True, genus_agg=True, topk=150),
    "K5_D_F": dict(genus_agg=True, topk=150),
    "K6_F_topk80": dict(topk=80),
    "K7_F_topk300": dict(topk=300),
}
ALL = {**IDEAS, **COMBOS}

if __name__ == "__main__":
    mode, names = sys.argv[1], sys.argv[2:]
    b = pickle.load(open(os.path.join(OUT, "v3_baseline.pkl"), "rb"))     # локальный файл, записанный нашим же стендом
    rho0 = depth_rho(b["oof"]); print("v3 rho(скор, глубина | здоровые): %.3f" % rho0[0], {k: round(v, 3) for k, v in rho0[1].items()}, flush=True)
    for nm in names:
        rec = Rec(make_fp(**ALL[nm]))
        r = evaluate(rec, nm + ("|confirm" if mode == "confirm" else ""), log="v4_F.csv", confirm=(mode == "confirm"))
        rho = depth_rho(rec.oof())
        pd.DataFrame([dict(name=nm, mode=mode, rho_mean=rho[0], **{"rho_" + k: v for k, v in rho[1].items()})]).to_csv(
            os.path.join(OUT, "v4_F_depth.csv"), mode="a", header=not os.path.exists(os.path.join(OUT, "v4_F_depth.csv")), index=False)
        print("   rho(скор, глубина | здоровые) %.3f (v3 %.3f)" % (rho[0], rho0[0]), flush=True)
