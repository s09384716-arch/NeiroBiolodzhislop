"""v4: пол как ещё одна ковариата в остатках (в train 64 % мужчин, в test 50 %; в США 83 % против 47 %). Одна идея, confirm=True."""
import numpy as np
import v3core as C

_orig = C.cov_design


def cov_design_gender(df, st, spec):
    return np.column_stack([_orig(df, st, spec), (df["gender"] == "M").to_numpy(float)])


def fp_gender(fit_df, pred_df):
    C.cov_design = cov_design_gender           # fit_preprocessing / apply_preprocessing берут cov_design из этого модуля
    try:
        return C.GutPipeline().fit(fit_df).predict(pred_df)
    finally:
        C.cov_design = _orig


if __name__ == "__main__":
    C.evaluate(fp_gender, "v3 + пол в остатках", log="v4_G.csv", confirm=True)
