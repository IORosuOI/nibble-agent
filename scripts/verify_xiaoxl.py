"""Recompute macros from ingredients for the xiaoxl recipes and flag where they agree with the labels.

Input : data/xiaoxl_nutrition_new.csv      (xiaoxl rows with nutrition, deduped against the Kaggle set)
Output: data/xiaoxl_nutrition_checked.csv  (+ calc_* columns and macro_verified / label_suspect flags)
Run from the project root:  python scripts/verify_xiaoxl.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval.nutrition import Calculator  # noqa: E402

calc = Calculator()
d = pd.read_csv("data/xiaoxl_nutrition_new.csv")
rows = []
for r in d.itertuples():
    lines = [x.strip() for x in str(r.ingredients).split(" ; ") if x.strip()]
    ps, cov, _ = calc.per_serving(lines, r.servings)
    rows.append((ps["kcal"], ps["protein"], ps["carb"], ps["fat"], cov))
c = pd.DataFrame(rows, columns=["calc_kcal", "calc_protein_g", "calc_carb_g", "calc_fat_g", "ingredient_coverage"])
d = pd.concat([d.reset_index(drop=True), c], axis=1)


def rel(a, b):
    return (a - b).abs() / b.clip(lower=1)


e_k = rel(d.calc_kcal, d.calories)
e_p = rel(d.calc_protein_g, d.protein_g)
e_c = rel(d.calc_carb_g, d.carbohydrates_g)
e_f = rel(d.calc_fat_g, d.fat_g)
atw = (4 * d.protein_g + 4 * d.carbohydrates_g + 9 * d.fat_g) / d.calories.clip(lower=1)
d["atwater_ok"] = (atw > 0.85) & (atw < 1.15)
d["well_covered"] = d.ingredient_coverage >= 0.8
d["macro_verified"] = d.well_covered & (e_k <= 0.25) & (e_p <= 0.4) & (e_c <= 0.4) & (e_f <= 0.4) & d.atwater_ok
d["label_suspect"] = d.well_covered & (e_k > 0.6)
d.to_csv("data/xiaoxl_nutrition_checked.csv", index=False, encoding="utf-8")
print(len(d), "rows |", int(d.macro_verified.sum()), "macro_verified |", int(d.label_suspect.sum()), "label_suspect")
