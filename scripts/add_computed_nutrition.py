"""Add USDA-computed *total* recipe macros to data/recipes_merged.jsonl.

Kaggle recipes have no servings, so only whole-recipe totals (and therefore calorie shares) can be computed.
Servings are not recoverable from the data, so for Kaggle rows they are set to round(total / 750 kcal), min 1
(`servings_estimated` = True). Recipes above MAX_TOTAL_KCAL (4000) are dropped.

Adds columns: calc_kcal_total, calc_protein_total, calc_carb_total, calc_fat_total, calc_coverage, nutrition_basis
  nutrition_basis = "label, agrees with USDA recompute"  (xiaoxl rows, per-serving label values)
                  = "computed, no servings"              (Kaggle rows with >= 80% ingredient coverage)
                  = "none"                               (everything else)
Run from the project root:  python scripts/add_computed_nutrition.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval.nutrition import Calculator  # noqa: E402

MIN_COVERAGE = 0.8
MAX_TOTAL_KCAL = 4000      # batch recipes are useless for a home-cooking app: drop anything bigger
KCAL_PER_SERVING = 750     # target serving size used to estimate servings for Kaggle rows
PATH = "data/recipes_merged.jsonl"

m = pd.read_json(PATH, lines=True)
calc = Calculator()
rows = []
for r in m.itertuples():
    tot, cov, _ = calc.macros(list(r.ingredients))
    rows.append((tot["kcal"], tot["protein"], tot["carb"], tot["fat"], cov))
c = pd.DataFrame(rows, columns=["calc_kcal_total", "calc_protein_total", "calc_carb_total", "calc_fat_total", "calc_coverage"])
for col in c.columns:
    m[col] = c[col].values

computed_ok = (m["source"] == "kaggle") & (m["calc_coverage"] >= MIN_COVERAGE) & (m["calc_kcal_total"] > 0)
m["nutrition_basis"] = "none"
m.loc[m["source"] == "xiaoxl", "nutrition_basis"] = "label, agrees with USDA recompute"
m.loc[computed_ok, "nutrition_basis"] = "computed, no servings"
# 1. Drop big-batch recipes (whole-recipe total above the cap; xiaoxl total = per-serving kcal x servings).
total = m["calc_kcal_total"].where(m["source"] == "kaggle", m["kcal"] * m["servings"])
big = total > MAX_TOTAL_KCAL
print(f"dropping {int(big.sum())} recipes with total > {MAX_TOTAL_KCAL} kcal:", m[big].groupby("source").size().to_dict())
m = m[~big].reset_index(drop=True)
computed_ok = computed_ok[~big].reset_index(drop=True)

# 2. Kaggle rows have no servings: estimate them so one serving is about KCAL_PER_SERVING (flagged, never a label).
m["servings_estimated"] = False
need = computed_ok & m["servings"].isna()
m.loc[need, "servings"] = (m.loc[need, "calc_kcal_total"] / KCAL_PER_SERVING).round().clip(lower=1)
m.loc[need, "servings_estimated"] = True
m.to_json(PATH, orient="records", lines=True, force_ascii=False)
print(m.groupby("nutrition_basis").size().to_dict())
