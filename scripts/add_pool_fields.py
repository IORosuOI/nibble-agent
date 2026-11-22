"""Add the retrieval fields to data/recipes_merged.jsonl (run AFTER add_computed_nutrition.py).

Adds, per recipe:
  ingredients_norm  list of plain singular ingredient names (recipe_eval/normalize.py), staples kept
  content_key       12-hex hash of the normalized title + sorted ingredients_norm: an identity that does not depend
                    on row order, for checking that a regenerated pool still holds the same recipes
  url               source link (xiaoxl rows: allrecipes.com; Kaggle rows: "" because that dataset has no links)
  allergens         allergens found in the ingredients or the steps, from data/allergen_map.json
                    (14 EU families: gluten, crustaceans, egg, fish, peanut, soy, milk, tree_nut, celery, mustard,
                    sesame, sulphites, lupin, molluscs; plus the convenience tags wheat and shellfish)
  steps_only        the subset of `allergens` found only in the steps (the ingredient list is incomplete)
  allergen_status   "verified" or "unverified". Unverified = an allergen shows up only in the steps, or fewer than 80% of
                    the ingredient lines are recognised in the nutrition database (so we cannot vouch for the list).
                    An allergic user should only be shown "verified" recipes.
  allergen_notes    why a recipe is unverified, e.g. ["steps_only:milk", "low_ingredient_coverage"]
Also rewrites data/recipe_allergens.json ({recipe_id: {allergens, steps_only, allergen_status}}).

Run from the project root:  python scripts/add_pool_fields.py
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval import allergens as A  # noqa: E402
from recipe_eval.normalize import ingredients_norm  # noqa: E402

POOL = "data/recipes_merged.jsonl"
XIAOXL = "data/xiaoxl_nutrition_checked.csv"
MIN_COVERAGE = 0.8

csv.field_size_limit(10 ** 9)
urls = {}
if Path(XIAOXL).exists():
    with open(XIAOXL, encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                key = (r["title"].strip().lower(), float(r["servings"]), round(float(r["calories"]), 1))
            except (TypeError, ValueError):
                continue
            urls.setdefault(key, r["url"])
else:
    print("note:", XIAOXL, "not found: xiaoxl urls will be empty")

rows = [json.loads(line) for line in open(POOL, encoding="utf8")]
out_allergens = {}
status_count, allergen_count = Counter(), Counter()
url_hits = xiaoxl_rows = 0
for r in rows:
    norm = ingredients_norm(r["ingredients"])
    r["ingredients_norm"] = norm
    title = " ".join(str(r["recipe_title"]).lower().split())
    r["content_key"] = hashlib.sha1("|".join([title] + sorted(norm)).encode("utf8")).hexdigest()[:12]

    url = ""
    if r["source"] == "xiaoxl":
        xiaoxl_rows += 1
        try:
            url = urls.get((str(r["recipe_title"]).strip().lower(), float(r["servings"]), round(float(r["kcal"]), 1)), "")
        except (TypeError, ValueError):
            url = ""
        url_hits += bool(url)
    r["url"] = url

    res = A.recipe_allergens(r["ingredients"], r["directions"])
    notes = [f"steps_only:{a}" for a in res["steps_only"]]
    cov = r.get("calc_coverage")
    if cov is None or cov != cov or cov < MIN_COVERAGE:
        notes.append("low_ingredient_coverage")
    r["allergens"] = res["allergens"]
    r["steps_only"] = res["steps_only"]
    r["allergen_status"] = "unverified" if notes else "verified"
    r["allergen_notes"] = notes
    status_count[r["allergen_status"]] += 1
    allergen_count.update(res["allergens"])
    out_allergens[r["recipe_id"]] = {"allergens": res["allergens"], "steps_only": res["steps_only"],
                                     "allergen_status": r["allergen_status"]}

with open(POOL, "w", encoding="utf8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("data/recipe_allergens.json", "w", encoding="utf8") as f:
    json.dump(out_allergens, f, ensure_ascii=False, separators=(",", ":"))

n = len(rows)
keys = Counter(r["content_key"] for r in rows)
print(f"{n} recipes | ingredients_norm empty: {sum(1 for r in rows if not r['ingredients_norm'])}")
print(f"url: {url_hits} of {xiaoxl_rows} xiaoxl recipes matched a link; Kaggle recipes have none")
print(f"content_key unique: {len(keys)} distinct of {n} (duplicates: {sum(1 for v in keys.values() if v > 1)} keys)")
print("allergen_status:", dict(status_count))
print("recipes with no allergen:", sum(1 for r in rows if not r["allergens"]))
print("by allergen:", dict(allergen_count.most_common()))
