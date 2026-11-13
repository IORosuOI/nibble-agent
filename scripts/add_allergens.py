"""Map every recipe in data/recipes_merged.jsonl to its allergens.

Output: data/recipe_allergens.json  {recipe_id: {"allergens": [...], "steps_only": [...]}}
  allergens  = found in the ingredient list or in the steps (the safe, wider answer)
  steps_only = found only in the steps (the ingredient list is probably incomplete; Kaggle lists are sometimes cut short)
Also prints how often the result agrees with the older dairy_free / gluten_free keyword tags.

Run from the project root:  python scripts/add_allergens.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval import allergens as A  # noqa: E402
from recipe_eval import labels as L  # noqa: E402

rows = [json.loads(line) for line in open("data/recipes_merged.jsonl", encoding="utf8")]
out = {}
count, steps_only = Counter(), Counter()
dairy_agree = gluten_agree = 0
for r in rows:
    res = A.recipe_allergens(r["ingredients"], r["directions"])
    out[r["recipe_id"]] = res
    count.update(res["allergens"])
    steps_only.update(res["steps_only"])
    kw = L.keyword_tags(r["ingredients"])
    listed = set(res["allergens"]) - set(res["steps_only"])
    dairy_agree += kw["dairy_free"] == ("milk" not in listed)
    gluten_agree += kw["gluten_free"] == ("gluten" not in listed)

with open("data/recipe_allergens.json", "w", encoding="utf8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

n = len(rows)
print(f"{n} recipes")
print("recipes with at least one allergen:", sum(1 for v in out.values() if v["allergens"]))
print("recipes with none:", sum(1 for v in out.values() if not v["allergens"]))
print("by allergen:", {k: v for k, v in count.most_common()})
print("found ONLY in steps:", dict(steps_only.most_common()), "| recipes affected:", sum(1 for v in out.values() if v["steps_only"]))
print(f"agreement with old tags (ingredients only): dairy_free {dairy_agree / n:.1%}, gluten_free {gluten_agree / n:.1%}")
