"""Build evaluation examples from data/recipes_merged.jsonl.

Each example is one test case for the recipe model:
  input    : ingredients the user has (+ optional distractors that need NOT be used) and 1-2 mood tags
  golden   : the reference recipe (with per-serving nutrition when available)

Tags come from recipe_eval.labels. Macro tags (high_protein, keto, ...) need nutrition, so those examples
come only from rows that have it; keyword tags (spicy, vegan, ...) can come from any row.

Run from the project root:  python scripts/build_examples.py [per_tag] [seed]
Output: data/eval_examples.jsonl
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval import core as M  # noqa: E402
from recipe_eval import labels as L  # noqa: E402
from recipe_eval import scoring as S  # noqa: E402

PER_TAG = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
random.seed(SEED)


def readable(line):
    """'2 (10 ounce) cans chopped tomatoes, drained' -> 'chopped tomatoes'."""
    return S.clean_name(line).strip()


m = pd.read_json("data/recipes_merged.jsonl", lines=True)
m["tags"] = [L.row_tags(r) for r in m.itertuples()]
m["names"] = m["ingredients"].map(lambda ings: list(dict.fromkeys(n for n in (readable(i) for i in ings) if n and not S.is_staple(n))))
m = m[m["names"].map(len) >= 3].reset_index(drop=True)
pool = sorted({n for names in m["names"] for n in names if 3 <= len(n) <= 30})

chosen = {}   # recipe_id -> requested tags
for tag in L.ALL_TAGS:
    cand = m[m["tags"].map(lambda ts: tag in ts)]
    cand = cand[~cand["recipe_id"].isin(chosen)]
    if tag in L.MACRO_TAGS:
        # macro tags: up to half from rows with real labels (xiaoxl), the rest from computed-share rows
        lab = cand[cand["nutrition_basis"].str.startswith("label")]
        comp = cand[~cand["nutrition_basis"].str.startswith("label")]
        n_lab = min(len(lab), max(PER_TAG // 2, PER_TAG - len(comp)))  # all labelled if there is no computed pool (low_calorie)
        picks = [lab.sample(n_lab, random_state=SEED)] if n_lab else []
        picks.append(comp.sample(min(len(comp), PER_TAG - n_lab), random_state=SEED))
        pick = pd.concat(picks)
    else:
        pick = cand.sample(min(PER_TAG, len(cand)), random_state=SEED)
    for r in pick.itertuples():
        extra = [t for t in r.tags if t != tag and random.random() < 0.35]  # sometimes a second mood
        chosen[r.recipe_id] = [tag] + extra[:1]

by_id = m.set_index("recipe_id")
rows = []
for k, (rid, tags) in enumerate(chosen.items()):
    r = by_id.loc[rid]
    distractors = random.sample([p for p in pool if p not in set(r["names"])], random.choice([0, 1, 2]))
    available = list(r["names"]) + distractors
    random.shuffle(available)
    golden = {"title": r["recipe_title"], "servings": None if pd.isna(r["servings"]) else float(r["servings"]),
              "ingredients": list(r["ingredients"]), "steps": list(r["directions"])}
    if r["nutrition_basis"].startswith("label"):
        golden["nutrition_per_serving"] = {k2: float(r[c]) for k2, c in
                                           [("kcal", "kcal"), ("protein_g", "protein_g"), ("carb_g", "carb_g"), ("fat_g", "fat_g")]}
    elif r["nutrition_basis"] == "computed, no servings":
        golden["computed_totals"] = {k2: float(r[c]) for k2, c in
                                     [("kcal", "calc_kcal_total"), ("protein_g", "calc_protein_total"),
                                      ("carb_g", "calc_carb_total"), ("fat_g", "calc_fat_total")]}
    rows.append({"example_id": f"e{k:04d}", "recipe_id": rid, "source": r["source"], "nutrition_basis": r["nutrition_basis"],
                 "request": {"ingredients": available, "tags": tags},
                 "distractors": distractors, "golden": golden})

Path("data").mkdir(exist_ok=True)
with open("data/eval_examples.jsonl", "w", encoding="utf8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(len(rows), "examples saved to data/eval_examples.jsonl")
print("by primary tag:", pd.Series([r["request"]["tags"][0] for r in rows]).value_counts().to_dict())
print("by source:", pd.Series([r["source"] for r in rows]).value_counts().to_dict())
