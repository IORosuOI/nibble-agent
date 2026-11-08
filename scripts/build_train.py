"""Build training/validation files from data/recipes_merged.jsonl, EXCLUDING every recipe used in data/eval_examples.jsonl.

Each line: {"recipe_id", "source", "nutrition_basis", "request": {"ingredients", "tags"}, "distractors",
            "messages": [user prompt, assistant JSON recipe], "target": {...recipe...}}
The request is built the same way as in build_examples.py (ingredients the user has + 0-2 distractors + 1-2 mood tags).

Run from the project root:  python scripts/build_train.py [val_fraction] [seed]
Output: data/train_examples.jsonl, data/val_examples.jsonl
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval import labels as L  # noqa: E402
from recipe_eval import prompt as P  # noqa: E402
from recipe_eval import scoring as S  # noqa: E402

VAL_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
rng = random.Random(SEED)

eval_ids = {json.loads(l)["recipe_id"] for l in open("data/eval_examples.jsonl", encoding="utf8")}
m = pd.read_json("data/recipes_merged.jsonl", lines=True)
m["tags"] = [L.row_tags(r) for r in m.itertuples()]
m["names"] = m["ingredients"].map(lambda ings: list(dict.fromkeys(
    n for n in (S.clean_name(i).strip() for i in ings) if n and not S.is_staple(n))))
pool = sorted({n for names in m["names"] for n in names if 3 <= len(n) <= 30})

n_all = len(m)
m = m[~m["recipe_id"].isin(eval_ids)]
n_after_eval = len(m)
m = m[(m["names"].map(len) >= 3) & (m["tags"].map(len) >= 1)].reset_index(drop=True)

tag_freq = pd.Series([t for ts in m["tags"] for t in ts]).value_counts()
rows = []
for r in m.itertuples():
    # prefer rarer tags so keto / high_protein / spicy are not drowned out by vegetarian
    weights = [1.0 / tag_freq[t] for t in r.tags]
    first = rng.choices(r.tags, weights=weights, k=1)[0]
    tags = [first]
    rest = [t for t in r.tags if t != first]
    if rest and rng.random() < 0.35:
        tags.append(rng.choice(rest))
    distractors = rng.sample([p for p in pool if p not in set(r.names)], rng.choice([0, 1, 2]))
    available = list(r.names) + distractors
    rng.shuffle(available)
    request = {"ingredients": available, "tags": tags}
    target = {"title": r.recipe_title, "servings": None if pd.isna(r.servings) else float(r.servings),
              "ingredients": list(r.ingredients), "steps": list(r.directions)}
    rows.append({"recipe_id": r.recipe_id, "source": r.source, "nutrition_basis": r.nutrition_basis,
                 "request": request, "distractors": distractors,
                 "messages": [{"role": "user", "content": P.render_prompt(request)},
                              {"role": "assistant", "content": P.target_json(target)}],
                 "target": target})

rng.shuffle(rows)
n_val = int(len(rows) * VAL_FRAC)
val, train = rows[:n_val], rows[n_val:]
for name, part in (("train", train), ("val", val)):
    with open(f"data/{name}_examples.jsonl", "w", encoding="utf8") as f:
        for row in part:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

assert not ({r["recipe_id"] for r in rows} & eval_ids), "eval recipe leaked into training data"
print(f"merged {n_all} -> minus {len(eval_ids)} eval recipes = {n_after_eval} -> usable {len(rows)}")
print(f"train {len(train)} | val {len(val)} | eval recipes excluded: {len(eval_ids)}")
print("tags in train (first tag):", pd.Series([r["request"]["tags"][0] for r in train]).value_counts().to_dict())
