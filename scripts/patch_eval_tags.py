"""Re-check the tags of data/eval_examples.jsonl against the current labels and fix only what is wrong.

Why: the keyword tags used to read only the ingredient list. Some source lists are cut short, so a recipe whose steps use
beef or flour could be tagged vegetarian or gluten free. Tags now also read the steps (recipe_eval/labels.py), and some
test cases no longer hold. A full rebuild would change most cases, so this script keeps every valid case:
  - a case whose requested tags are all still true is kept as is;
  - a case with a wrong SECOND tag keeps the recipe and drops that tag;
  - a case whose PRIMARY tag is wrong gets a new recipe with the same primary tag (and the same kind of nutrition source),
    built the same way as in build_examples.py. `example_id` stays, `recipe_id` changes.
Writes data/eval_examples.jsonl in place and data/eval_patch_report.json (what changed).

Run from the project root (after add_pool_fields.py):  python scripts/patch_eval_tags.py [seed]
"""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval import labels as L  # noqa: E402
from recipe_eval import scoring as S  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
rng = random.Random(SEED)

pool = {}
for line in open("data/recipes_merged.jsonl", encoding="utf8"):
    r = json.loads(line)
    pool[r["recipe_id"]] = r
examples = [json.loads(line) for line in open("data/eval_examples.jsonl", encoding="utf8")]
used = {e["recipe_id"] for e in examples}


def names_of(r):
    out = []
    for i in r["ingredients"]:
        n = S.clean_name(i).strip()
        if n and not S.is_staple(n) and n not in out:
            out.append(n)
    return out


all_names = sorted({n for r in pool.values() for n in names_of(r) if 3 <= len(n) <= 30})


def tags_of(r):
    return set(L.row_tags(NS(**r)))


def build(example_id, r, tags):
    names = names_of(r)
    distractors = rng.sample([p for p in all_names if p not in set(names)], rng.choice([0, 1, 2]))
    available = names + distractors
    rng.shuffle(available)
    golden = {"title": r["recipe_title"], "servings": r["servings"], "ingredients": list(r["ingredients"]),
              "steps": list(r["directions"])}
    if r["nutrition_basis"].startswith("label"):
        golden["nutrition_per_serving"] = {"kcal": float(r["kcal"]), "protein_g": float(r["protein_g"]),
                                           "carb_g": float(r["carb_g"]), "fat_g": float(r["fat_g"])}
    elif r["nutrition_basis"] == "computed, no servings":
        golden["computed_totals"] = {"kcal": float(r["calc_kcal_total"]), "protein_g": float(r["calc_protein_total"]),
                                     "carb_g": float(r["calc_carb_total"]), "fat_g": float(r["calc_fat_total"])}
    return {"example_id": example_id, "recipe_id": r["recipe_id"], "source": r["source"],
            "nutrition_basis": r["nutrition_basis"], "request": {"ingredients": available, "tags": tags},
            "distractors": distractors, "golden": golden}


def same_group(a, b):
    return a.startswith("label") == b.startswith("label")


report = {"kept": 0, "second_tag_dropped": [], "replaced": []}
out = []
for e in examples:
    r = pool[e["recipe_id"]]
    now = tags_of(r)
    tags = e["request"]["tags"]
    if all(t in now for t in tags):
        report["kept"] += 1
        out.append(e)
    elif tags[0] in now:                                      # only the second tag is wrong
        e["request"]["tags"] = [tags[0]]
        report["second_tag_dropped"].append({"example_id": e["example_id"], "dropped": tags[1]})
        out.append(e)
    else:                                                     # the primary tag is wrong: new recipe, same primary tag
        primary = tags[0]
        cands = [p for p in pool.values() if p["recipe_id"] not in used and same_group(p["nutrition_basis"], e["nutrition_basis"])
                 and len(names_of(p)) >= 3]
        rng.shuffle(cands)
        new = None
        for p in cands:
            tg = tags_of(p)
            if primary in tg:
                new = p
                extra = [t for t in L.ALL_TAGS if t in tg and t != primary and rng.random() < 0.35]
                new_tags = [primary] + extra[:1]
                break
        assert new is not None, f"no replacement for {e['example_id']} ({primary})"
        used.add(new["recipe_id"])
        out.append(build(e["example_id"], new, new_tags))
        report["replaced"].append({"example_id": e["example_id"], "old_recipe_id": e["recipe_id"], "new_recipe_id": new["recipe_id"],
                                   "primary_tag": primary, "old_tags": tags})

with open("data/eval_examples.jsonl", "w", encoding="utf8") as f:
    for e in out:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
json.dump(report, open("data/eval_patch_report.json", "w", encoding="utf8"), indent=1)

bad = [e["example_id"] for e in out if not all(t in tags_of(pool[e["recipe_id"]]) for t in e["request"]["tags"])]
print(f"kept {report['kept']} | second tag dropped {len(report['second_tag_dropped'])} | replaced {len(report['replaced'])}")
print("cases still invalid after the patch:", len(bad), "| total", len(out), "| unique recipes", len({e['recipe_id'] for e in out}))
