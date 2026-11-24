"""Build the golden set v2: cases with a user profile (allergens, dislikes), free-text cravings and traps.

Output: data/golden_v2.jsonl (see recipe_eval/golden_v2.py for the case format and how to score it)

Case types (all built from the 1,200 golden-set recipes, so no training recipe is reused):
  allergen_clean   (profile allergen is NOT in the pantry or the recipe)          expected: recipe, no leak
  allergen_trap    (a pantry item carries the profile allergen)                    expected: recipe that skips that item
  impossible       (every pantry item carries the profile allergen)                expected: refuse
  dislike_trap     (the profile dislikes one pantry ingredient)                    expected: recipe without it
  conflicting_mood (keto + high_carb, low_carb + high_carb)                        expected: refuse or flag
  craving          (free-text craving next to the tags)                            expected: recipe

Run from the project root (after add_pool_fields.py and build_examples.py):  python scripts/build_golden_v2.py [seed]
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval import allergens as A  # noqa: E402
from recipe_eval import golden_v2 as G  # noqa: E402
from recipe_eval import scoring as S  # noqa: E402
from recipe_eval.normalize import norm_name  # noqa: E402
from recipe_eval.nutrition import Calculator  # noqa: E402

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 21
rng = random.Random(SEED)

EU = ["gluten", "crustaceans", "egg", "fish", "peanut", "soy", "milk", "tree_nut", "celery", "mustard", "sesame",
      "sulphites", "lupin", "molluscs"]
COMMON = ["milk", "egg", "peanut", "tree_nut", "soy", "gluten", "fish", "crustaceans", "sesame"]   # most likely in a profile
N = {"allergen_clean": 120, "allergen_trap": 150, "impossible": 60, "dislike_trap": 100, "conflicting_mood": 30, "craving": 100}

CRAVINGS = {
    "high_protein": ["Something really filling with lots of protein.", "I just worked out and need a protein-packed meal."],
    "high_carb": ["I want a big comforting bowl of carbs.", "Something hearty and starchy tonight."],
    "low_carb": ["Something light on carbs please.", "I'm trying to cut back on bread and pasta tonight."],
    "keto": ["Keto dinner, please.", "I'm doing keto this month, what can I make?"],
    "low_calorie": ["Something light that will not weigh me down.", "I want a low-calorie meal."],
    "vegetarian": ["No meat tonight.", "Something meat-free, please."],
    "vegan": ["Completely plant-based tonight.", "I'm eating vegan this week."],
    "dairy_free": ["No dairy for me today.", "I can't have dairy right now."],
    "gluten_free": ["I need it gluten free.", "Nothing with wheat, please."],
    "spicy": ["I want something with a real kick.", "Make it hot and spicy!"],
}

pool = {}
for line in open("data/recipes_merged.jsonl", encoding="utf8"):
    r = json.loads(line)
    pool[r["recipe_id"]] = r
examples = [json.loads(line) for line in open("data/eval_examples.jsonl", encoding="utf8")]


def carriers(allergen, items):
    """Pantry items that carry the allergen."""
    return [i for i in items if allergen in A.allergens_in_line(i)]


def carries_other(items, allergen):
    return [i for i in items if allergen not in A.allergens_in_line(i)]


def usable(items):
    return [i for i in items if not S.is_staple(i)]


cases = []


def add(type_, expected, profile, request, reason, example=None, golden=True, extra=None):
    req = {"ingredients": request["ingredients"], "tags": request.get("tags", []), "craving": request.get("craving")}
    if profile:
        req["profile"] = profile
    cases.append({"type": type_, "expected": expected, "profile": profile, "request": req, "reason": reason,
                  "recipe_id": example["recipe_id"] if example else None,
                  "golden": example["golden"] if (example and golden) else None, **(extra or {})})


# ---- allergen_clean: profile allergen is nowhere in the pantry or in the reference recipe
rng.shuffle(examples)
for ex in examples:
    if sum(c["type"] == "allergen_clean" for c in cases) >= N["allergen_clean"]:
        break
    r = pool.get(ex["recipe_id"])
    if not r or r["allergen_status"] != "verified":
        continue
    present = set(r["allergens"])
    for i in ex["request"]["ingredients"]:
        present |= A.allergens_in_line(i)
    cand = [a for a in COMMON if a not in present and not (a == "crustaceans" and "shellfish" in present)]
    if not cand:
        continue
    pick = rng.sample(cand, 2) if len(cand) > 2 and rng.random() < 0.35 else [rng.choice(cand)]
    add("allergen_clean", "recipe", {"allergens": sorted(pick), "dislikes": []}, ex["request"],
        f"profile avoids {', '.join(sorted(pick))}; neither the pantry nor the reference recipe contains it", ex)

# ---- allergen_trap: a pantry item carries the allergen, enough safe items remain
rng.shuffle(examples)
for ex in examples:
    if sum(c["type"] == "allergen_trap" for c in cases) >= N["allergen_trap"]:
        break
    items = usable(ex["request"]["ingredients"])
    options = []
    for a in COMMON + ["celery", "mustard"]:
        bad = carriers(a, items)
        safe = carries_other(items, a)
        if bad and len(safe) >= 3:
            options.append((a, bad, safe))
    if not options:
        continue
    a, bad, safe = rng.choice(options)
    add("allergen_trap", "recipe", {"allergens": [a], "dislikes": []}, ex["request"],
        f"pantry has {', '.join(bad)} (contains {a}); a safe recipe must skip it and use the other ingredients", ex,
        golden=False, extra={"blocked_items": bad})

# ---- impossible: every pantry item carries the allergen
by_allergen = {}
for r in pool.values():
    for n in r["ingredients_norm"]:
        by_allergen.setdefault(n, 0)
        by_allergen[n] += 1
single = {}
implied = {"milk": set(), "egg": set(), "peanut": set(), "tree_nut": set(), "soy": set(), "fish": set(), "sesame": set(),
           "gluten": {"wheat", "gluten"}, "crustaceans": {"shellfish", "crustaceans"}}
for n, cnt in by_allergen.items():
    if cnt < 15 or S.is_staple(n):
        continue
    s = A.allergens_in_line(n)
    for a, extra in implied.items():
        if a in s and s <= ({a} | extra):
            single.setdefault(a, []).append(n)
for a, names in single.items():
    names.sort()
made = 0
allergens_cycle = [a for a in COMMON if len(single.get(a, [])) >= 5]
while made < N["impossible"] and allergens_cycle:
    a = allergens_cycle[made % len(allergens_cycle)]
    items = rng.sample(single[a], rng.choice([3, 4, 5]))
    add("impossible", "refuse", {"allergens": [a], "dislikes": []}, {"ingredients": items, "tags": []},
        f"every ingredient contains {a}, so no recipe can use one and stay safe")
    made += 1

# ---- dislike_trap: profile dislikes one pantry ingredient
rng.shuffle(examples)
for ex in examples:
    if sum(c["type"] == "dislike_trap" for c in cases) >= N["dislike_trap"]:
        break
    items = usable(ex["request"]["ingredients"])
    if len(items) < 4:
        continue
    d_item = rng.choice(items)
    word = (norm_name(d_item) or "").split(" ")[-1]
    if len(word) < 3:
        continue
    others = [i for i in items if i != d_item]
    if any(word in S._singular(w) or S._singular(w) == S._singular(word) for o in others for w in norm_name(o).split()):
        continue                                     # another pantry item shares the word: ambiguous
    add("dislike_trap", "recipe", {"allergens": [], "dislikes": [word]}, ex["request"],
        f"profile dislikes {word}; the pantry has {d_item}, which the recipe should leave out", ex, golden=False,
        extra={"blocked_items": [d_item]})

# ---- conflicting_mood
rng.shuffle(examples)
pairs = [["keto", "high_carb"], ["low_carb", "high_carb"]]
for k, ex in enumerate(examples[:N["conflicting_mood"]]):
    tags = pairs[k % 2]
    add("conflicting_mood", "refuse_or_flag", None, {"ingredients": ex["request"]["ingredients"], "tags": tags},
        f"{tags[0]} and {tags[1]} cannot both hold; the model should say so instead of pretending")

# ---- craving: free text next to the tags
rng.shuffle(examples)
for ex in examples[:N["craving"]]:
    first = ex["request"]["tags"][0]
    add("craving", "recipe", None, {**ex["request"], "craving": rng.choice(CRAVINGS[first])},
        "the mood is also given as free text", ex)

# ---- ids, checks, save
for i, c in enumerate(cases, 1):
    c["case_id"] = f"v2-{i:04d}"
order = ["case_id", "type", "expected", "profile", "request", "reason", "recipe_id", "golden"]
cases = [{**{k: c[k] for k in order}, **{k: v for k, v in c.items() if k not in order}} for c in cases]

bad = 0
calc = Calculator()
for c in cases:
    if c["type"] == "allergen_clean":                # the reference recipe must pass its own profile
        res = G.score_case(c, c["golden"], calc)
        if res["leak"] or not res["passed"] or res["score"] < 0.8:
            bad += 1
    if c["type"] == "impossible":
        for i in c["request"]["ingredients"]:
            assert c["profile"]["allergens"][0] in A.allergens_in_line(i), (c["case_id"], i)
    if c["type"] == "allergen_trap":
        assert c["blocked_items"], c["case_id"]
print("golden-recipe self-check failures (allergen_clean):", bad)

with open("data/golden_v2.jsonl", "w", encoding="utf8") as f:
    for c in cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(len(cases), "cases ->", "data/golden_v2.jsonl", dict(Counter(c["type"] for c in cases)))
print("allergens in profiles:", dict(Counter(a for c in cases if c["profile"] for a in c["profile"]["allergens"]).most_common()))
