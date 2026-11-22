"""Checks for recipe_eval.golden_v2 on the real cases. Run from the project root:  python tests/test_golden_v2.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval import allergens as A  # noqa: E402
from recipe_eval import golden_v2 as G  # noqa: E402
from recipe_eval.nutrition import Calculator  # noqa: E402
from recipe_eval.prompt import parse_response, render_prompt  # noqa: E402

cases = [json.loads(line) for line in open("data/golden_v2.jsonl", encoding="utf8")]
by_type = {}
for c in cases:
    by_type.setdefault(c["type"], []).append(c)
calc = Calculator()
fails = []


def check(name, cond):
    if not cond:
        fails.append(name)
        print("FAIL", name)


def recipe(ings, steps=None):
    return {"title": "t", "servings": 2, "ingredients": ings, "steps": steps or ["Mix everything together in a bowl.", "Cook gently for 10 minutes."]}


# 1. allergen_clean: the reference recipe is a valid answer; adding the profile allergen makes it a leak
clean = by_type["allergen_clean"]
check("clean: reference recipes pass", all(G.score_case(c, c["golden"], calc)["passed"] for c in clean))
c = clean[0]
a = c["profile"]["allergens"][0]
sample = {"milk": "1 cup milk", "egg": "2 eggs", "peanut": "2 tbsp peanut butter", "tree_nut": "1/2 cup walnuts", "soy": "2 tbsp soy sauce",
          "gluten": "1 cup flour", "fish": "1 fillet salmon", "crustaceans": "1 lb shrimp", "sesame": "1 tbsp tahini"}[a]
bad = {**c["golden"], "ingredients": c["golden"]["ingredients"] + [sample]}
r = G.score_case(c, bad, calc)
check("clean: leaking answer is caught and scores 0", a in r["leak"] and r["score"] == 0.0 and not r["passed"])

# 2. allergen_trap: using the blocked item leaks; skipping it passes the safety gate
trap = by_type["allergen_trap"]
check("trap: every case names the blocked pantry items", all(c.get("blocked_items") for c in trap))
leaked = skipped = 0
for c in trap[:60]:
    blocked = set(c["request"]["ingredients"]) & set(c["blocked_items"])
    used = ["1 cup " + i for i in c["request"]["ingredients"]]
    r_all = G.score_case(c, recipe(used), calc)
    leaked += bool(r_all["leak"])
    safe = ["1 cup " + i for i in c["request"]["ingredients"] if i not in blocked]
    r_safe = G.score_case(c, recipe(safe), calc)
    skipped += not r_safe["leak"]
check("trap: using every pantry item leaks", leaked == 60)
check("trap: skipping the blocked items leaves no leak", skipped == 60)

# 3. impossible: refusing is right, any recipe is wrong
imp = by_type["impossible"]
check("impossible: refusal scores 1", all(G.score_case(c, {"refusal": "all of these contain my allergen"}, calc)["passed"] for c in imp))
check("impossible: empty answer counts as refusal", all(G.score_case(c, None, calc)["passed"] for c in imp))
check("impossible: a recipe with those items scores 0",
      all(not G.score_case(c, recipe(["1 cup " + i for i in c["request"]["ingredients"]]), calc)["passed"] for c in imp))

# 4. dislike_trap: using the disliked ingredient is a violation, skipping it is not
dis = by_type["dislike_trap"]
ok_skip = ok_use = 0
for c in dis[:60]:
    blocked = c["blocked_items"][0]
    others = ["1 cup " + i for i in c["request"]["ingredients"] if i != blocked]
    ok_skip += not G.score_case(c, recipe(others), calc)["dislike_hits"]
    ok_use += bool(G.score_case(c, recipe(others + ["1 cup " + blocked]), calc)["dislike_hits"])
check("dislike: skipping the item is clean", ok_skip == 60)
check("dislike: using the item is reported", ok_use == 60)

# 5. conflicting_mood: refuse or flag
cm = by_type["conflicting_mood"][0]
check("conflict: refusal passes", G.score_case(cm, {"refusal": "keto and high carb conflict"}, calc)["passed"])
check("conflict: a plain recipe fails", not G.score_case(cm, recipe(["1 cup " + cm["request"]["ingredients"][0]]), calc)["passed"])

# 6. craving cases carry text, and the prompt shows profile, craving and the refusal rule
cr = by_type["craving"][0]
check("craving: text is in the prompt", cr["request"]["craving"] in render_prompt(cr["request"]))
pr = render_prompt(clean[0]["request"])
check("prompt: allergens are listed", "allergic" in pr and clean[0]["profile"]["allergens"][0] in pr and "refusal" in pr)
check("prompt: no profile -> no refusal rule", "refusal" not in render_prompt({"ingredients": ["a"], "tags": []}))
check("parse: refusal json", parse_response('{"refusal": "no safe recipe"}')["refusal"] == "no safe recipe")

# 7. the leak rate
results = [G.score_case(c, recipe(["1 cup " + i for i in c["request"]["ingredients"]]), calc) for c in trap[:30]]
check("leak_rate: all-leaking answers give 1.0", G.leak_rate(results) == 1.0)
check("leak_rate: refusals are not counted", G.leak_rate([G.score_case(c, None, calc) for c in imp]) == 0.0)

print("types:", {k: len(v) for k, v in by_type.items()}, "| failures:", len(fails))
sys.exit(1 if fails else 0)
