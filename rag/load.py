"""Read recipes from the dataset files into one common shape, and build the text that gets embedded.

Two sources:
  load_eval() - the 1,200 golden recipes in data/eval_examples.jsonl (in git; use to build and test the pipeline)
  load_pool() - the full pool, data/recipes_merged.jsonl (23,395 recipes; not in git, get it from the dataset owner)

Every recipe comes out as a dict:
  recipe_id, title, ingredients (lines with amounts), steps, ingredient_names (staples removed),
  source, category, nutrition_basis, servings, servings_estimated, nutrition_per_serving, nutrition_totals,
  single_step, tags, allergens, allergens_steps_only, allergens_known, eval_holdout
"""
import json
import math

from recipe_eval import allergens as A
from recipe_eval import labels as L
from recipe_eval import scoring as S

from .common import ROOT

EVAL_PATH = ROOT / "data" / "eval_examples.jsonl"
POOL_PATH = ROOT / "data" / "recipes_merged.jsonl"
ALLERGENS_PATH = ROOT / "data" / "recipe_allergens.json"
NUTRIENTS = ("kcal", "protein_g", "carb_g", "fat_g")


def _read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _num(x):
    """JSON number or None (pandas writes missing values as null; guard against NaN anyway)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(x)


def eval_ids(path=EVAL_PATH):
    """recipe_ids of the golden test set. Searches during evaluation must exclude these."""
    return {r["recipe_id"] for r in _read_jsonl(path)}


def ingredient_names(lines):
    """'2 (10 ounce) cans chopped tomatoes, drained' -> 'chopped tomatoes'; staples (salt, water, oil) dropped, order kept."""
    names = (S.clean_name(line).strip() for line in lines)
    return list(dict.fromkeys(n for n in names if n and not S.is_staple(n)))


def _tags(ingredients, basis, per_serving, totals):
    """Same rules as recipe_eval.labels.row_tags, so a recipe gets the same mood tags as in the dataset."""
    if basis.startswith("label") and per_serving:
        return L.all_tags(ingredients, *(per_serving[k] for k in NUTRIENTS))
    if basis == "computed, no servings" and totals:
        return L.all_tags(ingredients, *(totals[k] for k in NUTRIENTS), None,
                          per_serving=False, margin=L.COMPUTED_MARGIN)
    return L.all_tags(ingredients)


def _finish(recipes, holdout):
    """Add tags, allergens and the eval-holdout flag to every recipe."""
    try:
        known = json.loads(ALLERGENS_PATH.read_text(encoding="utf8"))
    except OSError:
        known = {}
    for r in recipes:
        r["ingredient_names"] = ingredient_names(r["ingredients"])
        r["tags"] = _tags(r["ingredients"], r["nutrition_basis"], r["nutrition_per_serving"], r["nutrition_totals"])
        # Allergens: the dataset file if it has this recipe, otherwise the same matcher run here.
        a = known.get(r["recipe_id"]) or A.recipe_allergens(r["ingredients"], r["steps"])
        r["allergens"] = a["allergens"]
        r["allergens_steps_only"] = a["steps_only"]
        r["allergens_known"] = True
        r["eval_holdout"] = r["recipe_id"] in holdout
    return recipes


def load_eval(path=EVAL_PATH):
    """The golden recipes of the test set, in the common shape. All are marked eval_holdout."""
    out = []
    for e in _read_jsonl(path):
        g = e["golden"]
        per = g.get("nutrition_per_serving")
        tot = g.get("computed_totals")
        out.append({
            "recipe_id": e["recipe_id"],
            "title": g["title"],
            "ingredients": list(g["ingredients"]),
            "steps": list(g["steps"]),
            "source": e["source"],
            "category": None,   # not stored in the test set
            "nutrition_basis": e["nutrition_basis"],
            "servings": _num(g.get("servings")),
            "servings_estimated": None,
            "nutrition_per_serving": {k: _num(per[k]) for k in NUTRIENTS} if per else None,
            "nutrition_totals": {k: _num(tot[k]) for k in NUTRIENTS} if tot else None,
            "single_step": len(g["steps"]) < 2,
        })
    return _finish(out, {r["recipe_id"] for r in out})


def load_pool(path=POOL_PATH):
    """The full recipe pool in the common shape. Recipes that are in the test set are marked eval_holdout.
    Column names follow recipes.ipynb (merge cell) and scripts/add_computed_nutrition.py."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. It is not in git; get it from the dataset owner "
                                "(see DATASET_SUMMARY.md) and put it in data/.")
    out = []
    for r in _read_jsonl(path):
        basis = r.get("nutrition_basis") or "none"
        per = {k: _num(r.get(k)) for k in NUTRIENTS} if basis.startswith("label") else None
        tot = None
        if basis == "computed, no servings":
            tot = {"kcal": _num(r.get("calc_kcal_total")), "protein_g": _num(r.get("calc_protein_total")),
                   "carb_g": _num(r.get("calc_carb_total")), "fat_g": _num(r.get("calc_fat_total"))}
        out.append({
            "recipe_id": r["recipe_id"],
            "title": r["recipe_title"],
            "ingredients": list(r["ingredients"]),
            "steps": list(r["directions"]),
            "source": r["source"],
            "category": r.get("subcategory") or r.get("category"),
            "nutrition_basis": basis,
            "servings": _num(r.get("servings")),
            "servings_estimated": r.get("servings_estimated"),
            "nutrition_per_serving": per,
            "nutrition_totals": tot,
            "single_step": bool(r.get("single_step", len(r["directions"]) < 2)),
        })
    holdout = eval_ids() if EVAL_PATH.exists() else set()
    return _finish(out, holdout)


def load(source):
    """source: 'eval' or 'pool'."""
    return {"eval": load_eval, "pool": load_pool}[source]()


def chunk_text(r):
    """The text that is embedded for one recipe (one chunk per recipe, see section 3)."""
    return (f"{r['title']}\n"
            f"Ingredients: {', '.join(r['ingredient_names'])}\n"
            f"Directions: {' '.join(r['steps'])}")
