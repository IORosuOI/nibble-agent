"""Checks for recipe_eval.allergens. Run from the project root:  python tests/test_allergens.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipe_eval.allergens import allergens_in_line as f  # noqa: E402
from recipe_eval.allergens import normalize_allergen  # noqa: E402

CASES = {
    "2 tbsp soy sauce": {"soy", "wheat", "gluten"},
    "1 cup all-purpose flour": {"wheat", "gluten"},
    "1 cup almond flour": {"tree_nut"},
    "1 cup gluten-free flour": set(),
    "2 eggs": {"egg"},
    "1 eggplant, diced": set(),
    "1 cup buckwheat": set(),
    "2 tbsp peanut butter": {"peanut"},
    "1 cup almond milk": {"tree_nut"},
    "1 can coconut milk": set(),
    "1 cup whole milk": {"milk"},
    "1/2 cup butter, softened": {"milk"},
    "1 cup vegan butter": set(),
    "1 lb shrimp, peeled": {"crustaceans", "shellfish"},
    "1 can clams": {"molluscs", "shellfish"},
    "1 cup white wine": {"sulphites"},
    "2 stalks celery": {"celery"},
    "1 tbsp dijon mustard": {"mustard"},
    "1 cup lupin flour": {"lupin"},
    "1 chicken bouillon cube": {"celery"},
    "1 tbsp curry powder": {"mustard"},
    "1 crab apple": set(),
    "1 cup oyster mushrooms": set(),
    "2 tbsp worcestershire sauce": {"fish"},
    "1 tsp cream of tartar": set(),
    "8 oz egg noodles": {"egg", "wheat", "gluten"},
    "1 bunch swiss chard": set(),
    "1/2 cup chopped walnuts": {"tree_nut"},
    "1 tsp ground nutmeg": set(),
    "1 can water chestnuts": set(),
    "1 cup rice": set(),
    "2 slices bread": {"wheat", "gluten"},
    "1/4 cup tahini": {"sesame"},
    "1 cup sliced strawberries": set(),
    "1 bottle beer": {"gluten"},
    "1 cup kale": set(),
    # EU Annex II: oats are a cereal containing gluten; spelt, khorasan (kamut) and durum are wheat
    "1 cup rolled oats": {"gluten"},
    "1/2 cup quick-cooking oatmeal": {"gluten"},
    "1 cup oat flour": {"gluten"},
    "1 cup oat milk": {"gluten"},
    "1 cup gluten-free rolled oats": set(),
    "1 cup granola": {"gluten"},
    "1 cup spelt": {"wheat", "gluten"},
    "1 cup kamut": {"wheat", "gluten"},
    "1 cup durum semolina": {"wheat", "gluten"},
    "1 cup freekeh": {"wheat", "gluten"},
    "8 oz goat meat": set(),
    "8 oz imitation crab": {"fish", "crustaceans", "shellfish"},
    "1 cup Romano (cranberry) beans": set(),
    "1/2 cup grated romano": {"milk"},
    # sentences from recipe steps: tools and verbs are not ingredients
    "Cool cookies on the cookie sheet.": {"wheat", "gluten"},        # cookies are, the sheet is not (cookies matches)
    "Place on a cookie sheet.": set(),
    "Line a muffin tin with paper liners.": set(),
    "Cut in the butter with a pastry blender.": {"milk"},
    "Roll out the mixture and roll up tightly.": set(),
    "Spoon the filling into the rolls.": {"wheat", "gluten"},
    "Grease a loaf pan.": set(),
}

bad = 0
for text, want in CASES.items():
    got = f(text)
    if got != want:
        bad += 1
        print("MISMATCH", repr(text), "got", sorted(got), "want", sorted(want))

# every official EU family name resolves to its code
fam = json.load(open(Path(__file__).resolve().parent.parent / "data" / "allergen_families.json", encoding="utf8"))["families"]
assert len(fam) == 14
for code, info in fam.items():
    if normalize_allergen(info["eu_name"]) != code:
        bad += 1
        print("MISMATCH eu_name", info["eu_name"])
print("cases", len(CASES), "bad", bad)
sys.exit(1 if bad else 0)
