"""Preference ("mood") labels for recipes.

Two kinds of tags:
  * macro tags   -> need per-serving nutrition (kcal, protein, carbs, fat, fiber)
  * keyword tags -> derived from ingredient text only (approximate, see KEYWORD_CAVEATS)

All thresholds live in THRESHOLDS so you can change them in one place.
"""
import re

from . import allergens as _A

THRESHOLDS = {
    # Macro tags use shares of calories, so they need no servings and work for computed-only recipes too.
    "high_protein_share": 0.25,    # protein >= 25% of calories
    "high_carb_share": 0.55,       # carbs   >= 55% of calories
    "low_carb_share": 0.20,        # carbs   <= 20% of calories
    "keto_carb_share": 0.10,       # carbs   <= 10% of calories ...
    "keto_fat_share": 0.60,        # ... and fat >= 60% of calories
    # Per-serving only (needs real servings, so xiaoxl rows only):
    "low_calorie_kcal": 400.0,
}
# Computed shares (no label) are about +/-3 points off, so a recipe must clear a threshold by this much to get the tag.
COMPUTED_MARGIN = 0.03

MACRO_TAGS = ["high_protein", "high_carb", "low_carb", "keto", "low_calorie"]
KEYWORD_TAGS = ["vegetarian", "vegan", "dairy_free", "gluten_free", "spicy"]
ALL_TAGS = MACRO_TAGS + KEYWORD_TAGS

KEYWORD_CAVEATS = (
    "Keyword tags read only the ingredient list: 'gluten_free' and 'dairy_free' ignore hidden sources "
    "(e.g. stock cubes, soy sauce is treated as gluten), and 'spicy' means a chile/heat ingredient is present."
)

_MEAT = ("chicken|beef|pork|bacon|ham|sausage|turkey|lamb|veal|steak|meatball|prosciutto|salami|pepperoni|venison|"
         "duck|chorizo|brisket|ribs|hot dog|bratwurst|pancetta|lard|bologna|liverwurst|oxtail|goose|rabbit|bison|jerky")
_FISH = ("fish|salmon|tuna|shrimp|crab|lobster|cod|tilapia|anchovy|anchovies|clam|mussel|scallop|oyster|squid|calamari|"
         "sardine|trout|halibut|catfish|mackerel|prawn|crawfish|crayfish|smelt|snapper|swordfish|haddock|bass|herring")
_DAIRY = (r"milk|cream|butter|cheese|yogurt|yoghurt|buttermilk|ghee|whey|custard|half-and-half|half and half|"
          r"parmesan|mozzarella|cheddar|ricotta|mascarpone|feta|kefir|ice cream|casein|evaporated milk|condensed milk")
_DAIRY_FREE_OK = re.compile(
    r"(coconut|almond|soy|oat|rice|cashew|hemp|nut)\s+(milk|cream|butter|yogurt|cheese)|peanut butter|cocoa butter|"
    r"butternut|shea butter|cream of tartar|nut butter|sunflower butter|apple butter|vegan (butter|cheese)|margarine|"
    r"dairy-free|non-dairy|nondairy")
_EGG = r"\beggs?\b(?!plant)|mayonnaise|\bmayo\b|meringue|egg white|egg yolk"
_OTHER_ANIMAL = r"honey|gelatin|gelatine|lard|tallow|fish sauce|worcestershire|oyster sauce|rennet|isinglass|shellac|anchov"
_GLUTEN = (r"\bwheat\b|\bflour\b|bread|pasta|noodle|spaghetti|macaroni|couscous|barley|\brye\b|semolina|bulgur|seitan|"
           r"soy sauce|\bbeer\b|cracker|tortilla|cake mix|bisquick|panko|breadcrumb|bread crumb|biscuit|pita|bagel|"
           r"croissant|dumpling|lasagna|ravioli|orzo|farro|malt|pie crust|puff pastry|phyllo|cookie|brownie mix|"
           r"cereal|graham|pretzel|ramen|udon|breading|cornbread|stuffing mix|oats\b|rolled oats")
_GLUTEN_OK = re.compile(r"gluten-free|gluten free|rice flour|almond flour|coconut flour|corn flour|cornflour|chickpea flour|"
                        r"tapioca flour|potato flour|oat flour certified|rice noodle|rice pasta|corn tortilla|cornstarch|"
                        r"buckwheat flour|arrowroot|tamari|gluten-free oats|sorghum flour|cassava flour|masa harina|"
                        r"nut flour|flaxseed meal|rice paper")
_SPICY = (r"cayenne|jalape.{1,2}o|serrano|habanero|chipotle|sriracha|hot sauce|red pepper flakes|crushed red pepper|"
          r"chili flakes|chile flakes|gochujang|harissa|curry paste|scotch bonnet|tabasco|thai chil|chili pepper|chile pepper|"
          r"hot pepper|hot chil|ghost pepper|sambal|chili paste|chili garlic|chili oil|peri-peri|piri piri|"
          r"cayenne pepper|pepper flakes|red chil|fresno|chiles?\b(?! powder)|calabrian|"
          r"wasabi|hot curry|vindaloo|jerk seasoning|berbere|aleppo")


def _text(ingredients):
    return " ; ".join(ingredients).lower()


def _has(rx, text):
    return re.search(rx, text) is not None


# "vegetarian sausage", "veggie burger", "meatless crumbles" are not meat
_PLANT_MOD = re.compile(r"\b(vegan|vegetarian|veggie|meatless|plant[- ]based|meat[- ]free|imitation|faux|mock)\s+(?:\w+\s+){0,2}")


def keyword_tags(ingredients, steps=None):
    """Return dict tag -> bool from the ingredient lines (list of strings) and, when given, the recipe steps.

    Steps are read too because some source ingredient lists are cut short: a recipe whose steps say "brown the beef"
    is not vegetarian even if the list has no beef. Allergen-based tags (dairy_free, gluten_free, vegan) use the same
    matcher as data/recipe_allergens.json."""
    t = _PLANT_MOD.sub(" ", _text(ingredients))
    step_text = _PLANT_MOD.sub(" ", " ; ".join(str(s) for s in steps).lower()) if steps else ""
    meat = _has(r"\b(" + _MEAT + r")\b", t)
    fish = _has(r"\b(" + _FISH + r")\b", t)
    dairy_text = _DAIRY_FREE_OK.sub(" ", t)
    dairy = _has(r"\b(" + _DAIRY + r")\b", dairy_text)
    egg = _has(_EGG, t)
    other_animal = _has(r"(" + _OTHER_ANIMAL + r")", t)
    gluten_text = _GLUTEN_OK.sub(" ", t)
    gluten = _has(_GLUTEN, gluten_text)
    if step_text:
        meat = meat or _has(r"\b(" + _MEAT + r")\b", step_text)
        fish = fish or _has(r"\b(" + _FISH + r")\b", step_text)
        other_animal = other_animal or _has(r"(" + _OTHER_ANIMAL + r")", step_text)
        in_steps = set()
        for s in steps:
            for sentence in re.split(r"(?<=[.!?])\s+", str(s)):
                in_steps |= _A.allergens_in_line(sentence)
        dairy = dairy or "milk" in in_steps
        egg = egg or "egg" in in_steps
        gluten = gluten or "gluten" in in_steps
    return {
        "vegetarian": not (meat or fish or _has(r"gelatin|gelatine|lard|fish sauce|anchov|worcestershire|oyster sauce", t)),
        "vegan": not (meat or fish or dairy or egg or other_animal),
        "dairy_free": not dairy,
        "gluten_free": not gluten,
        "spicy": _has(_SPICY, _text([i for i in ingredients if "optional" not in i.lower()])),
    }


def calorie_shares(kcal, protein_g, carb_g, fat_g):
    """-> (protein, carb, fat) shares of calories, or None if nutrition is missing. Works with totals or per-serving values."""
    vals = [kcal, protein_g, carb_g, fat_g]
    if any(v is None or v != v for v in vals) or kcal <= 0:
        return None
    return 4 * protein_g / kcal, 4 * carb_g / kcal, 9 * fat_g / kcal


def macro_tags(kcal, protein_g, carb_g, fat_g, fiber_g=None, per_serving=True, margin=0.0, th=THRESHOLDS):
    """Return dict tag -> bool, or {} if nutrition is missing.

    per_serving=False means the values are whole-recipe totals (servings unknown): only share-based tags are produced.
    margin: how far past a threshold the shares must be (use COMPUTED_MARGIN for computed, unlabelled nutrition).
    """
    s = calorie_shares(kcal, protein_g, carb_g, fat_g)
    if s is None:
        return {}
    p, c, f = s
    out = {
        "high_protein": p >= th["high_protein_share"] + margin,
        "high_carb": c >= th["high_carb_share"] + margin,
        "low_carb": c <= th["low_carb_share"] - margin,
        "keto": c <= th["keto_carb_share"] - margin and f >= th["keto_fat_share"] + margin,
    }
    if per_serving:
        out["low_calorie"] = kcal <= th["low_calorie_kcal"]
    return out


def all_tags(ingredients, kcal=None, protein_g=None, carb_g=None, fat_g=None, fiber_g=None, per_serving=True, margin=0.0,
             steps=None):
    """Union of macro tags (when nutrition is present) and keyword tags -> list of tag names that are True."""
    d = keyword_tags(ingredients, steps)
    d.update(macro_tags(kcal, protein_g, carb_g, fat_g, fiber_g, per_serving=per_serving, margin=margin))
    return [t for t in ALL_TAGS if d.get(t)]


def row_tags(r):
    """Tags for one row of data/recipes_merged.jsonl (a namedtuple from DataFrame.itertuples()).
    The nutrition source decides how macro tags are derived (see `nutrition_basis`).
    Keyword tags also read the steps (`directions`) when the row has them."""
    steps = getattr(r, "directions", None)
    if r.nutrition_basis.startswith("label"):          # real per-serving label values (xiaoxl)
        return all_tags(r.ingredients, r.kcal, r.protein_g, r.carb_g, r.fat_g, r.fiber_g, steps=steps)
    if r.nutrition_basis == "computed, no servings":   # USDA totals only: calorie shares, with a safety margin
        return all_tags(r.ingredients, r.calc_kcal_total, r.calc_protein_total, r.calc_carb_total, r.calc_fat_total,
                        None, per_serving=False, margin=COMPUTED_MARGIN, steps=steps)
    return all_tags(r.ingredients, steps=steps)         # keyword tags only
