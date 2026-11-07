"""Score a model's recipe against the request (available ingredients + mood tags).

Request  : {"ingredients": [str], "tags": [str]}               (tags from recipe_eval.labels.ALL_TAGS)
Response : {"title": str, "servings": number, "ingredients": [str lines with amounts], "steps": [str]}

Three scores in [0, 1]:
  compliance  - hard gate: every non-staple ingredient used must come from the available list
  preference  - how well the recipe satisfies each requested tag (macro tags use computed nutrition)
  feasibility - rule checks: steps present, ingredients used in steps, sane temperatures/times, amounts given

overall = 0 if compliance hard-fails, else 0.3*compliance + 0.4*preference + 0.3*feasibility.

Nutrition for a response is computed with recipe_eval.nutrition (USDA); it is only accurate to roughly
+/-15-20% (median), so macro thresholds get a tolerance band (TOL) instead of exact cutoffs.
"""
import re

from . import core as M
from . import labels as L

WEIGHTS = {"compliance": 0.3, "preference": 0.4, "feasibility": 0.3}
TOL = 0.15  # leniency on macro thresholds, matches the calculator's typical error

# Ingredients a user is assumed to always have. Edit to taste.
STAPLES = {"salt", "pepper", "black pepper", "water", "ice", "cooking spray", "oil", "vegetable oil",
           "olive oil", "canola oil", "cooking oil"}


UNITWORDS = {"can", "cans", "jar", "jars", "package", "packages", "pkg", "container", "containers", "bottle", "bottles",
             "envelope", "envelopes", "packet", "packets", "ounce", "ounces", "oz", "fluid", "pinch", "dash", "clove", "cloves",
             "slice", "slices", "stalk", "stalks", "sprig", "sprigs", "bunch", "head", "heads", "large", "small", "medium",
             "whole", "piece", "pieces", "bag", "bags", "box", "boxes", "stick", "sticks", "cup", "cups", "tablespoon",
             "tablespoons", "teaspoon", "teaspoons", "pound", "pounds", "gram", "grams", "quart", "pint", "splash", "drop",
             "drops", "handful", "fillet", "fillets", "inch"}
STAPLE_WORDS = {"salt", "pepper", "water", "oil", "ice", "spray", "black", "white", "ground", "kosher", "sea", "table",
                "olive", "vegetable", "canola", "cooking", "freshly", "cracked", "extra", "virgin", "coarse", "fine", "nonstick"}
STAPLE_CORE = {"salt", "pepper", "water", "oil", "ice", "spray"}
_ADJ_COMMA = re.compile(r"\b(skinless|boneless)\s*,\s*")  # "skinless, boneless chicken" -> "boneless chicken"


def _singular(w):
    """tomatoes->tomato, cherries->cherry, peaches->peach, eggs->egg (both sides of a comparison get the same treatment)."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("oes", "ches", "shes", "xes", "sses")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def clean_name(line):
    """'2 (10 ounce) cans chopped tomatoes, drained' -> 'chopped tomatoes' (amounts, units, packaging removed)."""
    qty, unit, name, hint = M.parse(_ADJ_COMMA.sub("", line.lower()))
    words = [w for w in re.sub(r"[^a-z ]", " ", name).split() if w not in UNITWORDS]
    return " ".join(words)


def name_tokens(line):
    """Ingredient line -> set of normalised tokens (amounts, units, prep words dropped)."""
    words = [w for w in clean_name(line).split() if w not in M.STOP]
    return {_singular(w) for w in words}


def is_staple(line):
    toks = name_tokens(line)
    if not toks:
        return True
    return toks <= STAPLE_WORDS and bool(toks & STAPLE_CORE)


def _matches(out_toks, avail_list):
    if not out_toks:
        return True
    for a in avail_list:
        if a and (out_toks <= a or a <= out_toks):
            return True
    return False


def compliance(available, response_ingredients):
    """-> dict(score, hard_fail, extras, usage). `available` are ingredient names/lines the user has."""
    avail = [name_tokens(a) for a in available]
    extras = []
    used = set()
    non_staple = 0
    for line in response_ingredients:
        if is_staple(line):
            continue
        non_staple += 1
        toks = name_tokens(line)
        hit = [i for i, a in enumerate(avail) if a and (toks <= a or a <= toks)]
        if hit:
            used.update(hit)
        else:
            extras.append(line)
    score = 1.0 if non_staple == 0 else 1.0 - len(extras) / non_staple
    usage = len(used) / max(len(avail), 1)
    return {"score": score, "hard_fail": bool(extras), "extras": extras, "usage": usage}


def _ramp_up(x, target):
    """1 when x >= target*(1-TOL), else linear down to 0 at 0."""
    t = target * (1 - TOL)
    return 1.0 if x >= t else max(0.0, x / t)


def _ramp_down(x, limit, slack):
    """1 when x <= limit*(1+TOL), else linear down to 0 over `slack` more."""
    t = limit * (1 + TOL)
    return 1.0 if x <= t else max(0.0, 1 - (x - t) / slack)


def tag_score(tag, response_ingredients, macros):
    """Return (score in [0,1], detail). macros = per-serving dict(kcal, protein, carb, fat, fiber) or None."""
    th = L.THRESHOLDS
    if tag in L.KEYWORD_TAGS:
        ok = L.keyword_tags(response_ingredients)[tag]
        return (1.0 if ok else 0.0), "ingredient check"
    if macros is None or macros["kcal"] <= 0:
        return 0.0, "no nutrition computed"
    kcal, p, c, f = macros["kcal"], macros["protein"], macros["carb"], macros["fat"]
    # Macro tags are scored on calorie SHARES: independent of the servings the model claims.
    ps, cs, fs = 4 * p / kcal, 4 * c / kcal, 9 * f / kcal
    if tag == "high_protein":
        return _ramp_up(ps, th["high_protein_share"]), f"{ps:.0%} of kcal from protein"
    if tag == "high_carb":
        return _ramp_up(cs, th["high_carb_share"]), f"{cs:.0%} of kcal from carbs"
    if tag == "low_carb":
        return _ramp_down(cs, th["low_carb_share"], 0.20), f"{cs:.0%} of kcal from carbs"
    if tag == "keto":
        s = min(_ramp_down(cs, th["keto_carb_share"], 0.20), _ramp_up(fs, th["keto_fat_share"]))
        return s, f"{cs:.0%} of kcal from carbs, {fs:.0%} from fat"
    if tag == "low_calorie":  # per serving: depends on servings (see score_response(reference_servings=...))
        return _ramp_down(kcal, th["low_calorie_kcal"], 400.0), f"{kcal:.0f} kcal"
    return 0.0, "unknown tag"


def preference(tags, response_ingredients, macros):
    if not tags:
        return {"score": 1.0, "per_tag": {}}
    per = {}
    for t in tags:
        s, d = tag_score(t, response_ingredients, macros)
        per[t] = {"score": round(s, 3), "detail": d}
    return {"score": sum(v["score"] for v in per.values()) / len(per), "per_tag": per}


_TEMP = re.compile(r"(\d{2,3})\s*(?:degrees?|°|º)?\s*(f|c)\b", re.I)
_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\b", re.I)


def feasibility(response):
    """Rule checks -> dict(score, checks). Each check scores 0..1."""
    steps = [s for s in response.get("steps", []) if str(s).strip()]
    ings = response.get("ingredients", [])
    text = " ".join(steps).lower()
    checks = {}
    checks["has_steps"] = 1.0 if len(steps) >= 2 else (0.5 if len(steps) == 1 and len(text) > 80 else 0.0)
    # every non-staple ingredient shows up in the steps
    non_staple = [i for i in ings if not is_staple(i)]
    if non_staple:
        seen = sum(1 for i in non_staple if any(t in text for t in name_tokens(i)))
        checks["ingredients_used_in_steps"] = seen / len(non_staple)
    else:
        checks["ingredients_used_in_steps"] = 1.0
    # sane temperatures (oven/pan): 150-550 F or 65-290 C
    bad_t = 0
    temps = _TEMP.findall(text)
    for v, u in temps:
        v = int(v)
        if u.lower() == "f" and not 100 <= v <= 600:
            bad_t += 1
        if u.lower() == "c" and not 40 <= v <= 320:
            bad_t += 1
    checks["sane_temperatures"] = 1.0 if not temps else 1 - bad_t / len(temps)
    # sane times: single step under 12 hours
    times = _TIME.findall(text)
    bad_m = 0
    for v, u in times:
        mins = float(v) * (60 if u.lower().startswith(("h")) else 1)
        if mins > 720 or mins <= 0:
            bad_m += 1
    checks["sane_times"] = 1.0 if not times else 1 - bad_m / len(times)
    # baking/roasting steps need a temperature
    needs_temp = bool(re.search(r"\b(bake|roast|preheat)\b", text))
    checks["oven_temperature_given"] = 1.0 if (not needs_temp or temps) else 0.0
    # ingredient lines carry an amount
    amt = [i for i in ings if not is_staple(i)]
    if amt:
        has = sum(1 for i in amt if re.search(r"\d|¼|½|¾|⅓|⅔|pinch|dash|handful", i.lower()))
        checks["amounts_given"] = has / len(amt)
    else:
        checks["amounts_given"] = 1.0
    return {"score": sum(checks.values()) / len(checks), "checks": {k: round(v, 3) for k, v in checks.items()}}


LLM_JUDGE_PROMPT = """You are judging whether a recipe is cookable by a home cook.
Rate 1-5 (5 = clearly cookable as written). Consider: are steps in a logical order, do cooking times/temperatures make sense,
is any essential step or ingredient missing, are amounts sensible for the stated servings.
Recipe:
{recipe}
Answer with one integer 1-5 and one sentence of reasoning."""


def score_response(request, response, calc=None, llm_feasibility=None, reference_servings=None):
    """Full score. `calc` = recipe_eval.nutrition.Calculator (needed for macro tags).
    `llm_feasibility` optional 1-5 rating from a judge model; averaged into the feasibility score if given.
    `reference_servings`: the golden recipe's servings. Only the `low_calorie` tag depends on servings; when given, the
    model's claimed servings are clamped to 0.67x-1.5x of it so claiming a different serving count cannot game that tag.
    (The other macro tags use calorie shares and ignore servings.)"""
    comp = compliance(request["ingredients"], response.get("ingredients", []))
    macros = None
    macro_requested = any(t in L.MACRO_TAGS for t in request.get("tags", []))
    if macro_requested and calc is not None:
        servings = float(response.get("servings") or 1)
        if reference_servings:
            servings = min(max(servings, 0.67 * reference_servings), 1.5 * reference_servings)
        ps, cov, miss = calc.per_serving(response.get("ingredients", []), servings)
        ps["fiber"] = 0.0
        macros = ps
        macros["coverage"] = cov
    pref = preference(request.get("tags", []), response.get("ingredients", []), macros)
    feas = feasibility(response)
    feas_score = feas["score"]
    if llm_feasibility is not None:
        feas_score = 0.5 * feas_score + 0.5 * (llm_feasibility - 1) / 4
    overall = 0.0 if comp["hard_fail"] else (
        WEIGHTS["compliance"] * comp["score"] + WEIGHTS["preference"] * pref["score"] + WEIGHTS["feasibility"] * feas_score)
    return {"overall": round(overall, 4), "compliance": comp, "preference": pref,
            "feasibility": {**feas, "score": feas_score}, "macros": macros}
