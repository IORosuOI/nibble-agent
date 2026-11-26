"""Find allergens in ingredient lines or recipe text, using data/allergen_map.json.

    allergens_in_line("2 tbsp soy sauce")           -> {"soy", "wheat", "gluten"}
    allergens_in_lines(["1 cup flour", "1 egg"])    -> {"wheat", "gluten", "egg"}
    recipe_allergens(ingredient_lines, step_lines)  -> {"allergens": [...], "steps_only": [...]}

The list is hand-written and approximate. It errs on the side of flagging. It cannot see hidden ingredients
(lecithin, a store-bought sauce) or cross-contact.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "allergen_map.json"


def _norm(text):
    t = text.lower().replace("-", " ").replace("&", " and ")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    return " " + re.sub(r"\s+", " ", t).strip() + " "


def _pattern(phrase):
    # whole words; plural s/es allowed on the last word
    return re.compile(r"(?<![a-z0-9])" + re.escape(_norm(phrase).strip()) + r"(?:s|es)?(?![a-z0-9])")


@lru_cache(maxsize=1)
def _load():
    m = json.load(open(MAP_PATH, encoding="utf8"))
    return {
        "implied": m["implied"],
        "compound": [(_pattern(k), set(v)) for k, v in m["compound_ingredients"].items()],
        "keywords": [(a, _pattern(k)) for a, ks in m["keywords"].items() for k in ks],
        "not": [(_pattern(k), set(v)) for k, v in m["not_allergen"].items()],
        "free": [(_pattern(k), set(v)) for k, v in m["free_from_phrases"].items()],
        "names": list(m["allergens"].keys()),
    }


def allergens_in_line(line):
    """Set of allergen names found in one ingredient line or one sentence."""
    d = _load()
    t = _norm(line)
    found = set()
    # spans that do NOT carry a given allergen (for example "peanut butter" is not dairy)
    blocked = {}
    for pat, alls in d["not"]:
        for mt in pat.finditer(t):
            for a in alls:
                blocked.setdefault(a, []).append(mt.span())
    for pat, alls in d["compound"]:
        if pat.search(t):
            found |= alls
    for a, pat in d["keywords"]:
        for mt in pat.finditer(t):
            s, e = mt.span()
            if any(bs <= s and e <= be for bs, be in blocked.get(a, [])):
                continue
            found.add(a)
            break
    for pat, alls in d["free"]:        # "gluten free flour" -> no wheat/gluten from this line
        if pat.search(t):
            found -= alls
    for a, implied in d["implied"].items():
        if a in found:
            found |= set(implied)
    return found


def allergens_in_lines(lines):
    out = set()
    for ln in lines:
        out |= allergens_in_line(str(ln))
    return out


def recipe_allergens(ingredient_lines, step_lines=()):
    """allergens: everything found in the ingredient list or the steps.
    steps_only: found in the steps but not in the ingredient list (a sign the ingredient list is incomplete)."""
    listed = allergens_in_lines(ingredient_lines)
    in_steps = set()
    for s in step_lines:
        for sentence in re.split(r"(?<=[.!?])\s+", str(s)):
            in_steps |= allergens_in_line(sentence)
    return {"allergens": sorted(listed | in_steps), "steps_only": sorted(in_steps - listed)}


FAMILIES_PATH = MAP_PATH.with_name("allergen_families.json")


@lru_cache(maxsize=1)
def _aliases():
    fam = json.load(open(FAMILIES_PATH, encoding="utf8"))["families"]
    out = {}
    for code, info in fam.items():
        out[code] = code
        for a in [info["eu_name"]] + info["aliases"]:
            out.setdefault(a.lower(), code)
    out.setdefault("shellfish", "shellfish")
    out.setdefault("wheat", "wheat")
    return out


def normalize_allergen(name):
    """'Peanuts' / 'dairy' / 'tree nuts' -> the code used in the data ('peanut', 'milk', 'tree_nut'). Unknown -> None."""
    return _aliases().get(str(name).strip().lower().replace("_", " ")) or _aliases().get(str(name).strip().lower())


def normalize_allergens(names):
    """List of profile allergens -> sorted list of codes. Raises ValueError for a name we do not know."""
    out = set()
    for n in names:
        code = normalize_allergen(n)
        if code is None:
            raise ValueError(f"unknown allergen {n!r}; see data/allergen_families.json")
        out.add(code)
    return sorted(out)


def leaks(profile_allergens, ingredient_lines, step_lines=()):
    """Allergens from the profile that appear in the recipe's ingredients or steps (sorted list; [] means safe)."""
    wanted = set(normalize_allergens(profile_allergens))
    found = set(recipe_allergens(ingredient_lines, step_lines)["allergens"])
    if "shellfish" in wanted:
        wanted |= {"crustaceans", "molluscs"}
    return sorted(wanted & found)
