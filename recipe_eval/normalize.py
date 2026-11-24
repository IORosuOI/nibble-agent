"""Turn ingredient lines into plain, comparable ingredient names (the `ingredients_norm` field of the pool).

    norm_names("2 cloves garlic, peeled and lightly crushed")  -> ["garlic"]
    norm_names("1 cup shredded Cheddar cheese")                -> ["cheddar cheese"]
    norm_names("2 tablespoons fresh basil, diced")             -> ["basil"]
    norm_names("1/2 pound morel mushrooms, chopped")           -> ["morel mushroom"]
    norm_names("salt and ground black pepper to taste")        -> ["salt", "black pepper"]
    ingredients_norm(lines)                                    -> ["garlic", "cheddar cheese", ...] (no duplicates, in order)

Amounts, units, packaging and preparation words are removed, only the first of "A or B" is kept, and every word is
singular, so "mushrooms" and "mushroom" compare equal. Staples (salt, oil...) are KEPT: filter them out with
`recipe_eval.scoring.is_staple` if you do not want them.
"""
import re

from . import core as M
from . import scoring as S

# words that look plural but are not
_KEEP = {"hummus", "couscous", "asparagus", "molasses", "grits", "swiss", "citrus", "lemongrass", "watercress", "bass",
         "brussels", "cress", "tahini", "harissa", "quinoa", "pasta", "chorizo", "teriyaki", "sriracha", "gnocchi"}
_NOISE = re.compile(r"\b(to taste|as needed|for garnish|for serving|for frying|optional|divided|plus more|or more|"
                    r"about|approximately|each|piece|pieces)\b")
_OR = re.compile(r"\bor\b")
_CUBE = re.compile(r"\bcubes?\b")
_SALT_PEPPER = re.compile(r"^\s*salt( and)?\s.*\bpepper\b")


def _sing(word):
    if word in _KEEP or word.endswith(("us", "is", "ss")):
        return word
    return S._singular(word)


def norm_names(line):
    """One ingredient line -> list of plain singular names (usually one; 'salt and pepper' gives two)."""
    name = S.clean_name(str(line))
    name = _NOISE.sub(" ", name)
    name = _OR.split(name)[0]                       # "butter or margarine" -> "butter"
    name = _CUBE.sub(" ", name)                     # "9 cubes chicken bouillon"
    if _SALT_PEPPER.match(name):                    # "salt and ground black pepper"
        return ["salt", "black pepper"]
    words = [w for w in name.split() if w not in M.STOP and len(w) > 1]
    if not words:                                   # "half and half" is all stop words: keep it as written
        left = [w for w in name.split() if len(w) > 1 and w != "and"]
        return [" ".join(left)] if left == ["half", "half"] else []
    return [" ".join(_sing(w) for w in words).strip()]


def norm_name(line):
    """First name of a line, or '' if nothing is left."""
    names = norm_names(line)
    return names[0] if names else ""


def ingredients_norm(lines):
    """Ingredient lines -> list of plain names, no duplicates, in recipe order."""
    out, seen = [], set()
    for ln in lines:
        for n in norm_names(ln):
            if n and n not in seen:
                seen.add(n)
                out.append(n)
    return out
