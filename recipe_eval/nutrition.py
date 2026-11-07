"""Compute per-recipe macros (kcal, protein, fat, carbs) from an ingredient list using USDA SR Legacy.

Matching order for each ingredient name:
  1. hand-written overrides (recipe_eval/overrides.py)
  2. cached model-assisted matches (data/ingredient_map.json)
  3. word-overlap fallback (core.best_food)

Usage:
    from recipe_eval import nutrition
    calc = nutrition.Calculator()
    totals, coverage, missed = calc.macros(["2 cups all-purpose flour", "1 egg"])

`coverage` is the fraction of ingredient lines that could be resolved to grams of a USDA food.
Treat results as roughly +/-15-20% (median); use tolerance bands, not exact comparisons.
"""
import json
import re
from pathlib import Path

import numpy as np

from . import core as M
from .overrides import O

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = ROOT / "data" / "ingredient_map.json"
SKIP = re.compile(
    r"to taste|for garnish|as needed|for (deep |pan |shallow )?frying|for deep-frying|"
    r"for greasing|for the pan|for dusting|for coating"
)


def key_of(name):
    return " ".join(w for w in re.sub(r"[^a-z ]", " ", name).split() if w not in M.STOP)


class Calculator:
    def __init__(self):
        self.F, self.P = M.load()
        idx = {int(f): i for i, f in enumerate(self.F.fdc_id)}
        res = {}
        if MAP_PATH.exists():
            for name, e in json.load(open(MAP_PATH, encoding="utf8")).items():
                if e["id"] in idx:
                    res[name] = (idx[e["id"]], e.get("unit_g"))
        for name, (rx, ug) in O.items():  # overrides win over cached matches
            if rx == "ZERO":
                res[name] = ("ZERO", None)
                continue
            m = self.F.description.str.contains(rx, regex=True)
            if m.any():
                res[name] = (int(np.argmax(m.values)), ug)
        self.res = res

    def macros(self, lines):
        F, P, res = self.F, self.P, self.res
        tot = dict(kcal=0.0, protein=0.0, fat=0.0, carb=0.0)
        ok = 0
        miss = []
        lines = [l for l in lines if not SKIP.search(l.lower())]
        for ln in lines:
            qty, unit, name, hint = M.parse(re.sub(r"\b(skinless|boneless)\s*,\s*", "", ln.lower()))
            k = key_of(name)
            r = res.get(k) or res.get(re.sub(r"s$", "", k))
            if r is None:
                bf = M.best_food(name, F)
                if bf:
                    r = (bf[0], None)
            if r is None:
                miss.append(ln)
                continue
            if r[0] == "ZERO":
                ok += 1
                continue
            i, ug = r
            fid = F.fdc_id.iloc[i]
            if hint:
                g = qty
            elif unit in M.MASS:
                g = (qty or 1.0) * M.MASS[unit]
            elif unit is None and ug:
                g = (qty or 1.0) * float(ug)
            else:
                g = M.grams(qty, unit, name, hint, fid, P, F)
            if g is None or not np.isfinite(g) or g > 20000:
                miss.append(ln)
                continue
            row = F.iloc[i]
            for kk in tot:
                tot[kk] += row[kk] * g / 100.0
            ok += 1
        return tot, ok / max(len(lines), 1), miss

    def per_serving(self, lines, servings):
        tot, cov, miss = self.macros(lines)
        s = max(float(servings), 1e-9)
        return {k: v / s for k, v in tot.items()}, cov, miss
