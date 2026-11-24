"""Search the recipe index: dense similarity search with hard filters (allergens, recipes already shown, test set).

From code (this is what the MCP search_recipes tool will call):
    from rag.search import search
    hits = search("spicy chicken dinner", exclude_allergens=["milk"], k=5)

From the command line (project root, Qdrant running):
    python -m rag.search "spicy chicken dinner" --no milk peanut -k 5
    python -m rag.search --check        # allergen filter test over every recipe in the index, then sample queries
"""
import argparse
import sys

from qdrant_client import models

from .common import DENSE, get_client, get_model, point_id, settings

# bge-small-en-v1.5 recommends this prefix for short queries against longer passages (recipes get no prefix).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

ALLERGENS = ("milk", "egg", "fish", "shellfish", "tree_nut", "peanut", "soy", "sesame", "wheat", "gluten")
ALIASES = {
    "dairy": "milk", "lactose": "milk", "eggs": "egg", "seafood": "shellfish", "crustacean": "shellfish",
    "crustaceans": "shellfish", "mollusc": "shellfish", "molluscs": "shellfish", "nut": "tree_nut", "nuts": "tree_nut",
    "tree nut": "tree_nut", "tree nuts": "tree_nut", "peanuts": "peanut", "soya": "soy", "sesame seeds": "sesame",
}


def normalize_allergens(names):
    """User words -> the dataset's allergen names ('Dairy' -> 'milk'). Unknown names raise instead of being ignored,
    because silently dropping an allergen would let it through."""
    out = []
    for n in names:
        key = n.strip().lower().replace("-", " ")
        key = ALIASES.get(key, key.replace(" ", "_"))
        if key not in ALLERGENS:
            raise ValueError(f"unknown allergen {n!r}; known: {', '.join(ALLERGENS)}")
        out.append(key)
    return sorted(set(out))


def build_filter(exclude_allergens=(), exclude_recipe_ids=(), exclude_eval_set=False):
    """Hard filters, applied inside the vector search (never relaxed)."""
    must, must_not = [], []
    allergens = normalize_allergens(exclude_allergens)
    if allergens:
        must_not.append(models.FieldCondition(key="allergens", match=models.MatchAny(any=allergens)))
        must.append(models.FieldCondition(key="allergens_known", match=models.MatchValue(value=True)))
    if exclude_recipe_ids:
        must_not.append(models.HasIdCondition(has_id=[point_id(i) for i in exclude_recipe_ids]))
    if exclude_eval_set:
        must_not.append(models.FieldCondition(key="eval_holdout", match=models.MatchValue(value=True)))
    if not (must or must_not):
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def search(query, k=5, exclude_allergens=(), exclude_recipe_ids=(), exclude_eval_set=False,
           client=None, collection=None):
    """Top-k recipes for a free-text query. Returns dicts: score plus the stored recipe payload."""
    client = client or get_client()
    collection = collection or settings()["collection"]
    vector = next(iter(get_model().embed([QUERY_PREFIX + query]))).tolist()
    res = client.query_points(
        collection_name=collection, query=vector, using=DENSE, limit=k, with_payload=True,
        query_filter=build_filter(exclude_allergens, exclude_recipe_ids, exclude_eval_set))
    return [{"score": p.score, **p.payload} for p in res.points]


# Queries that naturally pull up recipes with each allergen, so the filter has something to remove.
CHECK_QUERIES = {
    "milk": ["creamy cheese pasta", "butter cake"],
    "egg": ["fluffy omelette", "chocolate chip cookies"],
    "fish": ["baked salmon fillet", "tuna salad"],
    "shellfish": ["garlic shrimp pasta", "crab cakes"],
    "tree_nut": ["walnut brownies", "almond biscotti"],
    "peanut": ["peanut butter cookies", "thai peanut noodles"],
    "soy": ["stir fry with soy sauce", "tofu curry"],
    "sesame": ["sesame noodles", "hummus with tahini"],
    "wheat": ["homemade bread", "spaghetti bolognese"],
    "gluten": ["beer battered fish", "flour tortillas"],
}


def check_allergen_filter(k=20, client=None, collection=None):
    """For each allergen, search without and with the filter. 'leaks' must be 0 everywhere. Returns True if it passes."""
    print(f"{'allergen':<10} {'query':<26} {'unfiltered hits with it':>24} {'filtered leaks':>15}")
    ok = True
    for allergen, queries in CHECK_QUERIES.items():
        for q in queries:
            plain = search(q, k=k, client=client, collection=collection)
            safe = search(q, k=k, exclude_allergens=[allergen], client=client, collection=collection)
            with_it = sum(allergen in h["allergens"] for h in plain)
            leaks = sum(allergen in h["allergens"] for h in safe)
            ok &= leaks == 0
            print(f"{allergen:<10} {q:<26} {with_it:>21}/{len(plain):<2} {leaks:>15}{'' if leaks == 0 else '  <-- LEAK'}")
    print("PASS: no excluded allergen came back" if ok else "FAIL: an excluded allergen came back")
    return ok


# Typical multi-allergen profiles, checked on top of every single allergen.
CHECK_COMBOS = (("milk", "egg"), ("milk", "egg", "peanut", "tree_nut"), ("gluten", "milk"), ("fish", "shellfish", "sesame", "soy"))


def _scroll_ids(client, collection, flt=None):
    """IDs of every point that passes a filter (the whole collection, not just a top k)."""
    ids, offset = set(), None
    while True:
        points, offset = client.scroll(collection, scroll_filter=flt, limit=2000, offset=offset,
                                       with_payload=False, with_vectors=False)
        ids.update(str(p.id) for p in points)
        if offset is None:
            return ids


def check_allergen_filter_exhaustive(client=None, collection=None, combos=CHECK_COMBOS):
    """Proof over the whole index, not a sample of top-k results. Two checks:
      1. Every point's stored allergens equal the allergens recomputed from its own stored ingredients and steps.
      2. For each allergen (and each combo), the set of points that pass the filter is exactly the set of
         recipes whose recomputed allergens do not include it: no leaks, and nothing safe wrongly removed.
    Returns True if both pass."""
    from recipe_eval import allergens as A
    client = client or get_client()
    collection = collection or settings()["collection"]

    truth, rid, wrong, offset = {}, {}, [], None
    while True:
        points, offset = client.scroll(collection, limit=1000, offset=offset, with_vectors=False,
                                       with_payload=["recipe_id", "ingredients", "steps", "allergens", "allergens_known"])
        for p in points:
            pid = str(p.id)
            computed = A.recipe_allergens(p.payload["ingredients"], p.payload["steps"])["allergens"]
            truth[pid], rid[pid] = set(computed), p.payload["recipe_id"]
            if sorted(p.payload["allergens"]) != computed or not p.payload.get("allergens_known"):
                wrong.append(p.payload["recipe_id"])
        if offset is None:
            break
    n = len(truth)
    print(f"1. stored allergens vs recomputed from each recipe's own text: {n - len(wrong)}/{n} agree"
          + (f"  <-- WRONG: {', '.join(wrong[:5])}" if wrong else ""))

    print(f"2. {'excluded':<36} {'pass filter':>12} {'expected':>9} {'leaks':>6} {'wrongly removed':>16}")
    ok = not wrong
    for group in [(a,) for a in ALLERGENS] + list(combos):
        got = _scroll_ids(client, collection, build_filter(exclude_allergens=group))
        expected = {pid for pid, al in truth.items() if not al & set(group)}
        leaks, removed = got - expected, expected - got
        ok &= not leaks and not removed
        flag = "" if not (leaks or removed) else f"  <-- e.g. {', '.join(rid[p] for p in list(leaks or removed)[:3])}"
        print(f"   {', '.join(group):<36} {len(got):>12} {len(expected):>9} {len(leaks):>6} {len(removed):>16}{flag}")
    print(f"PASS: the filter is exact on all {n} recipes" if ok else "FAIL: see the rows marked above")
    return ok


def _print_hits(hits):
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h['title']}  [{h['recipe_id']}, {h['source']}, score {h['score']:.3f}]")
        print(f"   allergens: {', '.join(h['allergens']) or 'none'}   tags: {', '.join(h['tags']) or 'none'}")
        print(f"   ingredients: {', '.join(h['ingredient_names'][:10])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no", nargs="+", default=[], metavar="ALLERGEN", help="allergens to exclude, e.g. --no milk peanut")
    ap.add_argument("--exclude-eval", action="store_true", help="leave out the 1,200 test-set recipes")
    ap.add_argument("--check", action="store_true", help="run the allergen filter tests (whole index, then sample queries)")
    args = ap.parse_args()
    if args.check:
        exact = check_allergen_filter_exhaustive()
        print()
        sample = check_allergen_filter()
        sys.exit(0 if exact and sample else 1)
    if not args.query:
        ap.error("give a query, or --check")
    _print_hits(search(args.query, k=args.k, exclude_allergens=args.no, exclude_eval_set=args.exclude_eval))


if __name__ == "__main__":
    main()
