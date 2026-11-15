"""Search the recipe index: dense similarity search with hard filters (allergens, recipes already shown, test set).

From code (this is what the MCP search_recipes tool will call):
    from rag.search import search
    hits = search("spicy chicken dinner", exclude_allergens=["milk"], k=5)

From the command line (project root, Qdrant running):
    python -m rag.search "spicy chicken dinner" --no milk peanut -k 5
    python -m rag.search --check        # allergen filter test: no excluded allergen may ever come back
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
    ap.add_argument("--check", action="store_true", help="run the allergen filter test")
    args = ap.parse_args()
    if args.check:
        sys.exit(0 if check_allergen_filter() else 1)
    if not args.query:
        ap.error("give a query, or --check")
    _print_hits(search(args.query, k=args.k, exclude_allergens=args.no, exclude_eval_set=args.exclude_eval))


if __name__ == "__main__":
    main()
