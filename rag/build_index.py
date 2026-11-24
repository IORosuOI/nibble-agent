"""Build the Qdrant recipe index: create the collection and payload indexes, embed every recipe, upload.

Run from the project root (Qdrant must be running: docker compose up -d):
    python -m rag.build_index --source eval              # 1,200 golden recipes, to test the pipeline
    python -m rag.build_index --source pool --recreate   # the full 23,395-recipe pool
Options: --limit N (first N recipes only), --batch-size N, --url URL (default from config.yaml).

Re-running is safe: point IDs come from recipe_id, so existing recipes are overwritten, not duplicated.
"""
import argparse
import time
from glob import glob

from qdrant_client import models

from .common import DENSE, EMBED_DIM, MODEL_CACHE, SPARSE, get_client, get_model, point_id, settings
from .load import chunk_text, load

# Payload fields that filters use. Qdrant must index them BEFORE the bulk upload so filtered search stays fast.
PAYLOAD_INDEXES = {
    "recipe_id": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "tags": models.PayloadSchemaType.KEYWORD,
    "allergens": models.PayloadSchemaType.KEYWORD,         # hard filter: must_not contain the user's allergens
    "allergens_known": models.PayloadSchemaType.BOOL,      # only recipes whose allergens were checked pass an allergen filter
    "eval_holdout": models.PayloadSchemaType.BOOL,         # test-set recipes, excluded during evaluation
    # "ingredients_norm" (dislike filter) is added once the normalized ingredient field exists.
}
MAX_TOKENS = 512   # bge-small truncates longer chunks


def create_collection(client, name, recreate=False):
    if client.collection_exists(name):
        if not recreate:
            print(f"collection '{name}' exists, updating it (use --recreate to start from empty)")
            return
        client.delete_collection(name)
        print(f"deleted collection '{name}'")
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE: models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)},
        # Declared now so BM25 can be added later without rebuilding the collection.
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field, field_schema=schema)
    print(f"created collection '{name}' with payload indexes: {', '.join(PAYLOAD_INDEXES)}")


def payload(r):
    """Everything the app needs to show and filter a recipe, so search needs no second lookup."""
    return {k: r[k] for k in (
        "recipe_id", "title", "ingredients", "steps", "ingredient_names", "source", "category",
        "nutrition_basis", "servings", "servings_estimated", "nutrition_per_serving", "nutrition_totals",
        "single_step", "tags", "allergens", "allergens_steps_only", "allergens_known", "eval_holdout")}


def token_report(texts):
    """How many chunks are longer than the model's 512-token limit (their end is cut off when embedded)."""
    try:
        from tokenizers import Tokenizer
        path = glob(str(MODEL_CACHE / "**" / "tokenizer.json"), recursive=True)[0]
        tok = Tokenizer.from_file(path)
        tok.no_truncation()
        tok.no_padding()
    except (ImportError, IndexError) as e:
        print(f"token report skipped ({e!r})")
        return
    lengths = sorted(len(enc.ids) for enc in tok.encode_batch(texts))
    over = sum(n > MAX_TOKENS for n in lengths)
    print(f"tokens per chunk: median {lengths[len(lengths) // 2]}, max {lengths[-1]}; "
          f"{over} of {len(lengths)} ({over / len(lengths):.1%}) over {MAX_TOKENS} and truncated")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["eval", "pool"], default="eval")
    ap.add_argument("--recreate", action="store_true", help="delete and rebuild the collection")
    ap.add_argument("--limit", type=int, help="index only the first N recipes")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--url", help="Qdrant URL (default from config.yaml); ':memory:' for a throwaway test")
    args = ap.parse_args()

    name = settings()["collection"]
    t0 = time.perf_counter()
    recipes = load(args.source)[: args.limit]
    # Shortest first: each batch is padded to its longest text, so similar lengths per batch embed about 2x faster.
    recipes.sort(key=lambda r: len(chunk_text(r)))
    texts = [chunk_text(r) for r in recipes]
    print(f"loaded {len(recipes)} recipes from '{args.source}' in {time.perf_counter() - t0:.1f}s")

    client = get_client(args.url)
    create_collection(client, name, args.recreate)
    model = get_model()
    token_report(texts)

    t1 = time.perf_counter()
    for start in range(0, len(recipes), args.batch_size):
        batch = recipes[start: start + args.batch_size]
        vectors = model.embed(texts[start: start + args.batch_size], batch_size=args.batch_size)
        client.upsert(name, wait=True, points=[
            models.PointStruct(id=point_id(r["recipe_id"]), vector={DENSE: v.tolist()}, payload=payload(r))
            for r, v in zip(batch, vectors)])
        done = start + len(batch)
        rate = done / (time.perf_counter() - t1)
        print(f"  {done}/{len(recipes)} indexed, {rate:.0f} recipes/s", end="\r", flush=True)
    secs = time.perf_counter() - t1
    print(f"\nembedded and uploaded {len(recipes)} recipes in {secs:.0f}s ({len(recipes) / max(secs, 1e-9):.0f}/s)")
    print(f"collection '{name}' now holds {client.count(name, exact=True).count} points")
    return client


if __name__ == "__main__":
    main()
