# rag

Qdrant indexing, retrieval and the RAG experiments (section 3).

Owner: Mihai

## How to run

All commands run from the project root, in the project's virtual environment.

**1. Install the packages** (`qdrant-client` and `fastembed` are in `requirements.txt`):

```
pip install -r requirements.txt
```

**2. Start Qdrant** (needs Docker Desktop running):

```
docker compose up -d
```

Qdrant then runs at http://localhost:6333, with a dashboard at http://localhost:6333/dashboard. Its data is kept in
`qdrant_storage/` (not in git), so the index survives restarts and the container starts again with Docker.
`docker compose down` stops it without deleting the data.

**3. Get the data.** The full pool, `data/recipes_merged.jsonl`, is not in git (see `DATASET_SUMMARY.md`); get it from
the dataset owner and put it in `data/`. The 1,200 test recipes in `data/eval_examples.jsonl` are in git and are enough
to try the pipeline.

**4. Build the index.** The first run downloads the embedding model (about 70 MB, into `.cache/`, not in git).

```
python -m rag.build_index --source eval --recreate   # 1,200 test recipes, a few minutes
python -m rag.build_index --source pool --recreate   # all 23,395 recipes, about 40 minutes on a laptop CPU
```

`--recreate` deletes the collection first. Without it, recipes are overwritten in place (point IDs come from
`recipe_id`), but recipes that are no longer in the data stay in the index. Use `--recreate` when switching between
`eval` and `pool`. Other options: `--limit N` (first N recipes only), `--url :memory:` (throwaway in-process test, no
Docker needed).

**5. Search:**

```
python -m rag.search "spicy chicken dinner" --no milk peanut -k 5
python -m rag.search "pasta" --exclude-eval     # leave out the 1,200 test recipes (use this during evaluation)
```

`--no` takes allergens in plain words ("dairy", "nuts" and "peanuts" work). An allergen the data doesn't tag raises an
error instead of being ignored. The tagged allergens are milk, egg, fish, shellfish, tree_nut, peanut, soy, sesame,
wheat and gluten.

**6. Check the allergen filter** (after every rebuild):

```
python -m rag.search --check
```

It checks every recipe in the index: stored allergens must match the recipe's own text, and for each allergen (and
some common combinations) the recipes that pass the filter must be exactly the safe ones. It exits with code 1 on any
failure.

**From code** (this is what the `search_recipes` MCP tool will call):

```python
from rag.search import search

hits = search("spicy chicken dinner", exclude_allergens=["milk"], exclude_recipe_ids=["r00042"], k=5)
for h in hits:
    print(h["title"], h["score"], h["allergens"])
```

Each hit is the full stored recipe (`recipe_id`, `title`, `ingredients`, `steps`, `servings`, nutrition, `tags`,
`allergens`, `source`, ...) plus its `score`.

## Files

| File | What it does |
| --- | --- |
| `common.py` | Settings (Qdrant URL and collection from `config.yaml`), embedding model, point IDs |
| `load.py` | Reads the pool or the test set into one shape; adds tags, allergens and the test-set flag; builds the text that is embedded |
| `build_index.py` | Creates the collection and payload indexes, embeds every recipe, uploads |
| `search.py` | Search with hard filters (allergens, recipes already shown, test set) and the allergen filter checks |

## Reports

- [Week 7: recipe index in Qdrant](docs/week7-rag-index.md): setup, data, allergen filter results, open issues.
