"""Settings and helpers shared by the indexing and search code."""
import uuid
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384 dims, 512-token limit, MIT
EMBED_DIM = 384
DENSE = "dense"    # name of the dense vector in the collection
SPARSE = "bm25"    # name of the sparse vector (declared now, filled in when BM25 is added)
MODEL_CACHE = ROOT / ".cache" / "fastembed"   # keeps the model download out of the temp folder

# Fixed namespace so the same recipe_id always maps to the same Qdrant point ID (re-indexing overwrites, never duplicates).
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "nibble-agent/recipes")


def point_id(recipe_id):
    """'r00042' -> a stable UUID string, used as the Qdrant point ID."""
    return str(uuid.uuid5(_ID_NAMESPACE, recipe_id))


def settings():
    """Qdrant URL and collection name from config.yaml (falls back to the defaults if the file or key is missing)."""
    url, collection = "http://localhost:6333", "recipes"
    try:
        import yaml
        q = (yaml.safe_load(CONFIG_PATH.read_text(encoding="utf8")) or {}).get("qdrant", {})
        url, collection = q.get("url", url), q.get("collection", collection)
    except (ImportError, OSError):
        pass
    return {"url": url, "collection": collection}


def get_client(url=None):
    """Qdrant client. Pass ':memory:' for a throwaway in-process database (tests, no Docker needed)."""
    from qdrant_client import QdrantClient
    url = url or settings()["url"]
    return QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(url=url, timeout=60)


@lru_cache(maxsize=1)
def get_model():
    """The embedding model, loaded once per process (downloads about 70 MB on first use)."""
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_CACHE))
