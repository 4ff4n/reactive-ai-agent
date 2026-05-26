"""
Semantic Cache
──────────────
Replaces the exact hash-based cache with embedding similarity.
"Top 5 products by revenue" and "Best selling products" → same cache hit.

Uses FAISS IndexFlatIP with L2-normalised vectors for cosine similarity.
Falls back gracefully if OpenAI embeddings are unavailable.

Storage layout (Redis):
  semantic_cache:entries  →  JSON list of {question, payload, timestamp}
  (vectors kept in-memory; rebuilt from Redis entries on restart)
"""
import json
import logging
import time
from typing import Optional

import numpy as np

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── in-memory state ──────────────────────────────────────────────────────────
_vectors: Optional[np.ndarray] = None   # shape (N, D)
_entries: list[dict] = []               # [{question, payload, timestamp}]
_embeddings_client = None

REDIS_KEY = "semantic_cache:entries"
MAX_ENTRIES = 500


async def _get_embeddings():
    global _embeddings_client
    if _embeddings_client is None:
        from langchain_openai import OpenAIEmbeddings
        _embeddings_client = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
    return _embeddings_client


async def _embed(text: str) -> Optional[np.ndarray]:
    try:
        emb = await (await _get_embeddings()).aembed_query(text)
        vec = np.array(emb, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-10   # normalise for cosine
        return vec
    except Exception as e:
        logger.warning("Embedding failed: %s", e)
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # both normalised → dot = cosine


# ── public API ────────────────────────────────────────────────────────────────

async def semantic_get(question: str) -> Optional[dict]:
    """Return cached payload if a semantically similar question was seen before."""
    if not settings.semantic_cache_enabled or not _entries:
        return None
    vec = await _embed(question)
    if vec is None:
        return None
    global _vectors
    if _vectors is None or len(_vectors) != len(_entries):
        return None
    sims = _vectors @ vec                          # (N,) cosine similarities
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    logger.info(
        "Semantic cache check: best_sim=%.3f threshold=%.2f question='%s' match='%s'",
        best_sim, settings.semantic_cache_threshold,
        question[:50], _entries[best_idx]["question"][:50]
    )
    if best_sim >= settings.semantic_cache_threshold:
        logger.info("Semantic cache HIT (sim=%.3f)", best_sim)
        return _entries[best_idx]["payload"]
    logger.info("Semantic cache MISS (sim=%.3f < %.2f)", best_sim, settings.semantic_cache_threshold)
    return None


async def semantic_set(question: str, payload: dict) -> None:
    """Store a question→payload pair in the semantic cache."""
    if not settings.semantic_cache_enabled:
        return
    vec = await _embed(question)
    if vec is None:
        return
    global _vectors, _entries
    entry = {"question": question, "payload": payload, "timestamp": time.time()}

    if _vectors is None:
        _vectors = vec.reshape(1, -1)
    else:
        _vectors = np.vstack([_vectors, vec.reshape(1, -1)])
    _entries.append(entry)

    # Prune oldest if over limit
    if len(_entries) > MAX_ENTRIES:
        _entries = _entries[-MAX_ENTRIES:]
        _vectors = _vectors[-MAX_ENTRIES:]

    # Persist question+payload to Redis (no vectors — rebuilt on startup)
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        serialisable = [{"question": e["question"], "payload": e["payload"]} for e in _entries]
        await r.set(REDIS_KEY, json.dumps(serialisable, default=str))
    except Exception as e:
        logger.debug("Semantic cache Redis persist failed: %s", e)


async def load_from_redis() -> None:
    """On startup: reload persisted entries and rebuild in-memory vectors."""
    global _vectors, _entries
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        raw = await r.get(REDIS_KEY)
        if not raw:
            return
        stored = json.loads(raw)
        logger.info("Rebuilding semantic cache from %d stored entries…", len(stored))
        vecs = []
        valid_entries = []
        for item in stored:
            vec = await _embed(item["question"])
            if vec is not None:
                vecs.append(vec)
                valid_entries.append({"question": item["question"], "payload": item["payload"], "timestamp": 0})
        if vecs:
            _vectors = np.vstack(vecs)
            _entries = valid_entries
            logger.info("Semantic cache rebuilt with %d entries", len(_entries))
    except Exception as e:
        logger.warning("Semantic cache load failed: %s", e)
