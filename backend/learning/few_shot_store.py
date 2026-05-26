"""
Auto Few-Shot Learning Store
─────────────────────────────
Every successful SQL query is stored as a (question, sql, explanation) example.
On each new query, the top-K most semantically similar past examples are
retrieved and injected into the system prompt as dynamic few-shot context.

The system gets smarter with every use.

Storage: JSON file at data/few_shot_examples.json
In-memory: FAISS-style cosine similarity over OpenAI embeddings
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── in-memory state ──────────────────────────────────────────────────────────
_vectors: Optional[np.ndarray] = None
_examples: list[dict] = []   # [{question, sql_query, explanation}]
_embeddings_client = None

STORE_PATH = Path(settings.few_shot_store_path)
MAX_EXAMPLES = 200


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
        vec /= np.linalg.norm(vec) + 1e-10
        return vec
    except Exception as e:
        logger.warning("Few-shot embed failed: %s", e)
        return None


# ── public API ────────────────────────────────────────────────────────────────

async def retrieve_similar(question: str) -> list[dict]:
    """
    Return the top-K most similar past examples for injection into the prompt.
    Returns [] if store is empty or embeddings fail.
    """
    if not _examples:
        return []
    vec = await _embed(question)
    if vec is None or _vectors is None:
        return []
    sims = _vectors @ vec
    k = min(settings.few_shot_top_k, len(_examples))
    top_indices = np.argsort(sims)[::-1][:k]
    return [_examples[i] for i in top_indices]


async def add_example(question: str, sql_query: str, explanation: str) -> None:
    """
    Store a successful query as a new few-shot example.
    Deduplicates by SQL query to avoid redundant entries.
    """
    global _vectors, _examples

    # Deduplicate
    if any(e["sql_query"] == sql_query for e in _examples):
        return

    vec = await _embed(question)
    if vec is None:
        return

    example = {"question": question, "sql_query": sql_query, "explanation": explanation}
    _examples.append(example)

    if _vectors is None:
        _vectors = vec.reshape(1, -1)
    else:
        _vectors = np.vstack([_vectors, vec.reshape(1, -1)])

    # Prune
    if len(_examples) > MAX_EXAMPLES:
        _examples = _examples[-MAX_EXAMPLES:]
        _vectors = _vectors[-MAX_EXAMPLES:]

    _persist()
    logger.info("Few-shot store: added example (%d total)", len(_examples))


def _persist() -> None:
    """Save examples (without vectors) to JSON file."""
    try:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STORE_PATH.write_text(json.dumps(_examples, indent=2))
    except Exception as e:
        logger.warning("Few-shot persist failed: %s", e)


async def load() -> None:
    """On startup: load examples and rebuild vectors."""
    global _vectors, _examples
    if not STORE_PATH.exists():
        logger.info("Few-shot store: no saved examples yet")
        return
    try:
        stored = json.loads(STORE_PATH.read_text())
        logger.info("Rebuilding few-shot index from %d examples…", len(stored))
        vecs, valid = [], []
        for ex in stored:
            vec = await _embed(ex["question"])
            if vec is not None:
                vecs.append(vec)
                valid.append(ex)
        if vecs:
            _vectors = np.vstack(vecs)
            _examples = valid
            logger.info("Few-shot store ready: %d examples", len(_examples))
    except Exception as e:
        logger.warning("Few-shot load failed: %s", e)


def format_for_prompt(examples: list[dict]) -> str:
    """Render retrieved examples as text to inject into the system prompt."""
    if not examples:
        return ""
    lines = ["\nDynamic examples from past successful queries:"]
    for ex in examples:
        lines.append(f"Q: {ex['question']}")
        lines.append(f"SQL: {ex['sql_query']}")
        lines.append(f"Explanation: {ex['explanation']}\n")
    return "\n".join(lines)
