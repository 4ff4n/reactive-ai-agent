"""
Semantic-ish query cache keyed on (session_id, prompt_hash).
Uses Redis with TTL. Falls back to an in-process dict.
"""
import hashlib
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_local_cache: dict[str, str] = {}
CACHE_TTL = 3600  # 1 hour for query results


def _make_key(session_id: str, prompt: str) -> str:
    digest = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:16]
    return f"cache:{session_id}:{digest}"


class QueryCache:
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self._redis = redis_client

    async def get(self, session_id: str, prompt: str) -> Optional[str]:
        key = _make_key(session_id, prompt)
        try:
            if self._redis:
                val = await self._redis.get(key)
                if val:
                    logger.debug("Cache HIT for key %s", key)
                    return val
        except Exception:
            pass
        return _local_cache.get(key)

    async def set(self, session_id: str, prompt: str, response: str) -> None:
        key = _make_key(session_id, prompt)
        try:
            if self._redis:
                await self._redis.set(key, response, ex=CACHE_TTL)
                return
        except Exception:
            pass
        _local_cache[key] = response

    async def invalidate(self, session_id: str, prompt: str) -> None:
        key = _make_key(session_id, prompt)
        try:
            if self._redis:
                await self._redis.delete(key)
        except Exception:
            pass
        _local_cache.pop(key, None)


_cache_instance: Optional[QueryCache] = None


async def get_cache() -> QueryCache:
    global _cache_instance
    if _cache_instance is None:
        try:
            redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await redis_client.ping()
            _cache_instance = QueryCache(redis_client)
        except Exception:
            _cache_instance = QueryCache(redis_client=None)
    return _cache_instance
