"""
Per-session conversation memory stored in Redis.
Falls back to an in-process dict when Redis is unavailable.
Implements:
  - Rolling window of last N message pairs
  - TTL-based expiry (default 30 min inactivity)
  - Summarisation placeholder (hook for ConversationSummaryMemory)
"""
import json
import logging
from typing import Optional

import redis.asyncio as aioredis

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# In-process fallback store: session_id → list[dict]
_local_store: dict[str, list[dict]] = {}


class SessionMemory:
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self._redis = redis_client
        self._window = settings.memory_window_size
        self._ttl = settings.session_ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:history"

    # ── read ─────────────────────────────────────────────────────────────────

    async def get_history(self, session_id: str) -> list[dict]:
        """Return the rolling window of message dicts [{role, content}, ...]."""
        try:
            if self._redis:
                raw = await self._redis.get(self._key(session_id))
                if raw:
                    return json.loads(raw)[-self._window * 2:]
                return []
        except Exception as e:
            logger.warning("Redis read failed, using local store: %s", e)
        return _local_store.get(session_id, [])[-self._window * 2:]

    # ── write ────────────────────────────────────────────────────────────────

    async def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Append a user/assistant pair and refresh TTL."""
        history = await self.get_history(session_id)
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        # Keep only the rolling window
        history = history[-self._window * 2:]
        try:
            if self._redis:
                await self._redis.set(
                    self._key(session_id),
                    json.dumps(history),
                    ex=self._ttl,
                )
                return
        except Exception as e:
            logger.warning("Redis write failed, using local store: %s", e)
        _local_store[session_id] = history

    async def clear(self, session_id: str) -> None:
        try:
            if self._redis:
                await self._redis.delete(self._key(session_id))
        except Exception:
            pass
        _local_store.pop(session_id, None)

    # ── LangChain-compatible message list ────────────────────────────────────

    async def as_langchain_messages(self, session_id: str):
        """Return history as LangChain HumanMessage/AIMessage list."""
        from langchain_core.messages import HumanMessage, AIMessage
        history = await self.get_history(session_id)
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages


# ── singleton factory ─────────────────────────────────────────────────────────

_memory_instance: Optional[SessionMemory] = None


async def get_memory() -> SessionMemory:
    global _memory_instance
    if _memory_instance is None:
        try:
            redis_client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await redis_client.ping()
            logger.info("Session memory: using Redis at %s", settings.redis_url)
            _memory_instance = SessionMemory(redis_client)
        except Exception as e:
            logger.warning("Redis unavailable (%s); using in-process memory.", e)
            _memory_instance = SessionMemory(redis_client=None)
    return _memory_instance
