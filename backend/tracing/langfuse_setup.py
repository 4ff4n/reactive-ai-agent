"""
LangFuse Observability — compatible with langfuse v2.x and v4.x
In v4, langfuse reads credentials from environment variables automatically.
We set them explicitly before creating the handler to be safe.
"""
import logging
import os
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class _NoOpHandler:
    pass


def _set_langfuse_env():
    """Ensure langfuse env vars are set so v4 can auto-configure."""
    if settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    if settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    if settings.langfuse_host:
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)


def get_langfuse_handler(
    session_id: str,
    question: str,
    trace_name: str = "agent-query",
) -> object:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return _NoOpHandler()

    _set_langfuse_env()

    try:
        # langfuse v4.x
        from langfuse.langchain import CallbackHandler
    except ImportError:
        try:
            # langfuse v2.x fallback
            from langfuse.callback import CallbackHandler
        except ImportError:
            logger.warning("langfuse LangChain handler not found — tracing disabled")
            return _NoOpHandler()

    try:
        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            session_id=session_id,
            trace_name=trace_name,
            metadata={"question": question[:200], "app": "reactive-ai-agent"},
        )
        return handler
    except Exception as e:
        logger.warning("LangFuse handler init failed: %s", e)
        return _NoOpHandler()


def build_callbacks(handler) -> list:
    if isinstance(handler, _NoOpHandler):
        return []
    return [handler]
