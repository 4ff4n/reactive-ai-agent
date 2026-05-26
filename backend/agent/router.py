"""
Agent Router
────────────
1. Checks the query cache first.
2. Classifies intent: SQL query vs. general/knowledge question.
3. Routes to SQLAgent → on failure falls back to RAGAgent.
4. Stores the result in cache and session memory.
"""
import json
import logging
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncConnection

from backend.agent.rag_agent import get_rag_agent
from backend.agent.sql_agent import get_sql_agent
from backend.cache.query_cache import get_cache
from backend.config import get_settings
from backend.memory.session import get_memory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)
settings = get_settings()

CLASSIFIER_PROMPT = """Classify the following user message into one of two categories:
  SQL   – the user wants to query data, retrieve metrics, numbers, lists, or analytics from a database
  RAG   – the user wants general information, help, explanations, or asks a knowledge question

Reply with ONLY the single word: SQL or RAG

Message: {question}"""


async def classify_intent(question: str) -> str:
    llm = ChatOpenAI(
        model=settings.model_fast,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(CLASSIFIER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"question": question})
    intent = result.strip().upper()
    return "SQL" if "SQL" in intent else "RAG"


async def route_query(
    question: str,
    session_id: str,
    conn: AsyncConnection,
) -> dict[str, Any]:
    """
    Main routing entrypoint.
    Returns a unified response dict:
      {
        "type": "sql" | "rag" | "error",
        "answer": str,          # always present (natural-language reply)
        "sql_query": str,       # SQL type only
        "explanation": str,
        "columns": list,
        "rows": list,
        "row_count": int,
        "from_cache": bool,
      }
    """
    cache = await get_cache()
    memory = await get_memory()

    # ── cache check ───────────────────────────────────────────────────────
    cached = await cache.get(session_id, question)
    if cached:
        try:
            result = json.loads(cached)
            result["from_cache"] = True
            return result
        except Exception:
            pass

    # ── build history ─────────────────────────────────────────────────────
    history = await memory.as_langchain_messages(session_id)

    # ── classify intent ───────────────────────────────────────────────────
    intent = await classify_intent(question)
    logger.info("Intent for session=%s: %s", session_id, intent)

    result: dict[str, Any] = {"from_cache": False}

    if intent == "SQL":
        try:
            sql_agent = await get_sql_agent(conn)
            sql_result = await sql_agent.generate(question, history, conn)

            if "error" in sql_result:
                # SQL failed → fall back to RAG
                logger.warning("SQL failed, falling back to RAG: %s", sql_result["error"])
                rag_agent = get_rag_agent()
                answer = await rag_agent.answer(question, history)
                result.update({
                    "type": "rag",
                    "answer": f"I couldn't generate a valid query. Here's what I know:\n\n{answer}",
                    "fallback_reason": sql_result["error"],
                })
            else:
                rows_summary = _rows_to_text(sql_result)
                result.update({
                    "type": "sql",
                    "answer": f"{sql_result['explanation']}\n\n{rows_summary}",
                    **sql_result,
                })
        except Exception as e:
            logger.exception("SQLAgent crash: %s", e)
            rag_agent = get_rag_agent()
            answer = await rag_agent.answer(question, history)
            result.update({"type": "rag", "answer": answer})

    else:  # RAG
        rag_agent = get_rag_agent()
        answer = await rag_agent.answer(question, history)
        result.update({"type": "rag", "answer": answer})

    # ── cache + memory ────────────────────────────────────────────────────
    await cache.set(session_id, question, json.dumps(result, default=str))
    await memory.add_turn(session_id, question, result["answer"])

    return result


async def route_query_stream(
    question: str,
    session_id: str,
    conn: AsyncConnection,
) -> AsyncIterator[str]:
    """
    Streaming variant — yields text chunks.
    SQL results are emitted as a JSON block first, then explanation streams.
    """
    cache = await get_cache()
    memory = await get_memory()

    cached = await cache.get(session_id, question)
    if cached:
        try:
            data = json.loads(cached)
            yield json.dumps({"event": "cached", "data": data})
            return
        except Exception:
            pass

    history = await memory.as_langchain_messages(session_id)
    intent = await classify_intent(question)

    full_response = ""
    cache_payload: dict = {}   # what gets stored — full SQL result or RAG text

    if intent == "SQL":
        try:
            sql_agent = await get_sql_agent(conn)
            sql_result = await sql_agent.generate(question, history, conn)

            if "error" not in sql_result:
                # Emit structured SQL result
                yield json.dumps({"event": "sql_result", "data": sql_result}, default=str) + "\n"
                full_response = sql_result.get("explanation", "")
                # Cache the FULL result so replays show the complete table
                cache_payload = {**sql_result, "type": "sql", "from_cache": True}
            else:
                # Fall back to RAG streaming
                rag_agent = get_rag_agent()
                async for chunk in rag_agent.answer_stream(question, history):
                    full_response += chunk
                    yield json.dumps({"event": "token", "data": chunk}) + "\n"
                cache_payload = {"type": "rag", "answer": full_response, "from_cache": True}
        except Exception as e:
            logger.exception("Stream SQL error: %s", e)
            yield json.dumps({"event": "error", "data": str(e)}) + "\n"
            return
    else:
        rag_agent = get_rag_agent()
        async for chunk in rag_agent.answer_stream(question, history):
            full_response += chunk
            yield json.dumps({"event": "token", "data": chunk}) + "\n"
        cache_payload = {"type": "rag", "answer": full_response, "from_cache": True}

    yield json.dumps({"event": "done"}) + "\n"

    if full_response:
        await cache.set(session_id, question, json.dumps(cache_payload, default=str))
        await memory.add_turn(session_id, question, full_response)


def _rows_to_text(sql_result: dict) -> str:
    rows = sql_result.get("rows", [])
    if not rows:
        return "_No rows returned._"
    lines = []
    for row in rows[:10]:
        lines.append("  " + ", ".join(f"{k}: {v}" for k, v in row.items()))
    suffix = f"\n  … ({sql_result['row_count']} rows total)" if sql_result["row_count"] > 10 else ""
    return "**Results:**\n" + "\n".join(lines) + suffix
