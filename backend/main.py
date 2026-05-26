"""
Reactive AI Agent — FastAPI Application (v2 — LangGraph + Advanced Features)
=============================================================================
WebSocket handler now:
  1. Checks semantic cache (embedding similarity) → serve instantly if hit
  2. Loads session memory
  3. Runs LangGraph state machine (classify → sql/rag → heal → chart → finalize)
  4. Streams TTS audio chunks alongside text
  5. Saves result to semantic cache + session memory
"""
import base64
import json
import logging
import uuid
from contextlib import asynccontextmanager

import sqlalchemy as sa
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.cache import semantic_cache
from backend.config import get_settings
from backend.database.connection import engine, init_db
from backend.graph.agent_graph import get_graph, run_graph_stream
from backend.learning import few_shot_store
from backend.memory.session import get_memory
from backend.voice.stt import transcribe
from backend.voice.tts import synthesise, chunk_synthesise, TTSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Reactive AI Agent v2 (LangGraph)…")
    await init_db()

    # Pre-warm RAG index
    try:
        from backend.agent.rag_agent import get_rag_agent
        await get_rag_agent().ensure_ready()
        logger.info("RAG index ready")
    except Exception as e:
        logger.warning("RAG warm-up failed: %s", e)

    # Load semantic cache from Redis
    try:
        await semantic_cache.load_from_redis()
        logger.info("Semantic cache loaded")
    except Exception as e:
        logger.warning("Semantic cache load failed: %s", e)

    # Load few-shot examples
    try:
        await few_shot_store.load()
    except Exception as e:
        logger.warning("Few-shot store load failed: %s", e)

    # Pre-compile graph
    get_graph()
    logger.info("LangGraph compiled and ready")

    yield
    logger.info("Shutdown complete")


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Reactive AI Agent v2",
    version="2.0.0",
    description="LangGraph-powered NL interface over e-commerce data",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

class TTSRequest(BaseModel):
    text: str
    engine: TTSEngine = TTSEngine.GTTS
    language: str = "en"


# ── WebSocket streaming chat ──────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("WS connected: session=%s", session_id)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                question = payload.get("question", "").strip()
            except json.JSONDecodeError:
                question = data.strip()

            if not question:
                await websocket.send_text(json.dumps({"event": "error", "data": "Empty question"}))
                continue

            await websocket.send_text(json.dumps({"event": "start", "session_id": session_id}))

            # ── 1. Semantic cache check ───────────────────────────────────
            cached = await semantic_cache.semantic_get(question)
            if cached:
                await websocket.send_text(json.dumps({"event": "cached", "data": cached}, default=str))
                await websocket.send_text(json.dumps({"event": "done"}))
                continue

            # ── 2. Session memory ─────────────────────────────────────────
            memory = await get_memory()
            history = await memory.as_langchain_messages(session_id)

            # ── 3. Run graph + stream events ──────────────────────────────
            final_state: dict = {}
            try:
                async with engine.connect() as conn:
                    async for chunk in run_graph_stream(question, session_id, history, conn):
                        await websocket.send_text(chunk)
                        # Track final state from done event (we'll re-derive from graph output)
            except Exception as e:
                logger.exception("Graph stream error: %s", e)
                await websocket.send_text(json.dumps({"event": "error", "data": str(e)}))
            finally:
                await websocket.send_text(json.dumps({"event": "done"}))

    except WebSocketDisconnect:
        logger.info("WS disconnected: session=%s", session_id)
    except Exception as e:
        logger.exception("WS fatal error: %s", e)
        try:
            await websocket.send_text(json.dumps({"event": "error", "data": str(e)}))
        except Exception:
            pass


# ── REST chat (non-streaming) ─────────────────────────────────────────────────

@app.post("/chat/{session_id}")
async def chat(session_id: str, request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cached = await semantic_cache.semantic_get(request.question)
    if cached:
        return {**cached, "from_cache": True}

    memory = await get_memory()
    history = await memory.as_langchain_messages(session_id)

    from backend.graph.agent_graph import AgentState
    from backend.graph.agent_graph import get_graph as _get_graph

    graph = _get_graph()
    initial: AgentState = {
        "question": request.question, "session_id": session_id, "history": history,
        "intent": "", "sql_query": "", "sql_explanation": "", "sql_columns": [],
        "sql_rows": [], "sql_row_count": 0, "sql_error": "", "retry_count": 0,
        "heal_context": "", "rag_answer": "", "chart_config": None,
        "response_type": "", "final_answer": "",
    }
    async with engine.connect() as conn:
        result = await graph.ainvoke(initial, config={"configurable": {"conn": conn}})

    response = {
        "type":         result.get("response_type"),
        "answer":       result.get("final_answer"),
        "sql_query":    result.get("sql_query"),
        "explanation":  result.get("sql_explanation"),
        "columns":      result.get("sql_columns"),
        "rows":         result.get("sql_rows"),
        "row_count":    result.get("sql_row_count"),
        "chart_config": result.get("chart_config"),
        "heal_attempts": result.get("retry_count", 0),
        "from_cache":   False,
    }

    # Cache + memory
    await semantic_cache.semantic_set(request.question, response)
    await memory.add_turn(session_id, request.question, result.get("final_answer", ""))

    return response


# ── Voice endpoints ───────────────────────────────────────────────────────────

@app.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    ext = (file.filename or "audio.webm").rsplit(".", 1)[-1].lower()
    return await transcribe(audio_bytes, audio_format=ext)


@app.post("/voice/synthesise")
async def voice_synthesise(request: TTSRequest):
    audio_bytes = await synthesise(request.text, engine=request.engine, language=request.language)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")
    return {"audio_base64": base64.b64encode(audio_bytes).decode(), "format": "mp3"}


@app.post("/voice/synthesise/stream")
async def voice_synthesise_stream(request: TTSRequest):
    """Chunked TTS: returns list of audio chunks for sequential playback."""
    chunks = await chunk_synthesise(request.text, chunk_size=settings.tts_chunk_words)
    return {
        "chunks": [base64.b64encode(c).decode() for c in chunks],
        "count": len(chunks),
        "format": "mp3",
    }


@app.post("/voice/chat/{session_id}")
async def voice_chat(session_id: str, file: UploadFile = File(...)):
    audio_bytes = await file.read()
    ext = (file.filename or "audio.webm").rsplit(".", 1)[-1].lower()

    stt_result = await transcribe(audio_bytes, audio_format=ext)
    question = stt_result.get("text", "").strip()
    if not question:
        error_msg = stt_result.get("error", "Could not transcribe audio")
        logger.warning("Voice transcription empty: %s", error_msg)
        raise HTTPException(status_code=422, detail=error_msg)

    memory = await get_memory()
    history = await memory.as_langchain_messages(session_id)

    from backend.graph.agent_graph import AgentState, get_graph as _get_graph
    graph = _get_graph()
    initial: AgentState = {
        "question": question, "session_id": session_id, "history": history,
        "intent": "", "sql_query": "", "sql_explanation": "", "sql_columns": [],
        "sql_rows": [], "sql_row_count": 0, "sql_error": "", "retry_count": 0,
        "heal_context": "", "rag_answer": "", "chart_config": None,
        "response_type": "", "final_answer": "",
    }
    async with engine.connect() as conn:
        result = await graph.ainvoke(initial, config={"configurable": {"conn": conn}})

    resp_type   = result.get("response_type", "rag")
    explanation = result.get("sql_explanation", "")
    rag_answer  = result.get("rag_answer", "")
    answer_text = explanation if resp_type == "sql" else rag_answer
    if not answer_text:
        answer_text = result.get("final_answer", "I couldn't find an answer.")

    # Streaming TTS chunks
    audio_chunks = await chunk_synthesise(answer_text, chunk_size=settings.tts_chunk_words)

    # Build chart payload with rows included (frontend needs both)
    chart_data = None
    if result.get("chart_config") and result.get("sql_rows"):
        chart_data = {**result["chart_config"], "rows": result["sql_rows"]}

    return {
        "question":    question,
        "answer":      answer_text,
        "agent_type":  resp_type,
        # Full SQL result so the frontend can render the table
        "sql_query":   result.get("sql_query", ""),
        "explanation": explanation,
        "columns":     result.get("sql_columns", []),
        "rows":        result.get("sql_rows", []),
        "row_count":   result.get("sql_row_count", 0),
        "chart_data":  chart_data,
        "audio_chunks": [base64.b64encode(c).decode() for c in audio_chunks],
        "audio_format": "mp3",
    }


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.post("/admin/test/self-heal")
async def test_self_heal():
    """
    Injects a deliberately broken SQL query into the graph to test self-healing.
    Skips the LLM generation step and starts at execute_sql with a bad query.
    Returns: {healed, heal_attempts, original_error, final_sql, rows}
    """
    from backend.graph.agent_graph import AgentState, get_graph as _get_graph

    graph = _get_graph()

    # Realistic broken SQL — right tables, wrong column name.
    # GPT-4 can fix this by checking the schema (revenue → line_total).
    broken_sql = "SELECT p.name, SUM(p.revenue) AS total FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.name ORDER BY total DESC LIMIT 5"
    initial: AgentState = {
        "question":       "test self-heal",
        "session_id":     "test-heal",
        "history":        [],
        "intent":         "SQL",
        "sql_query":      broken_sql,
        "sql_explanation": "Deliberately broken query for testing",
        "sql_columns":    [],
        "sql_rows":       [],
        "sql_row_count":  0,
        "sql_error":      "",
        "retry_count":    0,
        "heal_context":   "",
        "rag_answer":     "",
        "chart_config":   None,
        "response_type":  "",
        "final_answer":   "",
    }

    async with engine.connect() as conn:
        result = await graph.ainvoke(
            initial,
            config={"configurable": {"conn": conn}},
        )

    healed     = result.get("retry_count", 0) > 0
    final_sql  = result.get("sql_query", "")
    error      = result.get("sql_error", "")
    resp_type  = result.get("response_type", "")

    return {
        "healed":          healed,
        "heal_attempts":   result.get("retry_count", 0),
        "original_sql":    broken_sql,
        "final_sql":       final_sql,
        "original_error":  "column nonexistent_column does not exist (expected)",
        "response_type":   resp_type,
        "row_count":       result.get("sql_row_count", 0),
        "rows_sample":     result.get("sql_rows", [])[:3],
        "fell_back_to_rag": resp_type == "rag",
        "rag_answer":      result.get("rag_answer", "") if resp_type == "rag" else None,
    }


@app.post("/admin/seed")
async def reseed_database():
    try:
        from backend.database.seed import seed
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text(
                "TRUNCATE reviews, order_items, orders, products, customers, categories RESTART IDENTITY CASCADE"
            ))
            await conn.commit()
        from backend.database.connection import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await seed(session)
        return {"status": "ok", "message": "Database re-seeded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/session/{session_id}")
async def clear_session(session_id: str):
    memory = await get_memory()
    await memory.clear(session_id)
    return {"status": "ok", "session_id": session_id}


@app.get("/admin/new-session")
async def new_session():
    return {"session_id": str(uuid.uuid4())}


@app.delete("/admin/cache")
async def clear_cache():
    """Clear the semantic cache (in-memory + Redis)."""
    import redis.asyncio as aioredis
    from backend.cache.semantic_cache import REDIS_KEY
    import backend.cache.semantic_cache as sc
    sc._vectors = None
    sc._entries = []
    try:
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        await r.delete(REDIS_KEY)
    except Exception:
        pass
    return {"status": "ok", "message": "Semantic cache cleared"}


@app.get("/admin/stats")
async def stats():
    from backend.cache.semantic_cache import _entries as sc_entries
    from backend.learning.few_shot_store import _examples as fs_examples
    return {
        "semantic_cache_entries": len(sc_entries),
        "few_shot_examples": len(fs_examples),
    }


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"status": "ok", "database": "up" if db_ok else "down", "version": "2.0.0"}


# ── Serve UI ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = frontend_path / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text())
    return HTMLResponse("<h1>AI Agent v2</h1><p>See /docs</p>")
