# DataMind — Reactive AI Agent

> Natural language interface over an e-commerce database. Ask questions in plain English, get SQL results, charts, and voice responses — no SQL or BI tooling required.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1-orange)
![LangFuse](https://img.shields.io/badge/LangFuse-Tracing-purple)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

### Media

![QueryDemo](./media/sql.png)
![RagDemo](./media/rag.png)
![VoiceDemo](./media/voice.png)
![SemanticCacheDemo](./media/semantic-cache.png)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Database Schema](#database-schema)
- [Advanced Features](#advanced-features)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Features

| Feature | Description |
|---|---|
| **Natural Language SQL** | Ask business questions in plain English — agent generates, validates, and executes SQL |
| **LangGraph Pipeline** | Stateful graph: classify → generate → validate → execute → chart → finalise |
| **Self-Healing SQL** | Automatically fixes broken queries using the error message as feedback (up to 2 retries) |
| **Semantic Cache** | Embedding-based similarity cache — paraphrases hit the same cached result |
| **Auto Few-Shot Learning** | Stores every successful query; injects top-K similar examples into future prompts |
| **RAG Fallback** | FAISS + OpenAI embeddings for knowledge questions that don't need database queries |
| **Chart Generation** | Auto-detects chart type (bar/line) from result data and renders inline |
| **Session Memory** | Redis-backed rolling conversation window (last 6 turns, 30 min TTL) |
| **Voice I/O** | Full voice pipeline — Whisper STT (English, hallucination-filtered) + gTTS TTS with chunked streaming. Returns full SQL table + chart for voice queries |
| **LangFuse Tracing** | Full observability — token counts, latency, cost per query per node |
| **WebSocket Streaming** | Token-by-token streaming with live pipeline progress bar |

---

## Architecture

```
Browser / Voice Client
        │
        ▼
FastAPI  (WebSocket /ws/{session_id}  +  REST /chat/{session_id})
        │
        ├── Semantic Cache  ──► Redis (embedding similarity, threshold 0.87)
        │         │ MISS
        │         ▼
        └── LangGraph State Machine
                  │
                  ├── classify_node       (GPT-3.5 — SQL vs RAG)
                  │
                  ├── [SQL path]
                  │     ├── generate_sql_node   (GPT-4o + schema + few-shot)
                  │     ├── execute_sql_node    (asyncpg + EXPLAIN dry-run)
                  │     ├── heal_sql_node       (GPT-4o — fixes errors, max 2 retries)
                  │     └── detect_chart_node   (heuristic — bar / line)
                  │
                  ├── [RAG path]
                  │     └── rag_node            (FAISS top-K + GPT-3.5)
                  │
                  └── finalize_node
                        ├── Save to semantic cache
                        ├── Save to few-shot store
                        └── Update session memory
        │
        ├── Session Memory   ──► Redis (rolling window, TTL 30 min)
        ├── Few-Shot Store   ──► FAISS + JSON (auto-learned examples)
        ├── LangFuse         ──► cloud.langfuse.com (traces, cost, latency)
        └── Voice Pipeline
              ├── STT: OpenAI Whisper API (language="en", hallucination filter, min duration 1s)
              └── TTS: gTTS chunked streaming (25 words/chunk, sequential audio queue)
```

---

## Project Structure

```
reactive-ai-agent/
│
├── backend/
│   ├── main.py                      # FastAPI app — WebSocket, REST, admin endpoints
│   ├── config.py                    # Pydantic settings (reads .env)
│   │
│   ├── graph/
│   │   └── agent_graph.py           # LangGraph state machine (all agent nodes + edges)
│   │
│   ├── agent/
│   │   ├── sql_agent.py             # SQL generation helpers (schema introspection)
│   │   └── rag_agent.py             # FAISS RAG agent
│   │
│   ├── cache/
│   │   ├── query_cache.py           # Legacy hash cache (kept for compatibility)
│   │   └── semantic_cache.py        # FAISS embedding similarity cache
│   │
│   ├── learning/
│   │   └── few_shot_store.py        # Auto few-shot learning store (FAISS + JSON)
│   │
│   ├── memory/
│   │   └── session.py               # Redis rolling-window session memory
│   │
│   ├── tracing/
│   │   └── langfuse_setup.py        # LangFuse callback handler (v2 + v4 compatible)
│   │
│   ├── database/
│   │   ├── connection.py            # Async SQLAlchemy engine
│   │   ├── models.py                # ORM models (customers, products, orders, etc.)
│   │   └── seed.py                  # Faker synthetic data generator
│   │
│   └── voice/
│       ├── stt.py                   # OpenAI Whisper STT (English-forced, hallucination filter, min-duration check)
│       └── tts.py                   # gTTS / Coqui / pyttsx3 TTS
│
├── frontend/
│   └── index.html                   # Dark terminal UI (Chart.js, WebSocket, voice)
│
├── data/
│   ├── docs/
│   │   └── ecommerce_knowledge_base.md   # RAG source document
│   └── few_shot_examples.json            # Auto-generated, grows with usage
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/)
- An [OpenAI API key](https://platform.openai.com/api-keys) with GPT-4 access
- *(Optional)* A free [LangFuse account](https://cloud.langfuse.com) for tracing

### 1. Clone the repository

```bash
git clone https://github.com/your-username/reactive-ai-agent.git
cd reactive-ai-agent
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-...
```

For LangFuse tracing (optional but recommended):

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 3. Build and start

```bash
docker compose up --build
```

This starts three containers: `postgres`, `redis`, and `backend`.

### 4. Seed synthetic data

In a second terminal:

```bash
docker compose exec backend python -m backend.database.seed
```

Expected output:
```
✓ 8 categories
✓ 41 products
✓ 500 customers
✓ ~1500 orders
✓ 400 reviews
✅ Seed complete!
```

### 5. Open the UI

```
http://localhost:8000
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `DATABASE_URL` | ✅ | `postgresql+asyncpg://agent:agentpass@localhost:5432/ecommerce` | Postgres connection string |
| `REDIS_URL` | ✅ | `redis://localhost:6379` | Redis connection string |
| `LANGFUSE_PUBLIC_KEY` | ⬜ | — | LangFuse public key (tracing) |
| `LANGFUSE_SECRET_KEY` | ⬜ | — | LangFuse secret key (tracing) |
| `LANGFUSE_HOST` | ⬜ | `https://cloud.langfuse.com` | LangFuse host |
| `SEMANTIC_CACHE_ENABLED` | ⬜ | `true` | Enable semantic similarity cache |
| `SEMANTIC_CACHE_THRESHOLD` | ⬜ | `0.87` | Cosine similarity threshold (0–1) |
| `FEW_SHOT_TOP_K` | ⬜ | `3` | Examples to inject per SQL query |
| `SQL_MAX_RETRIES` | ⬜ | `2` | Max self-heal retries on SQL error |
| `SESSION_TTL_SECONDS` | ⬜ | `1800` | Session memory TTL (seconds) |
| `MEMORY_WINDOW_SIZE` | ⬜ | `6` | Rolling memory window (turns) |
| `TTS_STREAMING_ENABLED` | ⬜ | `true` | Enable chunked TTS streaming |
| `TTS_CHUNK_WORDS` | ⬜ | `25` | Words per TTS audio chunk |

---

## API Reference

### WebSocket — Streaming Chat

```
WS /ws/{session_id}
```

**Send:**
```json
{"question": "What are the top 5 products by revenue?"}
```

**Events received:**

| Event | Payload | Description |
|---|---|---|
| `start` | `{session_id}` | Query started |
| `node_start` | `{node: "generate_sql"}` | Graph node activated |
| `token` | `{data: "chunk..."}` | RAG streaming token |
| `sql_result` | `{sql_query, explanation, columns, rows, row_count, heal_attempts}` | SQL executed |
| `chart` | `{type, x_col, y_cols, rows}` | Chart data |
| `cached` | `{type, answer, ...}` | Served from semantic cache |
| `done` | — | Query complete |
| `error` | `{data: "message"}` | Error occurred |

### REST Chat

```http
POST /chat/{session_id}
Content-Type: application/json

{"question": "How many orders last month?"}
```

### Voice Endpoints

```http
POST /voice/transcribe              # Audio file → {text, language}
POST /voice/synthesise              # {text, engine, language} → {audio_base64}
POST /voice/synthesise/stream       # {text} → {chunks: [...], count, format}
POST /voice/chat/{session_id}       # Audio in → {question, answer, sql_query, columns, rows, row_count, chart_data, audio_chunks}
```

### Admin Endpoints

```http
POST   /admin/seed                  # Re-seed synthetic data
DELETE /admin/session/{session_id}  # Clear session memory
DELETE /admin/cache                 # Clear semantic cache
GET    /admin/stats                 # {semantic_cache_entries, few_shot_examples}
GET    /admin/new-session           # Generate new session UUID
POST   /admin/test/self-heal        # Test self-healing SQL pipeline
GET    /health                      # {status, database, version}
```

Full interactive API docs: **http://localhost:8000/docs**

---

## Database Schema

| Table | ~Rows | Key Columns |
|---|---|---|
| `categories` | 8 | `id`, `name`, `description` |
| `products` | 41 | `id`, `name`, `sku`, `category_id`, `price`, `cost`, `stock`, `is_active` |
| `customers` | 500 | `id`, `email`, `first_name`, `last_name`, `country`, `city`, `is_premium` |
| `orders` | ~1,500 | `id`, `customer_id`, `status`, `total_amount`, `discount_amount`, `shipping_country` |
| `order_items` | ~4,500 | `id`, `order_id`, `product_id`, `quantity`, `unit_price`, `line_total` |
| `reviews` | ~400 | `id`, `customer_id`, `product_id`, `rating` (1–5), `comment` |

**Order statuses:** `pending` → `confirmed` → `shipped` → `delivered` / `cancelled` / `refunded`

---

## Advanced Features

### LangGraph Pipeline

The agent runs as a stateful graph. Each node handles one responsibility:

```
classify_node   → GPT-3.5 classifies SQL vs RAG
generate_sql    → GPT-4o generates SQL with schema + few-shot examples
execute_sql     → EXPLAIN dry-run + asyncpg execution
heal_sql        → GPT-4o fixes broken SQL using the error message
detect_chart    → Heuristic detects bar/line chart from column types
rag_node        → FAISS top-K retrieval + GPT-3.5 answer
finalize_node   → Saves to cache + few-shot store + session memory
```

### Self-Healing SQL

When a SQL query fails EXPLAIN or execution, the error is fed back to GPT-4 with the original query and schema. Up to `SQL_MAX_RETRIES` (default 2) healing attempts are made before falling back to RAG.

Test it:
```bash
curl -X POST http://localhost:8000/admin/test/self-heal | python3 -m json.tool
```

### Semantic Cache

Queries are embedded with `text-embedding-3-small` and stored as normalised vectors. On each new query, cosine similarity is computed against all stored vectors. A match above `SEMANTIC_CACHE_THRESHOLD` (default 0.87) returns the cached result instantly — no LLM call, no DB query.

```
"Top 5 products by revenue"     →  stored (similarity: 1.0)
"best selling products"         →  cache HIT (similarity: 0.891)
"top products by sales"         →  cache HIT (similarity: 0.873)
"how many customers do we have" →  cache MISS (similarity: 0.41)
```

### Auto Few-Shot Learning

Every successful SQL query is embedded and stored in `data/few_shot_examples.json`. On each new SQL query, the top-K most semantically similar past examples are injected into the system prompt dynamically. The agent improves with every use.

Check the store:
```bash
curl http://localhost:8000/admin/stats
# {"semantic_cache_entries": 12, "few_shot_examples": 28}
```

### LangFuse Observability

With LangFuse credentials set, every query generates a full trace at [cloud.langfuse.com](https://cloud.langfuse.com):

- Token usage and cost per node (classify, generate_sql, rag, etc.)
- Latency breakdown per step
- Full prompt and completion for every LLM call
- Session-grouped conversation history
- Model usage comparison (GPT-3.5 vs GPT-4o)

---

## Voice Input/Output

### How it works

```
User speaks  →  Browser MediaRecorder (WebM)
                      │
                      ▼
              POST /voice/chat/{session_id}
                      │
              pydub + ffmpeg
              (WebM → 16kHz mono WAV)
                      │
              OpenAI Whisper API
              language="en", temperature=0
                      │
              Hallucination filter
              (rejects "you", "thank you", clips < 1s)
                      │
              LangGraph pipeline
              (same as text queries)
                      │
              Returns: question + SQL table
                     + chart + audio_chunks
                      │
              gTTS chunked TTS
              (25 words/chunk → MP3 base64)
                      │
              Frontend audio queue
              (sequential playback)
```

### Usage tips

- Hold the 🎙 button, **wait 1 second**, then speak clearly
- Speak a **full sentence** — single words are often misrecognised
- Use a **quiet environment** — background noise causes Whisper to output "you" or silence
- Release the button **after** finishing, not during speech
- The response renders the **same full result** as a text query — SQL table, chart, and audio

### Hallucination filter

Whisper outputs `"you"`, `"thank you"`, or random text in wrong languages when given silence or very short clips. The STT module rejects these automatically:

| Check | Threshold | Action on fail |
|---|---|---|
| Minimum duration | 1.0 seconds | Reject, return error message |
| Minimum word count | 2 words | Reject, return error message |
| Hallucination phrase list | `"you"`, `"thank you"`, `"bye"`, etc. | Reject, return error message |
| Language | Forced to English | Prevents Japanese/Kannada hallucinations |

---

## Testing

### Smoke test

```bash
# Health check
curl http://localhost:8000/health | python3 -m json.tool

# REST chat
curl -X POST http://localhost:8000/chat/test-session \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers do we have?"}' | python3 -m json.tool

# Stats
curl http://localhost:8000/admin/stats | python3 -m json.tool
```

### Feature checklist

| Feature | Test query | Expected |
|---|---|---|
| SQL + chart | `Top 5 products by revenue` | Table + bar chart |
| RAG | `What is a premium customer?` | Text answer, no SQL |
| Semantic cache | Ask same question twice | ⚡ cached badge on 2nd |
| Self-healing | `POST /admin/test/self-heal` | `healed: true, heal_attempts: 1` |
| Session memory | Ask follow-up without context | Correctly references prior answer |
| Voice | Click 🎙, speak clearly, release | SQL table + chart + audio playback — same as text queries |
| LangFuse | Ask any question | Trace appears at cloud.langfuse.com |

### Useful log filters

```bash
# All features at once
docker compose logs backend -f | grep -iE "intent|heal|semantic cache|saved to|few-shot|chart"

# Semantic cache hits/misses
docker compose logs backend -f | grep -iE "semantic cache"

# Self-healing
docker compose logs backend -f | grep -iE "heal|execution error"

# LangFuse
docker compose logs backend -f | grep -i "langfuse"
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `pydantic extra_forbidden` | `.env` has vars not in Settings | Add `extra = "ignore"` to Config class |
| `database "agent" does not exist` | Postgres healthcheck uses wrong DB | Add `-d ecommerce` to `pg_isready` |
| `MissingGreenlet` | Schema inspect outside `run_sync` | Wrap all inspect calls inside `run_sync(_do_inspect)` |
| `LLM returned non-JSON` | GPT-3.5 ignoring JSON format | Use GPT-4 + `response_format: json_object` for SQL |
| `Object of type Decimal is not JSON serializable` | Postgres Decimal not handled | Add `default=str` to `json.dumps` calls |
| `langfuse not installed` warning | Wrong import path (v2 vs v4) | Use `from langfuse.langchain import CallbackHandler` |
| Semantic cache never hits | Threshold too high | Lower `SEMANTIC_CACHE_THRESHOLD` to `0.87` |
| Cache hits always 0 | `response_type` not set to `"sql"` | Set it in `execute_sql_node` on success |
| Empty response bubbles | `astream` using wrong stream mode | Use `stream_mode="updates"` |
| Voice transcribes as `"you"` | Whisper hallucinating on silence | Speak clearly after 1s pause; filter applied automatically |
| Voice transcribes in wrong language | No language hint to Whisper | `language="en"` now forced in `stt.py` |
| Voice returns text only, no table | `voice/chat` not returning SQL fields | Endpoint now returns full `columns`, `rows`, `chart_data` |
| Voice audio too short error | Recording < 1 second | Hold mic button for at least 1–2 seconds before speaking |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| Agent orchestration | LangGraph + LangChain |
| LLMs | GPT-4o (SQL/heal), GPT-3.5-turbo (classify/RAG) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Database | PostgreSQL 16 + asyncpg + SQLAlchemy 2.0 |
| Vector store | FAISS (RAG + semantic cache + few-shot) |
| Cache / Memory | Redis 7 |
| Observability | LangFuse |
| Voice STT | OpenAI Whisper API |
| Voice TTS | gTTS (Google Text-to-Speech) |
| Frontend | Vanilla JS + Chart.js + WebSocket |
| Containerisation | Docker + Docker Compose |
| Data generation | Faker |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.
