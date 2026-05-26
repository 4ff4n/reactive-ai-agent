"""
LangGraph Agent — State Machine
─────────────────────────────────

Flow:
  START
    ↓
  classify_intent
    ↓
  ┌─────────────────────────────┐
  │ SQL path          RAG path  │
  │                             │
  generate_sql       rag_answer │
    ↓                    ↓      │
  execute_sql         finalize  │
    ↓                           │
  [success] → detect_chart      │
  [fail]    → heal_sql (×2)     │
  [give up] → rag_answer        │
    ↓                           │
  finalize ───────────────────► END

State flows forward-only; heal_sql loops back into execute_sql.
DB connection is passed via RunnableConfig["configurable"]["conn"].
"""
import json
import logging
import re
from typing import Any, AsyncIterator, Optional, TypedDict

import sqlalchemy as sa
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.agent.rag_agent import get_rag_agent
from backend.config import get_settings
from backend.learning import few_shot_store
from backend.tracing.langfuse_setup import build_callbacks, get_langfuse_handler

logger = logging.getLogger(__name__)
settings = get_settings()


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # inputs
    question: str
    session_id: str
    history: list[BaseMessage]

    # classification
    intent: str                     # "SQL" | "RAG"

    # SQL flow
    sql_query: str
    sql_explanation: str
    sql_columns: list[str]
    sql_rows: list[dict]
    sql_row_count: int
    sql_error: str
    retry_count: int
    heal_context: str               # error message fed back for healing

    # RAG flow
    rag_answer: str

    # chart
    chart_config: Optional[dict]

    # output
    response_type: str              # "sql" | "rag"
    final_answer: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _llm(smart: bool = True, streaming: bool = False, json_mode: bool = False) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": settings.model_smart if smart else settings.model_fast,
        "api_key": settings.openai_api_key,
        "temperature": 0,
        "streaming": streaming,
    }
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)


def _parse_sql_response(raw: str) -> tuple[str, str]:
    """Extract (sql_query, explanation) from LLM JSON response."""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    parsed = json.loads(clean)
    return parsed["sql_query"].strip().rstrip(";"), parsed.get("explanation", "")


# ── Schema introspection (cached module-level) ────────────────────────────────

_schema_cache: str = ""


async def _get_schema(conn) -> str:
    global _schema_cache
    if _schema_cache:
        return _schema_cache

    def _inspect(sync_conn) -> str:
        insp = sa.inspect(sync_conn)
        lines = []
        for table in insp.get_table_names():
            cols = insp.get_columns(table)
            col_strs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            lines.append(f"  {table}({col_strs})")
        return "Tables:\n" + "\n".join(lines)

    _schema_cache = await conn.run_sync(_inspect)
    return _schema_cache


# ── Static few-shot examples ──────────────────────────────────────────────────

STATIC_EXAMPLES = """
Q: Top 5 best-selling products by revenue
SQL: SELECT p.name, SUM(oi.line_total) AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.id JOIN orders o ON oi.order_id = o.id WHERE o.status NOT IN ('cancelled','refunded') GROUP BY p.name ORDER BY revenue DESC LIMIT 5
Explanation: Sums line_total per product excluding cancelled/refunded orders.

Q: How many orders were placed last month?
SQL: SELECT COUNT(*) AS order_count FROM orders WHERE created_at >= date_trunc('month', NOW() - INTERVAL '1 month') AND created_at < date_trunc('month', NOW())
Explanation: Counts orders in the previous calendar month using date_trunc.

Q: Order fulfillment rate
SQL: SELECT ROUND(100.0 * SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) / COUNT(*), 2) AS fulfillment_rate_pct FROM orders
Explanation: fulfilled_orders / total_orders × 100.

Q: Top 10 customers by lifetime value
SQL: SELECT c.first_name || ' ' || c.last_name AS customer, SUM(o.total_amount) AS lifetime_value FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.status = 'delivered' GROUP BY c.id ORDER BY lifetime_value DESC LIMIT 10
Explanation: Sums delivered order totals per customer.
"""

SQL_SYSTEM = """You are an expert PostgreSQL analyst for an e-commerce company.
Schema:
{schema}

Business definitions:
- fulfillment rate = delivered_orders / total_orders
- revenue = SUM(order_items.line_total) excluding cancelled/refunded
- LTV = SUM(orders.total_amount) for delivered orders per customer
- AOV = AVG(orders.total_amount)
- premium customer = customers.is_premium = true

Rules:
1. LIMIT 50 unless user specifies otherwise.
2. Name all columns explicitly — no SELECT *.
3. Use date_trunc / INTERVAL for dates — never hard-code years.
4. Respond ONLY with valid JSON (no markdown fences):
   {{"sql_query": "...", "explanation": "..."}}

Static examples:
{static_examples}
{dynamic_examples}"""


# ── Graph nodes ───────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are a routing classifier for an e-commerce database agent.

SQL  → the question needs to query a database (products, orders, customers, revenue, counts, rankings, trends, lists, metrics, comparisons, aggregations).
RAG  → the question asks for a definition, explanation, or general knowledge (what does X mean, how is Y calculated, what are the statuses).

Key rule: if the question could be answered by querying a database table, choose SQL — even if it sounds conversational.

Examples:
  "best selling products"              → SQL
  "top customers by spend"             → SQL
  "how many orders last month"         → SQL
  "show me revenue by country"         → SQL
  "which products are low in stock"    → SQL
  "what is a premium customer"         → RAG
  "what does AOV mean"                 → RAG
  "how is LTV calculated"              → RAG
  "what are the order statuses"        → RAG

Reply with ONLY one word: SQL or RAG

Message: {question}"""


async def classify_node(state: AgentState, config: RunnableConfig) -> dict:
    # Skip LLM call if intent already injected (e.g. from test endpoints)
    if state.get("intent") in ("SQL", "RAG"):
        logger.info("Intent pre-set to %s — skipping classification", state["intent"])
        return {"intent": state["intent"]}
    lf = get_langfuse_handler(state["session_id"], state["question"], "classify")
    prompt = ChatPromptTemplate.from_template(CLASSIFY_PROMPT)
    chain = prompt | _llm(smart=False) | StrOutputParser()
    result = await chain.ainvoke(
        {"question": state["question"]},
        config={"callbacks": build_callbacks(lf)},
    )
    intent = "SQL" if "SQL" in result.upper() else "RAG"
    logger.info("Intent: %s for: %s", intent, state["question"][:60])
    return {"intent": intent}


async def generate_sql_node(state: AgentState, config: RunnableConfig) -> dict:
    conn = config["configurable"]["conn"]
    schema = await _get_schema(conn)

    # Dynamic few-shot from learning store
    similar = await few_shot_store.retrieve_similar(state["question"])
    dynamic_examples = few_shot_store.format_for_prompt(similar)

    lf = get_langfuse_handler(state["session_id"], state["question"], "generate_sql")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_SYSTEM),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]).partial(
        schema=schema,
        static_examples=STATIC_EXAMPLES,
        dynamic_examples=dynamic_examples,
    )
    chain = prompt | _llm(smart=True, json_mode=True) | StrOutputParser()
    try:
        raw = await chain.ainvoke(
            {"question": state["question"], "history": state.get("history", [])},
            config={"callbacks": build_callbacks(lf)},
        )
        sql, explanation = _parse_sql_response(raw)
        return {"sql_query": sql, "sql_explanation": explanation, "sql_error": ""}
    except Exception as e:
        logger.warning("SQL generation failed: %s", e)
        return {"sql_query": "", "sql_explanation": "", "sql_error": str(e)}


async def execute_sql_node(state: AgentState, config: RunnableConfig) -> dict:
    conn = config["configurable"]["conn"]
    sql = state.get("sql_query", "")
    if not sql:
        return {"sql_error": "No SQL query to execute"}
    try:
        # Dry-run validation
        await conn.execute(sa.text(f"EXPLAIN {sql}"))
        # Execute
        result = await conn.execute(sa.text(sql))
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchmany(50)]
        return {
            "sql_columns": columns,
            "sql_rows": rows,
            "sql_row_count": len(rows),
            "sql_error": "",
            "response_type": "sql",   # mark as SQL so finalize_node branches correctly
        }
    except Exception as e:
        logger.warning("SQL execution error: %s", e)
        # Rollback the failed transaction so the connection is clean
        # for heal_sql_node which needs to query the schema
        try:
            await conn.rollback()
        except Exception:
            pass
        return {"sql_error": str(e), "heal_context": str(e)}


HEAL_SYSTEM = """You are a PostgreSQL expert. The SQL query below failed.
Fix the error and respond ONLY with JSON: {{"sql_query": "...", "explanation": "..."}}

Schema:
{schema}

Failed query:
{failed_sql}

Error:
{error}"""


async def heal_sql_node(state: AgentState, config: RunnableConfig) -> dict:
    conn = config["configurable"]["conn"]
    # Clear schema cache so it re-queries on the now-clean connection
    global _schema_cache
    _schema_cache = ""
    schema = await _get_schema(conn)
    lf = get_langfuse_handler(state["session_id"], state["question"], "heal_sql")
    prompt = ChatPromptTemplate.from_template(HEAL_SYSTEM)
    chain = prompt | _llm(smart=True, json_mode=True) | StrOutputParser()
    retry_count = state.get("retry_count", 0) + 1
    try:
        raw = await chain.ainvoke(
            {
                "schema": schema,
                "failed_sql": state.get("sql_query", ""),
                "error": state.get("heal_context", state.get("sql_error", "")),
            },
            config={"callbacks": build_callbacks(lf)},
        )
        sql, explanation = _parse_sql_response(raw)
        logger.info("Healed SQL (attempt %d): %s", retry_count, sql[:80])
        return {"sql_query": sql, "sql_explanation": explanation, "retry_count": retry_count}
    except Exception as e:
        logger.warning("SQL healing failed: %s", e)
        return {"sql_error": str(e), "retry_count": retry_count}


async def detect_chart_node(state: AgentState, config: RunnableConfig) -> dict:
    """Determine if the SQL result is suitable for visualisation."""
    from decimal import Decimal

    rows = state.get("sql_rows", [])
    columns = state.get("sql_columns", [])

    if len(rows) < 2 or not columns:
        return {"chart_config": None}

    def _is_numeric(val) -> bool:
        # Postgres NUMERIC → Decimal, integers → int, floats → float
        if isinstance(val, (int, float, Decimal)):
            return True
        if isinstance(val, str):
            try:
                float(val)
                return True
            except ValueError:
                return False
        return False

    # Classify columns — check first 5 non-null rows
    numeric_cols = [
        c for c in columns
        if all(
            _is_numeric(r.get(c))
            for r in rows[:5] if r.get(c) is not None
        ) and any(r.get(c) is not None for r in rows[:5])
    ]
    text_cols = [c for c in columns if c not in numeric_cols]

    logger.info(
        "detect_chart: cols=%s numeric=%s text=%s rows=%d",
        columns, numeric_cols, text_cols, len(rows)
    )

    if not numeric_cols:
        return {"chart_config": None}

    date_kw = {"date", "month", "year", "week", "day", "period", "time", "created"}
    time_col = next((c for c in text_cols if any(kw in c.lower() for kw in date_kw)), None)

    if time_col and len(rows) >= 3:
        chart_type = "line"
        x_col = time_col
    elif text_cols:
        chart_type = "bar"
        x_col = text_cols[0]
    else:
        # All numeric — use index as x axis
        chart_type = "bar"
        x_col = columns[0]
        numeric_cols = columns[1:4]

    if not numeric_cols:
        return {"chart_config": None}

    logger.info("Chart config: type=%s x=%s y=%s", chart_type, x_col, numeric_cols)
    return {
        "chart_config": {
            "type": chart_type,
            "x_col": x_col,
            "y_cols": numeric_cols[:3],
        }
    }


async def rag_node(state: AgentState, config: RunnableConfig) -> dict:
    lf = get_langfuse_handler(state["session_id"], state["question"], "rag")
    rag_agent = get_rag_agent()
    await rag_agent.ensure_ready()

    # Use LangFuse callback if available
    callbacks = build_callbacks(lf)
    answer = await rag_agent.answer(
        state["question"],
        state.get("history", []),
        callbacks=callbacks,
    )
    return {"rag_answer": answer, "response_type": "rag"}


async def finalize_node(state: AgentState, config: RunnableConfig) -> dict:
    """Build the final_answer string and store in few-shot learning."""
    if state.get("response_type") == "sql":
        explanation = state.get("sql_explanation", "")
        row_count = state.get("sql_row_count", 0)
        final = f"{explanation}\n\n{row_count} row(s) returned."

        # Auto few-shot: store successful query
        if state.get("sql_query") and not state.get("sql_error"):
            await few_shot_store.add_example(
                question=state["question"],
                sql_query=state["sql_query"],
                explanation=explanation,
            )
    else:
        final = state.get("rag_answer", "I couldn't find an answer.")

    return {"final_answer": final}


# ── Routing functions ─────────────────────────────────────────────────────────

def route_classify(state: AgentState) -> str:
    if state.get("intent") != "SQL":
        return "rag"
    # If SQL query already injected, skip generation and go straight to execution
    if state.get("sql_query"):
        return "execute_sql"
    return "generate_sql"


def route_execute(state: AgentState) -> str:
    if not state.get("sql_error"):
        return "detect_chart"
    if state.get("retry_count", 0) < settings.sql_max_retries:
        return "heal_sql"
    logger.warning("SQL max retries reached — falling back to RAG")
    return "rag"


def route_generate(state: AgentState) -> str:
    return "execute_sql" if state.get("sql_query") else "rag"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    wf = StateGraph(AgentState)

    wf.add_node("classify",      classify_node)
    wf.add_node("generate_sql",  generate_sql_node)
    wf.add_node("execute_sql",   execute_sql_node)
    wf.add_node("heal_sql",      heal_sql_node)
    wf.add_node("detect_chart",  detect_chart_node)
    wf.add_node("rag",           rag_node)
    wf.add_node("finalize",      finalize_node)

    wf.set_entry_point("classify")
    wf.add_conditional_edges("classify",     route_classify,  {"generate_sql": "generate_sql", "rag": "rag", "execute_sql": "execute_sql"})
    wf.add_conditional_edges("generate_sql", route_generate,  {"execute_sql": "execute_sql",   "rag": "rag"})
    wf.add_conditional_edges("execute_sql",  route_execute,   {"detect_chart": "detect_chart", "heal_sql": "heal_sql", "rag": "rag"})
    wf.add_edge("heal_sql",     "execute_sql")
    wf.add_edge("detect_chart", "finalize")
    wf.add_edge("rag",          "finalize")
    wf.add_edge("finalize",     END)

    return wf.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Streaming runner ──────────────────────────────────────────────────────────

async def run_graph_stream(
    question: str,
    session_id: str,
    history: list[BaseMessage],
    conn,
) -> AsyncIterator[str]:
    """
    Runs the graph and yields WebSocket-friendly JSON event strings.
    Events:
      {"event": "node_start",  "node": "..."}
      {"event": "sql_result",  "data": {...}}
      {"event": "token",       "data": "..."}   ← RAG streaming
      {"event": "chart",       "data": {...}}
      {"event": "done"}
    """
    graph = get_graph()
    initial_state: AgentState = {
        "question":       question,
        "session_id":     session_id,
        "history":        history,
        "intent":         "",
        "sql_query":      "",
        "sql_explanation": "",
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

    # stream_mode="updates" yields {node_name: {changed_keys}} per step.
    # Without it, astream yields the FULL state dict (keys = "question","intent" etc)
    # which breaks our node_name detection and state merging.
    final_state: dict = dict(initial_state)  # seed with initial values

    async for node_chunk in graph.astream(
        initial_state,
        config={"configurable": {"conn": conn}},
        stream_mode="updates",
    ):
        for node_name, node_output in node_chunk.items():
            # Notify frontend which node just ran
            yield json.dumps({"event": "node_start", "node": node_name}) + "\n"
            # Merge this node's output into accumulated final state
            if isinstance(node_output, dict):
                final_state.update(node_output)

    # Emit results based on accumulated final state
    if final_state.get("sql_columns"):
        sql_payload = {
            "sql_query":     final_state.get("sql_query", ""),
            "explanation":   final_state.get("sql_explanation", ""),
            "columns":       final_state.get("sql_columns", []),
            "rows":          final_state.get("sql_rows", []),
            "row_count":     final_state.get("sql_row_count", 0),
            "heal_attempts": final_state.get("retry_count", 0),
        }
        yield json.dumps({"event": "sql_result", "data": sql_payload}, default=str) + "\n"

        if final_state.get("chart_config"):
            yield json.dumps({
                "event": "chart",
                "data": {**final_state["chart_config"], "rows": final_state.get("sql_rows", [])},
            }, default=str) + "\n"

    elif final_state.get("rag_answer"):
        # Pseudo-stream RAG answer word by word
        answer = final_state["rag_answer"]
        words = answer.split()
        for i in range(0, len(words), 4):
            chunk = " ".join(words[i:i + 4]) + " "
            yield json.dumps({"event": "token", "data": chunk}) + "\n"

    # Save to semantic cache + session memory
    # Use sql_explanation for SQL results, rag_answer for RAG — don't rely on final_answer
    from backend.cache.semantic_cache import semantic_set
    from backend.memory.session import get_memory

    resp_type = final_state.get("response_type", "rag")
    if resp_type == "sql":
        answer_text = final_state.get("sql_explanation", "") or final_state.get("final_answer", "")
    else:
        answer_text = final_state.get("rag_answer", "") or final_state.get("final_answer", "")

    if answer_text or final_state.get("sql_columns"):
        cache_payload = {
            "type":        resp_type,
            "answer":      answer_text,
            "sql_query":   final_state.get("sql_query", ""),
            "explanation": final_state.get("sql_explanation", ""),
            "columns":     final_state.get("sql_columns", []),
            "rows":        final_state.get("sql_rows", []),
            "row_count":   final_state.get("sql_row_count", 0),
            "chart_config": final_state.get("chart_config"),
        }
        await semantic_set(question, cache_payload)
        logger.info("Saved to semantic cache: type=%s question=%s", resp_type, question[:60])
        memory = await get_memory()
        await memory.add_turn(session_id, question, answer_text)

    yield json.dumps({"event": "done"}) + "\n"
