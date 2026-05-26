"""
SQL Agent
─────────
1. Introspects the live DB schema on startup
2. Routes to GPT-3.5 or GPT-4 based on prompt complexity
3. Generates SQL via strict JSON output: {sql_query, explanation}
4. Validates SQL with a simulated dry-run (EXPLAIN)
5. Executes and returns rows + natural-language explanation
"""
import json
import logging
import re
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from backend.config import get_settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)
settings = get_settings()

# ── few-shot examples ────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "question": "What are the top 5 best-selling products by revenue?",
        "sql_query": (
            "SELECT p.name, SUM(oi.line_total) AS revenue "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.id "
            "JOIN orders o ON oi.order_id = o.id "
            "WHERE o.status != 'cancelled' "
            "GROUP BY p.name ORDER BY revenue DESC LIMIT 5"
        ),
        "explanation": "Joins order_items, products and orders, excludes cancelled orders, sums revenue per product.",
    },
    {
        "question": "How many orders were placed last month?",
        "sql_query": (
            "SELECT COUNT(*) AS order_count FROM orders "
            "WHERE created_at >= date_trunc('month', NOW() - INTERVAL '1 month') "
            "AND created_at < date_trunc('month', NOW())"
        ),
        "explanation": "Counts orders whose created_at falls within the previous calendar month.",
    },
    {
        "question": "Which customers have spent the most overall?",
        "sql_query": (
            "SELECT c.first_name || ' ' || c.last_name AS customer, "
            "SUM(o.total_amount) AS lifetime_value "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "WHERE o.status = 'delivered' "
            "GROUP BY c.id ORDER BY lifetime_value DESC LIMIT 10"
        ),
        "explanation": "Sums delivered order totals per customer and returns the top 10.",
    },
    {
        "question": "What is the average order value by country?",
        "sql_query": (
            "SELECT shipping_country, ROUND(AVG(total_amount),2) AS avg_order_value, COUNT(*) AS num_orders "
            "FROM orders WHERE status NOT IN ('cancelled','refunded') "
            "GROUP BY shipping_country ORDER BY avg_order_value DESC"
        ),
        "explanation": "Groups non-cancelled/refunded orders by shipping country and calculates average value.",
    },
    {
        "question": "What is the order fulfillment rate?",
        "sql_query": (
            "SELECT ROUND(100.0 * SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) / COUNT(*), 2) "
            "AS fulfillment_rate_pct FROM orders"
        ),
        "explanation": "Fulfillment rate = delivered_orders / total_orders × 100.",
    },
]


def _format_few_shots() -> str:
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f"Q: {ex['question']}")
        lines.append(f"SQL: {ex['sql_query']}")
        lines.append(f"Explanation: {ex['explanation']}\n")
    return "\n".join(lines)


# ── schema introspection ─────────────────────────────────────────────────────

async def introspect_schema(conn: AsyncConnection) -> str:
    """Return a compact schema string for injection into the system prompt."""

    def _do_inspect(sync_conn) -> str:
        # All inspection must happen inside run_sync — never outside it
        insp = sa.inspect(sync_conn)
        lines = []
        for table in insp.get_table_names():
            cols = insp.get_columns(table)
            col_strs = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            lines.append(f"  {table}({col_strs})")
        return "Tables:\n" + "\n".join(lines)

    return await conn.run_sync(_do_inspect)


# ── LLM selection ────────────────────────────────────────────────────────────

def _choose_model(prompt: str) -> ChatOpenAI:
    word_count = len(prompt.split())
    # Always use the smart model for SQL — GPT-3.5 does not reliably
    # follow the strict JSON output format required for SQL generation.
    model_name = settings.model_smart
    logger.info("Routing to %s (prompt=%d words)", model_name, word_count)
    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,
        temperature=0,
        streaming=False,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# ── system prompt ────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are an expert SQL analyst for an e-commerce company.
You have access to a PostgreSQL database with the following schema:

{schema}

Business term definitions:
- "order fulfillment rate" = fulfilled_orders / total_orders
- "revenue" = SUM(order_items.line_total) excluding cancelled/refunded orders
- "lifetime value (LTV)" = SUM(orders.total_amount) for delivered orders per customer
- "AOV (average order value)" = AVG(orders.total_amount)
- "premium customer" = customers.is_premium = true

Rules:
1. Always use LIMIT 50 unless the user asks for all rows or a specific count.
2. Never use SELECT * — always name columns explicitly.
3. For date comparisons use date_trunc or INTERVAL; never hard-code years.
4. Always respond with ONLY valid JSON in this exact format (no markdown fences):
   {{"sql_query": "<your SQL>", "explanation": "<plain English explanation>"}}

Few-shot examples:
{few_shots}
"""


# ── SQL Agent class ──────────────────────────────────────────────────────────

class SQLAgent:
    def __init__(self, schema: str):
        self._schema = schema
        self._few_shots = _format_few_shots()

    def _build_prompt(self) -> ChatPromptTemplate:
        # Use .partial() so LangChain handles ALL substitution.
        # Never pre-format with str.format() — it un-escapes {{ }} braces.
        return ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ]).partial(schema=self._schema, few_shots=self._few_shots)

    async def generate(
        self,
        question: str,
        history: list,
        conn: AsyncConnection,
    ) -> dict[str, Any]:
        """
        Full pipeline:
          1. Generate SQL + explanation from LLM
          2. Validate with EXPLAIN (dry-run)
          3. Execute and return results
        """
        llm = _choose_model(question)
        chain = self._build_prompt() | llm | StrOutputParser()
        raw = await chain.ainvoke({"question": question, "history": history})

        # ── parse JSON response ───────────────────────────────────────────
        try:
            # Strip any accidental markdown fences
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(clean)
            sql = parsed["sql_query"].strip().rstrip(";")
            explanation = parsed.get("explanation", "")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("LLM returned non-JSON: %s | raw=%s", e, raw[:300])
            return {"error": "SQL generation failed — could not parse LLM output.", "raw": raw}

        # ── dry-run validation ────────────────────────────────────────────
        try:
            await conn.execute(sa.text(f"EXPLAIN {sql}"))
        except Exception as e:
            logger.warning("SQL validation failed: %s | sql=%s", e, sql)
            return {"error": f"Generated SQL is invalid: {e}", "sql_query": sql}

        # ── execute with row cap ──────────────────────────────────────────
        try:
            result = await conn.execute(sa.text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(50)]
        except Exception as e:
            logger.error("Query execution error: %s", e)
            return {"error": f"Query execution failed: {e}", "sql_query": sql}

        return {
            "sql_query":   sql,
            "explanation": explanation,
            "columns":     columns,
            "rows":        rows,
            "row_count":   len(rows),
        }


# ── module-level singleton ────────────────────────────────────────────────────

_sql_agent: SQLAgent | None = None


async def get_sql_agent(conn: AsyncConnection) -> SQLAgent:
    global _sql_agent
    if _sql_agent is None:
        schema = await introspect_schema(conn)
        _sql_agent = SQLAgent(schema)
        logger.info("SQLAgent initialised with schema:\n%s", schema)
    return _sql_agent
