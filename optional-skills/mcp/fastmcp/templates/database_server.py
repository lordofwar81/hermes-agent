from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

from fastmcp import FastMCP


mcp = FastMCP("__SERVER_NAME__")

DATABASE_PATH = os.getenv("SQLITE_PATH", "./app.db")
MAX_ROWS = int(os.getenv("SQLITE_MAX_ROWS", "200"))
TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DATABASE_PATH}?mode=ro", uri=True)


def _reject_mutation(sql: str) -> None:
    normalized = sql.strip().lower()
    if not normalized.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    # Block dangerous keywords
    dangerous = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "attach",
        "detach",
        "reindex",
        "vacuum",
        "pragma",
    ]
    for word in dangerous:
        if re.search(rf"\b{word}\b", normalized):
            raise ValueError(f"Query contains forbidden keyword: {word}")

    # Block subqueries with UNION/INTERSECT/EXCEPT
    if re.search(r"\b(union|intersect|except)\b", normalized):
        raise ValueError("UNION/INTERSECT/EXCEPT not allowed")

    # Block comments that might hide malicious code
    if "--" in sql or "/*" in sql:
        raise ValueError("Comments not allowed in queries")


def _validate_sql_query(sql: str) -> str:
    """Validate and sanitize SQL query."""
    _reject_mutation(sql)

    # Strip and normalize
    sql = sql.strip().rstrip(";")

    # Check for balanced parentheses
    if sql.count("(") != sql.count(")"):
        raise ValueError("Unbalanced parentheses in query")

    # Validate table names in FROM clause (basic check)
    from_match = re.search(r"\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", sql, re.IGNORECASE)
    if from_match:
        table_name = from_match.group(1)
        if not TABLE_NAME_RE.fullmatch(table_name):
            raise ValueError(f"Invalid table name in query: {table_name}")

    return sql


def _validate_table_name(table_name: str) -> str:
    if not TABLE_NAME_RE.fullmatch(table_name):
        raise ValueError("Invalid table name")
    return table_name


@mcp.tool
def list_tables() -> list[str]:
    """List user-defined SQLite tables."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


@mcp.tool
def describe_table(table_name: str) -> list[dict[str, Any]]:
    """Describe columns for a SQLite table."""
    safe_table_name = _validate_table_name(table_name)
    with _connect() as conn:
        rows = conn.execute(f"PRAGMA table_info({safe_table_name})").fetchall()
    return [
        {
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": bool(row[5]),
        }
        for row in rows
    ]


@mcp.tool
def query(sql: str, limit: int = 50) -> dict[str, Any]:
    """Run a read-only SELECT query and return rows plus column names."""
    safe_sql = _validate_sql_query(sql)
    safe_limit = max(0, min(limit, MAX_ROWS))
    wrapped_sql = f"SELECT * FROM ({safe_sql}) LIMIT {safe_limit}"
    with _connect() as conn:
        cursor = conn.execute(wrapped_sql)
        columns = [column[0] for column in cursor.description or []]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return {"limit": safe_limit, "columns": columns, "rows": rows}


if __name__ == "__main__":
    mcp.run()
