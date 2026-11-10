"""SQLite cache. Each element table: UNIQUE(account_id, uid) where uid is the natural key, so re-fetching
overlapping windows is an idempotent upsert and history accumulates."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .parse import N, SPECS, ParsedFile

COMMON = {"account_id": "TEXT NOT NULL", "uid": "TEXT NOT NULL", "stmt_from": "TEXT", "stmt_to": "TEXT",
          "currency": "TEXT", "fx_rate_to_base": "REAL"}
TAIL = {"raw_json": "TEXT", "raw_file": "TEXT", "updated_at": "TEXT"}


def _tables() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for spec in SPECS.values():
        cols = out.setdefault(spec.table, {})
        for c, (_, kind) in spec.cols.items():
            cols[c] = "REAL" if kind == N else "TEXT"
    out["elements"] = {"tag": "TEXT"}
    return out


TABLES = _tables()
INDEXES = {"nav": "report_date", "change_in_nav": "from_date", "cash_transactions": "type", "trades": "trade_date",
           "open_positions": "report_date", "fifo_summary": "stmt_to", "elements": "tag"}

EXTRA_DDL = """
CREATE TABLE IF NOT EXISTS accounts (account_id TEXT PRIMARY KEY, label TEXT NOT NULL, base_currency TEXT,
    last_refreshed TEXT, history_from TEXT, history_to TEXT);
CREATE TABLE IF NOT EXISTS statements (account_id TEXT NOT NULL, from_date TEXT, to_date TEXT, period TEXT,
    when_generated TEXT, raw_file TEXT, updated_at TEXT, UNIQUE(account_id, from_date, to_date));
CREATE TABLE IF NOT EXISTS refresh_log (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, account_id TEXT,
    started_at TEXT, finished_at TEXT, status TEXT, rows_by_table TEXT, raw_files TEXT, error TEXT,
    reconciliation TEXT);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(EXTRA_DDL)
            for t, cols in TABLES.items():
                body = ", ".join(f"{c} {ty}" for c, ty in {**COMMON, **cols, **TAIL}.items())
                conn.execute(f"CREATE TABLE IF NOT EXISTS {t} (id INTEGER PRIMARY KEY AUTOINCREMENT, {body}, "
                             "UNIQUE(account_id, uid))")
                have = {r["name"] for r in conn.execute(f"PRAGMA table_info({t})")}
                for c, ty in {**cols, **TAIL}.items():
                    if c not in have:
                        conn.execute(f"ALTER TABLE {t} ADD COLUMN {c} {ty}")
                if t in INDEXES:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{t} ON {t}(account_id, {INDEXES[t]})")
        conn.close()

    @staticmethod
    def store(conn: sqlite3.Connection, pf: ParsedFile) -> dict[str, int]:
        now = utcnow()
        counts: dict[str, int] = {}
        for t, rows in pf.tables.items():
            cols = list({**COMMON, **TABLES[t], **TAIL})
            sql = (f"INSERT INTO {t} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                   f"ON CONFLICT(account_id, uid) DO UPDATE SET "
                   + ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("account_id", "uid")))
            conn.executemany(sql, [tuple(now if c == "updated_at" else r.get(c) for c in cols) for r in rows])
            counts[t] = len(rows)
        conn.executemany(
            "INSERT INTO statements (account_id, from_date, to_date, period, when_generated, raw_file, updated_at) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(account_id, from_date, to_date) DO UPDATE SET period=excluded.period, "
            "when_generated=excluded.when_generated, raw_file=excluded.raw_file, updated_at=excluded.updated_at",
            [(s["account_id"], s["from_date"], s["to_date"], s["period"], s["when_generated"], s["raw_file"], now)
             for s in pf.statements])
        return counts

    def log(self, label: str, account_id: str | None, started: str, status: str, rows: dict[str, int],
            raw_files: list[str], error: str | None, reconciliation: Any) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO refresh_log (label, account_id, started_at, finished_at, status, rows_by_table, "
                         "raw_files, error, reconciliation) VALUES (?,?,?,?,?,?,?,?,?)",
                         (label, account_id, started, utcnow(), status, json.dumps(rows), json.dumps(raw_files),
                          error, json.dumps(reconciliation, default=str)))
        conn.close()


# ----------------------------------------------------------------------------- read-only SQL

_ALLOWED = {getattr(sqlite3, n, v) for n, v in
            (("SQLITE_SELECT", 21), ("SQLITE_READ", 20), ("SQLITE_FUNCTION", 31), ("SQLITE_RECURSIVE", 33))}


def readonly_query(db_path: Path, sql: str, max_rows: int = 1000) -> dict[str, Any]:
    stmt = (sql or "").strip().rstrip(";").strip()
    body = re.sub(r"^(\s*(--[^\n]*\n|/\*.*?\*/))*\s*", "", stmt, flags=re.S)
    if not re.match(r"(?i)(select|with)\b", body):
        raise ValueError("only a single SELECT (or WITH ... SELECT) statement is allowed")
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY)
        try:
            cur = conn.execute(stmt)
        except (sqlite3.ProgrammingError, sqlite3.DatabaseError) as e:
            raise ValueError(f"rejected: {e}") from None
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
    finally:
        conn.close()
    return {"columns": cols, "rows": [dict(zip(cols, r)) for r in rows[:max_rows]],
            "row_count": min(len(rows), max_rows), "truncated": len(rows) > max_rows}
