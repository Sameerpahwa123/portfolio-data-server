"""Refresh orchestration (fetch -> raw file -> parse -> upsert -> reconcile) and the read tools."""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from . import analytics as an
from .config import AccountConfig, Config
from .db import Database, readonly_query, utcnow
from .flex import FlexClient, FlexError, save_raw
from .metrics import AccountMetrics, compute, manual_metrics, table
from .parse import ParseError, parse_datetime_stamp, parse_flex

log = logging.getLogger(__name__)
CHUNK_DAYS = 365
OVERLAP_DAYS = 5  # re-pull a few days so late corrections from IBKR are picked up


class ReconciliationError(RuntimeError):
    pass


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


MAX_STEP_BACK = 3  # days; covers a weekend plus a late statement


def chunks(start: date, end: date, days: int = CHUNK_DAYS) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        out.append((cur, stop))
        cur = stop + timedelta(days=1)
    return out


class PortfolioService:
    def __init__(self, config: Config, client: FlexClient | None = None, today: Callable[[], date] = date.today) -> None:
        self.config = config
        self.db = Database(config.db_path)
        self.db.init()
        self._client = client
        self._today = today
        self._lock = threading.Lock()

    @property
    def client(self) -> FlexClient:
        if self._client is None:
            self._client = FlexClient()
        return self._client

    # ------------------------------------------------------------------------- ingest

    def ingest_file(self, label: str, path: Path) -> tuple[set[str], dict[str, int]]:
        content = path.read_bytes()
        pf = parse_flex(content, raw_file=path.name, reference=parse_datetime_stamp(path.name))
        with closing(self.db.connect()) as conn, conn:
            counts = self.db.store(conn, pf)
            for aid in pf.account_ids:
                conn.execute("INSERT INTO accounts (account_id, label) VALUES (?, ?) "
                             "ON CONFLICT(account_id) DO UPDATE SET label=excluded.label", (aid, label))
        return pf.account_ids, counts

    def _fetch(self, acc: AccountConfig, fd: date | None = None, td: date | None = None) -> Path:
        content = self.client.fetch(acc.token, acc.query_id, _ymd(fd) if fd else None, _ymd(td) if td else None)
        return save_raw(self.config.raw_dir, acc.label, content, _ymd(fd) if fd else None, _ymd(td) if td else None)

    def _fetch_latest(self, acc: AccountConfig, fd: date, td: date) -> Path:
        """Fetch a window ending at the newest statement IBKR has published.

        A day's statement only appears some hours after the US close, and until then IBKR rejects any
        window that includes it with code 1003. Step the end date back rather than failing the refresh.
        """
        for back in range(MAX_STEP_BACK + 1):
            end = td - timedelta(days=back)
            try:
                return self._fetch(acc, fd, end)
            except FlexError as e:
                if e.code != "1003" or back == MAX_STEP_BACK or end <= fd:
                    raise
                log.info("[%s] statement to %s not published yet (1003); trying a day earlier", acc.label, end)
                self.client.sleep(2)
        raise AssertionError("unreachable")

    def _account_id(self, conn: Any, label: str) -> str | None:
        r = conn.execute("SELECT account_id FROM accounts WHERE label=?", (label,)).fetchone()
        return r["account_id"] if r else None

    def _refresh_one(self, acc: AccountConfig) -> dict[str, Any]:
        started = utcnow()
        files: list[str] = []
        counts: Counter[str] = Counter()
        aid: str | None = None
        metrics: AccountMetrics | None = None
        status, error = "ok", None
        try:
            yesterday = self._today() - timedelta(days=1)
            with closing(self.db.connect()) as conn:
                aid = self._account_id(conn, acc.label)
                last = conn.execute("SELECT MAX(to_date) FROM statements WHERE account_id=?", (aid,)).fetchone()[0] if aid else None
            if last:
                start = date.fromisoformat(last) - timedelta(days=OVERLAP_DAYS)
            else:  # first run: default query tells us when the account was opened, then backfill to it
                p = self._fetch(acc)
                files.append(p.name)
                ids, c = self.ingest_file(acc.label, p)
                counts.update(c)
                aid = sorted(ids)[0]
                with closing(self.db.connect()) as conn:
                    info = conn.execute("SELECT date_opened, date_funded FROM account_info WHERE account_id=?", (aid,)).fetchone()
                opened = min(filter(None, [info["date_opened"], info["date_funded"]]), default=None) if info else None
                if not opened:
                    raise FlexError("AccountInformation.dateOpened missing: tick Account Information in the Flex query")
                start = date.fromisoformat(opened) - timedelta(days=1)
            windows = chunks(start, yesterday)
            for i, (fd, td) in enumerate(windows):
                if files or i:
                    self.client.sleep(2)
                log.info("[%s] fetching %s..%s", acc.label, fd, td)
                # only the final window may end early: stepping back an earlier one would leave a gap
                p = self._fetch_latest(acc, fd, td) if i == len(windows) - 1 else self._fetch(acc, fd, td)
                files.append(p.name)
                _, c = self.ingest_file(acc.label, p)
                counts.update(c)
            with closing(self.db.connect()) as conn, conn:
                conn.execute("UPDATE accounts SET last_refreshed=?, base_currency=(SELECT currency FROM account_info "
                             "WHERE account_id=?), history_from=(SELECT MIN(from_date) FROM statements WHERE account_id=?), "
                             "history_to=(SELECT MAX(to_date) FROM statements WHERE account_id=?) WHERE account_id=?",
                             (utcnow(), aid, aid, aid, aid))
                metrics = compute(conn, aid, acc.label, self.config.base_currency)
            if not metrics.reconciled:
                status, error = "reconciliation_failed", "metrics do not reconcile with IBKR totals (see table)"
        except (FlexError, ParseError, OSError) as e:
            status, error = "error", self.config.redact(f"{type(e).__name__}: {e}")
        result = {"account": acc.label, "account_id": aid, "status": status, "error": error, "raw_files": files,
                  "rows_by_table": dict(sorted(counts.items())),
                  "reconciliation": metrics.to_dict()["checks"] if metrics else None,
                  "residual": round(metrics.residual, 2) if metrics else None}
        if metrics:
            result["reconciliation_table"] = table([metrics])
        self.db.log(acc.label, aid, started, status, dict(counts), files, error, result["reconciliation"])
        return result

    def refresh(self, account: str = "all") -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            return {"status": "error", "error": "a refresh is already running"}
        try:
            accs = list(self.config.accounts) if account in ("all", "", None) else [self.config.account(account)]
            results = []
            for i, acc in enumerate(accs):
                if i:
                    self.client.sleep(2)
                results.append(self._refresh_one(acc))
            ok = all(r["status"] == "ok" for r in results)
            return {"status": "ok" if ok else "failed", "results": results}
        finally:
            self._lock.release()

    def reparse(self) -> dict[str, Any]:
        """Rebuild the cache from ./raw without contacting IBKR (oldest file first)."""
        labels = sorted((a.label for a in self.config.accounts), key=len, reverse=True)
        files = sorted(self.config.raw_dir.glob("*.xml"), key=lambda p: re.sub(r"^[^_]+_", "", p.name))
        counts: Counter[str] = Counter()
        done = []
        for p in files:
            label = next((lbl for lbl in labels if p.name.startswith(lbl + "_")), None)
            if label:
                counts.update(self.ingest_file(label, p)[1])
                done.append(p.name)
        with closing(self.db.connect()) as conn, conn:
            conn.execute("UPDATE accounts SET base_currency=(SELECT currency FROM account_info a WHERE "
                         "a.account_id=accounts.account_id), history_from=(SELECT MIN(from_date) FROM statements s "
                         "WHERE s.account_id=accounts.account_id), history_to=(SELECT MAX(to_date) FROM statements s "
                         "WHERE s.account_id=accounts.account_id)")
        return {"files": done, "rows_by_table": dict(sorted(counts.items()))}

    # ------------------------------------------------------------------------- reads

    def _accounts(self, conn: Any, account: str) -> list[an.Acct]:
        rows = conn.execute("SELECT * FROM accounts ORDER BY label").fetchall()
        if not rows:
            raise LookupError("no cached data: run refresh() first")
        accts = [an.Acct(r["account_id"], r["label"], r["history_to"]) for r in rows]
        if account in ("all", "", None):
            return accts
        sel = [a for a in accts if account in (a.label, a.account_id)]
        if not sel:
            raise ValueError(f"unknown account {account!r}; use all, " + ", ".join(a.label for a in accts))
        return sel

    def _read(self, account: str, fn: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
        with closing(self.db.connect()) as conn:
            accts = self._accounts(conn, account)
            return {"accounts_included": [a.label for a in accts], "currency": self.config.base_currency,
                    **fn(conn, accts, *args)}

    def metrics(self, account: str = "all", include_manual: bool = True) -> list[AccountMetrics]:
        """IBKR accounts (computed + reconciled) followed by manual accounts (as typed in manual_accounts.json)."""
        everything = account in ("all", "", None)
        manual = [manual_metrics(a) for a in self.config.manual if include_manual and (everything or account == a.label)]
        out: list[AccountMetrics] = []
        if everything or not manual:
            with closing(self.db.connect()) as conn:
                try:
                    accts = self._accounts(conn, account)
                except LookupError:
                    if not manual:
                        raise
                    accts = []
                out = [compute(conn, a.account_id, a.label, self.config.base_currency) for a in accts]
        return out + manual

    def get_accounts(self) -> dict[str, Any]:
        ms = self.metrics()
        keys = ("deposited", "current_value", "net_pnl", "realized", "unrealized", "dividends", "interest",
                "other_fees", "fx_conversion", "fees_paid", "residual")
        combined = {k: round(sum(getattr(m, k) for m in ms), 2) for k in keys}
        dep = sum(m.deposited for m in ms)
        combined["net_pnl_pct"] = round(combined["net_pnl"] / dep * 100, 2) if dep else None
        combined["realized_plus_unrealized"] = round(sum(m.trading_pnl for m in ms), 2)
        combined["realized_plus_unrealized_pct"] = round(sum(m.trading_pnl for m in ms) / dep * 100, 2) if dep else None
        combined["realized_pct"] = round(combined["realized"] / dep * 100, 2) if dep else None
        return {"currency": self.config.base_currency,
                "accounts": [{k: v for k, v in m.to_dict().items() if k != "checks"} for m in ms],
                "combined": combined,
                "all_reconciled": all(m.reconciled for m in ms if not m.is_manual),
                "manual_accounts": [m.label for m in ms if m.is_manual],
                "note": "manual accounts are typed in (manual_accounts.json) and cannot be reconciled; "
                        "their whole net P&L is realized"}

    def get_positions(self, account: str = "all") -> dict[str, Any]:
        return self._read(account, an.positions)

    def get_trades(self, account: str = "all", start: str | None = None, end: str | None = None,
                   symbol: str | None = None, asset_class: str | None = None) -> dict[str, Any]:
        return self._read(account, an.trades, an.norm_date(start), an.norm_date(end), symbol, asset_class)

    def get_round_trips(self, account: str = "all") -> dict[str, Any]:
        return self._read(account, an.round_trips)

    def get_cash_flows(self, account: str = "all", type: str | None = None, start: str | None = None,
                       end: str | None = None) -> dict[str, Any]:
        return self._read(account, an.cash_flows, type, an.norm_date(start), an.norm_date(end))

    def get_dividends(self, account: str = "all", year: int | None = None) -> dict[str, Any]:
        return self._read(account, an.dividends, int(year) if year else None)

    def get_pnl(self, account: str = "all", start: str | None = None, end: str | None = None,
                group_by: str = "symbol") -> dict[str, Any]:
        return self._read(account, an.pnl, an.norm_date(start), an.norm_date(end), group_by)

    def get_performance(self, account: str = "all") -> dict[str, Any]:
        return self._read(account, an.performance)

    def query_sql(self, sql: str) -> dict[str, Any]:
        if not self.config.db_path.exists():
            raise LookupError("no database yet: run refresh() first")
        return readonly_query(self.config.db_path, sql)

    def refresh_status(self) -> dict[str, Any]:
        out = []
        with closing(self.db.connect()) as conn:
            for acc in self.config.accounts:
                last = conn.execute("SELECT * FROM refresh_log WHERE label=? ORDER BY id DESC LIMIT 1", (acc.label,)).fetchone()
                a = conn.execute("SELECT * FROM accounts WHERE label=?", (acc.label,)).fetchone()
                out.append({"account": acc.label, "account_id": a["account_id"] if a else None,
                            "last_refreshed": a["last_refreshed"] if a else None,
                            "history_from": a["history_from"] if a else None, "history_to": a["history_to"] if a else None,
                            "last_status": last["status"] if last else None, "last_error": last["error"] if last else None,
                            "last_finished_at": last["finished_at"] if last else None})
        return {"accounts": out, "db_path": str(self.config.db_path)}
