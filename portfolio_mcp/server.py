"""MCP server (stdio). Read-only: no order placement or account-changing tools. Logs go to stderr."""

import logging
import os
import sys
from functools import partial
from typing import Any, Callable

import anyio

try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2 (FastMCP renamed)
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]

from .config import load_config
from .dashboard import write
from .service import PortfolioService

log = logging.getLogger("portfolio_mcp")

INSTRUCTIONS = """Read-only IBKR portfolio analytics (accounts: "options", "gold", or "all"). Data is cached from
IBKR Flex statements and reconciled against IBKR's own ChangeInNAV/FIFO totals on every refresh. All money is USD.
Dates are YYYY-MM-DD. This server cannot place, modify or cancel orders."""


def build_server(service: PortfolioService) -> Any:
    mcp = _Server(name="portfolio-mcp", instructions=INSTRUCTIONS)

    def safe(fn: Callable[..., dict[str, Any]], *a: Any) -> dict[str, Any]:
        try:
            return fn(*a)
        except (ValueError, LookupError) as e:
            return {"error": service.config.redact(str(e))}
        except Exception as e:
            log.exception("tool failed")
            return {"error": service.config.redact(f"{type(e).__name__}: {e}")}

    @mcp.tool()
    async def refresh(account: str = "all") -> dict[str, Any]:
        """Pull new statements from IBKR (backfilling to account inception on first run), store them and
        reconcile against IBKR's totals. status is "failed" with a reconciliation table if anything is off by > $1."""
        return await anyio.to_thread.run_sync(partial(safe, service.refresh, account))

    @mcp.tool()
    def refresh_status() -> dict[str, Any]:
        """Last refresh time, status, error and cached history range per account."""
        return safe(service.refresh_status)

    @mcp.tool()
    def get_accounts() -> dict[str, Any]:
        """Per account and combined: deposited, current value, net P&L and its reconciled breakdown, fees."""
        return safe(service.get_accounts)

    @mcp.tool()
    def get_positions(account: str = "all") -> dict[str, Any]:
        """Open positions at the latest statement: qty, avg cost, mark, value, unrealized P&L (options: right/strike/expiry)."""
        return safe(service.get_positions, account)

    @mcp.tool()
    def get_trades(account: str = "all", start: str | None = None, end: str | None = None,
                   symbol: str | None = None, asset_class: str | None = None) -> dict[str, Any]:
        """Executions newest first. symbol matches symbol or underlying; asset_class STK, OPT, CASH (FX conversions)..."""
        return safe(service.get_trades, account, start, end, symbol, asset_class)

    @mcp.tool()
    def get_round_trips(account: str = "all") -> dict[str, Any]:
        """Executions grouped by order (spread legs together) with strategy, net debit/credit and net realized."""
        return safe(service.get_round_trips, account)

    @mcp.tool()
    def get_cash_flows(account: str = "all", type: str | None = None, start: str | None = None,
                       end: str | None = None) -> dict[str, Any]:
        """Cash transactions (DETAIL rows). type: deposit, withdrawal, dividend, withholding, interest, fee,
        or an exact IBKR type such as "Deposits/Withdrawals"."""
        return safe(service.get_cash_flows, account, type, start, end)

    @mcp.tool()
    def get_dividends(account: str = "all", year: int | None = None) -> dict[str, Any]:
        """Dividends received by symbol (gross, withholding, net) plus open accruals."""
        return safe(service.get_dividends, account, year)

    @mcp.tool()
    def get_pnl(account: str = "all", start: str | None = None, end: str | None = None,
                group_by: str = "symbol") -> dict[str, Any]:
        """P&L by symbol, asset_class or month: realized, unrealized, dividends, interest, fees, FX conversion, net."""
        return safe(service.get_pnl, account, start, end, group_by)

    @mcp.tool()
    def get_performance(account: str = "all") -> dict[str, Any]:
        """Daily NAV series, TWR for 1D/7D/MTD/YTD/inception and max drawdown."""
        return safe(service.get_performance, account)

    @mcp.tool()
    async def build_dashboard() -> dict[str, Any]:
        """Refresh all accounts, then regenerate dashboard.html. Returns the path and the numbers shown."""

        def run() -> dict[str, Any]:
            r = service.refresh()
            ms = service.metrics()
            return {"path": str(write(ms)), "refresh_status": r.get("status"),
                    "errors": [x["error"] for x in r.get("results", []) if x.get("error")],
                    "accounts": [{k: v for k, v in m.to_dict().items() if k != "checks"} for m in ms]}

        return await anyio.to_thread.run_sync(partial(safe, run))

    @mcp.tool()
    def query_sql(sql: str) -> dict[str, Any]:
        """One read-only SELECT on the cache. Tables: accounts, statements, account_info, nav, change_in_nav,
        cash_transactions, trades, open_positions, fifo_summary, dividend_accruals, security_info, elements (all other
        IBKR sections, by tag), refresh_log. Every row has raw_json with all IBKR attributes."""
        return safe(service.query_sql, sql)

    return mcp


def setup_logging() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("httpx", "httpcore"):  # httpx logs request URLs, which contain the token
        logging.getLogger(n).setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    build_server(PortfolioService(load_config())).run()


if __name__ == "__main__":
    main()
