# Portfolio data server

An MCP server that lets an AI assistant answer questions about brokerage records, plus a dashboard and a scheduled daily update.

- **12 MCP tools** for querying positions, history and performance from Interactive Brokers Flex statements, stored in SQLite.
- **Reconciliation:** calculated positions are checked against broker statements, and any unreconciled residual is shown on the dashboard instead of being hidden. On my two accounts it reconciles to a $0.00 residual.
- **Tests:** `pytest` runs against a synthetic Flex statement in `tests/fixtures` (fake account IDs), with no real account data.

## Layout

| Path | What it does |
|---|---|
| `portfolio_mcp/` | Flex download and parsing, database, metrics, analytics, MCP server |
| `server.py` | Starts the MCP server |
| `daily_update.py` | Scheduled refresh (see `scripts/` for a macOS launchd example) |
| `dashboard.py`, `pie.py` | Static HTML dashboard and allocation chart |
| `research/` | A small gold-ETF distribution study with public price data |
| `tests/` | Automated tests |

## Setup

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    cp .env.example .env               # add your own Flex token and query ID
    cp manual_accounts.example.json manual_accounts.json
    .venv/bin/python -m pytest

Read-only: it downloads statements and never places orders. Your `.env`, database and statements stay on your machine and are git-ignored.

Built with Claude Code.
