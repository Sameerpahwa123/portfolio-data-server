"""Entry point: python server.py runs the portfolio MCP server over stdio."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio_mcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
