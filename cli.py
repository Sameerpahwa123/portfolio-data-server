"""Run any tool from the shell: .venv/bin/python cli.py get_accounts | refresh '{"account":"gold"}' | reparse"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio_mcp.config import load_config  # noqa: E402
from portfolio_mcp.server import setup_logging  # noqa: E402
from portfolio_mcp.service import PortfolioService  # noqa: E402

if __name__ == "__main__":
    setup_logging()
    svc = PortfolioService(load_config())
    kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(getattr(svc, sys.argv[1])(**kwargs), indent=2, default=str))
