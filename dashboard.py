"""Regenerate ./dashboard.html from the current cache (no refresh), open it, and print the numbers.

    .venv/bin/python dashboard.py [--refresh] [--no-open]
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio_mcp.config import load_config  # noqa: E402
from portfolio_mcp.dashboard import summary_text, write  # noqa: E402
from portfolio_mcp.metrics import table  # noqa: E402
from portfolio_mcp.server import setup_logging  # noqa: E402
from portfolio_mcp.service import PortfolioService  # noqa: E402


def main() -> int:
    setup_logging()
    service = PortfolioService(load_config())
    if "--refresh" in sys.argv:
        r = service.refresh()
        for x in r["results"]:
            print(f"refresh {x['account']}: {x['status']}" + (f" ({x['error']})" if x["error"] else ""))
    metrics = service.metrics()
    path = write(metrics)
    print("\nRECONCILIATION (IBKR accounts)\n" + table([m for m in metrics if not m.is_manual]))
    print(f"\nDASHBOARD  {path}\n" + summary_text(metrics))
    dep = sum(m.deposited for m in metrics)
    tr, rl, nt = (sum(m.trading_pnl for m in metrics), sum(m.realized for m in metrics), sum(m.net_pnl for m in metrics))
    print(f"\nTOTAL  deposited ${dep:,.2f}")
    print(f"  NET (realized + unrealized)  {tr:+,.2f}  ({tr / dep * 100:+.2f}%)")
    print(f"  REALIZED                     {rl:+,.2f}  ({rl / dep * 100:+.2f}%)")
    print(f"  all-in net P&L (incl. dividends, interest, fees)  {nt:+,.2f}  ({nt / dep * 100:+.2f}%)")
    if "--no-open" not in sys.argv and sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    return 0 if all(m.reconciled for m in metrics if not m.is_manual) else 1


if __name__ == "__main__":
    raise SystemExit(main())
