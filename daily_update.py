"""Daily job: refresh IBKR Flex data, then rebuild dashboard.html and pie.html (without opening them).

Run by launchd at 01:00 and 11:00 Dubai time (see scripts/com.portfolio-mcp.daily.plist). The 11:00 run
exists because IBKR publishes the day that just closed some hours after 01:00. Manual run:

    .venv/bin/python daily_update.py

Output goes to logs/daily.log. No secrets are logged.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from portfolio_mcp.config import load_config  # noqa: E402
from portfolio_mcp.server import setup_logging  # noqa: E402
from portfolio_mcp.service import PortfolioService  # noqa: E402


def main() -> int:
    setup_logging()
    print(f"=== daily update {datetime.now().astimezone().isoformat(timespec='seconds')} ===", flush=True)
    r = PortfolioService(load_config()).refresh()
    for x in r["results"]:
        print(f"refresh {x['account']}: {x['status']}" + (f" ({x['error']})" if x.get("error") else ""), flush=True)
    rc = 0 if r["status"] == "ok" else 1
    # Rebuild both pages even if a refresh failed, so they show the latest cached data.
    for script in ("dashboard.py", "pie.py"):
        p = subprocess.run([sys.executable, str(ROOT / script), "--no-open"], cwd=ROOT)
        print(f"{script}: exit {p.returncode}", flush=True)
        rc = rc or p.returncode
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
