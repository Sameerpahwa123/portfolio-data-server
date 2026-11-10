"""Phase 1: download the configured Flex query for each account into ./raw (no parsing, no DB).

    .venv/bin/python scripts/fetch_raw.py [label ...] [--fd YYYYMMDD --td YYYYMMDD]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_mcp.config import load_config  # noqa: E402
from portfolio_mcp.flex import FlexClient, FlexError, save_raw  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--fd")
    ap.add_argument("--td")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx logs URLs, which contain the token
    cfg = load_config()
    client = FlexClient()
    rc = 0
    for i, acc in enumerate(a for a in cfg.accounts if not args.labels or a.label in args.labels):
        if i:
            client.sleep(2)
        try:
            content = client.fetch(acc.token, acc.query_id, args.fd, args.td)
            print(f"{acc.label}: saved {save_raw(cfg.raw_dir, acc.label, content, args.fd, args.td)} ({len(content):,} bytes)")
        except FlexError as e:
            print(f"{acc.label}: ERROR {cfg.redact(str(e))}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
