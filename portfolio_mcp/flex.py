"""IBKR Flex Web Service v3 client (read-only: downloads statements, nothing else)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
from lxml import etree

log = logging.getLogger(__name__)

SEND_REQUEST_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
USER_AGENT = "portfolio-mcp/1.0"


class FlexError(Exception):
    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _root(content: bytes) -> etree._Element:
    try:
        root = etree.fromstring(content, etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True))
    except etree.XMLSyntaxError:
        root = None
    if root is None:
        raise FlexError(f"Non-XML response from IBKR: {content[:200].decode('utf-8', 'replace')!r}")
    return root


def _err(root: etree._Element) -> tuple[str | None, str]:
    code = (root.findtext(".//ErrorCode") or "").strip() or None
    msg = (root.findtext(".//ErrorMessage") or "").strip() or f"Status={root.findtext('.//Status')}"
    return code, msg


class FlexClient:
    def __init__(self, http: httpx.Client | None = None, *, sleep: Callable[[float], None] = time.sleep,
                 poll_budget_s: float = 90.0, send_url: str = SEND_REQUEST_URL) -> None:
        self.http = http or httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0), follow_redirects=True)
        self.http.headers["User-Agent"] = USER_AGENT
        self.sleep = sleep
        self.poll_budget = poll_budget_s
        self.send_url = send_url

    def _get(self, url: str, params: dict[str, str], token: str) -> bytes:
        try:
            r = self.http.get(url, params=params)
        except httpx.HTTPError as e:
            raise FlexError(f"HTTP error: {type(e).__name__}: {str(e).replace(token, '***')}") from None
        if r.status_code != 200:
            raise FlexError(f"HTTP {r.status_code} from IBKR")
        return r.content

    def send_request(self, token: str, query_id: str, fd: str | None = None, td: str | None = None) -> tuple[str, str]:
        params = {"t": token, "q": query_id, "v": "3"}
        if fd and td:
            params |= {"fd": fd, "td": td}
        for attempt in range(4):
            root = _root(self._get(self.send_url, params, token))
            ref, url = (root.findtext(".//ReferenceCode") or "").strip(), (root.findtext(".//Url") or "").strip()
            if (root.findtext(".//Status") or "").strip() == "Success" and ref and url:
                return ref, url
            code, msg = _err(root)
            if code == "1018" and attempt < 3:
                wait = 10.0 * (attempt + 1)
                log.warning("rate limited (1018) on SendRequest; waiting %.0fs", wait)
                self.sleep(wait)
                continue
            raise FlexError(f"SendRequest failed: ErrorCode {code}: {msg}", code)
        raise FlexError("SendRequest failed: rate limited", "1018")

    def get_statement(self, token: str, ref: str, url: str) -> bytes:
        params = {"t": token, "q": ref, "v": "3"}
        waited, delay, rl = 0.0, 2.0, 0
        while True:
            content = self._get(url, params, token)
            root = _root(content)
            if etree.QName(root).localname == "FlexQueryResponse":
                return content
            code, msg = _err(root)
            if code == "1019" and waited < self.poll_budget:
                wait = min(delay, self.poll_budget - waited)
                log.info("statement generating (1019); retry in %.0fs", wait)
                self.sleep(wait)
                waited, delay = waited + wait, delay * 2
                continue
            if code == "1018" and rl < 3:
                rl += 1
                log.warning("rate limited (1018) on GetStatement; waiting %ds", 10 * rl)
                self.sleep(10.0 * rl)
                continue
            raise FlexError(f"GetStatement failed: ErrorCode {code}: {msg}", code)

    def fetch(self, token: str, query_id: str, fd: str | None = None, td: str | None = None) -> bytes:
        ref, url = self.send_request(token, query_id, fd, td)
        return self.get_statement(token, ref, url)


def save_raw(raw_dir: Path, label: str, content: bytes, fd: str | None = None, td: str | None = None) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{label}_{ts}.xml" if not fd else f"{label}_{ts}_{fd}-{td}.xml"
    path = raw_dir / name
    n = 1
    while path.exists():
        path = raw_dir / name.replace(".xml", f"_{n}.xml")
        n += 1
    path.write_bytes(content)
    path.chmod(0o600)
    return path
