from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from portfolio_mcp.config import AccountConfig, Config
from portfolio_mcp.flex import FlexClient
from portfolio_mcp.service import PortfolioService

FIXTURE = (Path(__file__).parent / "fixtures" / "flex_daily.xml").read_bytes()
TOKENS = {"options": "111122223333444455556666", "gold": "999988887777666655554444"}
QUERIES = {"Q-options": b"U9000001", "Q-gold": b"U9000002"}  # gold = same fixture, other account id


class FakeIbkr:
    def __init__(self, content_by_query: dict[str, bytes] | None = None, pending: int = 1) -> None:
        self.content = content_by_query or {q: FIXTURE.replace(b"U9000001", aid) for q, aid in QUERIES.items()}
        self.pending = pending
        self.polls: dict[str, int] = {}
        self.requests: list[httpx.Request] = []
        self.unpublished: set[str] = set()  # td values rejected with 1003 (statement not ready yet)

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        p = req.url.params
        if req.url.path.endswith("/SendRequest"):
            if p.get("t") not in TOKENS.values():
                return httpx.Response(200, content=b"<FlexStatementResponse><Status>Fail</Status><ErrorCode>1012"
                                                   b"</ErrorCode><ErrorMessage>Token has expired.</ErrorMessage></FlexStatementResponse>")
            if p.get("td") in self.unpublished:
                return httpx.Response(200, content=b"<FlexStatementResponse><Status>Fail</Status><ErrorCode>1003"
                                                   b"</ErrorCode><ErrorMessage>Statement is not available.</ErrorMessage></FlexStatementResponse>")
            ref = f"{p['q']}~{p.get('fd', '')}~{p.get('td', '')}"
            return httpx.Response(200, content=f"<FlexStatementResponse><Status>Success</Status><ReferenceCode>{ref}"
                                               f"</ReferenceCode><Url>https://x.test/GetStatement</Url></FlexStatementResponse>".encode())
        ref = p["q"]
        n = self.polls[ref] = self.polls.get(ref, 0) + 1
        if n <= self.pending:
            return httpx.Response(200, content=b"<FlexStatementResponse><Status>Warn</Status><ErrorCode>1019</ErrorCode>"
                                               b"<ErrorMessage>Statement generation in progress.</ErrorMessage></FlexStatementResponse>")
        return httpx.Response(200, content=self.content[ref.split("~")[0]])


@pytest.fixture
def fake() -> FakeIbkr:
    return FakeIbkr()


@pytest.fixture
def sleeps() -> list[float]:
    return []


def make_service(tmp_path: Path, fake: FakeIbkr, sleeps: list[float]) -> PortfolioService:
    cfg = Config(tuple(AccountConfig(lbl, tok, f"Q-{lbl}") for lbl, tok in TOKENS.items()), "USD",
                 tmp_path / "portfolio.db", tmp_path / "raw")
    client = FlexClient(httpx.Client(transport=httpx.MockTransport(fake)), sleep=sleeps.append)
    return PortfolioService(cfg, client, today=lambda: date(2026, 8, 7))


@pytest.fixture
def service(tmp_path: Path, fake: FakeIbkr, sleeps: list[float]) -> PortfolioService:
    return make_service(tmp_path, fake, sleeps)


@pytest.fixture
def loaded(service: PortfolioService) -> PortfolioService:
    r = service.refresh()
    assert r["status"] == "ok", r
    return service
