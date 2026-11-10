from contextlib import closing
from datetime import date

import pytest

from portfolio_mcp import dashboard as dash
from portfolio_mcp.parse import DateParser, detect_slash_order, num, parse_flex
from portfolio_mcp.server import build_server
from portfolio_mcp.service import PortfolioService, chunks
from tests.conftest import FIXTURE, TOKENS, FakeIbkr, make_service


def count(svc: PortfolioService, sql: str) -> int:
    with closing(svc.db.connect()) as c:
        return c.execute(sql).fetchone()[0]


# --------------------------------------------------------------------------- parsing


def test_values_and_dates() -> None:
    assert num("3,674") == 3674 and num("--") is None and num("") is None and num("(1.5)") == -1.5
    dp = DateParser("DMY")
    assert dp.date("03/08/2026") == "2026-08-03" and dp.date("20260803") == "2026-08-03"
    assert dp.datetime("05/08/2026;202000") == "2026-08-05 20:20:00"
    assert dp.datetime("20260805;202000") == "2026-08-05 20:20:00"
    assert detect_slash_order(b'x="25/08/2026"') == "DMY" and detect_slash_order(b'x="08/25/2026"') == "MDY"
    assert detect_slash_order(b'<FlexStatement toDate="02/11/2026">', date(2026, 11, 3)) == "DMY"


def test_parse_fixture() -> None:
    pf = parse_flex(FIXTURE)
    assert pf.account_ids == {"U9000001"} and len(pf.statements) == 4
    c = pf.counts()
    assert c["trades"] == 8 and c["cash_transactions"] == 10 and c["change_in_nav"] == 4
    cash = next(r for r in pf.tables["trades"] if r["asset_category"] == "CASH")
    assert cash["ib_commission"] == -2 and cash["ib_commission_currency"] == "USD" and cash["currency"] == "AED"
    put = next(r for r in pf.tables["trades"] if r["trade_id"] == "8003")
    assert (put["put_call"], put["strike"], put["expiry"], put["multiplier"], put["underlying_symbol"]) == \
        ("P", 100, "2026-09-18", 100, "XYZ")
    assert "elements" in c  # unmodelled sections are still stored


def test_chunks() -> None:
    cs = chunks(date(2024, 1, 1), date(2026, 3, 1))
    assert cs[0] == (date(2024, 1, 1), date(2024, 12, 30)) and cs[-1][1] == date(2026, 3, 1)
    assert all((b - a).days < 365 for a, b in cs)


# --------------------------------------------------------------------------- refresh


def test_first_refresh_backfills_from_inception(service: PortfolioService, fake: FakeIbkr, sleeps: list[float]) -> None:
    r = service.refresh("options")
    assert r["status"] == "ok", r
    sends = [q for q in fake.requests if q.url.path.endswith("SendRequest")]
    assert "fd" not in sends[0].url.params  # discovery pull
    assert (sends[1].url.params["fd"], sends[1].url.params["td"]) == ("20260802", "20260806")  # dateOpened-1 .. yesterday
    assert all(q.headers["User-Agent"] == "portfolio-mcp/1.0" for q in fake.requests)
    assert 2 in sleeps  # 1019 backoff
    files = sorted(p.name for p in service.config.raw_dir.glob("*.xml"))
    assert len(files) == 2 and files[0].startswith("options_2026") and "_20260802-20260806" in files[1]
    assert TOKENS["options"] not in str(r)


def test_refresh_steps_back_when_newest_statement_unpublished(loaded: PortfolioService, fake: FakeIbkr) -> None:
    fake.unpublished = {"20260806"}  # yesterday's statement is not out yet (the 1 AM Dubai case)
    fake.requests.clear()
    r = loaded.refresh("options")
    assert r["status"] == "ok", r
    assert [q.url.params["td"] for q in fake.requests if q.url.path.endswith("SendRequest")] == ["20260806", "20260805"]


def test_refresh_gives_up_after_stepping_back(loaded: PortfolioService, fake: FakeIbkr) -> None:
    fake.unpublished = {f"202608{d:02d}" for d in range(1, 7)}
    r = loaded.refresh("options")
    assert r["status"] == "failed" and "1003" in r["results"][0]["error"]


def test_refresh_idempotent(loaded: PortfolioService) -> None:
    before = {t: count(loaded, f"SELECT COUNT(*) FROM {t}") for t in ("trades", "cash_transactions", "nav", "change_in_nav")}
    assert loaded.refresh()["status"] == "ok"  # incremental re-pull
    after = {t: count(loaded, f"SELECT COUNT(*) FROM {t}") for t in before}
    assert before == after == {"trades": 16, "cash_transactions": 20, "nav": 10, "change_in_nav": 8}


def test_bad_token_surfaces_error(tmp_path, fake: FakeIbkr, sleeps) -> None:
    svc = make_service(tmp_path, fake, sleeps)
    object.__setattr__(svc.config.accounts[0], "token", "wrong-token-123")
    r = svc.refresh("options")
    assert r["status"] == "failed" and "1012" in r["results"][0]["error"] and "wrong-token-123" not in str(r)


# --------------------------------------------------------------------------- metrics + reconciliation


def test_metrics(loaded: PortfolioService) -> None:
    m = loaded.metrics("options")[0]
    assert round(m.deposited, 2) == 1000.28  # AED 3,674 * 0.27226, DETAIL rows only
    assert round(m.current_value, 2) == 1026.58 and round(m.net_pnl, 2) == 26.30
    assert round(m.realized, 2) == 25.20  # 18 + 48.60 - 41.40; CLOSED_LOT and CASH excluded
    assert m.unrealized == 0
    assert round(m.dividends, 2) == 4.25 and round(m.interest, 2) == 0.10 and round(m.other_fees, 2) == -1.00
    assert round(m.fx_conversion, 2) == -2.25
    assert round(m.trade_commissions, 2) == -6.80  # $2 FX commission counted in USD, not * 0.27226
    assert round(m.fees_paid, 2) == 7.80
    assert abs(m.residual) < 0.005 and m.reconciled and m.from_inception
    assert all(c.ok for c in m.checks)
    assert m.history_from == "2026-08-03" and m.nav_date == "2026-08-06"


def test_reconciliation_failure_is_loud(tmp_path, sleeps) -> None:
    bad = FIXTURE.replace(b'depositsWithdrawals="1000.28324"', b'depositsWithdrawals="1200"')
    fake = FakeIbkr({"Q-options": bad, "Q-gold": FIXTURE.replace(b"U9000001", b"U9000002")})
    svc = make_service(tmp_path, fake, sleeps)
    r = svc.refresh("options")
    res = r["results"][0]
    assert r["status"] == "failed" and res["status"] == "reconciliation_failed"
    assert "MISMATCH" in res["reconciliation_table"] and "deposited" in res["reconciliation_table"]
    assert count(svc, "SELECT COUNT(*) FROM refresh_log WHERE status='reconciliation_failed'") == 1


def test_unreconciled_residual_shown_on_dashboard(loaded: PortfolioService) -> None:
    m = loaded.metrics("gold")[0]
    m.residual = 12.34
    lines = dict(dash.breakdown(m))
    assert lines["Unreconciled"] == 12.34 and "Other" not in lines


# --------------------------------------------------------------------------- tools


def test_tools(loaded: PortfolioService) -> None:
    acc = loaded.get_accounts()
    assert acc["all_reconciled"] and acc["combined"]["deposited"] == 2000.57
    pos = loaded.get_positions("options")
    assert pos["positions"] == []  # flat at the latest statement
    t = loaded.get_trades("options", symbol="XYZ", asset_class="OPT")["trades"]
    assert len(t) == 4 and t[0]["date_time"] >= t[-1]["date_time"]
    rt = {g["order_id"]: g for g in loaded.get_round_trips("options")["round_trips"]}
    assert rt["11"]["strategy"] == "vertical put spread" and rt["11"]["net_cash_usd"] == 98.6 and rt["11"]["debit_credit"] == "credit"
    assert rt["12"]["net_realized_usd"] == 7.2 and rt["12"]["net_cash_usd"] == -91.4
    cf = loaded.get_cash_flows("options", "deposit")
    assert cf["count"] == 1 and cf["cash_flows"][0]["currency"] == "AED"
    d = loaded.get_dividends("options", 2026)
    assert d["received"][0] == {"symbol": "XYZ", "gross_usd": 5.0, "withholding_usd": -0.75, "net_usd": 4.25, "payments": 1}
    p = loaded.get_pnl("options")
    assert p["totals"]["net"] == 26.3 and p["totals"]["realized"] == 25.2
    perf = loaded.get_performance("options")
    assert perf["twr_pct"]["inception"] == pytest.approx(2.8607, abs=1e-3)  # 1.017635 * 1.011668 * 0.999124 - 1
    assert perf["max_drawdown_pct"] == pytest.approx(-0.0876, abs=1e-3)
    assert loaded.get_performance("all")["latest_nav_usd"] == 2053.17


def test_query_sql_read_only(loaded: PortfolioService) -> None:
    r = loaded.query_sql("SELECT json_extract(raw_json,'$.ibOrderID') o FROM trades WHERE trade_id='8003' LIMIT 1")
    assert r["rows"] == [{"o": "11"}]
    for bad in ("DELETE FROM trades", "SELECT 1; DROP TABLE trades", "PRAGMA table_info(trades)",
                "WITH x AS (SELECT 1) UPDATE trades SET quantity=0", "ATTACH 'x.db' AS x"):
        with pytest.raises(ValueError):
            loaded.query_sql(bad)


def test_server_tools(service: PortfolioService) -> None:
    import anyio

    names = {t.name for t in anyio.run(build_server(service).list_tools)}
    assert names == {"refresh", "refresh_status", "get_accounts", "get_positions", "get_trades", "get_round_trips",
                     "get_cash_flows", "get_dividends", "get_pnl", "get_performance", "build_dashboard", "query_sql"}


# --------------------------------------------------------------------------- dashboard


def test_dashboard(loaded: PortfolioService, tmp_path) -> None:
    path = dash.write(loaded.metrics(), tmp_path / "d.html")
    doc = path.read_text()
    assert doc.index(">GOLD<") < doc.index(">OPTIONS<")
    for s in ("$1,000.28", "$1,026.58", "+$26.30", "+2.63%", "+$25.20", "+$4.25", "-$2.25", "$7.80",
              "commissions already included in realized P&amp;L", "history from 2026-08-03", "NAV as of 2026-08-06"):
        assert s in doc, s
    assert "Other<" not in doc and "Unreconciled" not in doc
    low = doc.lower()
    assert not any(x in low for x in ("<script", "http", "color:#f", "red", "green"))
    assert "border-left:1px solid #000" in doc and "max-width:700px" in doc
    assert dash.money(-939.06, True) == "-$939.06" and dash.money(1234.56, True) == "+$1,234.56"


# --------------------------------------------------------------------------- manual account (Capital.com)


def test_manual_account(tmp_path, fake: FakeIbkr, sleeps) -> None:
    import dataclasses
    import json as _json

    from portfolio_mcp.config import load_manual

    f = tmp_path / "manual_accounts.json"
    f.write_text(_json.dumps([{"label": "capital.com", "title": "CAPITAL.COM", "currency": "USD",
                               "deposited": 4931, "current_value": 4514, "as_of": "2026-11-03"}]))
    svc = make_service(tmp_path, fake, sleeps)
    svc.config = dataclasses.replace(svc.config, manual=load_manual(f))
    assert svc.refresh()["status"] == "ok"  # refresh ignores manual accounts
    ms = {m.label: m for m in svc.metrics()}
    cap = ms["capital.com"]
    assert cap.is_manual and not cap.reconciled
    assert (cap.deposited, cap.current_value, cap.net_pnl, cap.realized, cap.unrealized) == (4931, 4514, -417, -417, 0)
    assert round(cap.net_pct, 2) == -8.46
    acc = svc.get_accounts()
    assert acc["all_reconciled"] and acc["manual_accounts"] == ["capital.com"]
    assert acc["combined"]["deposited"] == round(2 * 1000.28324 + 4931, 2)
    assert [m.label for m in svc.metrics("capital.com")] == ["capital.com"]
    doc = dash.render(list(ms.values()))
    assert doc.index(">GOLD<") < doc.index(">OPTIONS<") < doc.index(">CAPITAL.COM<")
    assert "--cols:3" in doc and "-$417.00" in doc and "-8.46%" in doc and "Manual entry · balance as of 2026-11-03" in doc
    cap_html = doc[doc.index(">CAPITAL.COM<"):]
    assert "Unrealized" not in cap_html and "Realized P&amp;L" in cap_html
    with pytest.raises(ValueError):
        f.write_text(_json.dumps([{"label": "x", "deposited": 1, "current_value": 1, "as_of": "2026-01-01", "currency": "AED"}]))
        load_manual(f)


def test_return_percentages(loaded: PortfolioService, tmp_path) -> None:
    m = loaded.metrics("options")[0]
    assert round(m.trading_pnl, 2) == 25.20  # realized 25.20 + unrealized 0 (dividends/fees/fx excluded)
    assert round(m.trading_pct, 2) == round(25.20 / 1000.28324 * 100, 2) == 2.52
    assert round(m.realized_pct, 2) == 2.52 and round(m.net_pct, 2) == 2.63  # net includes dividends etc.
    d = m.to_dict()
    assert d["realized_plus_unrealized_pct"] == 2.52 and d["realized_pct"] == 2.52
    c = loaded.get_accounts()["combined"]
    assert c["realized_plus_unrealized_pct"] == 2.52 and c["realized_pct"] == 2.52
    doc = dash.write(loaded.metrics(), tmp_path / "d.html").read_text()
    assert "Return on deposited" not in doc  # per-column block replaced by one portfolio total
    total = doc[doc.index('class="total"'):]
    assert "Total net · realized + unrealized" in total and "Total realized" in total
    assert total.count("+$50.40") == 2 and total.count("+2.52%") == 2  # both accounts: 2 x 25.20 on 2 x 1,000.28
    assert "all-in net P&amp;L incl. dividends, interest &amp; fees +$52.60 (+2.63%)" in total


def test_positions_block_on_dashboard(loaded: PortfolioService) -> None:
    m = loaded.metrics("options")[0]
    assert m.positions == [] and dash.positions_block(m) == ""  # flat account: no block
    m.positions = [
        {"symbol": "IAUI", "qty": 68.7967, "avg_cost": 54.2832, "mark": 45.55, "cost": 3734.50, "value": 3133.69,
         "unrealized": -600.81, "unrealized_pct": -16.09, "share_of_unrealized": 66.1},
        {"symbol": "GLD", "qty": 7.1826, "avg_cost": 404.6251, "mark": 361.65, "cost": 2906.26, "value": 2597.59,
         "unrealized": -308.67, "unrealized_pct": -10.62, "share_of_unrealized": 33.9}]
    html_ = dash.section("GOLD", m)
    block = html_[html_.index("Unrealized by holding"):]
    for s in (">IAUI<", ">GLD<", "68.7967", "$54.28", "$45.55", "$3,734.50", "$3,133.69", "-$600.81", "-16.09%",
              "66.1%", "$404.63", "$361.65", "-$308.67", "-10.62%", "33.9%", "Avg cost", "break-even price"):
        assert s in block, s
    assert html_.index("Unrealized by holding") < html_.index("Cumulative fees paid")
