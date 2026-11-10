"""Read tools over the cache. Money is USD (account base): amount * fxRateToBase per row."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from .metrics import (COMMISSION_ADJ, DEPOSIT, DIVIDEND_TYPES, INTEREST_TYPES, OTHER_FEES, WITHHOLDING, _cash_rows,
                      commission_usd, tile)
from .parse import DateParser

OPT = {"OPT", "FOP"}


@dataclass(frozen=True)
class Acct:
    account_id: str
    label: str
    history_to: str | None


def r2(v: float | None) -> float | None:
    return None if v is None else round(v, 2)


def norm_date(v: str | None) -> str | None:
    if not v:
        return None
    d = DateParser("DMY").date(str(v))
    if d is None:
        raise ValueError(f"invalid date {v!r}: use YYYY-MM-DD")
    return d


def _fx(r: sqlite3.Row) -> float:
    return r["fx_rate_to_base"] if r["fx_rate_to_base"] is not None else 1.0


def _in(dt: str | None, start: str | None, end: str | None) -> bool:
    d = (dt or "")[:10]
    return (not start or d >= start) and (not end or d <= end)


def _opt(r: sqlite3.Row) -> dict[str, Any]:
    if r["asset_category"] not in OPT:
        return {}
    return {"right": {"C": "CALL", "P": "PUT"}.get(r["put_call"] or "", r["put_call"]), "strike": r["strike"],
            "expiry": r["expiry"], "multiplier": r["multiplier"]}


# ----------------------------------------------------------------------------- positions


def latest_positions(conn: sqlite3.Connection, accts: list[Acct]) -> list[tuple[Acct, sqlite3.Row]]:
    out = []
    for a in accts:
        for r in conn.execute("SELECT * FROM open_positions WHERE account_id=? AND report_date=? AND "
                              "COALESCE(level_of_detail,'SUMMARY')='SUMMARY' AND COALESCE(position,0)!=0",
                              (a.account_id, a.history_to)):
            out.append((a, r))
    return out


def positions(conn: sqlite3.Connection, accts: list[Acct]) -> dict[str, Any]:
    items = []
    for a, r in latest_positions(conn, accts):
        cost, unr = r["cost_basis_money"], r["fifo_pnl_unrealized"]
        items.append({"account": a.label, "symbol": r["symbol"], "underlying": r["underlying_symbol"] or r["symbol"],
                      "asset_class": r["asset_category"], "currency": r["currency"], "qty": r["position"],
                      "avg_cost": r["cost_basis_price"], "mark": r["mark_price"],
                      "value_usd": r2((r["position_value"] or 0) * _fx(r)),
                      "unrealized_usd": r2((unr or 0) * _fx(r)),
                      "unrealized_pct": r2(unr / abs(cost) * 100) if unr is not None and cost else None,
                      "as_of": r["report_date"], **_opt(r)})
    items.sort(key=lambda p: -abs(p["value_usd"] or 0))
    return {"positions": items, "total_value_usd": r2(sum(p["value_usd"] or 0 for p in items)),
            "total_unrealized_usd": r2(sum(p["unrealized_usd"] or 0 for p in items))}


# ----------------------------------------------------------------------------- trades


def _execs(conn: sqlite3.Connection, accts: list[Acct]) -> list[tuple[Acct, sqlite3.Row]]:
    out = []
    for a in accts:
        out += [(a, r) for r in conn.execute("SELECT * FROM trades WHERE account_id=? AND level_of_detail='EXECUTION'",
                                              (a.account_id,))]
    return out


def trades(conn: sqlite3.Connection, accts: list[Acct], start: str | None, end: str | None,
           symbol: str | None, asset_class: str | None) -> dict[str, Any]:
    items = []
    for a, r in _execs(conn, accts):
        if not _in(r["date_time"] or r["trade_date"], start, end):
            continue
        if symbol and symbol.upper() not in {(r["symbol"] or "").upper(), (r["underlying_symbol"] or "").upper()}:
            continue
        if asset_class and (r["asset_category"] or "").upper() != asset_class.upper():
            continue
        items.append({"account": a.label, "date_time": r["date_time"], "symbol": r["symbol"],
                      "underlying": r["underlying_symbol"] or r["symbol"], "asset_class": r["asset_category"],
                      "buy_sell": r["buy_sell"], "open_close": r["open_close"], "qty": r["quantity"],
                      "price": r["trade_price"], "currency": r["currency"],
                      "proceeds_usd": r2((r["proceeds"] or 0) * _fx(r)), "commission_usd": r2(commission_usd(r)),
                      "realized_usd": r2((r["fifo_pnl_realized"] or 0) * _fx(r)), "order_id": r["order_id"],
                      "trade_id": r["trade_id"], **_opt(r)})
    items.sort(key=lambda t: t["date_time"] or "", reverse=True)
    return {"trades": items, "count": len(items),
            "realized_usd": r2(sum(t["realized_usd"] or 0 for t in items if t["asset_class"] != "CASH")),
            "commissions_usd": r2(sum(t["commission_usd"] or 0 for t in items))}


def _strategy(legs: list[dict[str, Any]]) -> str:
    opts = [leg for leg in legs if leg["right"]]
    if not opts:
        return "stock" if legs and legs[0]["asset_class"] == "STK" else (legs[0]["asset_class"] or "?").lower()
    if len(opts) < len(legs):
        return f"stock + {len(opts)}-leg option"
    rights, exps, strikes = {o["right"] for o in opts}, {o["expiry"] for o in opts}, {o["strike"] for o in opts}
    side = lambda o: "long" if o["qty"] > 0 else "short"  # noqa: E731
    if len(opts) == 1:
        return f"{side(opts[0])} {opts[0]['right'].lower()}"
    if len(opts) == 2 and len(rights) == 1:
        return f"vertical {next(iter(rights)).lower()} spread" if len(exps) == 1 else (
            f"calendar {next(iter(rights)).lower()} spread" if len(strikes) == 1 else f"diagonal {next(iter(rights)).lower()} spread")
    if len(opts) == 2:
        return ("straddle" if len(strikes) == 1 else "strangle") if side(opts[0]) == side(opts[1]) else "risk reversal"
    if len(opts) == 4 and len(rights) == 2 and len(exps) == 1:
        return "iron condor"
    return f"{len(opts)}-leg combo"


def round_trips(conn: sqlite3.Connection, accts: list[Acct]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[tuple[Acct, sqlite3.Row]]] = defaultdict(list)
    for a, r in _execs(conn, accts):
        if r["asset_category"] == "CASH":
            continue
        key = (a.account_id, "o", r["order_id"]) if r["order_id"] else (
            a.account_id, "t", r["date_time"] or "", r["underlying_symbol"] or r["symbol"] or "")
        groups[key].append((a, r))
    out = []
    for rows in groups.values():
        legs: dict[str, dict[str, Any]] = {}
        for _, r in rows:
            leg = legs.setdefault(r["conid"] or r["symbol"], {
                "symbol": r["symbol"], "asset_class": r["asset_category"],
                "right": {"C": "CALL", "P": "PUT"}.get(r["put_call"] or "") if r["asset_category"] in OPT else None,
                "strike": r["strike"], "expiry": r["expiry"], "qty": 0.0, "px_qty": 0.0})
            leg["qty"] += r["quantity"] or 0
            leg["px_qty"] += (r["trade_price"] or 0) * abs(r["quantity"] or 0)
        legs_out = []
        for leg in legs.values():
            pq = leg.pop("px_qty")
            leg["avg_price"] = round(pq / abs(leg["qty"]), 4) if leg["qty"] else None
            legs_out.append(leg)
        legs_out.sort(key=lambda x: (x["expiry"] or "", x["right"] or "", x["strike"] or 0))
        net = sum((r["proceeds"] or 0) * _fx(r) + commission_usd(r) for _, r in rows)
        realized = sum((r["fifo_pnl_realized"] or 0) * _fx(r) for _, r in rows)
        a0, r0 = rows[0]
        ocs = sorted({r["open_close"] for _, r in rows if r["open_close"]})
        out.append({"account": a0.label, "date_time": min(r["date_time"] or "" for _, r in rows),
                    "underlying": r0["underlying_symbol"] or r0["symbol"], "order_id": r0["order_id"],
                    "strategy": _strategy(legs_out), "open_close": "/".join(ocs) or None, "legs": legs_out,
                    "net_cash_usd": r2(net), "debit_credit": "credit" if net > 0 else "debit",
                    "net_realized_usd": r2(realized)})
    out.sort(key=lambda g: g["date_time"], reverse=True)
    return {"round_trips": out, "count": len(out),
            "note": "net_cash = proceeds + commissions; realized is IBKR FIFO, net of commissions"}


# ----------------------------------------------------------------------------- cash


_ALIASES = {"deposit": [DEPOSIT], "deposits": [DEPOSIT], "withdrawal": [DEPOSIT], "dividend": list(DIVIDEND_TYPES),
            "dividends": list(DIVIDEND_TYPES), "withholding": [WITHHOLDING], "interest": list(INTEREST_TYPES),
            "fee": [OTHER_FEES], "fees": [OTHER_FEES], "commission": [COMMISSION_ADJ]}


def cash_flows(conn: sqlite3.Connection, accts: list[Acct], type_: str | None, start: str | None,
               end: str | None) -> dict[str, Any]:
    want = _ALIASES.get((type_ or "").lower(), [type_] if type_ else None)
    items, totals = [], defaultdict(float)
    for a in accts:
        for r in _cash_rows(conn, a.account_id):
            if want and r["type"] not in want:
                continue
            if type_ and type_.lower() == "withdrawal" and (r["amount"] or 0) >= 0:
                continue
            if type_ and type_.lower() in ("deposit", "deposits") and (r["amount"] or 0) < 0:
                continue
            if not _in(r["date_time"] or r["report_date"], start, end):
                continue
            v = (r["amount"] or 0) * _fx(r)
            totals[r["type"]] += v
            items.append({"account": a.label, "date": (r["date_time"] or r["report_date"] or "")[:10], "type": r["type"],
                          "symbol": r["symbol"], "description": r["description"], "amount": r["amount"],
                          "currency": r["currency"], "fx_rate_to_base": r["fx_rate_to_base"], "amount_usd": r2(v)})
    items.sort(key=lambda x: x["date"], reverse=True)
    return {"cash_flows": items, "count": len(items), "totals_usd": {k: r2(v) for k, v in sorted(totals.items())}}


def dividends(conn: sqlite3.Connection, accts: list[Acct], year: int | None) -> dict[str, Any]:
    by: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for a in accts:
        for r in _cash_rows(conn, a.account_id):
            if r["type"] not in (*DIVIDEND_TYPES, WITHHOLDING):
                continue
            d = (r["date_time"] or r["report_date"] or "")[:10]
            if year and not d.startswith(str(year)):
                continue
            k = "withholding" if r["type"] == WITHHOLDING else "gross"
            by[r["symbol"] or "?"][k] += (r["amount"] or 0) * _fx(r)
            if k == "gross":
                by[r["symbol"] or "?"]["payments"] += 1
    received = sorted(({"symbol": s, "gross_usd": r2(v["gross"]), "withholding_usd": r2(v["withholding"]),
                        "net_usd": r2(v["gross"] + v["withholding"]), "payments": int(v["payments"])} for s, v in by.items()),
                      key=lambda x: -(x["gross_usd"] or 0))
    accrued = []
    for a in accts:
        for r in conn.execute("SELECT * FROM dividend_accruals WHERE account_id=? AND stmt_to=?", (a.account_id, a.history_to)):
            if year and not (r["pay_date"] or "").startswith(str(year)):
                continue
            accrued.append({"account": a.label, "symbol": r["symbol"], "ex_date": r["ex_date"], "pay_date": r["pay_date"],
                            "quantity": r["quantity"], "gross_usd": r2((r["gross_amount"] or 0) * _fx(r)),
                            "tax_usd": r2(-abs(r["tax"] or 0) * _fx(r)), "net_usd": r2((r["net_amount"] or 0) * _fx(r))})
    return {"year": year, "received": received,
            "received_total_usd": {k: r2(sum(x[f"{k}_usd"] or 0 for x in received)) for k in ("gross", "withholding", "net")},
            "accrued": accrued, "accrued_net_usd": r2(sum(x["net_usd"] or 0 for x in accrued))}


# ----------------------------------------------------------------------------- P&L


def pnl(conn: sqlite3.Connection, accts: list[Acct], start: str | None, end: str | None, group_by: str) -> dict[str, Any]:
    if group_by not in ("symbol", "asset_class", "month"):
        raise ValueError("group_by must be symbol, asset_class or month")
    g: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def key(sym: str | None, ac: str | None, d: str | None) -> str:
        return {"symbol": sym or "(cash)", "asset_class": ac or "CASH", "month": (d or "")[:7] or "?"}[group_by]

    for _, r in _execs(conn, accts):
        d = r["date_time"] or r["trade_date"]
        if not _in(d, start, end):
            continue
        k = key(r["underlying_symbol"] or r["symbol"], r["asset_category"], d)
        if r["asset_category"] != "CASH":
            g[k]["realized"] += (r["fifo_pnl_realized"] or 0) * _fx(r)
        g[k]["commissions"] += commission_usd(r)
    fields = {**{t: "dividends" for t in DIVIDEND_TYPES}, WITHHOLDING: "withholding", OTHER_FEES: "other_fees",
              COMMISSION_ADJ: "commission_adjustments", **{t: "interest" for t in INTEREST_TYPES}}
    for a in accts:
        for r in _cash_rows(conn, a.account_id):
            d = r["date_time"] or r["report_date"]
            if r["type"] in fields and _in(d, start, end):
                ac = "STK" if r["type"] in (*DIVIDEND_TYPES, WITHHOLDING) else "CASH"
                g[key(r["symbol"], ac, d)][fields[r["type"]]] += (r["amount"] or 0) * _fx(r)
        rows = conn.execute("SELECT * FROM fifo_summary WHERE account_id=? AND asset_category='CASH' AND conid IS NOT NULL",
                            (a.account_id,)).fetchall()
        keep = tile({(r["stmt_from"], r["stmt_to"]) for r in rows})
        for r in rows:
            if (r["stmt_from"], r["stmt_to"]) in keep and _in(r["stmt_to"], start, end):
                g[key("(fx conversion)", "CASH", r["stmt_to"])]["fx_conversion"] += r["total_realized_pnl"] or 0
    for _, r in latest_positions(conn, accts):
        k = key(r["underlying_symbol"] or r["symbol"], r["asset_category"], "current unrealized")
        g[k if group_by != "month" else "current (unrealized)"]["unrealized"] += (r["fifo_pnl_unrealized"] or 0) * _fx(r)
    cols = ("realized", "unrealized", "dividends", "withholding", "interest", "other_fees", "commission_adjustments",
            "fx_conversion")
    groups = []
    for k, v in g.items():
        row = {group_by: k, **{c: r2(v.get(c, 0.0)) for c in (*cols, "commissions")}}
        row["net"] = r2(sum(v.get(c, 0.0) for c in cols))
        groups.append(row)
    groups.sort(key=(lambda x: x["month"]) if group_by == "month" else (lambda x: -abs(x["net"] or 0)))
    return {"group_by": group_by, "start": start, "end": end, "groups": groups,
            "totals": {c: r2(sum(x[c] or 0 for x in groups)) for c in (*cols, "commissions", "net")},
            "notes": ["net = realized + unrealized + dividends + withholding + interest + other fees + "
                      "commission adjustments + fx conversion; commissions are already inside realized"]}


# ----------------------------------------------------------------------------- performance


def _daily(conn: sqlite3.Connection, a: Acct) -> list[dict[str, Any]]:
    navs = {r["report_date"]: r["total"] for r in conn.execute(
        "SELECT report_date, total FROM nav WHERE account_id=? AND total IS NOT NULL", (a.account_id,))}
    cin = conn.execute("SELECT * FROM change_in_nav WHERE account_id=? AND from_date=to_date", (a.account_id,)).fetchall()
    by = {r["from_date"]: r for r in cin}
    first_funded = min((d for d, v in navs.items() if v), default=None)
    out = []
    for d in sorted(navs):
        if first_funded is None or d < first_funded:
            continue
        c = by.get(d)
        out.append({"date": d, "nav": navs[d], "flow": (c["deposits_withdrawals"] or 0) if c else 0.0,
                    "twr": (c["twr_pct"] / 100) if c and c["twr_pct"] is not None else None})
    return out


def performance(conn: sqlite3.Connection, accts: list[Acct]) -> dict[str, Any]:
    series = [_daily(conn, a) for a in accts]
    if len(accts) == 1:
        pts = series[0]
        for i, p in enumerate(pts):
            if p["twr"] is None:
                prev = pts[i - 1]["nav"] if i else 0
                den = prev + p["flow"]
                p["twr"] = (p["nav"] - prev - p["flow"]) / den if i and den > 0 else 0.0
        source = "IBKR ChangeInNAV.twr (daily)"
    else:  # combine: forward-fill NAV, sum flows, recompute returns
        dates = sorted({p["date"] for s in series for p in s})
        maps = [{p["date"]: p for p in s} for s in series]
        last = [0.0] * len(maps)
        pts = []
        for d in dates:
            flow = 0.0
            for i, m in enumerate(maps):
                if d in m:
                    late_join = not last[i] and pts  # account starts mid-series: its NAV is an inflow
                    flow += m[d]["nav"] if late_join else m[d]["flow"]
                    last[i] = m[d]["nav"]
            nav = sum(last)
            prev = pts[-1]["nav"] if pts else 0.0
            den = prev + flow
            pts.append({"date": d, "nav": nav, "flow": flow,
                        "twr": (nav - prev - flow) / den if pts and den > 0 else 0.0})
        source = "computed from combined NAV and depositsWithdrawals (flows at start of day)"
    if not pts:
        return {"error": "no NAV history cached"}
    idx, peak, mdd, peak_d, trough = 1.0, 1.0, 0.0, pts[0]["date"], None
    for p in pts:
        idx *= 1 + p["twr"]
        p["index"] = idx
        if idx > peak:
            peak, peak_d = idx, p["date"]
        if idx / peak - 1 < mdd:
            mdd, trough, mdd_peak = idx / peak - 1, p["date"], peak_d
    end = date.fromisoformat(pts[-1]["date"])

    def window(after: str | None) -> float | None:
        sel = [p for p in pts[1:] if after is None or p["date"] > after]
        if not sel:
            return None
        g = 1.0
        for p in sel:
            g *= 1 + p["twr"]
        return round((g - 1) * 100, 4)

    return {
        "accounts": [a.label for a in accts], "from": pts[0]["date"], "to": pts[-1]["date"],
        "latest_nav_usd": r2(pts[-1]["nav"]), "twr_source": source,
        "twr_pct": {"1D": window(pts[-2]["date"]) if len(pts) > 1 else None,
                    "7D": window((end - timedelta(days=7)).isoformat()),
                    "MTD": window((end.replace(day=1) - timedelta(days=1)).isoformat()),
                    "YTD": window(date(end.year - 1, 12, 31).isoformat()), "inception": window(None)},
        "max_drawdown_pct": round(mdd * 100, 4),
        "max_drawdown_peak": mdd_peak if trough else None, "max_drawdown_trough": trough,
        "series": [{"date": p["date"], "nav": r2(p["nav"]), "flow": r2(p["flow"]),
                    "daily_return_pct": round(p["twr"] * 100, 4), "cum_twr_pct": round((p["index"] - 1) * 100, 4)}
                   for p in pts],
    }
