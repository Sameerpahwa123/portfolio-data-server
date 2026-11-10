"""Per-account metrics (USD, since the start of cached history) and reconciliation against IBKR totals.

net_pnl = current_value - deposited
        = realized + unrealized + dividends + interest + other_fees + commission_adjustments + fx_conversion + residual
residual must be within TOLERANCE; it is never hidden.

Facts established from the real statements (scripts/inspect_xml.py):
- CashTransaction comes as DETAIL and SUMMARY rows describing the same money: only DETAIL is summed.
- ibCommission is in ibCommissionCurrency (USD), not the trade currency (AED for USD.AED conversions).
- FX conversion costs (commission + spread) show up as CASH-category rows of FIFOPerformanceSummaryUnderlying;
  CASH Trade rows carry fifoPnlRealized=0.
- IBKR ChangeInNAV.mtm is gross of commissions: mtm = realized + unrealized + fx_conversion - commissions.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

TOLERANCE = 1.00
DEPOSIT = "Deposits/Withdrawals"
DIVIDEND_TYPES = ("Dividends", "Payment In Lieu Of Dividends")
WITHHOLDING = "Withholding Tax"
INTEREST_TYPES = ("Broker Interest Received", "Broker Interest Paid")
OTHER_FEES = "Other Fees"
COMMISSION_ADJ = "Commission Adjustments"
KNOWN_CASH_TYPES = {DEPOSIT, *DIVIDEND_TYPES, WITHHOLDING, *INTEREST_TYPES, OTHER_FEES, COMMISSION_ADJ}
UNMODELLED_CIN = ("asset_transfers", "internal_cash_transfers", "transaction_tax", "fx_translation",
                  "net_fx_trading", "other", "withholding_871m")


class FxMissing(ValueError):
    pass


def usd(amount: float | None, currency: str | None, fx: float | None, base: str = "USD") -> float:
    if amount is None:
        return 0.0
    if fx is not None:
        return amount * fx
    if (currency or base) == base:
        return amount
    raise FxMissing(f"no fxRateToBase for {currency} amount {amount}")


def commission_usd(r: sqlite3.Row, base: str = "USD") -> float:
    """ibCommission is denominated in ibCommissionCurrency, which can differ from the trade currency."""
    amt = r["ib_commission"] or 0.0
    ccy = r["ib_commission_currency"] or r["currency"] or base
    if ccy == base:
        return amt
    if ccy == r["currency"] and r["fx_rate_to_base"] is not None:
        return amt * r["fx_rate_to_base"]
    raise FxMissing(f"commission in {ccy} on a {r['currency']} trade without a usable rate")


def tile(windows: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Non-overlapping statement windows (single-day windows preferred) so nothing is summed twice."""
    chosen: list[tuple[str, str]] = []
    for w in sorted(windows, key=lambda w: (w[0] != w[1], w[0], w[1])):
        if all(w[1] < c[0] or w[0] > c[1] for c in chosen):
            chosen.append(w)
    return set(chosen)


@dataclass
class Check:
    metric: str
    mine: float | None
    ibkr: float | None
    note: str = ""
    enforce: bool = True

    @property
    def diff(self) -> float | None:
        return None if self.mine is None or self.ibkr is None else self.mine - self.ibkr

    @property
    def ok(self) -> bool:
        return not self.enforce or (self.diff is not None and abs(self.diff) <= TOLERANCE)


@dataclass
class AccountMetrics:
    label: str
    account_id: str
    base_currency: str = "USD"
    source: str = "ibkr"  # "ibkr" (reconciled against broker totals) or "manual" (typed in, unverifiable)
    note: str = ""
    history_from: str | None = None
    history_to: str | None = None
    from_inception: bool = False
    nav_date: str | None = None
    last_refreshed: str | None = None
    deposited: float = 0.0
    current_value: float = 0.0
    net_pnl: float = 0.0
    realized: float = 0.0
    unrealized: float = 0.0
    dividends_gross: float = 0.0
    withholding: float = 0.0
    dividends: float = 0.0
    interest: float = 0.0
    other_fees: float = 0.0
    commission_adjustments: float = 0.0
    fx_conversion: float = 0.0
    trade_commissions: float = 0.0
    fees_paid: float = 0.0
    residual: float = 0.0
    positions: list[dict[str, Any]] = field(default_factory=list)  # open positions at the latest statement, USD
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def net_pct(self) -> float | None:
        return self.net_pnl / self.deposited * 100 if self.deposited else None

    @property
    def trading_pnl(self) -> float:
        """Realized + unrealized only (excludes dividends, interest, fees, FX conversion costs)."""
        return self.realized + self.unrealized

    @property
    def trading_pct(self) -> float | None:
        return self.trading_pnl / self.deposited * 100 if self.deposited else None

    @property
    def realized_pct(self) -> float | None:
        return self.realized / self.deposited * 100 if self.deposited else None

    @property
    def is_manual(self) -> bool:
        return self.source == "manual"

    @property
    def reconciled(self) -> bool:
        """True only for broker data that passed every check; manual accounts are never 'reconciled'."""
        if self.is_manual:
            return False
        return all(c.ok for c in self.checks) and abs(self.residual) <= TOLERANCE and not self.warnings_blocking

    @property
    def warnings_blocking(self) -> list[str]:
        return [w for w in self.warnings if w.startswith("BLOCKING")]

    def components(self) -> list[tuple[str, float]]:
        return [("realized", self.realized), ("unrealized", self.unrealized), ("dividends", self.dividends),
                ("interest", self.interest), ("other_fees", self.other_fees),
                ("commission_adjustments", self.commission_adjustments), ("fx_conversion", self.fx_conversion)]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["checks"] = [{"metric": c.metric, "mine": _r(c.mine), "ibkr": _r(c.ibkr), "diff": _r(c.diff), "ok": c.ok,
                        "enforced": c.enforce, "note": c.note} for c in self.checks]
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 2)
        d["net_pnl_pct"] = None if self.net_pct is None else round(self.net_pct, 2)
        d["realized_plus_unrealized"] = round(self.trading_pnl, 2)
        d["realized_plus_unrealized_pct"] = None if self.trading_pct is None else round(self.trading_pct, 2)
        d["realized_pct"] = None if self.realized_pct is None else round(self.realized_pct, 2)
        d["reconciled"] = self.reconciled
        return d


def manual_metrics(acc: Any) -> AccountMetrics:
    """Manual account (config.ManualAccount): no positions, so the whole net P&L is realized."""
    m = AccountMetrics(acc.label, acc.label, acc.currency, source="manual", note=acc.note,
                       nav_date=acc.as_of, deposited=acc.deposited, current_value=acc.current_value)
    m.net_pnl = m.realized = m.current_value - m.deposited
    m.warnings.append("manual entry: not reconciled against broker data; fees and P&L split unavailable")
    return m


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 2)


def _cash_rows(conn: sqlite3.Connection, aid: str) -> list[sqlite3.Row]:
    rows = conn.execute("SELECT * FROM cash_transactions WHERE account_id=?", (aid,)).fetchall()
    if any(r["level_of_detail"] == "DETAIL" for r in rows):
        return [r for r in rows if r["level_of_detail"] == "DETAIL"]
    return [r for r in rows if not r["level_of_detail"]]


def compute(conn: sqlite3.Connection, account_id: str, label: str, base: str = "USD") -> AccountMetrics:
    m = AccountMetrics(label, account_id, base)
    acc = conn.execute("SELECT * FROM accounts WHERE account_id=?", (account_id,)).fetchone()
    info = conn.execute("SELECT * FROM account_info WHERE account_id=?", (account_id,)).fetchone()
    if acc:
        m.last_refreshed = acc["last_refreshed"]
    if info and info["currency"] and info["currency"] != base:
        m.warnings.append(f"BLOCKING: account base currency {info['currency']} != BASE_CURRENCY {base} (not supported)")

    stmts = conn.execute("SELECT MIN(from_date) f, MAX(to_date) t FROM statements WHERE account_id=?", (account_id,)).fetchone()
    m.history_from, m.history_to = stmts["f"], stmts["t"]
    # History covers inception when IBKR's earliest ChangeInNAV starts from zero (weekend funding dates
    # mean the first statement can be later than dateFunded).
    first = conn.execute("SELECT starting_value FROM change_in_nav WHERE account_id=? ORDER BY from_date LIMIT 1",
                         (account_id,)).fetchone()
    m.from_inception = bool(first and abs(first["starting_value"] or 0) < 0.005)
    if not m.from_inception:
        inception = min(filter(None, [info["date_opened"], info["date_funded"]]), default=None) if info else None
        m.warnings.append(f"cached history starts {m.history_from} with a non-zero NAV (account opened {inception}): "
                          "totals are not since inception; backfill further")

    try:
        # --- cash transactions (DETAIL only)
        cash = _cash_rows(conn, account_id)
        by: dict[str, float] = {}
        for r in cash:
            by[r["type"]] = by.get(r["type"], 0.0) + usd(r["amount"], r["currency"], r["fx_rate_to_base"], base)
        unknown = sorted(set(by) - KNOWN_CASH_TYPES)
        if unknown:
            m.warnings.append(f"BLOCKING: unhandled CashTransaction types {unknown} (amounts {[round(by[t], 2) for t in unknown]})")
        m.deposited = by.get(DEPOSIT, 0.0)
        m.dividends_gross = sum(by.get(t, 0.0) for t in DIVIDEND_TYPES)
        m.withholding = by.get(WITHHOLDING, 0.0)
        m.dividends = m.dividends_gross + m.withholding
        m.interest = sum(by.get(t, 0.0) for t in INTEREST_TYPES)
        m.other_fees = by.get(OTHER_FEES, 0.0)
        m.commission_adjustments = by.get(COMMISSION_ADJ, 0.0)

        # --- trades: EXECUTION rows only (CLOSED_LOT rows repeat the same P&L)
        execs = conn.execute("SELECT * FROM trades WHERE account_id=? AND level_of_detail='EXECUTION'", (account_id,)).fetchall()
        m.realized = sum(usd(r["fifo_pnl_realized"], r["currency"], r["fx_rate_to_base"], base)
                         for r in execs if r["asset_category"] != "CASH")
        cash_trade_realized = sum(usd(r["fifo_pnl_realized"], r["currency"], r["fx_rate_to_base"], base)
                                  for r in execs if r["asset_category"] == "CASH")
        if abs(cash_trade_realized) > 0.005:
            m.warnings.append(f"CASH trades carry fifoPnlRealized {cash_trade_realized:.2f} (counted via FIFO summary instead)")
        m.trade_commissions = sum(commission_usd(r, base) for r in execs)
    except FxMissing as e:
        m.warnings.append(f"BLOCKING: {e}")

    # --- latest snapshot: positions on the latest statement date (none that day = flat)
    nav = conn.execute("SELECT report_date, total FROM nav WHERE account_id=? AND total IS NOT NULL "
                       "ORDER BY report_date DESC LIMIT 1", (account_id,)).fetchone()
    if nav:
        m.nav_date, m.current_value = nav["report_date"], nav["total"]
    else:
        m.warnings.append("BLOCKING: no EquitySummaryByReportDateInBase rows")
    pos = conn.execute("SELECT * FROM open_positions WHERE account_id=? AND report_date=? "
                       "AND COALESCE(level_of_detail,'SUMMARY')='SUMMARY'", (account_id, m.history_to)).fetchall()
    m.unrealized = sum((r["fifo_pnl_unrealized"] or 0) * (r["fx_rate_to_base"] if r["fx_rate_to_base"] is not None else 1.0)
                       for r in pos)
    for r in pos:
        fx = r["fx_rate_to_base"] if r["fx_rate_to_base"] is not None else 1.0
        cost, unr = (r["cost_basis_money"] or 0) * fx, (r["fifo_pnl_unrealized"] or 0) * fx
        m.positions.append({
            "symbol": r["symbol"], "asset_category": r["asset_category"], "currency": r["currency"],
            "qty": r["position"], "avg_cost": r["cost_basis_price"], "mark": r["mark_price"],
            "cost": round(cost, 2), "value": round((r["position_value"] or 0) * fx, 2), "unrealized": round(unr, 2),
            "unrealized_pct": round(unr / abs(cost) * 100, 2) if cost else None,
            "share_of_unrealized": round(unr / m.unrealized * 100, 1) if m.unrealized else None})
    m.positions.sort(key=lambda p: -abs(p["unrealized"]))

    # --- IBKR's own totals over non-overlapping statement windows
    cin_rows = conn.execute("SELECT * FROM change_in_nav WHERE account_id=?", (account_id,)).fetchall()
    keep = tile({(r["from_date"], r["to_date"]) for r in cin_rows})
    cin = [r for r in cin_rows if (r["from_date"], r["to_date"]) in keep]
    S = lambda col: sum(r[col] or 0.0 for r in cin)  # noqa: E731
    fifo_rows = conn.execute("SELECT * FROM fifo_summary WHERE account_id=?", (account_id,)).fetchall()
    fkeep = tile({(r["stmt_from"], r["stmt_to"]) for r in fifo_rows})
    fifo = [r for r in fifo_rows if (r["stmt_from"], r["stmt_to"]) in fkeep and r["conid"]]  # conid-less = totals
    m.fx_conversion = sum(r["total_realized_pnl"] or 0 for r in fifo if r["asset_category"] == "CASH")
    fifo_last = [r for r in fifo if r["stmt_to"] == max((w[1] for w in fkeep), default=None)]

    m.net_pnl = m.current_value - m.deposited
    m.residual = m.net_pnl - sum(v for _, v in m.components())
    m.fees_paid = abs(m.trade_commissions + m.other_fees + m.commission_adjustments)

    last_cin = max(cin, key=lambda r: r["to_date"], default=None)
    m.checks = [
        Check("deposited", m.deposited, S("deposits_withdrawals"), "ChangeInNAV.depositsWithdrawals"),
        Check("dividends (gross)", m.dividends_gross, S("dividends"), "ChangeInNAV.dividends"),
        Check("withholding tax", m.withholding, S("withholding_tax"), "ChangeInNAV.withholdingTax"),
        Check("commissions", m.trade_commissions + m.commission_adjustments, S("commissions") + S("forex_commissions"),
              "sum(Trade.ibCommission) vs ChangeInNAV.commissions+forexCommissions"),
        Check("other fees", m.other_fees, S("other_fees"), "ChangeInNAV.otherFees"),
        Check("interest", m.interest, S("interest"), "ChangeInNAV.interest"),
        Check("current value", m.current_value, last_cin["ending_value"] if last_cin else None,
              f"ChangeInNAV.endingValue {last_cin['to_date'] if last_cin else ''}"),
        Check("realized (FIFO summary)", m.realized,
              sum(r["total_realized_pnl"] or 0 for r in fifo if r["asset_category"] != "CASH"),
              "FIFOPerformanceSummaryUnderlying.totalRealizedPnl, non-CASH"),
        Check("unrealized (FIFO summary)", m.unrealized,
              sum(r["total_unrealized_pnl"] or 0 for r in fifo_last if r["asset_category"] != "CASH"),
              "FIFOPerformanceSummaryUnderlying.totalUnrealizedPnl, latest day"),
        Check("mtm", m.realized + m.unrealized + m.fx_conversion - m.trade_commissions, S("mtm"),
              "realized+unrealized+fx_conversion-commissions vs ChangeInNAV.mtm (gross of commissions)",
              enforce=m.from_inception),
        Check("unmodelled ChangeInNAV components", 0.0, sum(S(c) for c in UNMODELLED_CIN),
              "must be ~0: " + ", ".join(f"{c}={S(c):.2f}" for c in UNMODELLED_CIN if abs(S(c)) >= 0.005) or "none"),
    ]
    if any(abs(r["realized"] or 0) > 0.005 for r in cin):
        m.checks.append(Check("ChangeInNAV.realized (info)", m.realized + m.fx_conversion, S("realized"),
                              "informational", enforce=False))
    return m


def table(metrics: list[AccountMetrics]) -> str:
    lines = [f"{'account':<9} {'metric':<36} {'mine':>12} {'ibkr':>12} {'diff':>9}  status"]
    for m in metrics:
        for c in m.checks:
            f = lambda v: "—" if v is None else f"{v:,.2f}"  # noqa: E731
            status = "OK" if c.ok and c.enforce else ("info" if not c.enforce else "MISMATCH")
            lines.append(f"{m.label:<9} {c.metric:<36} {f(c.mine):>12} {f(c.ibkr):>12} {f(c.diff):>9}  {status}")
        lines.append(f"{m.label:<9} {'residual (net - components)':<36} {m.residual:>12,.2f} {'0.00':>12} "
                     f"{m.residual:>9,.2f}  {'OK' if abs(m.residual) <= TOLERANCE else 'MISMATCH'}")
        for w in m.warnings:
            lines.append(f"{m.label:<9} note: {w}")
    return "\n".join(lines)
