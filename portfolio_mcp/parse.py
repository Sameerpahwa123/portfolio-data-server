"""Flex XML -> rows. Column/attribute names below are the ones discovered in the real statements
(scripts/inspect_xml.py). Every element with attributes is stored: typed tables for the elements the
metrics use, the generic `elements` table for everything else. raw_json always keeps all attributes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from lxml import etree

Row = dict[str, Any]
S, N, D, T = "s", "n", "d", "t"  # text, number, date, datetime


class ParseError(ValueError):
    pass


# ----------------------------------------------------------------------------- values

_EMPTY = {"", "--", "-", "n/a", "null", "none"}


def text(v: str | None) -> str | None:
    if v is None:
        return None
    s = v.strip()
    return None if s.lower() in _EMPTY else s


def num(v: str | None) -> float | None:
    s = text(v)
    if s is None:
        return None
    s = s.replace(",", "").replace(" ", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        f = float(s)
    except ValueError:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return -f if neg else f


@dataclass
class DateParser:
    """Flex dates are yyyyMMdd, or dd/MM/yyyy vs MM/dd/yyyy depending on the query's date format."""

    slash_order: str = "DMY"  # or "MDY"

    def date(self, v: str | None) -> str | None:
        s = text(v)
        if s is None:
            return None
        part = re.split(r"[;, T]", s, maxsplit=1)[0]
        if len(part) == 8 and part.isdigit():
            return f"{part[:4]}-{part[4:6]}-{part[6:]}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            return part
        m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", part)
        if m:
            a, b, y = (int(x) for x in m.groups())
            d, mo = (a, b) if self.slash_order == "DMY" else (b, a)
            try:
                return date(y, mo, d).isoformat()
            except ValueError:
                return None
        return None

    def datetime(self, v: str | None) -> str | None:
        s = text(v)
        if s is None:
            return None
        parts = re.split(r"[;, T]", s, maxsplit=1)
        d = self.date(parts[0])
        if d is None or len(parts) == 1:
            return d
        t = re.sub(r"\D", "", parts[1])[:6]
        return f"{d} {t[:2]}:{t[2:4]}:{t[4:6]}" if len(t) == 6 else d


def detect_slash_order(content: bytes, reference: date | None = None) -> str:
    pairs = [(int(a), int(b)) for a, b, _ in re.findall(rb"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", content)]
    if any(a > 12 for a, _ in pairs):
        return "DMY"
    if any(b > 12 for _, b in pairs):
        return "MDY"
    m = re.search(rb'<FlexStatement\b[^>]*\btoDate="(\d{1,2})/(\d{1,2})/(\d{4})"', content)
    if not m:
        return "DMY"
    a, b, y = (int(x) for x in m.groups())
    ref = reference or date.today()

    def gap(mo: int, d: int) -> float:
        try:
            delta = (ref - date(y, mo, d)).days
        except ValueError:
            return math.inf
        return delta if delta >= -1 else math.inf

    return "MDY" if gap(a, b) < gap(b, a) else "DMY"


# ----------------------------------------------------------------------------- table specs


@dataclass(frozen=True)
class Spec:
    table: str
    cols: dict[str, tuple[str, str]]  # column -> (attribute, kind)
    keys: tuple[tuple[str, ...], ...]  # candidate natural keys; first fully non-null wins, else hash


_SEC = {"conid": ("conid", S), "symbol": ("symbol", S), "description": ("description", S),
        "asset_category": ("assetCategory", S), "underlying_symbol": ("underlyingSymbol", S),
        "put_call": ("putCall", S), "strike": ("strike", N), "expiry": ("expiry", D), "multiplier": ("multiplier", N)}

_TRADE_COLS = {
    "trade_id": ("tradeID", S), "transaction_id": ("transactionID", S), "exec_id": ("ibExecID", S),
    "order_id": ("ibOrderID", S), "level_of_detail": ("levelOfDetail", S), "transaction_type": ("transactionType", S),
    "date_time": ("dateTime", T), "trade_date": ("tradeDate", D), "report_date": ("reportDate", D), **_SEC,
    "buy_sell": ("buySell", S), "open_close": ("openCloseIndicator", S), "quantity": ("quantity", N),
    "trade_price": ("tradePrice", N), "proceeds": ("proceeds", N), "ib_commission": ("ibCommission", N),
    "ib_commission_currency": ("ibCommissionCurrency", S), "taxes": ("taxes", N), "net_cash": ("netCash", N),
    "cost": ("cost", N), "fifo_pnl_realized": ("fifoPnlRealized", N), "mtm_pnl": ("mtmPnl", N),
    "order_type": ("orderType", S), "exchange": ("exchange", S), "notes": ("notes", S),
    "open_date_time": ("openDateTime", T),
}

SPECS: dict[str, Spec] = {
    "AccountInformation": Spec("account_info", {
        "name": ("name", S), "account_type": ("accountType", S), "date_opened": ("dateOpened", D),
        "date_funded": ("dateFunded", D), "date_closed": ("dateClosed", D)}, (("account_id",),)),
    "EquitySummaryByReportDateInBase": Spec("nav", {
        "report_date": ("reportDate", D), "total": ("total", N), "cash": ("cash", N), "stock": ("stock", N),
        "options": ("options", N), "funds": ("funds", N), "interest_accruals": ("interestAccruals", N),
        "dividend_accruals": ("dividendAccruals", N), "total_long": ("totalLong", N), "total_short": ("totalShort", N)},
        (("report_date",),)),
    "ChangeInNAV": Spec("change_in_nav", {
        "from_date": ("fromDate", D), "to_date": ("toDate", D), "starting_value": ("startingValue", N),
        "ending_value": ("endingValue", N), "mtm": ("mtm", N), "realized": ("realized", N),
        "change_in_unrealized": ("changeInUnrealized", N), "deposits_withdrawals": ("depositsWithdrawals", N),
        "asset_transfers": ("assetTransfers", N), "internal_cash_transfers": ("internalCashTransfers", N),
        "dividends": ("dividends", N), "withholding_tax": ("withholdingTax", N), "withholding_871m": ("withholding871m", N),
        "interest": ("interest", N), "commissions": ("commissions", N), "forex_commissions": ("forexCommissions", N),
        "other_fees": ("otherFees", N), "transaction_tax": ("transactionTax", N), "fx_translation": ("fxTranslation", N),
        "net_fx_trading": ("netFxTrading", N), "other": ("other", N), "twr_pct": ("twr", N)},
        (("from_date", "to_date"),)),
    "CashTransaction": Spec("cash_transactions", {
        "transaction_id": ("transactionID", S), "type": ("type", S), "level_of_detail": ("levelOfDetail", S),
        "date_time": ("dateTime", T), "report_date": ("reportDate", D), "settle_date": ("settleDate", D),
        "amount": ("amount", N), "action_id": ("actionID", S), **_SEC},
        (("level_of_detail", "transaction_id"), ("level_of_detail", "type", "date_time", "amount", "currency", "description"))),
    "Trade": Spec("trades", _TRADE_COLS,
                  (("level_of_detail", "trade_id", "open_date_time"), ("level_of_detail", "trade_id"),
                   ("level_of_detail", "transaction_id"))),
    "Lot": Spec("trades", _TRADE_COLS, (("level_of_detail", "trade_id", "open_date_time", "quantity"),)),
    "OpenPosition": Spec("open_positions", {
        "report_date": ("reportDate", D), "level_of_detail": ("levelOfDetail", S), **_SEC, "position": ("position", N),
        "mark_price": ("markPrice", N), "position_value": ("positionValue", N), "cost_basis_money": ("costBasisMoney", N),
        "cost_basis_price": ("costBasisPrice", N), "fifo_pnl_unrealized": ("fifoPnlUnrealized", N), "side": ("side", S),
        "percent_of_nav": ("percentOfNAV", N), "open_date_time": ("openDateTime", T)},
        (("report_date", "level_of_detail", "conid", "open_date_time"), ("report_date", "level_of_detail", "conid"))),
    "FIFOPerformanceSummaryUnderlying": Spec("fifo_summary", {
        "report_date": ("reportDate", D), **_SEC, "realized_st_profit": ("realizedSTProfit", N),
        "realized_st_loss": ("realizedSTLoss", N), "realized_lt_profit": ("realizedLTProfit", N),
        "realized_lt_loss": ("realizedLTLoss", N), "total_realized_pnl": ("totalRealizedPnl", N),
        "total_unrealized_pnl": ("totalUnrealizedPnl", N), "total_fifo_pnl": ("totalFifoPnl", N)},
        (("stmt_from", "stmt_to", "asset_category", "conid"), ("stmt_from", "stmt_to", "description"))),
    "OpenDividendAccrual": Spec("dividend_accruals", {
        "report_date": ("reportDate", D), "ex_date": ("exDate", D), "pay_date": ("payDate", D), **_SEC,
        "quantity": ("quantity", N), "gross_rate": ("grossRate", N), "gross_amount": ("grossAmount", N),
        "tax": ("tax", N), "fee": ("fee", N), "net_amount": ("netAmount", N), "code": ("code", S)},
        (("stmt_to", "conid", "ex_date", "pay_date"),)),
    "SecurityInfo": Spec("security_info", {**_SEC, "isin": ("isin", S), "listing_exchange": ("listingExchange", S)},
                         (("conid",),)),
}


# ----------------------------------------------------------------------------- parsing


@dataclass
class ParsedFile:
    tables: dict[str, list[Row]] = field(default_factory=dict)
    statements: list[Row] = field(default_factory=list)
    slash_order: str = "DMY"

    def add(self, table: str, row: Row) -> None:
        self.tables.setdefault(table, []).append(row)

    @property
    def account_ids(self) -> set[str]:
        return {s["account_id"] for s in self.statements}

    def counts(self) -> dict[str, int]:
        return {t: len(r) for t, r in sorted(self.tables.items())}


def _hash(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _uid(row: Row, keys: tuple[tuple[str, ...], ...], attrs: dict[str, str]) -> str:
    for key in keys:
        vals = [row.get(k) for k in key]
        if all(v not in (None, "") for v in vals):
            return "|".join(format(v, ".10g") if isinstance(v, float) else str(v) for v in vals)
    return "h:" + _hash(row.get("stmt_from"), row.get("stmt_to"), attrs)


def parse_flex(content: bytes, *, raw_file: str | None = None, reference: date | None = None) -> ParsedFile:
    try:
        root = etree.fromstring(content, etree.XMLParser(huge_tree=True, resolve_entities=False, no_network=True))
    except etree.XMLSyntaxError as e:
        raise ParseError(f"invalid XML: {e}") from e
    if root is None or etree.QName(root).localname != "FlexQueryResponse":
        raise ParseError("not a FlexQueryResponse")
    out = ParsedFile(slash_order=detect_slash_order(content, reference))
    dp = DateParser(out.slash_order)
    conv = {S: text, N: num, D: dp.date, T: dp.datetime}
    for stmt in root.iter("FlexStatement"):
        acct = text(stmt.get("accountId"))
        if not acct:
            raise ParseError("FlexStatement without accountId")
        s_from, s_to = dp.date(stmt.get("fromDate")), dp.date(stmt.get("toDate"))
        out.statements.append({"account_id": acct, "from_date": s_from, "to_date": s_to,
                               "period": text(stmt.get("period")), "when_generated": dp.datetime(stmt.get("whenGenerated")),
                               "raw_file": raw_file})
        for el in stmt.iterdescendants():
            if not isinstance(el.tag, str) or not el.attrib:
                continue
            tag = etree.QName(el).localname
            attrs = dict(el.attrib)
            row: Row = {"account_id": text(attrs.get("accountId")) or acct, "stmt_from": s_from, "stmt_to": s_to,
                        "currency": text(attrs.get("currency")), "fx_rate_to_base": num(attrs.get("fxRateToBase")),
                        "raw_json": json.dumps(attrs, sort_keys=True, ensure_ascii=False), "raw_file": raw_file}
            spec = SPECS.get(tag)
            if spec is None:
                row.update(tag=tag, uid=f"{tag}|h:{_hash(s_from, s_to, attrs)}")
                out.add("elements", row)
                continue
            for col, (attr, kind) in spec.cols.items():
                row[col] = conv[kind](attrs.get(attr))
            if "level_of_detail" in row and row["level_of_detail"]:
                row["level_of_detail"] = row["level_of_detail"].upper()
            if tag == "Lot" and not row.get("level_of_detail"):
                row["level_of_detail"] = "CLOSED_LOT"
            row["uid"] = _uid(row, spec.keys, attrs)
            out.add(spec.table, row)
    if not out.statements:
        raise ParseError("FlexQueryResponse contains no FlexStatement")
    return out


def parse_datetime_stamp(name: str) -> date | None:
    """Reference date from a raw filename like gold_20261103_165319.xml."""
    m = re.search(r"_(\d{8})_\d{6}", name)
    return datetime.strptime(m.group(1), "%Y%m%d").date() if m else None
