"""Static black-and-white dashboard: one self-contained HTML file, no JS, no CDN, no colors."""

from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

from .config import PROJECT_ROOT
from .metrics import TOLERANCE, AccountMetrics

COLUMNS = (("gold", "GOLD"), ("options", "OPTIONS"))  # IBKR columns, left to right; manual accounts follow
OUTPUT = PROJECT_ROOT / "dashboard.html"


def columns(metrics: list[AccountMetrics]) -> list[tuple[str, str]]:
    return list(COLUMNS) + [(m.label, m.label.upper()) for m in metrics if m.is_manual]


def _dubai() -> tzinfo:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Asia/Dubai")
    except Exception:
        return timezone(timedelta(hours=4), "GST")


def money(v: float, signed: bool = False) -> str:
    c = round(v * 100)
    sign = ("+" if c >= 0 else "-") if signed else ("-" if c < 0 else "")
    return f"{sign}${abs(c) / 100:,.2f}"


def pct(v: float | None) -> str:
    return "—" if v is None else f"{'+' if v >= 0 else '-'}{abs(v):.2f}%"


def dubai_time(iso: str | None) -> str:
    if not iso:
        return "never"
    return datetime.fromisoformat(iso).astimezone(_dubai()).strftime("%d %b %Y %H:%M")


def breakdown(m: AccountMetrics) -> list[tuple[str, float]]:
    """Lines shown under NET P&L. Dividends is a gold line but is also shown elsewhere when nonzero, so the
    lines always add up to NET P&L; anything left over beyond TOLERANCE is shown as Unreconciled."""
    if m.is_manual:  # no positions: everything is realized
        return [("Realized P&L", m.realized)]
    lines = [("Realized P&L", m.realized), ("Unrealized P&L", m.unrealized)]
    optional = [("Dividends", m.dividends, m.label == "gold"), ("Interest", m.interest, False),
                ("Other fees", m.other_fees, False), ("Commission adjustments", m.commission_adjustments, False),
                ("FX conversion costs", m.fx_conversion, False)]
    lines += [(n, v) for n, v, always in optional if always or round(v * 100) != 0]
    if abs(m.residual) > TOLERANCE:
        lines.append(("Unreconciled", m.residual))
    return lines


CSS = """*{box-sizing:border-box;margin:0;padding:0}
html,body{background:#fff;color:#000;min-height:100%}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.page{display:flex;flex-direction:column;min-height:100vh}
main{display:grid;grid-template-columns:repeat(var(--cols),1fr);flex:1}
.total{border-top:1px solid #000;padding:44px clamp(24px,3.2vw,56px) 48px}
.tg{display:grid;grid-template-columns:1fr 1fr;column-gap:56px;row-gap:32px}
.tn{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;font-weight:800;
font-size:clamp(40px,5vw,72px);line-height:1.05;letter-spacing:-.02em}
.tp{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;font-weight:800;
font-size:clamp(24px,2.6vw,36px);margin-top:6px}
.total .note{margin-top:28px}
section{padding:64px clamp(24px,3.2vw,56px) 36px;display:flex;flex-direction:column;min-width:0}
main>section+section{border-left:1px solid #000}
h1{font-size:13px;font-weight:600;letter-spacing:.32em;margin-bottom:56px}
.m{margin-bottom:40px}
.l{font-size:11px;font-weight:500;letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px}
.n{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;font-size:clamp(28px,2.8vw,40px);line-height:1.1}
.p{font-size:18px;margin-top:6px}
.b{margin:22px 0 0 24px;padding-left:20px;border-left:1px solid #000;max-width:440px}
.r{display:flex;justify-content:space-between;gap:24px;padding:7px 0}
.r .l{margin:0}
.r .n{font-size:17px}
.u .l,.u .n{font-weight:700}
.pt{border-collapse:collapse;width:100%;max-width:480px;margin-top:4px}
.pt th,.pt td{padding:7px 0;text-align:right;font-weight:400;vertical-align:baseline}
.pt th.l{text-align:left;font-size:11px;font-weight:500;letter-spacing:.18em;text-transform:uppercase;padding-right:12px}
.pt th.h{font-size:11px;font-weight:600;letter-spacing:.18em;border-bottom:1px solid #000;padding-bottom:9px}
.pt td{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;font-size:16px;
white-space:nowrap;padding-left:24px}
.pt tr.k td{font-weight:700}
.pt+.pt{margin-top:18px}
.rr{display:grid;grid-template-columns:minmax(0,1fr) auto 8.5ch;column-gap:16px;align-items:baseline;padding:7px 0}
.rr .l{margin:0}
.rr .n{font-size:17px;white-space:nowrap;text-align:right}
.note{font-size:11px;margin-top:8px}
footer{margin-top:auto;padding-top:32px;font-size:11px;letter-spacing:.03em;line-height:1.6}
@media (max-width:700px){main{grid-template-columns:1fr}section{padding:40px 24px}
section+section{border-left:0;border-top:1px solid #000}.n{font-size:32px}.tg{grid-template-columns:1fr}}"""


def total_block(metrics: list[AccountMetrics]) -> str:
    """Full-width strip under the columns: whole portfolio, % of everything deposited."""
    dep = sum(m.deposited for m in metrics)
    trading = sum(m.trading_pnl for m in metrics)
    realized = sum(m.realized for m in metrics)
    net = sum(m.net_pnl for m in metrics)
    p = lambda v: pct(v / dep * 100 if dep else None)  # noqa: E731
    names = " + ".join(m.label.upper() for m in metrics)
    big = [("Total net · realized + unrealized", trading), ("Total realized", realized)]
    cells = "".join(f'<div class="t"><div class="l">{html.escape(n)}</div><div class="tn">{money(v, True)}</div>'
                    f'<div class="tp">{p(v)}</div></div>' for n, v in big)
    return (f'<section class="total" aria-label="Portfolio total"><div class="tg">{cells}</div>'
            f'<p class="note">{html.escape(names)} · {money(dep)} deposited · {money(sum(m.current_value for m in metrics))} '
            f'current value · all-in net P&amp;L incl. dividends, interest &amp; fees {money(net, True)} ({p(net)})</p>'
            '</section>')


def positions_block(m: AccountMetrics, per_table: int = 3) -> str:
    """Open positions as a metric-by-holding table (one column per holding, at most per_table per table)."""
    if not m.positions:
        return ""

    def num(v: float | None, fmt: str) -> str:
        return "—" if v is None else format(v, fmt)

    rows = [("Shares", lambda p: num(p["qty"], ",.4f").rstrip("0").rstrip(".")),
            ("Avg cost", lambda p: "—" if p["avg_cost"] is None else f"${p['avg_cost']:,.2f}"),
            ("Price", lambda p: "—" if p["mark"] is None else f"${p['mark']:,.2f}"),
            ("Money put in", lambda p: money(p["cost"])),
            ("Current value", lambda p: money(p["value"])),
            ("Unrealized P&L", lambda p: money(p["unrealized"], True)),
            ("% of money put in", lambda p: pct(p["unrealized_pct"])),
            ("Share of unrealized", lambda p: "—" if p["share_of_unrealized"] is None else f"{p['share_of_unrealized']:.1f}%")]
    tables = []
    for i in range(0, len(m.positions), per_table):
        chunk = m.positions[i:i + per_table]
        head = "".join(f'<th class="h">{html.escape(p["symbol"] or "?")}</th>' for p in chunk)
        key_row = ' class="k"'
        body = "".join(
            f'<tr{key_row if name == "Unrealized P&L" else ""}><th class="l">{html.escape(name)}</th>'
            + "".join(f'<td>{fn(p)}</td>' for p in chunk) + "</tr>" for name, fn in rows)
        tables.append(f'<table class="pt"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>')
    return (f'<div class="m"><div class="l">Unrealized by holding</div>{"".join(tables)}'
            f'<p class="note">as of {html.escape(m.history_to or "—")} · avg cost includes commissions, so it is your break-even price</p></div>')


def section(title: str, m: AccountMetrics | None) -> str:
    e = html.escape
    if m is None:
        return f'<section><h1>{e(title)}</h1><p class="l">No data</p><footer>Last refreshed never</footer></section>'
    rows = "".join(f'<div class="r{" u" if n == "Unreconciled" else ""}"><span class="l">{e(n)}</span>'
                   f'<span class="n">{money(v, True)}</span></div>' for n, v in breakdown(m))
    if m.is_manual:
        return (f'<section aria-label="{e(title)}"><h1>{e(title)}</h1>'
                f'<div class="m"><div class="l">Deposited</div><div class="n">{money(m.deposited)}</div></div>'
                f'<div class="m"><div class="l">Current value</div><div class="n">{money(m.current_value)}</div></div>'
                f'<div class="m"><div class="l">Net P&amp;L</div><div class="n">{money(m.net_pnl, True)}</div>'
                f'<div class="n p">{pct(m.net_pct)}</div><div class="b">{rows}</div></div>'
                f'<div class="m"><div class="l">Cumulative fees paid</div><div class="n">—</div>'
                f'<p class="note">not available for a manual entry (already inside the realized loss)</p></div>'
                f'<footer>Manual entry · balance as of {e(m.nav_date or "—")} · not reconciled with broker data'
                f'</footer></section>')
    return (f'<section aria-label="{e(title)}"><h1>{e(title)}</h1>'
            f'<div class="m"><div class="l">Deposited</div><div class="n">{money(m.deposited)}</div></div>'
            f'<div class="m"><div class="l">Current value</div><div class="n">{money(m.current_value)}</div></div>'
            f'<div class="m"><div class="l">Net P&amp;L</div><div class="n">{money(m.net_pnl, True)}</div>'
            f'<div class="n p">{pct(m.net_pct)}</div><div class="b">{rows}</div></div>'
            f'{positions_block(m)}'
            f'<div class="m"><div class="l">Cumulative fees paid</div><div class="n">{money(m.fees_paid)}</div>'
            f'<p class="note">commissions already included in realized P&amp;L (FX conversion commissions in FX conversion costs)</p></div>'
            f'<footer>Last refreshed {e(dubai_time(m.last_refreshed))} (Dubai) · NAV as of {e(m.nav_date or "—")} · '
            f'history from {e(m.history_from or "—")}</footer></section>')


def render(metrics: list[AccountMetrics]) -> str:
    by = {m.label: m for m in metrics}
    cols = columns(metrics)
    body = "".join(section(title, by.get(label)) for label, title in cols)
    return ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Portfolio</title><style>{CSS}</style></head><body><div class="page">'
            f'<main style="--cols:{len(cols)}">{body}</main>{total_block(metrics)}</div></body></html>\n')


def write(metrics: list[AccountMetrics], path: Path = OUTPUT) -> Path:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(render(metrics), encoding="utf-8")
    tmp.replace(path)
    return path


def summary_text(metrics: list[AccountMetrics]) -> str:
    by = {m.label: m for m in metrics}
    cols = [(title, by.get(label)) for label, title in columns(metrics)]
    names = ["Deposited", "Current value", "Net P&L", "Net P&L %", "Realized + unrealized %", "Realized only %", "  Realized", "  Unrealized", "  Dividends",
             "  Interest", "  Other fees", "  FX conversion costs", "  Unreconciled (residual)", "Fees paid",
             "NAV as of", "History from", "Refreshed (Dubai)", "Reconciled"]

    def vals(m: AccountMetrics) -> list[str]:
        if m.is_manual:  # only deposited / value / realized are known for a manual entry
            na = "—"
            return [money(m.deposited), money(m.current_value), money(m.net_pnl, True), pct(m.net_pct),
                    pct(m.trading_pct), pct(m.realized_pct), money(m.realized, True), na, na, na, na, na, na, na, m.nav_date or na, na, "manual entry",
                    "n/a (manual)"]
        return [money(m.deposited), money(m.current_value), money(m.net_pnl, True), pct(m.net_pct),
                pct(m.trading_pct), pct(m.realized_pct), money(m.realized, True), money(m.unrealized, True), money(m.dividends, True), money(m.interest, True),
                money(m.other_fees, True), money(m.fx_conversion, True), money(m.residual, True), money(m.fees_paid),
                m.nav_date or "—", m.history_from or "—", "manual entry" if m.is_manual else dubai_time(m.last_refreshed),
                "n/a (manual)" if m.is_manual else ("YES" if m.reconciled else "NO")]

    table = [vals(m) if m else ["no data"] * len(names) for _, m in cols]
    w = max(len(n) for n in names)
    lines = [" " * w + "".join(f"{t:>22}" for t, _ in cols)]
    lines += [n.ljust(w) + "".join(f"{c[i]:>22}" for c in table) for i, n in enumerate(names)]
    return "\n".join(lines)
