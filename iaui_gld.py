"""IAUI vs GLD since IAUI's inception (Yahoo Finance via yfinance) -> ./iaui_gld.html (self-contained).

    .venv/bin/python iaui_gld.py [--no-open]

Charts: (1) price indexed to 100 (IAUI price, IAUI total return incl. distributions, GLD),
(2) daily-return scatter with fitted line, (3) small multiples on one month axis: distribution per share,
GLD price on ex-date, distribution as % of IAUI price.
"""

import html
import json
import math
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

OUT = Path(__file__).resolve().parent / "iaui_gld.html"
WHT = 0.30  # US withholding tax on IAUI distributions (matches IBKR: gold account withheld 30.0% of dividends)
NET = 1 - WHT
W, PAD_L, PAD_R = 960, 64, 24


# ----------------------------------------------------------------------------- data


def load() -> dict:
    h = yf.Ticker("IAUI").history(period="max", auto_adjust=False)
    g = yf.Ticker("GLD").history(start=h.index[0].date(), auto_adjust=False)
    px = pd.DataFrame({"iaui": h["Close"], "gld": g["Close"]}).dropna()
    px.index = px.index.tz_localize(None).normalize()
    div = yf.Ticker("IAUI").dividends
    div.index = div.index.tz_localize(None).normalize()
    gross = div.copy()
    div = gross * NET  # everything below uses what you actually receive after withholding
    shares, tr = 1.0, []
    for dt, p in px["iaui"].items():  # total return: reinvest each distribution on its ex-date
        if dt in div.index:
            shares *= 1 + div[dt] / p
        tr.append(shares * p)
    px["iaui_tr"] = tr
    r = px[["iaui", "gld"]].pct_change().dropna()
    beta, alpha = np.polyfit(r["gld"], r["iaui"], 1)
    dist = []
    for dt, amt in div.items():
        i = min(px.index.searchsorted(dt), len(px) - 1)
        p_i, p_g = px["iaui"].iloc[i], px["gld"].iloc[i]
        dist.append({"date": dt.date().isoformat(), "dist": float(amt), "gross": float(gross[dt]),
                     "iaui": float(p_i), "gld": float(p_g),
                     "pct": float(amt / p_i * 100)})
    dd = pd.DataFrame(dist)
    up, dn = r[r["gld"] > 0], r[r["gld"] < 0]
    first, last = px.iloc[0], px.iloc[-1]
    return {
        "px": px, "r": r, "dist": dist,
        "stats": {
            "start": px.index[0].date().isoformat(), "end": px.index[-1].date().isoformat(), "days": len(px),
            "corr_daily": float(r.corr().iloc[0, 1]), "corr_level": float(px[["iaui", "gld"]].corr().iloc[0, 1]),
            "beta": float(beta), "alpha": float(alpha),
            "up_capture": float(up["iaui"].mean() / up["gld"].mean()),
            "down_capture": float(dn["iaui"].mean() / dn["gld"].mean()),
            "vol_iaui": float(r["iaui"].std() * math.sqrt(252) * 100), "vol_gld": float(r["gld"].std() * math.sqrt(252) * 100),
            "ret_iaui": float((last["iaui"] / first["iaui"] - 1) * 100), "ret_gld": float((last["gld"] / first["gld"] - 1) * 100),
            "ret_iaui_tr": float((last["iaui_tr"] / first["iaui_tr"] - 1) * 100),
            "dist_total": float(div.sum()), "dist_total_gross": float(gross.sum()),
            "dist_max_gross": float(gross.max()), "dist_last_gross": float(gross.iloc[-1]), "dist_n": len(div),
            "corr_dist_gld": float(dd["dist"].corr(dd["gld"])), "corr_dist_iaui": float(dd["dist"].corr(dd["iaui"])),
            "pct_min": float(dd["pct"].min()), "pct_max": float(dd["pct"].max()), "pct_mean": float(dd["pct"].mean()),
            "dist_min": float(dd["dist"].min()), "dist_max": float(dd["dist"].max()),
            "dist_peak_date": dd.loc[dd["dist"].idxmax(), "date"], "dist_first": float(dd["dist"].iloc[0]), "dist_last": float(dd["dist"].iloc[-1]),
            "gld_peak": float(px["gld"].max()), "gld_peak_date": px["gld"].idxmax().date().isoformat(),
            "iaui_peak": float(px["iaui"].max()), "iaui_peak_date": px["iaui"].idxmax().date().isoformat(),
            "iaui_last": float(last["iaui"]), "gld_last": float(last["gld"]),
        },
    }


# ----------------------------------------------------------------------------- svg helpers


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo or abs(hi) or 1
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    start = math.floor(lo / step) * step
    return [round(start + i * step, 10) for i in range(int((hi - start) / step) + 2) if start + i * step <= hi + step * 0.999]


def fmt(v: float, kind: str) -> str:
    return {"usd": f"${v:,.2f}", "usd0": f"${v:,.0f}", "pct": f"{v:.2f}%", "idx": f"{v:.0f}", "dist": f"${v:.3f}"}[kind]


class Frame:
    """Plot frame: date x-axis (shared across small multiples), numeric y-axis."""

    def __init__(self, x0: date, x1: date, y_lo: float, y_hi: float, h: int, kind: str) -> None:
        self.x0, self.x1, self.h, self.kind = x0, x1, h, kind
        self.top, self.bot = 14, h - 28
        ticks = nice_ticks(y_lo, y_hi)
        self.ticks = ticks
        self.y_lo, self.y_hi = ticks[0], ticks[-1]

    def x(self, d: date | str) -> float:
        d = date.fromisoformat(d) if isinstance(d, str) else d
        return PAD_L + (d - self.x0).days / max(1, (self.x1 - self.x0).days) * (W - PAD_L - PAD_R)

    def y(self, v: float) -> float:
        return self.bot - (v - self.y_lo) / (self.y_hi - self.y_lo) * (self.bot - self.top)

    def axes(self, x_labels: bool = True) -> str:
        out = []
        for t in self.ticks:
            y = self.y(t)
            out.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                       f'<text class="ax" x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(t, self.kind)}</text>')
        if x_labels:
            m = date(self.x0.year, self.x0.month, 1)
            while m <= self.x1:
                if m >= self.x0 and m.month in (1, 4, 7, 10):
                    lab = m.strftime("%b %Y") if m.month == 1 or m == date(self.x0.year, self.x0.month, 1) else m.strftime("%b")
                    out.append(f'<text class="ax" x="{self.x(m):.1f}" y="{self.h - 8}" text-anchor="middle">{lab}</text>')
                m = date(m.year + (m.month // 12), m.month % 12 + 1, 1)
        out.append(f'<line class="base" x1="{PAD_L}" x2="{W - PAD_R}" y1="{self.bot}" y2="{self.bot}"/>')
        return "".join(out)


def path(frame: Frame, pts: list[tuple[str, float]]) -> str:
    return "M" + " L".join(f"{frame.x(d):.1f},{frame.y(v):.1f}" for d, v in pts)


# ----------------------------------------------------------------------------- charts


def bar_tip(p: dict) -> str:
    lines = [f"${p['dist']:.3f} per share after tax (gross ${p['gross']:.3f})", f"IAUI ${p['iaui']:.2f} · GLD ${p['gld']:.2f}", f"{p['pct']:.2f}% of IAUI price"]
    return html.escape(json.dumps({"t": p["date"], "l": lines}))


def dot_tip(d, g: float, i: float) -> str:
    return html.escape(json.dumps({"t": d.date().isoformat(), "l": [f"GLD {g:+.2f}%", f"IAUI {i:+.2f}%"]}))


def chart_indexed(data: dict) -> str:
    px = data["px"]
    base = px.iloc[0]
    series = [("IAUI price", 1, "iaui"), ("IAUI total return (incl. distributions)", 3, "iaui_tr"), ("GLD price", 2, "gld")]
    pts = {k: [(d.date().isoformat(), float(v / base[k] * 100)) for d, v in px[k].items()] for _, _, k in series}
    vals = [v for s in pts.values() for _, v in s]
    x0, x1 = px.index[0].date(), px.index[-1].date()
    f = Frame(x0, x1, min(vals), max(vals), 340, "idx")
    lines = "".join(f'<path class="ln" stroke="var(--series-{slot})" d="{path(f, pts[k])}"/>' for _, slot, k in series)
    labels = []
    for name, slot, k in series:  # direct label at the line end
        d, v = pts[k][-1]
        labels.append(f'<text class="dl" x="{f.x(d) + 6:.1f}" y="{f.y(v) + 4:.1f}">{html.escape(name.split(" (")[0])} '
                      f'{v - 100:+.1f}%</text>')
    hover = {"dates": [d for d, _ in pts["iaui"]],
             "series": [{"name": n, "slot": s, "vals": [round(v, 2) for _, v in pts[k]],
                         "raw": [round(float(x), 2) for x in px[k]]} for n, s, k in series],
             "x": [round(f.x(d), 1) for d, _ in pts["iaui"]], "y": [[round(f.y(v), 1) for _, v in pts[k]] for _, _, k in series],
             "top": f.top, "bot": f.bot}
    return (f'<svg class="ch" viewBox="0 0 {W + 210} 340" data-hover=\'{html.escape(json.dumps(hover))}\'>'
            f'<line class="hund" x1="{PAD_L}" x2="{W - PAD_R}" y1="{f.y(100):.1f}" y2="{f.y(100):.1f}"/>{f.axes()}{lines}'
            f'{"".join(labels)}<g class="cross"></g><rect class="ov" x="{PAD_L}" y="0" width="{W - PAD_L - PAD_R}" height="340"/>'
            "</svg>")


def chart_scatter(data: dict) -> str:
    r = data["r"] * 100
    s = data["stats"]
    lim = math.ceil(max(r.abs().max()) + 0.5)
    size = 440
    pad = 52

    def sx(v: float) -> float:
        return pad + (v + lim) / (2 * lim) * (size - pad - 16)

    def sy(v: float) -> float:
        return size - pad - (v + lim) / (2 * lim) * (size - pad - 16)

    ticks = [t for t in nice_ticks(-lim, lim, 6) if -lim <= t <= lim]
    grid = "".join(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{sy(lim):.1f}" y2="{sy(-lim):.1f}"/>'
                   f'<line class="grid" x1="{sx(-lim):.1f}" x2="{sx(lim):.1f}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                   f'<text class="ax" x="{sx(t):.1f}" y="{size - pad + 18}" text-anchor="middle">{t:g}%</text>'
                   f'<text class="ax" x="{pad - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}%</text>' for t in ticks)
    axes0 = (f'<line class="base" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{sy(lim):.1f}" y2="{sy(-lim):.1f}"/>'
             f'<line class="base" x1="{sx(-lim):.1f}" x2="{sx(lim):.1f}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    same = f'<line class="ref" x1="{sx(-lim):.1f}" y1="{sy(-lim):.1f}" x2="{sx(lim):.1f}" y2="{sy(lim):.1f}"/>'
    fit = (f'<line class="fit" x1="{sx(-lim):.1f}" y1="{sy(s["alpha"] * 100 - s["beta"] * lim):.1f}" '
           f'x2="{sx(lim):.1f}" y2="{sy(s["alpha"] * 100 + s["beta"] * lim):.1f}"/>')
    dots = "".join(
        f'<circle class="dot" cx="{sx(g):.1f}" cy="{sy(i):.1f}" r="4" data-tip=\'{dot_tip(d, g, i)}\'/>'
        for d, (i, g) in r[["iaui", "gld"]].iterrows())
    labels = (f'<text class="dl" x="{sx(lim) - 4:.1f}" y="{sy(lim) - 6:.1f}" text-anchor="end">if IAUI moved 1:1</text>'
              f'<text class="dl b" x="{sx(lim) - 4:.1f}" y="{sy(s["beta"] * lim) + 18:.1f}" text-anchor="end">'
              f'actual: IAUI ≈ {s["beta"]:.2f} × GLD</text>')
    titles = (f'<text class="ax" x="{(pad + size) / 2:.0f}" y="{size - 8}" text-anchor="middle">GLD daily move</text>'
              f'<text class="ax" transform="translate(12 {(size - pad) / 2:.0f}) rotate(-90)" text-anchor="middle">IAUI daily move</text>')
    return f'<svg class="ch sq" viewBox="0 0 {size} {size}">{grid}{axes0}{same}{dots}{fit}{labels}{titles}</svg>'


def chart_distributions(data: dict) -> str:
    dist = data["dist"]
    x0 = date.fromisoformat(dist[0]["date"]).replace(day=1)
    last = date.fromisoformat(dist[-1]["date"])
    x1 = date(last.year + (last.month // 12), last.month % 12 + 1, 1)
    out = []
    specs = [
        ("Distribution per IAUI share, after 30% tax", "dist", "dist", 0, True),
        ("GLD price on each ex-date", "gld", "usd0", 2, False),
        ("After-tax distribution as % of IAUI price (that month)", "pct", "pct", 0, False),
    ]
    for title, key, kind, slot, bars in specs:
        vals = [p[key] for p in dist]
        lo = 0 if bars else min(vals) - (max(vals) - min(vals)) * 0.25
        hi = max(vals) * (1.08 if bars else 1) + (0 if bars else (max(vals) - min(vals)) * 0.25)
        if key == "pct":
            lo, hi = 0.6, 0.8
        f = Frame(x0, x1, lo, hi, 170, kind)
        color = f"var(--series-{slot if slot else 1})"
        marks = []
        if bars:
            bw = 22
            for p in dist:
                x, y = f.x(p["date"]), f.y(p["dist"])
                hgt = f.bot - y
                marks.append(  # 4px rounded data-end, square at the baseline
                    f'<path class="bar" fill="{color}" d="M{x - bw / 2:.1f},{f.bot} V{y + 4:.1f} Q{x - bw / 2:.1f},{y:.1f} '
                    f'{x - bw / 2 + 4:.1f},{y:.1f} H{x + bw / 2 - 4:.1f} Q{x + bw / 2:.1f},{y:.1f} {x + bw / 2:.1f},{y + 4:.1f} '
                    f'V{f.bot} Z" data-tip=\'{bar_tip(p)}\'/>'
                    if hgt > 4 else "")
            peak = max(dist, key=lambda p: p["dist"])
            marks.append(f'<text class="dl" x="{f.x(peak["date"]):.1f}" y="{f.y(peak["dist"]) - 8:.1f}" '
                         f'text-anchor="middle">${peak["dist"]:.3f}</text>')
            marks.append(f'<text class="dl" x="{f.x(dist[-1]["date"]):.1f}" y="{f.y(dist[-1]["dist"]) - 8:.1f}" '
                         f'text-anchor="middle">${dist[-1]["dist"]:.3f}</text>')
        else:
            pts = [(p["date"], p[key]) for p in dist]
            marks.append(f'<path class="ln" stroke="{color}" d="{path(f, pts)}"/>')
            for p in dist:
                tip = {"t": p["date"], "l": [f"GLD ${p['gld']:.2f}" if key == "gld" else f"{p['pct']:.2f}% of IAUI ${p['iaui']:.2f}",
                                              f"distribution ${p['dist']:.3f}"]}
                marks.append(f'<circle class="mk" cx="{f.x(p["date"]):.1f}" cy="{f.y(p[key]):.1f}" r="4.5" '
                             f'fill="{color}" data-tip=\'{html.escape(json.dumps(tip))}\'/>')
            if key == "pct":
                y = f.y(NET)
                marks.insert(0, f'<line class="hund" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                                f'<text class="dl" x="{PAD_L + 6}" y="{y + 16:.1f}">{NET:.2f}% a month = 1% before tax</text>')
        out.append(f'<h3>{html.escape(title)}</h3><svg class="ch sm" viewBox="0 0 {W} 170">{f.axes(x_labels=key == "pct")}'
                   f'{"".join(marks)}</svg>')
    return "".join(out)


# ----------------------------------------------------------------------------- page


def render(data: dict) -> str:
    s = data["stats"]
    rows = "".join(f'<tr><td>{p["date"]}</td><td class="n">${p["gross"]:.3f}</td><td class="n">-${p["gross"] - p["dist"]:.3f}</td>'
                   f'<td class="n"><b>${p["dist"]:.3f}</b></td><td class="n">${p["iaui"]:.2f}</td>'
                   f'<td class="n">{p["pct"]:.2f}%</td><td class="n">{p["pct"] * 12:.1f}%</td><td class="n">${p["gld"]:.2f}</td></tr>'
                   for p in data["dist"])
    simple = f"""
<li><b>IAUI is a "gold with a paycheck" fund.</b> It holds gold exposure and sells options on it every month.
The money from selling those options is what it pays you each month. The price of that paycheck: when gold rallies, IAUI
gives up part of the gain.</li>
<li><b>Day to day, IAUI follows GLD very closely, but at about ¾ speed.</b> Their daily moves are
{s["corr_daily"]:.2f} correlated (1.00 = identical). On a typical day IAUI moves about {s["beta"]:.2f}× what GLD moves:
if GLD rises 1%, IAUI rises about {s["beta"]:.2f}%. It catches roughly {s["up_capture"] * 100:.0f}% of GLD's up days but
{s["down_capture"] * 100:.0f}% of the down days, so it gives up a little more on the way up than it saves on the way down.</li>
<li><b>The monthly payout is about 1% of IAUI's own price before tax, and you keep about {NET:.2f}% after the
{WHT * 100:.0f}% US withholding.</b> After tax, all {s["dist_n"]} payouts were between {s["pct_min"]:.2f}% and
{s["pct_max"]:.2f}% of the share price, about {s["pct_mean"] * 12:.1f}% a year in your pocket (about 12% before tax).
So the dollar amount is not fixed: it moves with IAUI's price, and IAUI's price moves with gold.</li>
<li><b>That is why payouts rose and then fell with gold.</b> As GLD climbed to ${s["gld_peak"]:,.0f} ({s["gld_peak_date"]}),
your after-tax payout grew from ${s["dist_first"]:.3f} (first payout) to ${s["dist_max"]:.3f} per share
(peak {s["dist_peak_date"]}). As gold fell back, it shrank to ${s["dist_last"]:.3f}. The payout tracks IAUI's price almost
perfectly ({s["corr_dist_iaui"]:.2f}) and GLD's less tightly ({s["corr_dist_gld"]:.2f}), because GLD only drives IAUI
at about ¾ strength.</li>
<li><b>The payouts come out of the share price.</b> Since launch, GLD is {s["ret_gld"]:+.1f}% but IAUI's price is
{s["ret_iaui"]:+.1f}%. Counting the {s["dist_n"]} payouts after tax (${s["dist_total"]:.2f} per share kept, of ${s["dist_total_gross"]:.2f}
paid), IAUI's total return is {s["ret_iaui_tr"]:+.1f}%. The price drifts down over time because ~12% a year is handed
out, and you only keep {NET * 100:.0f}% of it, so judge IAUI on price
<i>plus</i> payouts, never on price alone.</li>
<li><b>What it means for you:</b> IAUI is your gold bet with income attached. If gold rises, your payouts rise but IAUI lags
GLD. If gold falls, your payouts shrink and the price falls too, just a bit less than GLD. It is not a hedge against gold
falling, only a cushion.</li>"""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IAUI vs GLD</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#76756f;
 --grid:#e4e3df;--axis:#b9b8b3;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;--surface-1:#1a1a19;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a998f;--grid:#2e2e2c;--axis:#55544f;
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9a998f;--grid:#2e2e2c;--axis:#55544f;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}}
*{{box-sizing:border-box;margin:0}} body{{background:var(--surface-1)}}
.viz-root{{background:var(--surface-1);color:var(--text-primary);min-height:100vh;padding:40px 24px 64px;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;line-height:1.5}}
.wrap{{max-width:1100px;margin:0 auto}}
h1{{font-size:20px;font-weight:600}} h2{{font-size:16px;font-weight:600;margin-top:44px}}
h3{{font-size:13px;font-weight:600;margin:18px 0 2px;color:var(--text-secondary)}}
.sub{{color:var(--text-secondary);font-size:13px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:22px 0}}
.tile{{border:1px solid var(--grid);border-radius:8px;padding:12px 14px}}
.tile .k{{font-size:12px;color:var(--text-secondary)}} .tile .v{{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}}
.tile .h{{font-size:12px;color:var(--text-muted)}}
ul.simple{{padding-left:20px;margin-top:10px;font-size:15px;max-width:860px}} ul.simple li{{margin:8px 0}}
.legend{{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:6px 18px;margin:10px 0 4px;font-size:13px;color:var(--text-secondary)}}
.sw{{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}}
svg.ch{{display:block;width:100%;height:auto;overflow:visible}} svg.sq{{max-width:460px}}
.grid{{stroke:var(--grid);stroke-width:1}} .base{{stroke:var(--axis);stroke-width:1}}
.hund{{stroke:var(--axis);stroke-width:1;stroke-dasharray:3 4}}
.ax{{font-size:11px;fill:var(--text-muted);font-variant-numeric:tabular-nums}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.dl{{font-size:12px;fill:var(--text-secondary);font-variant-numeric:tabular-nums}} .dl.b{{fill:var(--text-primary);font-weight:600}}
.dot{{fill:var(--series-1);fill-opacity:.55;stroke:var(--surface-1);stroke-width:1}}
.dot:hover,.mk:hover,.bar:hover{{fill-opacity:1;stroke:var(--text-primary)}}
.mk{{stroke:var(--surface-1);stroke-width:2}}
.fit{{stroke:var(--text-primary);stroke-width:2}} .ref{{stroke:var(--axis);stroke-width:1;stroke-dasharray:4 4}}
.ov{{fill:transparent}} .cross line{{stroke:var(--axis);stroke-width:1}} .cross circle{{stroke:var(--surface-1);stroke-width:2}}
.split{{display:grid;grid-template-columns:460px 1fr;gap:32px;align-items:start}}
@media (max-width:860px){{.split{{grid-template-columns:1fr}}}}
.note{{font-size:13px;color:var(--text-secondary)}}
.tip{{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--grid);
 border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.5;box-shadow:0 4px 14px rgba(0,0,0,.12);opacity:0;
 font-variant-numeric:tabular-nums;z-index:9}}
.tip .m{{color:var(--text-secondary)}} .tip .s{{display:inline-block;width:10px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}}
details{{margin-top:20px;font-size:13px}} summary{{cursor:pointer;color:var(--text-secondary)}}
table{{border-collapse:collapse;width:100%;margin-top:10px}} td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}}
th{{font-weight:500;color:var(--text-secondary)}} td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>IAUI vs GLD since IAUI launched</h1>
<p class="sub">Daily closes {s["start"]} → {s["end"]} ({s["days"]} trading days) and all {s["dist_n"]} monthly IAUI
distributions · source: Yahoo Finance (yfinance)</p>
<div class="tiles">
<div class="tile"><div class="k">Daily-move correlation</div><div class="v">{s["corr_daily"]:.2f}</div><div class="h">1.00 = move identically</div></div>
<div class="tile"><div class="k">IAUI moves per 1% of GLD</div><div class="v">{s["beta"]:.2f}%</div><div class="h">beta, daily</div></div>
<div class="tile"><div class="k">Payout after tax, % of IAUI price</div><div class="v">{s["pct_mean"]:.2f}%</div><div class="h">per month · range {s["pct_min"]:.2f}–{s["pct_max"]:.2f}%</div></div>
<div class="tile"><div class="k">Since launch: GLD</div><div class="v">{s["ret_gld"]:+.1f}%</div><div class="h">price</div></div>
<div class="tile"><div class="k">Since launch: IAUI</div><div class="v">{s["ret_iaui_tr"]:+.1f}%</div><div class="h">price {s["ret_iaui"]:+.1f}% + payouts</div></div>
</div>
<h2>In simple words</h2><ul class="simple">{simple}</ul>
<h2>1 · Price since launch (start = 100)</h2>
<p class="note">All three start at 100 on {s["start"]} so they share one scale. The gap between the blue and aqua lines is the money IAUI paid out.</p>
<ul class="legend"><li><span class="sw" style="background:var(--series-1)"></span>IAUI price</li>
<li><span class="sw" style="background:var(--series-3)"></span>IAUI total return (after-tax distributions reinvested)</li>
<li><span class="sw" style="background:var(--series-2)"></span>GLD price</li></ul>
{chart_indexed(data)}
<div class="split"><div><h2>2 · Daily moves: IAUI vs GLD</h2>{chart_scatter(data)}</div>
<div><h2>&nbsp;</h2><p class="note" style="margin-top:28px">Each dot is one trading day. Dots hugging a line mean the two move together
(correlation {s["corr_daily"]:.2f}). The solid line's slope is {s["beta"]:.2f}: IAUI typically moves about ¾ as far as GLD.
The dashed line is what a 1:1 copy of GLD would look like. IAUI sits flatter because the monthly option sales cap part of the move.</p>
<p class="note" style="margin-top:12px">Yearly volatility: IAUI {s["vol_iaui"]:.1f}% vs GLD {s["vol_gld"]:.1f}%.
Gold's peak: GLD ${s["gld_peak"]:,.2f} on {s["gld_peak_date"]}; IAUI ${s["iaui_peak"]:,.2f} on {s["iaui_peak_date"]}.
Now: GLD ${s["gld_last"]:,.2f}, IAUI ${s["iaui_last"]:,.2f}.</p></div></div>
<h2>3 · How the monthly payout followed gold</h2>
<p class="note">Three small charts on the same month axis (one scale each, no double axis). Top: dollars paid per share.
Middle: GLD on the same dates. Bottom: after-tax payout ÷ IAUI price, which barely moves. All payouts are after {WHT * 100:.0f}% withholding.
The payout is a near-constant ~{NET:.1f}% (1% before tax) of
IAUI's price, so its dollar size rides gold up and down.</p>
{chart_distributions(data)}
<details open><summary>Table view: every distribution</summary><table>
<thead><tr><th>Ex-date</th><th class="n">Gross</th>
<th class="n">Tax ({WHT * 100:.0f}%)</th><th class="n">You keep</th><th class="n">IAUI price</th><th class="n">% of price</th>
<th class="n">Annualized</th><th class="n">GLD price</th></tr></thead><tbody>{rows}</tbody></table></details>
</div><div class="tip" id="tip"></div></div>
<script>
const tip=document.getElementById('tip');
function place(x,y){{const w=tip.offsetWidth,h=tip.offsetHeight;tip.style.left=Math.min(x+14,innerWidth-w-8)+'px';tip.style.top=Math.max(8,y-h-12)+'px';tip.style.opacity=1}}
document.querySelectorAll('[data-tip]').forEach(el=>{{el.addEventListener('mousemove',e=>{{const d=JSON.parse(el.dataset.tip);
 tip.innerHTML='<b>'+d.t+'</b><br>'+d.l.join('<br>');place(e.clientX,e.clientY)}});el.addEventListener('mouseleave',()=>tip.style.opacity=0)}});
document.querySelectorAll('svg[data-hover]').forEach(svg=>{{const H=JSON.parse(svg.dataset.hover),g=svg.querySelector('.cross'),ov=svg.querySelector('.ov');
 ov.addEventListener('mousemove',e=>{{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const q=p.matrixTransform(svg.getScreenCTM().inverse());
  let i=0,best=1e9;H.x.forEach((x,k)=>{{const d=Math.abs(x-q.x);if(d<best){{best=d;i=k}}}});
  g.innerHTML=`<line x1="${{H.x[i]}}" x2="${{H.x[i]}}" y1="${{H.top}}" y2="${{H.bot}}"/>`+H.series.map((s,k)=>`<circle cx="${{H.x[i]}}" cy="${{H.y[k][i]}}" r="4.5" fill="var(--series-${{s.slot}})"/>`).join('');
  tip.innerHTML='<b>'+H.dates[i]+'</b><br>'+H.series.map(s=>`<span class="s" style="background:var(--series-${{s.slot}})"></span>${{s.name.split(' (')[0]}}: ${{s.vals[i].toFixed(1)}} <span class="m">(${{s.name.startsWith('IAUI total')?'value $':'$'}}${{s.raw[i].toFixed(2)}})</span>`).join('<br>');
  place(e.clientX,e.clientY)}});
 ov.addEventListener('mouseleave',()=>{{g.innerHTML='';tip.style.opacity=0}})}});
</script></body></html>
"""


def main() -> int:
    data = load()
    OUT.write_text(render(data), encoding="utf-8")
    s = data["stats"]
    print(f"Wrote {OUT}")
    for k in ("corr_daily", "corr_level", "beta", "up_capture", "down_capture", "ret_gld", "ret_iaui", "ret_iaui_tr",
              "pct_mean", "corr_dist_iaui", "corr_dist_gld"):
        print(f"  {k:<16} {s[k]:.3f}")
    if "--no-open" not in sys.argv and sys.platform == "darwin":
        subprocess.run(["open", str(OUT)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
