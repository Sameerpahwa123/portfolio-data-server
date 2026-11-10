"""IAUI vs GLD research page -> research/iaui_gld.html (self-contained; data from Yahoo Finance via yfinance).

    .venv/bin/python research/iaui_gld.py [--no-open]

Charts: growth of $100 since IAUI launch (IAUI price, IAUI total return, GLD), daily-return scatter,
monthly distributions aligned with GLD and yield, distribution vs GLD price scatter, and a table view.
"""

from __future__ import annotations

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
import yfinance as yf

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "iaui_gld.html"
DATA = HERE / "data"
E = html.escape


# ----------------------------------------------------------------------------- data


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    h = {t: yf.Ticker(t).history(start="2025-06-01", auto_adjust=False, actions=True) for t in ("IAUI", "GLD")}
    for t in h:
        h[t].index = h[t].index.tz_localize(None).normalize()
    inception = date.fromtimestamp(yf.Ticker("IAUI").get_info().get("fundInceptionDate") or 0)
    df = pd.DataFrame({"iaui": h["IAUI"]["Close"], "iaui_tr": h["IAUI"]["Adj Close"], "gld": h["GLD"]["Close"]}).dropna()
    dist = h["IAUI"]["Dividends"][h["IAUI"]["Dividends"] > 0]
    rows = []
    for dt, amt in dist.items():
        i = min(df.index.searchsorted(dt), len(df) - 1)
        ex = df.index[i]
        rows.append({"ex": ex, "amount": float(amt), "iaui": float(df.iaui.iloc[i]), "gld": float(df.gld.iloc[i]),
                     "yield_pct": float(amt) * 12 / float(df.iaui.iloc[i]) * 100})
    m = pd.DataFrame(rows)
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "daily_prices.csv")
    m.to_csv(DATA / "iaui_distributions.csv", index=False)
    r = df.pct_change().dropna()
    beta, alpha = np.polyfit(r.gld, r.iaui, 1)
    up, dn = r.gld > 0, r.gld < 0
    stats = {
        "inception": inception.isoformat(), "start": df.index[0].date().isoformat(), "end": df.index[-1].date().isoformat(),
        "days": len(df), "corr_daily": r.iaui.corr(r.gld), "corr_daily_tr": r.iaui_tr.corr(r.gld),
        "corr_weekly": df.resample("W-FRI").last().pct_change().dropna().pipe(lambda w: w.iaui.corr(w.gld)),
        "corr_levels": df.iaui.corr(df.gld), "beta": beta, "alpha": alpha,
        "up_capture": r.iaui[up].mean() / r.gld[up].mean() * 100, "down_capture": r.iaui[dn].mean() / r.gld[dn].mean() * 100,
        "vol_iaui": r.iaui.std() * math.sqrt(252) * 100, "vol_gld": r.gld.std() * math.sqrt(252) * 100,
        "cum_iaui": (df.iaui.iloc[-1] / df.iaui.iloc[0] - 1) * 100, "cum_tr": (df.iaui_tr.iloc[-1] / df.iaui_tr.iloc[0] - 1) * 100,
        "cum_gld": (df.gld.iloc[-1] / df.gld.iloc[0] - 1) * 100, "dist_total": m.amount.sum(), "dist_n": len(m),
        "corr_dist_gld": m.amount.corr(m.gld), "corr_dist_iaui": m.amount.corr(m.iaui),
        "corr_yield_gld": m.yield_pct.corr(m.gld),
        "dist_beta": np.polyfit(m.gld, m.amount, 1)[0],
    }
    return df, m, stats


# ----------------------------------------------------------------------------- svg helpers


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round ticks that always enclose [lo, hi] (first tick <= lo, last tick >= hi)."""
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(k * mag for k in (1, 2, 2.5, 5, 10) if k * mag >= raw)
    first, last = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    return [round(first + i * step, 10) for i in range(round((last - first) / step) + 1)]


class Scale:
    def __init__(self, d0: float, d1: float, r0: float, r1: float) -> None:
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v: float) -> float:
        return self.r0 + (v - self.d0) / ((self.d1 - self.d0) or 1) * (self.r1 - self.r0)


def y_axis(ys: Scale, ticks: list[float], x0: float, x1: float, fmt) -> str:
    out = []
    for t in ticks:
        y = ys(t)
        out.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{E(fmt(t))}</text>')
    return "".join(out)


# ----------------------------------------------------------------------------- charts


def growth_chart(df: pd.DataFrame) -> str:
    W, H, L, R, T, B = 1000, 380, 56, 170, 16, 34
    idx = df / df.iloc[0] * 100
    series = [("gld", "GLD", "--s2"), ("iaui_tr", "IAUI total return", "--s3"), ("iaui", "IAUI price", "--s1")]
    lo, hi = float(idx.min().min()), float(idx.max().max())
    ticks = nice_ticks(lo - 2, hi + 2)
    xs = Scale(0, len(idx) - 1, L, W - R)
    ys = Scale(ticks[0], ticks[-1], H - B, T)
    parts = [y_axis(ys, ticks, L, W - R, lambda t: f"{t:g}")]
    parts.append(f'<line class="base" x1="{L}" x2="{W - R}" y1="{ys(100):.1f}" y2="{ys(100):.1f}"/>')
    # x ticks: first trading day of each quarter
    last_x = -1e9
    for i, d in enumerate(idx.index):
        if (i == 0 or (d.month in (1, 4, 7, 10) and idx.index[i - 1].month != d.month)) and xs(i) - last_x >= 80:
            parts.append(f'<text class="tick" x="{xs(i):.1f}" y="{H - B + 20}" text-anchor="middle">{d.strftime("%b %Y")}</text>')
            last_x = xs(i)
    ends = []
    for key, name, var in series:
        pts = " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(idx[key]))
        parts.append(f'<polyline class="line" points="{pts}" style="stroke:var({var})"/>')
        ends.append([ys(float(idx[key].iloc[-1])), name, var, float(idx[key].iloc[-1])])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels >= 30px apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 30)
    for y, name, var, v in ends:
        parts.append(f'<circle cx="{W - R + 10}" cy="{y - 4:.1f}" r="4" style="fill:var({var})"/>'
                     f'<text class="lab" x="{W - R + 20}" y="{y:.1f}">{E(name)}</text>'
                     f'<text class="val" x="{W - R + 20}" y="{y + 15:.1f}">{v:.1f} ({v - 100:+.1f}%)</text>')
    data = [{"d": d.strftime("%d %b %Y"), "x": round(xs(i), 1),
             **{k: round(float(idx[k].iloc[i]), 2) for k, _, _ in series},
             **{f"{k}_px": round(float(df[k].iloc[i]), 2) for k in ("iaui", "gld")}} for i, d in enumerate(idx.index)]
    parts.append(f'<line class="xhair" id="gx" x1="0" x2="0" y1="{T}" y2="{H - B}"/>')
    parts += [f'<circle class="hdot" id="gd_{k}" r="4.5" style="fill:var({v})"/>' for k, _, v in series]
    parts.append(f'<rect class="hit" id="ghit" x="{L}" y="{T}" width="{W - R - L}" height="{H - B - T}"/>')
    ymap = {"lo": ticks[0], "hi": ticks[-1], "top": T, "bot": H - B}
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Growth of 100 dollars since IAUI launch" '
            f'data-g=\'{E(json.dumps({"pts": data, "y": ymap}))}\' id="growth">{"".join(parts)}</svg>')


def scatter(xs_v: list[float], ys_v: list[float], tips: list[str], *, xlab: str, ylab: str, fit: tuple[float, float],
            diag: bool, fmt, labels: dict[int, str] | None = None, W: int = 480, H: int = 380, pad: float = 0.0,
            ident: str = "s") -> str:
    L, R, T, B = 60, 18, 16, 48
    xt = nice_ticks(min(xs_v) - pad, max(xs_v) + pad)
    yt = nice_ticks(min(ys_v) - pad, max(ys_v) + pad)
    xs, ys = Scale(xt[0], xt[-1], L, W - R), Scale(yt[0], yt[-1], H - B, T)
    parts = [f'<clipPath id="{ident}c"><rect x="{L}" y="{T}" width="{W - R - L}" height="{H - B - T}"/></clipPath>',
             y_axis(ys, yt, L, W - R, fmt[1])]
    for t in xt:
        parts.append(f'<text class="tick" x="{xs(t):.1f}" y="{H - B + 18}" text-anchor="middle">{E(fmt[0](t))}</text>')
    parts.append(f'<text class="axl" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">{E(xlab)}</text>'
                 f'<text class="axl" transform="translate(14,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle">{E(ylab)}</text>')
    if diag:  # y = x reference: points below it mean IAUI moved less than GLD
        a, b = max(xt[0], yt[0]), min(xt[-1], yt[-1])
        parts.append(f'<line class="ref" clip-path="url(#{ident}c)" x1="{xs(a):.1f}" y1="{ys(a):.1f}" x2="{xs(b):.1f}" y2="{ys(b):.1f}"/>'
                     f'<text class="refl" x="{xs(b) - 4:.1f}" y="{ys(b) + 14:.1f}" text-anchor="end">same move as GLD</text>')
    m, c = fit
    parts.append(f'<line class="fit" clip-path="url(#{ident}c)" x1="{xs(xt[0]):.1f}" y1="{ys(m * xt[0] + c):.1f}" x2="{xs(xt[-1]):.1f}" '
                 f'y2="{ys(m * xt[-1] + c):.1f}"/>')
    for i, (x, y) in enumerate(zip(xs_v, ys_v)):
        parts.append(f'<circle class="dot" cx="{xs(x):.1f}" cy="{ys(y):.1f}" r="4"/>'
                     f'<circle class="dhit" cx="{xs(x):.1f}" cy="{ys(y):.1f}" r="8" data-tip="{E(tips[i])}"/>')
        if labels and i in labels:
            right = xs(x) > W - R - 70
            parts.append(f'<text class="val" x="{xs(x) + (-8 if right else 8):.1f}" y="{ys(y) - 8:.1f}" '
                         f'text-anchor="{"end" if right else "start"}">{E(labels[i])}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{E(ylab)} vs {E(xlab)}" class="sc" id="{ident}">{"".join(parts)}</svg>'


def distribution_panels(m: pd.DataFrame) -> str:
    """Three panels on one shared month axis (small multiples, never a dual axis)."""
    W, L, R = 1000, 64, 24
    n = len(m)
    band = (W - L - R) / n
    cx = [L + band * (i + 0.5) for i in range(n)]
    panels = [("Distribution per IAUI share", "amount", lambda v: f"${v:.2f}", 0.0, 150, "bar"),
              ("GLD price on the ex-date", "gld", lambda v: f"${v:,.0f}", None, 130, "line"),
              ("IAUI annualized distribution yield (12 × monthly ÷ price)", "yield_pct", lambda v: f"{v:.1f}%", None, 120, "line")]
    parts, y0 = [], 10
    for title, key, fmt, floor, ph, kind in panels:
        vals = m[key].astype(float)
        ticks = nice_ticks(floor if floor is not None else float(vals.min()), float(vals.max()), 3)
        top, bot = y0 + 26, y0 + 26 + ph
        ys = Scale(ticks[0], ticks[-1], bot, top)
        parts.append(f'<text class="ptitle" x="{L}" y="{y0 + 12}">{E(title)}</text>')
        parts.append(y_axis(ys, ticks, L, W - R, fmt))
        var = "--s2" if key == "gld" else "--s1"
        if kind == "bar":
            bw = band - 2 - max(4, band * 0.28)  # thin bars, 2px+ surface gap
            for i, v in enumerate(vals):
                y = ys(v)
                h = bot - y
                r = min(4, bw / 2, h)
                x = cx[i] - bw / 2
                parts.append(f'<path class="bar" style="fill:var({var})" d="M{x:.1f},{bot} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
                             f'H{x + bw - r:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{bot} Z"/>')
            for i in (0, int(vals.idxmax()), n - 1):  # selective direct labels: first, peak, latest
                parts.append(f'<text class="val" x="{cx[i]:.1f}" y="{ys(vals.iloc[i]) - 6:.1f}" text-anchor="middle">{fmt(vals.iloc[i])}</text>')
        else:
            pts = " ".join(f"{cx[i]:.1f},{ys(v):.1f}" for i, v in enumerate(vals))
            parts.append(f'<polyline class="line" points="{pts}" style="stroke:var({var})"/>')
            parts += [f'<circle class="mk" cx="{cx[i]:.1f}" cy="{ys(v):.1f}" r="4" style="fill:var({var})"/>' for i, v in enumerate(vals)]
        y0 = bot + 22
    for i, d in enumerate(m.ex):
        if i % 2 == 0 or i == n - 1:
            parts.append(f'<text class="tick" x="{cx[i]:.1f}" y="{y0 + 4}" text-anchor="middle">{d.strftime("%b %y")}</text>')
    H = y0 + 16
    for i, row in m.iterrows():  # one hover column per month spanning all three panels
        tip = (f"<b>{row.ex.strftime('%d %b %Y')}</b> (ex-date)<br>Distribution ${row.amount:.3f}/share<br>"
               f"GLD ${row.gld:,.2f} · IAUI ${row.iaui:,.2f}<br>Annualized yield {row.yield_pct:.2f}%")
        parts.append(f'<rect class="col" x="{cx[i] - band / 2:.1f}" y="0" width="{band:.1f}" height="{H - 20}" data-tip="{E(tip)}"/>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Monthly distributions with GLD price and yield" id="dist">{"".join(parts)}</svg>'


# ----------------------------------------------------------------------------- page


def render(df: pd.DataFrame, m: pd.DataFrame, s: dict) -> str:
    r = df.pct_change().dropna() * 100
    tips = [f"<b>{d.strftime('%d %b %Y')}</b><br>GLD {g:+.2f}% · IAUI {i:+.2f}%" for d, g, i in zip(r.index, r.gld, r.iaui)]
    sc1 = scatter(list(r.gld), list(r.iaui), tips, xlab="GLD daily return", ylab="IAUI daily return (price)",
                  fit=(s["beta"], s["alpha"] * 100), diag=True, fmt=(lambda t: f"{t:+g}%", lambda t: f"{t:+g}%"), ident="sc1")
    peak = int(m.amount.idxmax())
    lab = {0: m.ex.iloc[0].strftime("%b %y"), peak: m.ex.iloc[peak].strftime("%b %y"), len(m) - 1: m.ex.iloc[-1].strftime("%b %y")}
    tips2 = [f"<b>{row.ex.strftime('%d %b %Y')}</b><br>${row.amount:.3f}/share · GLD ${row.gld:,.2f}" for _, row in m.iterrows()]
    fit2 = tuple(np.polyfit(m.gld, m.amount, 1))
    sc2 = scatter(list(m.gld), list(m.amount), tips2, xlab="GLD price on ex-date", ylab="IAUI distribution per share",
                  fit=fit2, diag=False, fmt=(lambda t: f"${t:,.0f}", lambda t: f"${t:.2f}"), labels=lab, ident="sc2")
    rows = "".join(f"<tr><td>{row.ex.strftime('%d %b %Y')}</td><td class='n'>${row.amount:.3f}</td><td class='n'>${row.iaui:,.2f}</td>"
                   f"<td class='n'>{row.yield_pct:.2f}%</td><td class='n'>${row.gld:,.2f}</td></tr>" for _, row in m.iterrows())
    tiles = [("Daily-return correlation", f"{s['corr_daily']:.2f}", "IAUI vs GLD, price returns"),
             ("Beta to GLD", f"{s['beta']:.2f}", "IAUI moves ~¾ of GLD's daily move"),
             ("Up / down capture", f"{s['up_capture']:.0f}% / {s['down_capture']:.0f}%", "of GLD's average up / down day"),
             ("Distributions paid", f"${s['dist_total']:.2f}", f"per share, {s['dist_n']} monthly payments")]
    tile_html = "".join(f'<div class="tile"><div class="tl">{E(a)}</div><div class="tv">{E(b)}</div><div class="ts">{E(c)}</div></div>'
                        for a, b, c in tiles)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IAUI vs GLD</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#76756f;
 --grid:#e9e8e4;--axis:#c9c8c2;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;--surface-1:#1a1a19;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a998f;--grid:#2c2c2a;--axis:#4a4a46;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9a998f;--grid:#2c2c2a;--axis:#4a4a46;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
*{{box-sizing:border-box;margin:0}} body{{background:var(--surface-1)}}
.viz-root{{background:var(--surface-1);color:var(--text-primary);padding:40px 24px 56px;min-height:100vh;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto}}
h1{{font-size:20px;font-weight:600}} h2{{font-size:15px;font-weight:600;margin:44px 0 4px}}
.sub,.cap{{color:var(--text-secondary);font-size:13px;margin-top:4px;line-height:1.5;max-width:860px}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:24px}}
@media (max-width:820px){{.tiles{{grid-template-columns:repeat(2,1fr)}}.two{{grid-template-columns:1fr!important}}}}
.tile{{border:1px solid var(--grid);border-radius:8px;padding:14px 16px}}
.tl{{font-size:12px;color:var(--text-secondary)}} .tv{{font-size:28px;font-weight:600;font-variant-numeric:tabular-nums;margin:4px 0}}
.ts{{font-size:12px;color:var(--text-muted)}}
.legend{{list-style:none;padding:0;display:flex;gap:18px;margin:12px 0 0;font-size:13px;color:var(--text-secondary)}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
svg{{display:block;width:100%;height:auto;overflow:visible;margin-top:8px}}
.grid{{stroke:var(--grid);stroke-width:1}} .base{{stroke:var(--axis);stroke-width:1}}
.tick{{font-size:11px;fill:var(--text-muted);font-variant-numeric:tabular-nums}}
.axl{{font-size:12px;fill:var(--text-secondary)}} .ptitle{{font-size:12px;font-weight:600;fill:var(--text-secondary)}}
.line{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}}
.lab{{font-size:13px;font-weight:600;fill:var(--text-primary)}} .val{{font-size:12px;fill:var(--text-secondary);font-variant-numeric:tabular-nums}}
.mk{{stroke:var(--surface-1);stroke-width:2}} .bar{{stroke:none}}
.dot{{fill:var(--s1);fill-opacity:.6;stroke:var(--surface-1);stroke-width:1}} .dhit{{fill:transparent}}
.dhit:hover+.dot,.dot.on{{fill-opacity:1}}
.fit{{stroke:var(--text-primary);stroke-width:2}} .ref{{stroke:var(--axis);stroke-width:1;stroke-dasharray:4 4}}
.refl{{font-size:11px;fill:var(--text-muted)}}
.xhair{{stroke:var(--axis);stroke-width:1;opacity:0}} .hdot{{stroke:var(--surface-1);stroke-width:2;opacity:0}}
.hit,.col{{fill:transparent}} .col:hover{{fill:var(--text-primary);fill-opacity:.05}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:32px}}
.tip{{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--grid);
 border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.55;box-shadow:0 4px 14px rgba(0,0,0,.14);opacity:0;z-index:5;
 font-variant-numeric:tabular-nums}}
details{{margin-top:40px;font-size:13px}} summary{{cursor:pointer;color:var(--text-secondary)}}
table{{border-collapse:collapse;width:100%;margin-top:10px}} td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}}
th{{font-weight:500;color:var(--text-secondary)}} td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
.src{{margin-top:28px;font-size:12px;color:var(--text-muted);line-height:1.5}}
</style></head>
<body><div class="viz-root"><div class="wrap">
<h1>IAUI vs GLD since IAUI's launch</h1>
<p class="sub">NEOS Gold High Income ETF (IAUI), inception {E(s['inception'])}. Daily closes {E(s['start'])} – {E(s['end'])}
({s['days']} trading days) and every monthly distribution, from Yahoo Finance.</p>
<div class="tiles">{tile_html}</div>

<h2>Growth of $100 since IAUI launch</h2>
<p class="cap">IAUI's share price is {s['cum_iaui']:+.1f}% while GLD is {s['cum_gld']:+.1f}%. Reinvesting the distributions
(total return) brings IAUI to {s['cum_tr']:+.1f}%: the income makes up for the falling share price, but still trails holding GLD.</p>
<ul class="legend"><li><span class="sw" style="background:var(--s2)"></span>GLD</li>
<li><span class="sw" style="background:var(--s3)"></span>IAUI total return (distributions reinvested)</li>
<li><span class="sw" style="background:var(--s1)"></span>IAUI price</li></ul>
{growth_chart(df)}

<div class="two">
<div><h2>Daily returns: IAUI vs GLD</h2>
<p class="cap">Correlation {s['corr_daily']:.2f} (weekly {s['corr_weekly']:.2f}). The fitted line's slope is the beta,
{s['beta']:.2f}: IAUI moves about ¾ as much as GLD, and every point under the dashed line is a day IAUI moved less.
Its upside is capped harder ({s['up_capture']:.0f}% capture) than its downside ({s['down_capture']:.0f}%). That is the
cost of selling options for income.</p>
{sc1}</div>
<div><h2>Distribution vs gold price</h2>
<p class="cap">Correlation {s['corr_dist_gld']:.2f} with GLD's price on the ex-date (about ${s['dist_beta'] * 100:.2f} more per
share for every $100 on GLD), but {s['corr_dist_iaui']:.2f} with IAUI's own price. The payout is set as a share of IAUI's price,
so it follows gold only as far as IAUI does. Only {s['dist_n']} months of data, so treat the slope as rough.</p>
{sc2}</div>
</div>

<h2>How the monthly distribution moved with gold</h2>
<p class="cap">The payout rose from ${m.amount.iloc[0]:.3f} to a ${m.amount.max():.3f} peak in
{E(m.ex.iloc[int(m.amount.idxmax())].strftime('%B %Y'))}, when GLD was near its high, and has fallen to ${m.amount.iloc[-1]:.3f}
as gold came off. The yield barely moved ({m.yield_pct.min():.1f}–{m.yield_pct.max():.1f}%): the dollar amount follows the price,
the percentage stays near 12%. Price-level correlations are inflated by shared trends (IAUI vs GLD price levels {s['corr_levels']:.2f});
the daily-return figure above is the reliable one.</p>
{distribution_panels(m)}

<details open><summary>Table view: monthly distributions</summary><table>
<thead><tr><th>Ex-date</th><th class="n">Distribution / share</th><th class="n">IAUI close</th><th class="n">Annualized yield</th>
<th class="n">GLD close</th></tr></thead><tbody>{rows}</tbody></table></details>
<p class="src">Source: Yahoo Finance via yfinance, fetched {date.today().isoformat()}. Prices are unadjusted closes; total return uses
Yahoo's dividend-adjusted close. Yield = 12 × distribution ÷ IAUI close on the ex-date. Raw data: research/data/*.csv.</p>
</div><div class="tip" id="tip"></div></div>
<script>
const tip=document.getElementById('tip');
function place(x,y){{const w=tip.offsetWidth,h=tip.offsetHeight;tip.style.left=Math.min(x+14,innerWidth-w-8)+'px';
 tip.style.top=(y-h-12<8?y+16:y-h-12)+'px';tip.style.opacity=1}}
function hide(){{tip.style.opacity=0}}
// crosshair on the growth chart
const g=document.getElementById('growth'),G=JSON.parse(g.dataset.g),hit=document.getElementById('ghit'),gx=document.getElementById('gx');
const keys=[['gld','GLD'],['iaui_tr','IAUI total return'],['iaui','IAUI price']];
function yOf(v){{return G.y.bot-(v-G.y.lo)/(G.y.hi-G.y.lo)*(G.y.bot-G.y.top)}}
hit.addEventListener('mousemove',ev=>{{const pt=g.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;
 const sx=pt.matrixTransform(g.getScreenCTM().inverse()).x;let best=G.pts[0];
 for(const p of G.pts)if(Math.abs(p.x-sx)<Math.abs(best.x-sx))best=p;
 gx.setAttribute('x1',best.x);gx.setAttribute('x2',best.x);gx.style.opacity=1;
 for(const [k] of keys){{const d=document.getElementById('gd_'+k);d.setAttribute('cx',best.x);d.setAttribute('cy',yOf(best[k]));d.style.opacity=1}}
 tip.innerHTML=`<b>${{best.d}}</b><br>`+keys.map(([k,n])=>`${{n}}: ${{best[k].toFixed(1)}} (${{(best[k]-100>=0?'+':'')+(best[k]-100).toFixed(1)}}%)`).join('<br>')
  +`<br><span style="color:var(--text-secondary)">IAUI $${{best.iaui_px.toFixed(2)}} · GLD $${{best.gld_px.toFixed(2)}}</span>`;
 place(ev.clientX,ev.clientY)}});
hit.addEventListener('mouseleave',()=>{{gx.style.opacity=0;document.querySelectorAll('.hdot').forEach(d=>d.style.opacity=0);hide()}});
// per-mark tooltips (scatter dots, month columns)
document.querySelectorAll('[data-tip]').forEach(el=>{{
 el.addEventListener('mousemove',ev=>{{tip.innerHTML=el.dataset.tip;place(ev.clientX,ev.clientY)}});
 el.addEventListener('mouseleave',hide)}});
</script></body></html>
"""


def main() -> int:
    df, m, stats = load()
    OUT.write_text(render(df, m, stats), encoding="utf-8")
    print(f"Wrote {OUT}")
    for k in ("corr_daily", "corr_weekly", "beta", "up_capture", "down_capture", "cum_iaui", "cum_tr", "cum_gld",
              "corr_dist_gld", "corr_dist_iaui", "corr_levels", "dist_total"):
        print(f"  {k:<16} {stats[k]:.3f}")
    if "--no-open" not in sys.argv and sys.platform == "darwin":
        subprocess.run(["open", str(OUT)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
