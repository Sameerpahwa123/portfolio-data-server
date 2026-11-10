"""Gold-crash simulation for the IAUI + GLD part of the portfolio -> ./sim_gold.html (self-contained).

    .venv/bin/python sim_gold.py [--no-open]

Question: if GLD falls to 50% of its all-time high, is it wise to reinvest the after-tax IAUI payouts into GLD
every month, and do the payouts hold up?

Model (calibrated on IAUI's full history from Yahoo Finance, validated: reproduces IAUI's actual total return):
  IAUI daily total return = alpha + beta * GLD daily return + noise
  IAUI pays PAYOUT of its price every ~21 trading days; you receive (1 - WHT) of it.
GLD paths: scenario waypoints in log-price, plus daily noise at GLD's realized volatility, pinned to the waypoints
(Brownian bridge); after the last waypoint the path wanders freely with no drift.
Strategies compared on every path:
  A  hold both, keep the payouts as cash          B  reinvest each payout into GLD (the plan)
  C  reinvest each payout into IAUI               D  switch all IAUI into GLD today, no payouts after
"""

from __future__ import annotations

import html
import json
import math
import subprocess
import sys
import warnings
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portfolio_mcp import analytics as an  # noqa: E402
from portfolio_mcp.config import load_config  # noqa: E402
from portfolio_mcp.service import PortfolioService  # noqa: E402

OUT = Path(__file__).resolve().parent / "sim_gold.html"
DAYS = 1260  # 5 years of trading days
EX_EVERY = 21  # IAUI pays monthly
WHT = 0.30
N_SCEN = 400  # noisy paths per named scenario
N_MC = 3000  # random scenarios
SEED = 7


# ----------------------------------------------------------------------------- calibration + holdings


def calibrate() -> dict:
    h = yf.Ticker("IAUI").history(period="max", auto_adjust=False)
    g = yf.Ticker("GLD").history(start=h.index[0].date(), auto_adjust=False)
    div = yf.Ticker("IAUI").dividends
    div.index = div.index.tz_localize(None).normalize()
    px = pd.DataFrame({"i": h["Close"], "g": g["Close"]}).dropna()
    px.index = px.index.tz_localize(None).normalize()
    dv = div.reindex(px.index).fillna(0)
    tri = ((px["i"] + dv) / px["i"].shift(1) - 1).dropna()
    rg = px["g"].pct_change().dropna()
    beta, alpha = np.polyfit(rg, tri, 1)
    resid = tri - (alpha + beta * rg)
    model_tr = float(np.prod(1 + alpha + beta * rg) - 1)
    pay = [amt / px["i"].iloc[min(px.index.searchsorted(dt), len(px) - 1)] for dt, amt in div.items()]
    gld_ath = float(px["g"].max())
    return {"alpha_d": float(alpha), "beta": float(beta), "resid_d": float(resid.std()),
            "gld_vol_d": float(rg.std()), "payout": float(np.mean(pay)), "payout_sd": float(np.std(pay)),
            "actual_tr": float(np.prod(1 + tri) - 1), "model_tr": model_tr,
            "gld_ath": gld_ath, "gld_ath_date": px["g"].idxmax().date().isoformat(),
            "gld_now": float(px["g"].iloc[-1]), "iaui_now": float(px["i"].iloc[-1]),
            "asof": px.index[-1].date().isoformat(), "since": px.index[0].date().isoformat()}


def holdings() -> dict:
    svc = PortfolioService(load_config())
    out = {}
    with closing(svc.db.connect()) as conn:
        for _, r in an.latest_positions(conn, svc._accounts(conn, "gold")):
            out[r["symbol"]] = {"qty": float(r["position"]), "cost": float(r["cost_basis_money"]),
                                "as_of": r["report_date"]}
    return out


# ----------------------------------------------------------------------------- paths


def bridge_paths(waypoints: list[tuple[int, float]], n: int, vol_d: float, rng: np.random.Generator) -> np.ndarray:
    """GLD prices (n, DAYS+1): log-linear skeleton through waypoints + noise pinned at each waypoint."""
    t = np.arange(DAYS + 1)
    wd = [w[0] for w in waypoints]
    wl = np.log([w[1] for w in waypoints])
    skel = np.interp(t, wd, wl)  # flat after the last waypoint
    noise = np.concatenate([np.zeros((n, 1)), np.cumsum(rng.normal(0, vol_d, (n, DAYS)), axis=1)], axis=1)
    pinned = noise.copy()
    for (a, _), (b, _) in zip(waypoints[:-1], waypoints[1:]):
        seg = slice(a, b + 1)
        frac = (t[seg] - a) / max(1, b - a)
        pinned[:, seg] = noise[:, seg] - (noise[:, [a]] + frac * (noise[:, [b]] - noise[:, [a]]))
    last = waypoints[-1][0]
    if last < DAYS:  # free random walk after the last waypoint, starting from the pinned point
        pinned[:, last:] = noise[:, last:] - noise[:, [last]]
    return np.exp(skel + pinned)


def scenarios(c: dict) -> list[dict]:
    now, bot, ath = c["gld_now"], c["gld_ath"] / 2, c["gld_ath"]
    m = 21  # trading days per month
    return [
        {"key": "control", "name": "No crash (control)", "desc": "GLD drifts sideways around today's price",
         "wp": [(0, now), (DAYS, now)]},
        {"key": "crash_low", "name": "Sharp crash, stays low", "desc": f"to ${bot:,.0f} in 3 months, then sideways",
         "wp": [(0, now), (3 * m, bot)]},
        {"key": "crash_v", "name": "Sharp crash, V recovery", "desc": "to the bottom in 3 months, back to today's price by year 2",
         "wp": [(0, now), (3 * m, bot), (24 * m, now)]},
        {"key": "crash_ath", "name": "Crash, then new highs", "desc": f"to the bottom in 3 months, back to the ${ath:,.0f} high by year 5",
         "wp": [(0, now), (3 * m, bot), (DAYS, ath)]},
        {"key": "grind2", "name": "Slow 2-year slide", "desc": "steady decline to the bottom over 24 months, then sideways",
         "wp": [(0, now), (24 * m, bot)]},
        {"key": "grind5", "name": "Very slow 5-year slide", "desc": "bottom reached only at year 5",
         "wp": [(0, now), (DAYS, bot)]},
        {"key": "choppy", "name": "Choppy bear market", "desc": "falls in waves with sharp bear-market rallies, bottom at year 3",
         "wp": [(0, now), (6 * m, 300), (12 * m, 350), (18 * m, 270), (24 * m, 320), (36 * m, bot)]},
        {"key": "overshoot", "name": "Overshoot to −60%", "desc": f"panic to ${ath * .4:,.0f} in 9 months, settles at ${bot:,.0f}",
         "wp": [(0, now), (9 * m, ath * 0.4), (24 * m, bot)]},
        {"key": "partial", "name": "Drop, partial recovery", "desc": "bottom at year 1, climbs back to $320 by year 5",
         "wp": [(0, now), (12 * m, bot), (DAYS, 320.0)]},
    ]


# ----------------------------------------------------------------------------- strategies


def commission(value: np.ndarray, price: np.ndarray) -> np.ndarray:
    """IBKR fixed pricing as seen on your own small trades: $0.005/share, min $1, capped at 1% of trade value."""
    return np.minimum(np.maximum(1.0, 0.005 * value / price), 0.01 * value)


def simulate(gld: np.ndarray, c: dict, hold: dict, alpha_d: float, rng: np.random.Generator) -> dict:
    n = gld.shape[0]
    rg = gld[:, 1:] / gld[:, :-1] - 1
    tr = alpha_d + c["beta"] * rg + rng.normal(0, c["resid_d"], rg.shape)
    iaui = np.empty_like(gld)
    iaui[:, 0] = c["iaui_now"]
    s0_i, s0_g = hold["IAUI"]["qty"], hold["GLD"]["qty"]
    sh = {k: np.full(n, s0_i) for k in "ABC"}
    gs = {k: np.full(n, s0_g) for k in "ABC"}
    cash = {k: np.zeros(n) for k in "ABCD"}
    # D: switch IAUI -> GLD today (two trades, commission each)
    sale = s0_i * c["iaui_now"]
    sale -= commission(np.array([sale]), np.array([c["iaui_now"]]))[0]
    buy = sale - commission(np.array([sale]), np.array([c["gld_now"]]))[0]
    gs_d = np.full(n, s0_g + buy / c["gld_now"])
    payouts_net = {k: np.zeros(n) for k in "AC"}
    monthly = {k: [] for k in "AC"}
    gld_bought, gld_spent = np.zeros(n), np.zeros(n)
    for t in range(1, DAYS + 1):
        iaui[:, t] = iaui[:, t - 1] * (1 + tr[:, t - 1])
        if t % EX_EVERY == 0:  # ex-date: price drops by the gross payout, you receive 70% of it
            gross_ps = c["payout"] * iaui[:, t]
            iaui[:, t] -= gross_ps
            net_ps = gross_ps * (1 - WHT)
            for k in "ABC":
                amt = net_ps * sh[k]
                if k == "A":
                    cash["A"] += amt
                elif k == "B":
                    cm = commission(amt, gld[:, t])
                    q = (amt - cm) / gld[:, t]
                    gs["B"] += q
                    gld_bought += q
                    gld_spent += amt
                else:
                    sh["C"] += (amt - commission(amt, iaui[:, t])) / iaui[:, t]
            payouts_net["A"] += net_ps * sh["A"]
            payouts_net["C"] += net_ps * (sh["C"] - (net_ps * sh["C"] / iaui[:, t]) * 0)  # paid on pre-DRIP shares
            monthly["A"].append(net_ps * sh["A"])
            monthly["C"].append(net_ps * sh["C"])
    gT, iT = gld[:, -1], iaui[:, -1]
    final = {k: sh[k] * iT + gs[k] * gT + cash[k] for k in "ABC"}
    final["D"] = gs_d * gT
    return {"final": final, "payouts": payouts_net, "monthly": {k: np.array(v).T for k, v in monthly.items()},
            "gld_bought": gld_bought, "gld_spent": gld_spent, "gld_T": gT, "iaui_T": iT, "iaui_path": iaui}


def q(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p))


# ----------------------------------------------------------------------------- run


def run() -> dict:
    rng = np.random.default_rng(SEED)
    c = calibrate()
    hold = holdings()
    today = hold["IAUI"]["qty"] * c["iaui_now"] + hold["GLD"]["qty"] * c["gld_now"]
    cost = hold["IAUI"]["cost"] + hold["GLD"]["cost"]
    now_monthly = hold["IAUI"]["qty"] * c["iaui_now"] * c["payout"] * (1 - WHT)
    res = []
    for s in scenarios(c):
        gld = bridge_paths(s["wp"], N_SCEN, c["gld_vol_d"], rng)
        r = simulate(gld, c, hold, c["alpha_d"], rng)
        f = r["final"]
        mA = r["monthly"]["A"]
        res.append({
            **{k: s[k] for k in ("key", "name", "desc")}, "wp": s["wp"],
            "gld_med": [float(x) for x in np.median(gld, axis=0)[::5]],
            "gld_samples": [[float(x) for x in gld[i, ::10]] for i in range(12)],
            "final": {k: {"p10": q(v, 10), "p50": q(v, 50), "p90": q(v, 90)} for k, v in f.items()},
            "adv": {k: {"p10": q(f[k] - f["A"], 10), "p50": q(f[k] - f["A"], 50), "p90": q(f[k] - f["A"], 90),
                        "win": float(np.mean(f[k] > f["A"]))} for k in "BCD"},
            "payout_total": q(r["payouts"]["A"], 50),
            "monthly_med": [float(x) for x in np.median(mA, axis=0)],
            "monthly_p10": [float(x) for x in np.percentile(mA, 10, axis=0)],
            "monthly_p90": [float(x) for x in np.percentile(mA, 90, axis=0)],
            "monthly_end": q(mA[:, -1], 50),
            "gld_bought": q(r["gld_bought"], 50), "gld_spent": q(r["gld_spent"], 50),
            "avg_buy": q(r["gld_spent"] / np.maximum(r["gld_bought"], 1e-9), 50),
            "gld_T": q(r["gld_T"], 50), "iaui_T": q(r["iaui_T"], 50),
        })
    # Monte Carlo: bottom always at 50% of the ATH; time to bottom 1-48 months; GLD at year 5 anywhere from the
    # bottom back to the old high
    T = rng.integers(21, 48 * 21 + 1, N_MC)
    E = rng.uniform(c["gld_ath"] / 2, c["gld_ath"], N_MC)
    gld = np.vstack([bridge_paths([(0, c["gld_now"]), (int(t), c["gld_ath"] / 2), (DAYS, float(e))], 1,
                                  c["gld_vol_d"], rng) for t, e in zip(T, E)])
    mc = {}
    for label, a_yr in (("fitted", c["alpha_d"] * 252), ("zero", 0.0), ("plus5", 0.05)):
        r = simulate(gld, c, hold, a_yr / 252, np.random.default_rng(SEED + 1))
        f = r["final"]
        mc[label] = {"alpha_yr": a_yr,
                     **{f"win_{k}": float(np.mean(f[k] > f["A"])) for k in "BCD"},
                     **{f"adv_{k}_p50": q(f[k] - f["A"], 50) for k in "BCD"},
                     "final_A_p50": q(f["A"], 50), "final_B_p50": q(f["B"], 50),
                     "monthly_end_p50": q(r["monthly"]["A"][:, -1], 50),
                     "monthly_end_ratio_p50": q(r["monthly"]["A"][:, -1] / now_monthly, 50)}
        if label == "fitted":
            mc["points"] = [[round(float(g), 2), round(float(d), 2), int(t)]
                            for g, d, t in zip(r["gld_T"], f["B"] - f["A"], T)]
            # where does B start beating A? smallest GLD end-price bucket with >50% wins
            bins = np.arange(250, 500, 25)
            idx = np.digitize(r["gld_T"], bins)
            mc["win_by_end"] = [{"from": float(bins[i - 1]) if i else float(c["gld_ath"] / 2),
                                 "win": float(np.mean((f["B"] > f["A"])[idx == i])), "n": int(np.sum(idx == i))}
                                for i in range(len(bins) + 1) if np.sum(idx == i) > 20]
            b1, a1 = np.polyfit(r["gld_T"], f["B"] - f["A"], 1)  # DCA advantage is ~linear in GLD's year-5 price
            mc["breakeven_gld"] = float(-a1 / b1)
            b2, a2 = np.polyfit(r["gld_T"], f["D"] - f["A"], 1)
            mc["switch_breakeven_gld"] = float(-a2 / b2)
    return {"cal": c, "hold": hold, "today": today, "cost": cost, "now_monthly": now_monthly,
            "scen": res, "mc": mc}


# ----------------------------------------------------------------------------- rendering

W = 960


def money(v: float, signed: bool = False) -> str:
    s = ("+" if v >= 0 else "−") if signed else ("−" if v < 0 else "")
    return f"{s}${abs(v):,.0f}"


def small_paths(s: dict, d: dict) -> str:
    """One small panel: GLD sample paths (muted) + median (GLD color)."""
    w, h, pl, pb = 300, 150, 40, 18
    lo, hi = 150, 620
    ys = lambda v: 8 + (math.log(hi) - math.log(max(v, 1))) / (math.log(hi) - math.log(lo)) * (h - pb - 8)  # noqa: E731
    xs = lambda i, n: pl + i / (n - 1) * (w - pl - 6)  # noqa: E731
    out = [f'<line class="grid" x1="{pl}" x2="{w - 6}" y1="{ys(v):.1f}" y2="{ys(v):.1f}"/>'
           f'<text class="ax" x="{pl - 5}" y="{ys(v) + 3:.1f}" text-anchor="end">${v}</text>' for v in (200, 300, 400, 500)]
    ath, bot = d["cal"]["gld_ath"], d["cal"]["gld_ath"] / 2
    out.append(f'<line class="ref" x1="{pl}" x2="{w - 6}" y1="{ys(bot):.1f}" y2="{ys(bot):.1f}"/>')
    for p in s["gld_samples"]:
        out.append('<path class="sp" d="M' + " L".join(f"{xs(i, len(p)):.1f},{ys(v):.1f}" for i, v in enumerate(p)) + '"/>')
    m = s["gld_med"]
    out.append('<path class="ln" stroke="var(--series-2)" d="M' + " L".join(f"{xs(i, len(m)):.1f},{ys(v):.1f}" for i, v in enumerate(m)) + '"/>')
    for yr in range(0, 6):
        out.append(f'<text class="ax" x="{xs(yr, 6):.1f}" y="{h - 3}" text-anchor="middle">{"now" if yr == 0 else f"y{yr}"}</text>')
    return (f'<div class="sm"><div class="smt">{html.escape(s["name"])}</div><div class="smd">{html.escape(s["desc"])}</div>'
            f'<svg viewBox="0 0 {w} {h}">{"".join(out)}</svg></div>')


def adv_chart(d: dict) -> str:
    """Grouped horizontal bars: median gain vs 'hold, keep payouts as cash', p10-p90 whisker."""
    rows = d["scen"]
    series = [("B", "Reinvest payouts into GLD", 1), ("C", "Reinvest payouts into IAUI", 3), ("D", "Switch IAUI to GLD today", 2)]
    vals = [r["adv"][k][p] for r in rows for k, _, _ in series for p in ("p10", "p90")]
    lo, hi = min(vals + [0]), max(vals + [0])
    span = hi - lo
    lo, hi = lo - span * 0.04, hi + span * 0.04
    pl, pr, row_h, bar_h, gap = 210, 70, 78, 16, 2
    h = 24 + len(rows) * row_h
    x = lambda v: pl + (v - lo) / (hi - lo) * (W - pl - pr)  # noqa: E731
    out = []
    step = 10 ** math.floor(math.log10(span / 4))
    step = min((s * step for s in (1, 2, 2.5, 5, 10) if span / (s * step) <= 7), default=step * 10)
    t = math.ceil(lo / step) * step
    while t <= hi:
        out.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="16" y2="{h - 4}"/>'
                   f'<text class="ax" x="{x(t):.1f}" y="11" text-anchor="middle">{money(t, True) if t else "$0"}</text>')
        t += step
    out.append(f'<line class="base" x1="{x(0):.1f}" x2="{x(0):.1f}" y1="16" y2="{h - 4}"/>')
    for i, r in enumerate(rows):
        y0 = 24 + i * row_h
        out.append(f'<text class="rl" x="{pl - 12}" y="{y0 + 22}" text-anchor="end">{html.escape(r["name"])}</text>')
        for j, (k, name, slot) in enumerate(series):
            a = r["adv"][k]
            yb = y0 + j * (bar_h + gap)
            x0, x1 = sorted((x(0), x(a["p50"])))
            tip = html.escape(json.dumps({"t": f'{r["name"]} · {name}', "l": [
                f'median {money(a["p50"], True)} vs keeping payouts as cash',
                f'10–90% of paths: {money(a["p10"], True)} to {money(a["p90"], True)}',
                f'better on {a["win"] * 100:.0f}% of paths']}))
            out.append(f'<line class="wh" x1="{x(a["p10"]):.1f}" x2="{x(a["p90"]):.1f}" y1="{yb + bar_h / 2:.1f}" y2="{yb + bar_h / 2:.1f}"/>')
            out.append(f'<rect class="bar" x="{x0:.1f}" y="{yb}" width="{max(1.5, x1 - x0):.1f}" height="{bar_h}" rx="3" '
                       f'fill="var(--series-{slot})" data-tip=\'{tip}\'/>')
            if k == "B":  # label just past the whisker so it never sits on a bar
                out.append(f'<text class="dl b" x="{x(a["p90"]) + 6:.1f}" y="{yb + 12}">{money(a["p50"], True)}</text>')
    return f'<svg class="ch" viewBox="0 0 {W} {h}">{"".join(out)}</svg>'


def payout_panels(d: dict) -> str:
    out = []
    now = d["now_monthly"]
    for s in d["scen"]:
        w, h, pl = 300, 130, 40
        med, p10, p90 = s["monthly_med"], s["monthly_p10"], s["monthly_p90"]
        hi = 32.0
        ys = lambda v: 8 + (hi - v) / hi * (h - 26)  # noqa: E731
        xs = lambda i: pl + i / (len(med) - 1) * (w - pl - 6)  # noqa: E731
        band = ("M" + " L".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(p90))
                + " L" + " L".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in reversed(list(enumerate(p10)))) + " Z")
        g = [f'<line class="grid" x1="{pl}" x2="{w - 6}" y1="{ys(v):.1f}" y2="{ys(v):.1f}"/>'
             f'<text class="ax" x="{pl - 5}" y="{ys(v) + 3:.1f}" text-anchor="end">${v}</text>' for v in (0, 10, 20, 30)]
        g.append(f'<line class="ref" x1="{pl}" x2="{w - 6}" y1="{ys(now):.1f}" y2="{ys(now):.1f}"/>')
        g.append(f'<path class="band" d="{band}"/>')
        g.append('<path class="ln" stroke="var(--series-1)" d="M' + " L".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(med)) + '"/>')
        g.append(f'<text class="dl b" x="{w - 6}" y="{ys(med[-1]) - 6:.1f}" text-anchor="end">${med[-1]:.0f}/mo</text>')
        for yr in range(0, 6):
            g.append(f'<text class="ax" x="{pl + yr / 5 * (w - pl - 6):.1f}" y="{h - 3}" text-anchor="middle">{"now" if yr == 0 else f"y{yr}"}</text>')
        out.append(f'<div class="sm"><div class="smt">{html.escape(s["name"])}</div>'
                   f'<div class="smd">{med[-1] / now * 100:.0f}% of today\'s payout after 5 years</div>'
                   f'<svg viewBox="0 0 {w} {h}">{"".join(g)}</svg></div>')
    return "".join(out)


def mc_scatter(d: dict) -> str:
    pts = d["mc"]["points"]
    w, h, pl, pb = 620, 360, 64, 34
    xs_lo, xs_hi = d["cal"]["gld_ath"] / 2 - 40, d["cal"]["gld_ath"] + 40
    ys_vals = [p[1] for p in pts]
    ylo, yhi = min(ys_vals), max(ys_vals)
    pad = (yhi - ylo) * 0.08
    ylo, yhi = ylo - pad, yhi + pad
    X = lambda v: pl + (v - xs_lo) / (xs_hi - xs_lo) * (w - pl - 10)  # noqa: E731
    Y = lambda v: 10 + (yhi - v) / (yhi - ylo) * (h - pb - 10)  # noqa: E731
    out = []
    for v in range(250, 551, 50):
        if xs_lo <= v <= xs_hi:
            out.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="10" y2="{h - pb}"/>'
                       f'<text class="ax" x="{X(v):.1f}" y="{h - pb + 16}" text-anchor="middle">${v}</text>')
    step = 50 if yhi - ylo < 600 else 100
    t = math.ceil(ylo / step) * step
    while t <= yhi:
        out.append(f'<line class="grid" x1="{pl}" x2="{w - 10}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                   f'<text class="ax" x="{pl - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{money(t, True) if t else "$0"}</text>')
        t += step
    out.append(f'<line class="base" x1="{pl}" x2="{w - 10}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    now = d["cal"]["gld_now"]
    out.append(f'<line class="ref" x1="{X(now):.1f}" x2="{X(now):.1f}" y1="10" y2="{h - pb}"/>'
               f'<text class="dl" x="{X(now) + 5:.1f}" y="22">GLD today ${now:,.0f}</text>')
    for g_, dlt, t_ in pts[:1500]:
        out.append(f'<circle class="dot" cx="{X(g_):.1f}" cy="{Y(dlt):.1f}" r="3"/>')
    out.append(f'<text class="ax" x="{(pl + w) / 2:.0f}" y="{h - 4}" text-anchor="middle">GLD price at year 5</text>')
    out.append(f'<text class="ax" transform="translate(14 {(h - pb) / 2:.0f}) rotate(-90)" text-anchor="middle">'
               'Reinvesting in GLD vs keeping cash</text>')
    return f'<svg class="ch sc" viewBox="0 0 {w} {h}">{"".join(out)}</svg>'


def render(d: dict) -> str:
    c, mc, sc = d["cal"], d["mc"], {s["key"]: s for s in d["scen"]}
    e = html.escape
    rows = "".join(
        f'<tr><td>{e(s["name"])}</td><td class="n">${s["gld_T"]:,.0f}</td>'
        f'<td class="n">{money(s["final"]["A"]["p50"])}</td><td class="n"><b>{money(s["final"]["B"]["p50"])}</b></td>'
        f'<td class="n">{money(s["final"]["C"]["p50"])}</td><td class="n">{money(s["final"]["D"]["p50"])}</td>'
        f'<td class="n">{money(s["adv"]["B"]["p50"], True)}</td><td class="n">{s["adv"]["B"]["win"] * 100:.0f}%</td>'
        f'<td class="n">{money(s["payout_total"])}</td><td class="n">${s["monthly_end"]:.2f}</td>'
        f'<td class="n">{s["gld_bought"]:.2f} @ ${s["avg_buy"]:,.0f}</td></tr>' for s in d["scen"])
    sens = "".join(
        f'<tr><td>{lbl}</td><td class="n">{mc[k]["alpha_yr"] * 100:+.1f}%</td><td class="n">{mc[k]["win_B"] * 100:.0f}%</td>'
        f'<td class="n">{money(mc[k]["adv_B_p50"], True)}</td><td class="n">{mc[k]["win_D"] * 100:.0f}%</td>'
        f'<td class="n">{money(mc[k]["adv_D_p50"], True)}</td><td class="n">{mc[k]["monthly_end_ratio_p50"] * 100:.0f}%</td></tr>'
        for k, lbl in (("fitted", "Fitted from IAUI's history (base case)"), ("zero", "IAUI earns its premium in full"),
                       ("plus5", "Optimistic: options pay well in a sideways market")))
    wins = "".join(f'<tr><td>${w["from"]:,.0f}+</td><td class="n">{w["win"] * 100:.0f}%</td><td class="n">{w["n"]}</td></tr>'
                   for w in mc["win_by_end"])
    bv = mc["breakeven_gld"]
    low = sc["crash_low"]
    ctrl = sc["control"]
    ath_s = sc["crash_ath"]
    dca_lose = [x for x in d["scen"] if x["adv"]["B"]["p50"] < 0]
    worst = min(d["scen"], key=lambda x: x["adv"]["B"]["p50"])
    sw_wins = sum(1 for x in d["scen"] if x["adv"]["D"]["p50"] > 0)
    drip_loses = sum(1 for x in d["scen"] if x["final"]["C"]["p50"] < x["final"]["B"]["p50"])
    mo = [x["monthly_end"] for x in d["scen"]]
    simple = f"""
<li><b>No, the payouts will not sustain.</b> Today you receive <b>${d["now_monthly"]:.2f}/month</b> after tax. In 5 years
it's lower in <b>every</b> scenario: about <b>${low["monthly_end"]:.0f}/month</b> if gold crashes and stays low,
${ctrl["monthly_end"]:.0f} if gold just goes sideways, and still only ${ath_s["monthly_end"]:.0f} even if gold climbs back to its old
high. The reason: IAUI pays out about 12% of its price a year but earns less than that, so its price keeps sliding. Since each
payout is ~1% of that price, the payout slides with it. (This already happened: since launch GLD is up ~17%, but IAUI's price
is down ~9%.)</li>
<li><b>Reinvesting the payouts into GLD is fine, but it's a small bet on gold recovering.</b> It moves only about
${low["payout_total"]:,.0f}–${ath_s["payout_total"]:,.0f} into GLD over 5 years. Whether that beats keeping the cash depends almost
only on <b>where GLD is in year 5</b>. The dividing line is about <b>${bv:,.0f}</b>: above it, buying GLD wins; below it, keeping
the cash wins. If gold stays near the bottom you lose a little ({money(low["adv"]["B"]["p50"], True)}; worst
{money(worst["adv"]["B"]["p50"], True)} in the "{e(worst["name"])}" case, because you keep buying all the way down). If it recovers
you gain ({money(sc["crash_v"]["adv"]["B"]["p50"], True)} with a V recovery, {money(ath_s["adv"]["B"]["p50"], True)} with new highs).
Either way the difference is a few hundred dollars on a ${d["today"]:,.0f} position.</li>
<li><b>GLD is the better place for the payouts than IAUI.</b> Buying GLD beat reinvesting back into IAUI in
<b>{drip_loses} of {len(d["scen"])}</b> scenarios, crash cases included.</li>
<li><b>The bigger finding is about IAUI itself.</b> Switching all your IAUI into GLD today came out ahead of holding IAUI in
<b>{sw_wins} of {len(d["scen"])}</b> scenarios, <b>even when gold crashes and stays low</b>
({money(low["adv"]["D"]["p50"], True)}). IAUI's crash cushion (it falls ~¾ as much as gold) is a one-time benefit. Its costs repeat
every year: the 30% tax on ~12% of payouts is ~3.7% a year gone, plus IAUI's own lag behind ¾-gold. Over 5 years those
costs outweigh the cushion. Only if IAUI earned a lot more in a falling market than it has so far (≈+5%/yr extra, see section 5)
would holding it win.</li>
<li><b>Verdict.</b> Reinvesting IAUI payouts into GLD every month is a <b>sensible, low-stakes</b> habit: it beats putting them
back into IAUI and only costs you a little if gold stays down. But <b>don't rely on the payouts</b>: they shrink in a crash and
even in a flat market. And the payouts aren't the real decision. For a multi-year gold holding, the 30% tax makes IAUI a
structurally weaker way to own gold than GLD for you. A simulation, not financial advice.</li>"""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold crash simulation · IAUI payouts into GLD</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#76756f;
 --grid:#e4e3df;--axis:#b9b8b3;--muted:#c9c8c3;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;--surface-1:#1a1a19;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a998f;--grid:#2e2e2c;--axis:#55544f;--muted:#4a4945;
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9a998f;--grid:#2e2e2c;--axis:#55544f;--muted:#4a4945;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}}
*{{box-sizing:border-box;margin:0}} body{{background:var(--surface-1)}}
.viz-root{{background:var(--surface-1);color:var(--text-primary);min-height:100vh;padding:40px 24px 64px;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;line-height:1.5}}
.wrap{{max-width:1100px;margin:0 auto}} h1{{font-size:20px;font-weight:600}} h2{{font-size:16px;font-weight:600;margin-top:44px}}
.sub,.note{{color:var(--text-secondary);font-size:13px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}}
.tile{{border:1px solid var(--grid);border-radius:8px;padding:12px 14px}} .tile .k{{font-size:12px;color:var(--text-secondary)}}
.tile .v{{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}} .tile .h{{font-size:12px;color:var(--text-muted)}}
ul.simple{{padding-left:20px;margin-top:10px;font-size:15px;max-width:880px}} ul.simple li{{margin:9px 0}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px 24px;margin-top:14px}}
@media (max-width:820px){{.grid3{{grid-template-columns:1fr}}}}
.sm svg{{width:100%;height:auto;display:block;overflow:visible}} .smt{{font-size:13px;font-weight:600}}
.smd{{font-size:12px;color:var(--text-secondary);min-height:2.6em}}
svg.ch{{display:block;width:100%;height:auto;overflow:visible}} svg.sc{{max-width:640px}}
.grid{{stroke:var(--grid);stroke-width:1}} .base{{stroke:var(--axis);stroke-width:1.5}}
.ref{{stroke:var(--text-muted);stroke-width:1;stroke-dasharray:3 4}}
.ax{{font-size:11px;fill:var(--text-muted);font-variant-numeric:tabular-nums}}
.rl{{font-size:13px;fill:var(--text-primary)}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}} .sp{{fill:none;stroke:var(--muted);stroke-width:1}}
.band{{fill:var(--series-1);fill-opacity:.14}}
.dl{{font-size:12px;fill:var(--text-secondary);font-variant-numeric:tabular-nums}} .dl.b{{fill:var(--text-primary);font-weight:600}}
.wh{{stroke:var(--text-muted);stroke-width:1.5}} .bar{{stroke:var(--surface-1);stroke-width:1}} .bar:hover{{stroke:var(--text-primary)}}
.dot{{fill:var(--series-1);fill-opacity:.35}}
.legend{{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:6px 18px;margin:10px 0 4px;font-size:13px;color:var(--text-secondary)}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.tip{{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--grid);
 border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.5;box-shadow:0 4px 14px rgba(0,0,0,.12);opacity:0;z-index:9}}
.split{{display:grid;grid-template-columns:640px 1fr;gap:28px;align-items:start}} @media (max-width:980px){{.split{{grid-template-columns:1fr}}}}
details{{margin-top:16px;font-size:13px}} summary{{cursor:pointer;color:var(--text-secondary)}}
table{{border-collapse:collapse;width:100%;margin-top:10px}} td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}}
th{{font-weight:500;color:var(--text-secondary)}} td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>What if gold halves from its high? Reinvesting IAUI payouts into GLD</h1>
<p class="sub">Your gold section today: {d["hold"]["IAUI"]["qty"]:.4f} IAUI + {d["hold"]["GLD"]["qty"]:.4f} GLD =
${d["today"]:,.2f} (prices {c["asof"]}). GLD high ${c["gld_ath"]:,.2f} ({c["gld_ath_date"]}) → 50% = ${c["gld_ath"] / 2:,.2f},
{(c["gld_ath"] / 2 / c["gld_now"] - 1) * 100:.0f}% below today. 5-year horizon · {N_SCEN} noisy paths per scenario +
{N_MC:,} random scenarios · payouts after {WHT * 100:.0f}% tax · 1% commission on each small buy (as on your IBKR trades).</p>
<div class="tiles">
<div class="tile"><div class="k">Your payout today</div><div class="v">${d["now_monthly"]:.2f}/mo</div><div class="h">after 30% tax</div></div>
<div class="tile"><div class="k">Crash & stays low: payout in 5y</div><div class="v">${low["monthly_end"]:.0f}/mo</div><div class="h">{low["monthly_end"] / d["now_monthly"] * 100:.0f}% of today</div></div>
<div class="tile"><div class="k">Switching IAUI → GLD wins</div><div class="v">{sw_wins} of {len(d["scen"])}</div><div class="h">scenarios, incl. crash & stays low</div></div>
<div class="tile"><div class="k">Break-even GLD in year 5</div><div class="v">${bv:,.0f}</div><div class="h">above it, DCA into GLD beats keeping cash</div></div>
</div>
<h2>In simple words</h2><ul class="simple">{simple}</ul>

<h2>1 · The gold paths simulated</h2>
<p class="note">Orange = typical (median) GLD path; grey = 12 of the {N_SCEN} noisy versions at gold's real volatility
({c["gld_vol_d"] * math.sqrt(252) * 100:.0f}%/yr). Dashed line = ${c["gld_ath"] / 2:,.0f}, half the all-time high. Log scale.</p>
<div class="grid3">{"".join(small_paths(s, d) for s in d["scen"])}</div>

<h2>2 · Is reinvesting into GLD better than keeping the payouts as cash?</h2>
<p class="note">Gain or loss after 5 years compared with simply holding IAUI + GLD and keeping each payout as cash.
Bar = median path, whisker = 10th–90th percentile. Hover for details.</p>
<ul class="legend"><li><span class="sw" style="background:var(--series-1)"></span>Reinvest payouts into GLD (your plan)</li>
<li><span class="sw" style="background:var(--series-3)"></span>Reinvest payouts into IAUI</li>
<li><span class="sw" style="background:var(--series-2)"></span>Switch all IAUI to GLD today</li></ul>
{adv_chart(d)}

<h2>3 · Will the IAUI payouts sustain?</h2>
<p class="note">Your monthly payout after tax (holding today's {d["hold"]["IAUI"]["qty"]:.2f} IAUI shares). Line = median,
shaded = 10th–90th percentile, dashed = today's ${d["now_monthly"]:.2f}.</p>
<div class="grid3">{payout_panels(d)}</div>

<h2>4 · {N_MC:,} random crash scenarios</h2>
<div class="split"><div>{mc_scatter(d)}</div><div>
<p class="note" style="margin-top:8px">Every scenario hits ${c["gld_ath"] / 2:,.0f} (half the high) at a random time between
1 month and 4 years, then ends year 5 anywhere between the bottom and the old high. Each dot is one scenario (1,500 shown):
how much better (+) or worse (−) reinvesting into GLD did than keeping the cash. The dots form a tight rising band:
<b>the result is decided by where gold ends</b>, not by how it got there. Break-even ≈ ${bv:,.0f}. Note the
{mc["fitted"]["win_B"] * 100:.0f}% overall win rate only holds if gold is equally likely to end anywhere from the bottom to the old
high. It's an assumption, not a forecast.</p>
<table><thead><tr><th>GLD at year 5</th><th class="n">GLD DCA wins</th><th class="n">paths</th></tr></thead><tbody>{wins}</tbody></table>
</div></div>

<h2>5 · How sure is this? IAUI's own return assumption</h2>
<p class="note">IAUI was modelled as {c["beta"]:.2f} × GLD's daily move {c["alpha_d"] * 252 * 100:+.1f}%/yr, fitted on
{c["since"]}–{c["asof"]}. That fit reproduces IAUI's actual total return ({c["model_tr"] * 100:+.1f}% vs
{c["actual_tr"] * 100:+.1f}% actual). But it covers only one gold rally, so the extra return in a falling or sideways market is
uncertain. The random scenarios rerun with friendlier assumptions:</p>
<table><thead><tr><th>IAUI assumption</th><th class="n">extra return/yr</th><th class="n">GLD DCA beats cash</th>
<th class="n">median gain</th><th class="n">Switch-to-GLD beats holding</th><th class="n">median gain</th>
<th class="n">payout in 5y vs today</th></tr></thead><tbody>{sens}</tbody></table>

<details open><summary>Table view: every scenario (medians, 5 years)</summary><table>
<thead><tr><th>Scenario</th><th class="n">GLD y5</th><th class="n">Keep cash</th><th class="n">DCA into GLD</th>
<th class="n">DRIP IAUI</th><th class="n">Switch to GLD</th><th class="n">DCA vs cash</th><th class="n">DCA wins</th>
<th class="n">Payouts received</th><th class="n">Payout/mo y5</th><th class="n">GLD bought</th></tr></thead><tbody>{rows}</tbody></table>
<p class="note">Values are the IAUI + GLD + payout cash after 5 years; today ${d["today"]:,.0f}, money put in ${d["cost"]:,.0f}.</p></details>
<p class="note" style="margin-top:22px">A model, not a forecast. Taxes on payouts are fixed at 30%, IAUI's payout at
{c["payout"] * 100:.2f}% of price a month (its record so far), no capital-gains tax, GLD's 0.40% fee is inside GLD's price.</p>
</div><div class="tip" id="tip"></div></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{{el.addEventListener('mousemove',ev=>{{const d=JSON.parse(el.dataset.tip);
 tip.innerHTML='<b>'+d.t+'</b><br>'+d.l.join('<br>');const w=tip.offsetWidth,h=tip.offsetHeight;
 tip.style.left=Math.min(ev.clientX+14,innerWidth-w-8)+'px';tip.style.top=Math.max(8,ev.clientY-h-12)+'px';tip.style.opacity=1}});
 el.addEventListener('mouseleave',()=>tip.style.opacity=0)}});
</script></body></html>
"""


def main() -> int:
    d = run()
    OUT.write_text(render(d), encoding="utf-8")
    c = d["cal"]
    print(f"Wrote {OUT}")
    print(f"calibration: beta {c['beta']:.3f} alpha {c['alpha_d'] * 252 * 100:+.2f}%/yr resid {c['resid_d'] * 100:.3f}%/d "
          f"payout {c['payout'] * 100:.3f}%  model TR {c['model_tr'] * 100:.2f}% vs actual {c['actual_tr'] * 100:.2f}%")
    print(f"today ${d['today']:,.2f}  cost ${d['cost']:,.2f}  payout now ${d['now_monthly']:.2f}/mo")
    print(f"{'scenario':<26}{'GLD y5':>8}{'cash':>8}{'DCA GLD':>9}{'DRIP':>8}{'switch':>8}{'B-A':>8}{'win':>6}{'paid':>7}{'mo y5':>7}")
    for s in d["scen"]:
        f = s["final"]
        print(f"{s['name']:<26}{s['gld_T']:>8.0f}{f['A']['p50']:>8.0f}{f['B']['p50']:>9.0f}{f['C']['p50']:>8.0f}"
              f"{f['D']['p50']:>8.0f}{s['adv']['B']['p50']:>+8.0f}{s['adv']['B']['win'] * 100:>5.0f}%{s['payout_total']:>7.0f}"
              f"{s['monthly_end']:>7.2f}")
    for k in ("fitted", "zero", "plus5"):
        m = d["mc"][k]
        print(f"MC {k:<7} alpha {m['alpha_yr'] * 100:+.1f}%: DCA>cash {m['win_B'] * 100:.0f}% (median {m['adv_B_p50']:+.0f}), "
              f"IAUI DRIP>cash {m['win_C'] * 100:.0f}%, switch>hold {m['win_D'] * 100:.0f}% (median {m['adv_D_p50']:+.0f}), "
              f"payout y5 {m['monthly_end_ratio_p50'] * 100:.0f}% of now")
    print("break-even GLD at y5 (DCA vs cash):", round(d["mc"]["breakeven_gld"], 1),
          "| switch vs hold:", round(d["mc"]["switch_breakeven_gld"], 1))
    if "--no-open" not in sys.argv and sys.platform == "darwin":
        subprocess.run(["open", str(OUT)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
