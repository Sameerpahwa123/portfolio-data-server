"""Allocation page -> ./pie.html (self-contained, no CDN). Two pies, same entities and colors:

  1. Current value: each position at market value, all IBKR cash as one "Cash" slice, manual accounts.
  2. Money put in:  each position at cost basis (what you paid for the shares still held, incl. commissions),
                    cash and manual accounts unchanged.

    .venv/bin/python pie.py [--no-open]
"""

import html
import json
import math
import subprocess
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portfolio_mcp import analytics as an  # noqa: E402
from portfolio_mcp.config import load_config  # noqa: E402
from portfolio_mcp.service import PortfolioService  # noqa: E402

OUT = Path(__file__).resolve().parent / "pie.html"
SLOTS = 8  # categorical slots defined in CSS (fixed order, never cycled)


def entities(svc: PortfolioService) -> tuple[list[dict], str]:
    """One entry per slice, in a fixed order (positions by value, Cash, manual accounts) so color follows entity."""
    positions: dict[str, dict] = {}
    cash = 0.0
    with closing(svc.db.connect()) as conn:
        accts = svc._accounts(conn, "all")
        as_of = max(a.history_to or "" for a in accts)
        for a, r in an.latest_positions(conn, accts):
            fx = an._fx(r)
            p = positions.setdefault(r["symbol"], {"name": r["symbol"], "kind": r["asset_category"], "value": 0.0,
                                                   "invested": 0.0, "qty": 0.0, "accounts": set()})
            p["value"] += (r["position_value"] or 0) * fx
            p["invested"] += (r["cost_basis_money"] or 0) * fx
            p["qty"] += r["position"] or 0
            p["accounts"].add(a.label)
        for a in accts:
            row = conn.execute("SELECT cash FROM nav WHERE account_id=? AND report_date=?",
                               (a.account_id, a.history_to)).fetchone()
            cash += (row["cash"] or 0) if row else 0
    out = sorted(positions.values(), key=lambda p: -abs(p["value"]))
    for p in out:
        pnl = p["value"] - p["invested"]
        p["account"] = ", ".join(sorted(p.pop("accounts")))
        p["detail"] = f"{p.pop('qty'):g} shares · unrealized {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}"
    # Manual accounts (Capital.com) hold only cash right now, so they are part of the Cash slice.
    manual = svc.config.manual
    ibkr_cash = cash
    cash += sum(m.current_value for m in manual)
    labels = [a.label for a in sorted(accts, key=lambda a: a.label)] + [m.label for m in manual]
    detail = f"IBKR ${ibkr_cash:,.2f}" + "".join(
        f" + {'Capital.com' if m.label == 'capital.com' else m.title} ${m.current_value:,.2f} (manual, {m.as_of})"
        for m in manual)
    out.append({"name": "Cash", "kind": "CASH", "value": cash, "invested": cash, "account": ", ".join(labels),
                "detail": detail, "manual": [(m.label, m.as_of) for m in manual]})
    if len(out) > SLOTS:  # fold the tail rather than inventing a 9th hue
        tail = out[SLOTS - 1:]
        out = out[:SLOTS - 1] + [{"name": "Other", "kind": "MIXED", "account": "",
                                  "value": sum(t["value"] for t in tail), "invested": sum(t["invested"] for t in tail),
                                  "detail": f"{len(tail)} holdings"}]
    return out, as_of


def pie_svg(parts: list[dict], key: str, label: str) -> str:
    e = html.escape
    total = sum(p[key] for p in parts)
    cx, cy, r = 260, 220, 150
    ang = -math.pi / 2
    paths, labels = [], []
    for i, p in enumerate(parts):
        v = p[key]
        if v <= 0:
            continue
        share = v / total
        a0, a1 = ang, ang + share * 2 * math.pi
        ang = a1
        x0, y0, x1, y1 = cx + r * math.cos(a0), cy + r * math.sin(a0), cx + r * math.cos(a1), cy + r * math.sin(a1)
        mid = (a0 + a1) / 2
        slot = f"var(--series-{i + 1})"
        extra = ""
        if p["kind"] not in ("CASH", "MANUAL"):
            extra = (f" · invested ${p['invested']:,.2f}" if key == "value" else f" · now worth ${p['value']:,.2f}")
        tip = json.dumps({"name": p["name"], "value": f"${v:,.2f}", "share": f"{share * 100:.1f}%",
                          "detail": f"{p['account']} · {p['detail']}{extra}", "what": label})
        if share > 0.9999:  # a single slice is a full circle
            shape = f'<circle cx="{cx}" cy="{cy}" r="{r}"'
        else:
            shape = (f'<path d="M{cx},{cy} L{x0:.2f},{y0:.2f} A{r},{r} 0 {1 if a1 - a0 > math.pi else 0} 1 '
                     f'{x1:.2f},{y1:.2f} Z"')
        paths.append(f'{shape} class="slice" fill="{slot}" tabindex="0" role="img" '
                     f'aria-label="{e(p["name"])}: ${v:,.2f}, {share * 100:.1f}%" data-tip=\'{e(tip)}\'/>')
        lx, ly = cx + (r + 32) * math.cos(mid), cy + (r + 32) * math.sin(mid)
        sx, sy = cx + (r + 6) * math.cos(mid), cy + (r + 6) * math.sin(mid)
        ex, ey = cx + (r + 24) * math.cos(mid), cy + (r + 24) * math.sin(mid)
        anchor = "start" if math.cos(mid) >= 0 else "end"
        labels.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" class="leader"/>'
                      f'<text x="{lx:.1f}" y="{ly - 6:.1f}" text-anchor="{anchor}" class="lab">{e(p["name"])}</text>'
                      f'<text x="{lx:.1f}" y="{ly + 12:.1f}" text-anchor="{anchor}" class="val">'
                      f'{share * 100:.1f}% · ${v:,.0f}</text>')
    return (f'<svg class="pie" viewBox="-30 0 580 440" role="group" aria-label="Pie chart: {e(label)}">'
            f'{"".join(paths)}{"".join(labels)}</svg>')


def render(parts: list[dict], as_of: str) -> str:
    e = html.escape
    tv, ti = sum(p["value"] for p in parts), sum(p["invested"] for p in parts)
    legend = "".join(f'<li><span class="sw" style="background:var(--series-{i + 1})"></span>{e(p["name"])}</li>'
                     for i, p in enumerate(parts))
    rows = "".join(
        f'<tr><td><span class="sw" style="background:var(--series-{i + 1})"></span>{e(p["name"])}</td>'
        f'<td>{e(p["account"])}</td><td class="n">${p["value"]:,.2f}</td><td class="n">{p["value"] / tv * 100:.1f}%</td>'
        f'<td class="n">${p["invested"]:,.2f}</td><td class="n">{p["invested"] / ti * 100:.1f}%</td></tr>'
        for i, p in enumerate(parts))
    rows += (f'<tr class="tot"><td>Total</td><td></td><td class="n">${tv:,.2f}</td><td class="n">100.0%</td>'
             f'<td class="n">${ti:,.2f}</td><td class="n">100.0%</td></tr>')
    manual = [m for p in parts for m in p.get("manual", [])]
    sub = f"IBKR as of {as_of} (latest statement)" + "".join(
        f" · {'Capital.com' if lbl == 'capital.com' else lbl} cash is a manual entry ({d}), not reconciled"
        for lbl, d in manual) + " · USD"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio allocation</title>
<style>
.viz-root{{color-scheme:light;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#76756f;
 --grid:#e4e3df;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-4:#eda100;--series-5:#e87ba4;
 --series-6:#008300;--series-7:#6250d6;--series-8:#e34948}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])) .viz-root{{color-scheme:dark;--surface-1:#1a1a19;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a998f;--grid:#383835;--series-1:#3987e5;--series-2:#d95926;
 --series-3:#199e70;--series-4:#c98500;--series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767}}}}
:root[data-theme="dark"] .viz-root{{color-scheme:dark;--surface-1:#1a1a19;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --text-muted:#9a998f;--grid:#383835;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;
 --series-5:#d55181;--series-6:#008300;--series-7:#9085e9;--series-8:#e66767}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--surface-1)}}
.viz-root{{background:var(--surface-1);color:var(--text-primary);min-height:100vh;padding:40px 24px;
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:18px;font-weight:600}}
h2{{font-size:15px;font-weight:600}}
.sub{{color:var(--text-secondary);font-size:13px;margin-top:4px}}
.legend{{list-style:none;padding:0;display:flex;flex-wrap:wrap;gap:6px 18px;margin:18px 0 8px;font-size:13px;color:var(--text-secondary)}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:24px 40px;margin-top:16px}}
@media (max-width:860px){{.charts{{grid-template-columns:1fr}}}}
.chart h2 span{{font-weight:400;color:var(--text-secondary)}}
.chart p{{color:var(--text-secondary);font-size:12px;margin-top:2px}}
svg{{display:block;width:100%;height:auto;overflow:visible}}
.slice{{stroke:var(--surface-1);stroke-width:2;stroke-linejoin:round;transition:opacity .12s;outline:none}}
svg.hovering .slice:not(.on){{opacity:.35}}
.leader{{stroke:var(--text-muted);stroke-width:1}}
.lab{{font-size:13px;font-weight:600;fill:var(--text-primary)}}
.val{{font-size:12px;fill:var(--text-secondary);font-variant-numeric:tabular-nums}}
.tip{{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--grid);
 border-radius:6px;padding:8px 10px;font-size:12px;line-height:1.5;box-shadow:0 4px 14px rgba(0,0,0,.12);opacity:0;
 transition:opacity .08s;max-width:320px}}
.tip b{{font-size:13px}} .tip .m{{color:var(--text-secondary)}}
details{{margin-top:12px;font-size:13px}} summary{{cursor:pointer;color:var(--text-secondary)}}
table{{border-collapse:collapse;width:100%;margin-top:10px}}
td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}}
th{{font-weight:500;color:var(--text-secondary)}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
tr.tot td{{font-weight:600;border-bottom:0}}
</style></head>
<body><div class="viz-root"><div class="wrap">
<h1>Portfolio allocation</h1>
<p class="sub">{e(sub)}</p>
<ul class="legend">{legend}</ul>
<div class="charts">
<div class="chart"><h2>Current value <span>· ${tv:,.2f}</span></h2><p>Positions at market value</p>
{pie_svg(parts, "value", "Current value")}</div>
<div class="chart"><h2>Money put in <span>· ${ti:,.2f}</span></h2>
<p>Positions at cost basis: what you paid for the shares you still hold, incl. commissions</p>
{pie_svg(parts, "invested", "Money put in")}</div>
</div>
<details open><summary>Table view</summary><table>
<thead><tr><th>Holding</th><th>Account</th><th class="n">Current value</th><th class="n">Share</th>
<th class="n">Money put in</th><th class="n">Share</th></tr></thead>
<tbody>{rows}</tbody></table></details>
</div><div class="tip" id="tip" role="status"></div></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('svg.pie').forEach(svg=>{{
 const slices=svg.querySelectorAll('.slice');
 function show(el,x,y){{const d=JSON.parse(el.dataset.tip);svg.classList.add('hovering');
  slices.forEach(s=>s.classList.toggle('on',s===el));
  tip.innerHTML=`<b>${{d.name}}</b> <span class="m">· ${{d.what}}</span><br>${{d.value}} · ${{d.share}}<br><span class="m">${{d.detail}}</span>`;
  const w=tip.offsetWidth,h=tip.offsetHeight;tip.style.left=Math.min(x+14,innerWidth-w-8)+'px';
  tip.style.top=Math.max(8,y-h-12)+'px';tip.style.opacity=1}}
 function hide(){{svg.classList.remove('hovering');tip.style.opacity=0}}
 slices.forEach(s=>{{s.addEventListener('mousemove',ev=>show(s,ev.clientX,ev.clientY));s.addEventListener('mouseleave',hide);
  s.addEventListener('focus',()=>{{const b=s.getBoundingClientRect();show(s,b.left+b.width/2,b.top+b.height/2)}});
  s.addEventListener('blur',hide)}})}});
</script></body></html>
"""


def main() -> int:
    svc = PortfolioService(load_config())
    parts, as_of = entities(svc)
    OUT.write_text(render(parts, as_of), encoding="utf-8")
    tv, ti = sum(p["value"] for p in parts), sum(p["invested"] for p in parts)
    print(f"Wrote {OUT}  (IBKR as of {as_of})")
    print(f"  {'':<14}{'current value':>16}{'share':>8}{'money put in':>16}{'share':>8}")
    for p in parts:
        print(f"  {p['name']:<14}{p['value']:>16,.2f}{p['value'] / tv * 100:>7.1f}%{p['invested']:>16,.2f}"
              f"{p['invested'] / ti * 100:>7.1f}%")
    print(f"  {'Total':<14}{tv:>16,.2f}{'':>8}{ti:>16,.2f}")
    if "--no-open" not in sys.argv and sys.platform == "darwin":
        subprocess.run(["open", str(OUT)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
