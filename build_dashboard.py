"""Build dashboard.html: how the incumbent (best y so far) progressed for each function.

Reads data/data.csv and writes a self-contained dashboard.html (inline SVG, no
external dependencies). Re-run after each round:

    python build_dashboard.py
"""
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "data.csv"
OUT_PATH = ROOT / "dashboard.html"

# Functions plotted on a log10 axis (positive, many orders of magnitude)
LOG_FUNCTIONS = {1}

# Colour-blind-safe categorical palette (works on light and dark backgrounds)
COLOURS = ["#4C78A8", "#F58518", "#54A24B", "#E45756",
           "#B279A2", "#9D755D", "#72B7B2", "#EECA3B"]


# --------------------------------------------------------------------------- data
def load_incumbents():
    data = pd.read_csv(DATA_PATH, float_precision="round_trip")
    out = {}
    for f, g in data.groupby("function"):
        best = g.groupby("round")["y"].max().sort_index()
        inc = best.cummax()
        dim = int(g[[c for c in g.columns if c.startswith("x")]].notna().any().sum())
        out[int(f)] = {"rounds": inc.index.to_numpy(), "inc": inc.to_numpy(), "dim": dim,
                       "n": len(g)}
    return out


# --------------------------------------------------------------------- formatting
def fmt_value(v):
    return _fmt_value(v).replace("-", "\u2212")


def _fmt_value(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a < 1e-3:
        m, e = f"{v:.2e}".split("e")
        return f"{m}e{int(e)}"
    if a >= 1000:
        return f"{v:,.1f}"
    if a >= 10:
        return f"{v:.3f}"
    return f"{v:.4f}"


def nice_ticks(lo, hi, target=5):
    span = hi - lo
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag)
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    t = first
    while t <= hi + 1e-9 * step:
        ticks.append(round(t, 12))
        t += step
    return ticks, step


def tick_label(v, step):
    if abs(v) < step * 1e-6:
        return "0"
    if step >= 1:
        text = f"{v:,.0f}"
    else:
        decimals = max(0, -math.floor(math.log10(step)))
        text = f"{v:.{decimals}f}"
    return text.replace("-", "\u2212")


def esc(s):
    return html.escape(str(s))


# ------------------------------------------------------------------------- charts
def step_path(xs, ys):
    """SVG path for a step-after line through (xs, ys) pixel coordinates."""
    d = f"M{xs[0]:.1f},{ys[0]:.1f}"
    for i in range(1, len(xs)):
        d += f" H{xs[i]:.1f} V{ys[i]:.1f}"
    return d


def build_chart(rounds, values, *, width, height, colour, y_title, log=False,
                y_range=None, y_ticks=None, label_fn=None, show_markers=True,
                series=None, aria=""):
    """Return an SVG string. `values` are already in the plotted scale.
    `series` (optional) = list of (rounds, values, colour, name) for a multi-line chart."""
    ml, mr, mt, mb = 66, 22, 14, 48
    pw, ph = width - ml - mr, height - mt - mb
    all_vals = np.concatenate([s[1] for s in series]) if series else np.asarray(values)
    lo, hi = (y_range if y_range else (float(all_vals.min()), float(all_vals.max())))
    x0, x1 = float(rounds[0]), float(rounds[-1])

    def px(r):
        return ml + (r - x0) / (x1 - x0) * pw

    def py(v):
        return mt + (1 - (v - lo) / (hi - lo)) * ph

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(aria)}" '
             f'class="chart" preserveAspectRatio="xMidYMid meet">']

    # gridlines + y tick labels
    for t, lab in y_ticks:
        y = py(t)
        parts.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{lab}</text>')

    # x ticks
    step_x = 1 if x1 <= 14 else 2
    for r in range(int(x0), int(x1) + 1, step_x):
        x = px(r)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt + ph}" y2="{mt + ph + 5}" class="axis"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt + ph + 20}" text-anchor="middle" class="tick">{r}</text>')

    parts.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" class="axis"/>')
    parts.append(f'<text x="{ml + pw / 2:.1f}" y="{height - 8}" text-anchor="middle" class="axtitle">Round</text>')
    parts.append(f'<text transform="translate(14,{mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
                 f'class="axtitle">{esc(y_title)}</text>')

    if series:
        for rs, vs, col, name in series:
            xs = [px(r) for r in rs]
            ys = [py(v) for v in vs]
            parts.append(f'<path d="{step_path(xs, ys)}" fill="none" stroke="{col}" stroke-width="2.6" '
                         f'stroke-linejoin="round" stroke-linecap="round"><title>{esc(name)}</title></path>')
    else:
        xs = [px(r) for r in rounds]
        ys = [py(v) for v in values]
        base = mt + ph
        area = step_path(xs, ys) + f" V{base} H{xs[0]:.1f} Z"
        parts.append(f'<path d="{area}" fill="{colour}" opacity="0.10"/>')
        parts.append(f'<path d="{step_path(xs, ys)}" fill="none" stroke="{colour}" stroke-width="3" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        if show_markers:
            prev = None
            for i, (r, v) in enumerate(zip(rounds, values)):
                if prev is None or v > prev + 1e-15 * max(1.0, abs(v)):
                    tip = f"R{int(r)}: {label_fn(i)}"
                    parts.append(f'<circle cx="{xs[i]:.1f}" cy="{ys[i]:.1f}" r="4.6" fill="{colour}" '
                                 f'class="halo"><title>{esc(tip)}</title></circle>')
                prev = v if prev is None else max(prev, v)
    parts.append("</svg>")
    return "".join(parts)


def log_ticks(lo, hi):
    first, last = math.ceil(lo), math.floor(hi)
    step = 1
    while (last - first) / step > 5:
        step += 1
    ticks = []
    e = first
    while e <= last:
        exponent = str(e).replace("-", "−")   # typographic minus, exponent only
        ticks.append((e, f'10<tspan class="sup" dy="-5">{exponent}</tspan>'))
        e += step
    return ticks


# --------------------------------------------------------------------------- page
def main():
    inc = load_incumbents()
    last_round = max(int(v["rounds"][-1]) for v in inc.values())

    # ---- per-function charts
    cards = []
    summary_series = []
    legend = []
    for f in range(1, 9):
        d = inc[f]
        colour = COLOURS[f - 1]
        rounds, raw = d["rounds"], d["inc"]
        log = f in LOG_FUNCTIONS
        vals = np.log10(raw) if log else raw
        pad = (vals.max() - vals.min()) * 0.10
        lo, hi = vals.min() - pad, vals.max() + pad
        if log:
            ticks = log_ticks(lo, hi)
            y_title = "Incumbent y (log scale)"
        else:
            tvals, step = nice_ticks(lo, hi)
            ticks = [(t, tick_label(t, step)) for t in tvals]
            y_title = "Incumbent y"
        label_fn = (lambda i, raw=raw: fmt_value(raw[i]))
        svg = build_chart(rounds, vals, width=520, height=310, colour=colour, y_title=y_title,
                          log=log, y_range=(lo, hi), y_ticks=ticks, label_fn=label_fn,
                          aria=f"Function {f}: incumbent from {fmt_value(raw[0])} to {fmt_value(raw[-1])}")

        improved = [int(r) for i, r in enumerate(rounds) if i > 0 and raw[i] > raw[i - 1]]
        best_round = improved[-1] if improved else 0
        since = int(rounds[-1]) - best_round
        status = ("new best in the latest round" if since == 0
                  else f"no new best for {since} round{'s' if since != 1 else ''}")
        cards.append(f"""
      <article class="card">
        <header>
          <div class="ctitle"><span class="dot" style="background:{colour}"></span>
            <h3>Function {f}</h3><span class="dim">{d['dim']}D</span></div>
        </header>
        {svg}
        <dl class="stats">
          <div><dt>Start</dt><dd>{fmt_value(raw[0])}</dd></div>
          <div><dt>Current best</dt><dd>{fmt_value(raw[-1])}</dd></div>
          <div><dt>Best found</dt><dd>R{best_round}</dd></div>
          <div><dt>New bests</dt><dd>{len(improved)}</dd></div>
        </dl>
        <p class="status">{status}{' · log scale' if log else ''}</p>
      </article>""")

        # normalised progress for the summary: 0% = starting best, 100% = current best
        span = vals[-1] - vals[0]
        prog = (vals - vals[0]) / span * 100 if span > 0 else np.zeros_like(vals)
        summary_series.append((rounds, prog, colour, f"Function {f}"))
        legend.append(f'<li><span class="sw" style="background:{colour}"></span>F{f}</li>')

    ticks = [(t, f"{t}%") for t in (0, 25, 50, 75, 100)]
    summary_svg = build_chart(summary_series[0][0], None, width=1000, height=380, colour=None,
                              y_title="Progress to current best (%)", y_range=(-4, 104),
                              y_ticks=ticks, series=summary_series,
                              aria="Normalised progress of the incumbent for all eight functions")

    total_obs = sum(v["n"] for v in inc.values())
    page = TEMPLATE.format(summary=summary_svg, legend="".join(legend), cards="".join(cards),
                           last_round=last_round, total_obs=total_obs)
    OUT_PATH.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT_PATH} (rounds 0-{last_round}, {total_obs} observations)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BBO Progress Dashboard</title>
<style>
  :root {{
    --bg: #f6f7f9; --card: #ffffff; --text: #1d2330; --muted: #5f6b7d;
    --line: #e3e7ee; --grid: #e8ebf1; --axis: #9aa4b5; --halo: #ffffff;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #11141a; --card: #1a1f29; --text: #e8ecf3; --muted: #9aa6b8;
      --line: #2a3140; --grid: #2a3140; --axis: #6b7689; --halo: #1a1f29;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.45; -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1240px; margin: 0 auto; padding: 32px 24px 48px; }}
  header.page {{ margin-bottom: 24px; }}
  h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); font-size: 15px; margin: 0; max-width: 820px; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 14px;
    padding: 20px 22px 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04);
  }}
  .summary {{ margin-bottom: 20px; }}
  .summary h2 {{ font-size: 18px; margin: 0 0 4px; }}
  .summary .note {{ margin: 0 0 12px; }}
  .legend {{ list-style: none; display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 8px; padding: 0; font-size: 14px; font-weight: 600; }}
  .legend li {{ display: flex; align-items: center; gap: 7px; }}
  .sw {{ width: 18px; height: 4px; border-radius: 2px; display: inline-block; }}
  .grid2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }}
  @media (max-width: 860px) {{ .grid2 {{ grid-template-columns: 1fr; }} .wrap {{ padding: 20px 16px 32px; }} }}
  .ctitle {{ display: flex; align-items: center; gap: 9px; }}
  h3 {{ font-size: 18px; margin: 0; }}
  .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
  .dim {{ font-size: 12px; font-weight: 600; color: var(--muted); border: 1px solid var(--line);
         border-radius: 999px; padding: 1px 8px; }}
  .note {{ color: var(--muted); font-size: 14px; margin: 4px 0 8px; }}
  svg.chart {{ width: 100%; height: auto; display: block; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1.2; }}
  .tick {{ fill: var(--muted); font-size: 13px; }}
  .tick .sup {{ font-size: 10px; }}
  .axtitle {{ fill: var(--muted); font-size: 13.5px; font-weight: 600; }}
  .halo {{ stroke: var(--halo); stroke-width: 2; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0 8px; }}
  .stats div {{ background: var(--bg); border-radius: 10px; padding: 8px 10px; }}
  dt {{ font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
  dd {{ margin: 2px 0 0; font-size: 15px; font-weight: 650; font-variant-numeric: tabular-nums; }}
  .status {{ margin: 0; font-size: 13px; color: var(--muted); }}
  footer {{ margin-top: 24px; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="page">
    <h1>Black-box optimisation: progress by function</h1>
    <p class="sub">The incumbent is the best y found so far. Data through round {last_round}
      ({total_obs} observations across eight functions). Round 0 is the best of each
      function's initial points.</p>
  </header>

  <section class="card summary">
    <h2>All functions: progress toward the current best</h2>
    <p class="note">Each line is rescaled from 0% (starting incumbent) to 100% (current incumbent), so functions with
      very different scales can be compared. Function 1 is rescaled on a log scale.</p>
    <ul class="legend">{legend}</ul>
    {summary}
  </section>

  <section class="grid2">{cards}
  </section>

  <footer>Generated by <code>build_dashboard.py</code> from <code>data/data.csv</code>.
    Lines step up only when a new best is observed; markers show those rounds.</footer>
</div>
</body>
</html>
"""

if __name__ == "__main__":
    main()
