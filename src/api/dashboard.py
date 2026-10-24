"""Accuracy dashboard: logged predictions scored against observed demand.

Rendered entirely in Python. Charts are SVG built with string formatting, so the
page carries no JavaScript and no charting library.
"""

import math

import pandas as pd

from src.ingest.database import read_predictions_vs_actual
from src.models.train import load_model

CHART_DAYS = 7

# Naive baseline, scored on the same held-out window as the model.
BASELINE_MAPE = 5.71
BASELINE_MAE = 303


# ---------------------------------------------------------------- data

def build_dashboard_data() -> dict:
    """Everything the page needs, computed with pandas."""
    df = read_predictions_vs_actual()
    scored = df.dropna(subset=["actual_mw"]).copy()

    if scored.empty:
        return {"empty": True}

    scored["abs_err"] = scored["error_mw"].abs()
    scored["abs_pct"] = (scored["abs_err"] / scored["actual_mw"]) * 100
    scored["brisbane"] = scored["target_utc"].dt.tz_convert("Australia/Brisbane")

    overall = {
        "n": int(len(scored)),
        "mae": scored["abs_err"].mean(),
        "mape": scored["abs_pct"].mean(),
        "bias": scored["error_mw"].mean(),
        "first": scored["brisbane"].min().strftime("%d %b"),
        "last": scored["brisbane"].max().strftime("%d %b %Y"),
    }

    scored["day"] = ((scored["hours_ahead"] - 1) // 24) + 1
    by_horizon = [
        {"day": int(d), "mape": g["abs_pct"].mean(),
         "mae": g["abs_err"].mean(), "n": int(len(g))}
        for d, g in scored.groupby("day")
    ]

    hourly = scored.groupby(scored["brisbane"].dt.hour)["abs_err"].mean()
    by_hour = [{"hour": int(h), "mae": float(v)} for h, v in hourly.items()]

    latest = (scored.sort_values("made_at_utc")
                    .groupby("target_utc", as_index=False)
                    .last())
    cutoff = latest["target_utc"].max() - pd.Timedelta(days=CHART_DAYS)
    window = latest[latest["target_utc"] >= cutoff].sort_values("target_utc")
    local = window["target_utc"].dt.tz_convert("Australia/Brisbane")

    series = {
        "labels": [t.strftime("%a %d %b, %H:%M") for t in local],
        "short": [t.strftime("%a %d") for t in local],
        "actual": [float(v) for v in window["actual_mw"]],
        "predicted": [float(v) for v in window["predicted_mw"]],
    }

    try:
        bundle = load_model()
        model = {
            "trained_at": pd.Timestamp(bundle["trained_at"])
                            .tz_convert("Australia/Brisbane")
                            .strftime("%d %b %Y"),
            "holdout_mape": bundle["metrics"]["MAPE"],
        }
    except Exception:
        model = None

    return {"empty": False, "overall": overall, "by_horizon": by_horizon,
            "by_hour": by_hour, "series": series, "model": model}


# ---------------------------------------------------------------- charts

def _nice_bounds(lo: float, hi: float, target_ticks: int = 5):
    """Round axis bounds outward to clean numbers. Returns (lo, hi, step)."""
    span = (hi - lo) or 1.0
    magnitude = 10 ** math.floor(math.log10(span / target_ticks))

    step = magnitude
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * magnitude
        if span / step <= target_ticks:
            break

    return math.floor(lo / step) * step, math.ceil(hi / step) * step, step


def line_chart(series: dict, W: int = 1180, H: int = 290) -> str:
    """Predicted and actual demand over time."""
    actual, predicted = series["actual"], series["predicted"]
    n = len(actual)
    if n < 2:
        return '<p class="muted">Not enough scored predictions yet to plot.</p>'

    left, right, top, bottom = 56, 14, 14, 28
    both = actual + predicted
    lo, hi, step = _nice_bounds(min(both), max(both))

    def x(i):
        return left + (i / (n - 1)) * (W - left - right)

    def y(v):
        return top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom)

    parts = []
    v = lo
    while v <= hi + step / 2:
        gy = y(v)
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{W - right}" y2="{gy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{gy:.1f}" text-anchor="end" '
                     f'dominant-baseline="middle" class="tick">{v:,.0f}</text>')
        v += step

    for k in range(6):
        i = round((n - 1) * k / 5)
        anchor = "start" if k == 0 else "end" if k == 5 else "middle"
        parts.append(f'<text x="{x(i):.1f}" y="{H - 8}" text-anchor="{anchor}" '
                     f'class="tick">{series["short"][i]}</text>')

    for values, colour in ((actual, "--actual"), (predicted, "--predicted")):
        pts = " ".join(f"{x(i):.1f},{y(val):.1f}" for i, val in enumerate(values))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var({colour})" '
                     f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Predicted and actual '
            f'demand over the last {CHART_DAYS} days">' + "".join(parts) + "</svg>")


def bar_chart(by_hour: list, W: int = 720, H: int = 250) -> str:
    """Mean absolute error by hour of day."""
    if not by_hour:
        return ""

    values = [h["mae"] for h in by_hour]
    left, right, top, bottom = 50, 12, 26, 26
    _, hi, step = _nice_bounds(0, max(values))

    band = (W - left - right) / len(by_hour)
    width = min(22, band - 3)
    worst = max(range(len(values)), key=lambda i: values[i])

    def y(v):
        return top + (1 - v / hi) * (H - top - bottom)

    parts = []
    v = 0.0
    while v <= hi + step / 2:
        gy = y(v)
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{W - right}" y2="{gy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{gy:.1f}" text-anchor="end" '
                     f'dominant-baseline="middle" class="tick">{v:,.0f}</text>')
        v += step

    baseline = y(0)
    for i, row in enumerate(by_hour):
        cx = left + band * i + band / 2
        bx, by = cx - width / 2, y(row["mae"])
        height = baseline - by
        r = min(4, height / 2, width / 2)      # rounded data-end, square baseline
        d = (f"M{bx:.1f},{baseline:.1f} L{bx:.1f},{by + r:.1f} "
             f"Q{bx:.1f},{by:.1f} {bx + r:.1f},{by:.1f} "
             f"L{bx + width - r:.1f},{by:.1f} "
             f"Q{bx + width:.1f},{by:.1f} {bx + width:.1f},{by + r:.1f} "
             f"L{bx + width:.1f},{baseline:.1f} Z")
        parts.append(f'<path d="{d}" fill="var(--actual)">'
                     f'<title>{row["hour"]:02d}:00 — {row["mae"]:,.0f} MW</title></path>')
        if i % 4 == 0:
            parts.append(f'<text x="{cx:.1f}" y="{H - 7}" text-anchor="middle" '
                         f'class="tick">{row["hour"]:02d}</text>')

    wx = left + band * worst + band / 2
    parts.append(f'<text x="{wx:.1f}" y="{y(values[worst]) - 9:.1f}" text-anchor="middle" '
                 f'class="peak">{values[worst]:,.0f} MW</text>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean absolute error by '
            f'hour of day, worst at {by_hour[worst]["hour"]:02d}:00">'
            + "".join(parts) + "</svg>")


# ---------------------------------------------------------------- page

CSS = """
:root {
  color-scheme: light;
  --page: #e8ebee; --card: #ffffff; --border: #d9dee3; --grid-c: #e9edf1;
  --ink: #15181c; --ink-2: #4a5159; --ink-3: #79818a;
  --actual: #2a78d6; --predicted: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #121418; --card: #1d2025; --border: #2c3036; --grid-c: #272b31;
    --ink: #eef1f4; --ink-2: #b4bcc4; --ink-3: #868e97;
    --actual: #3987e5; --predicted: #d95926;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; padding: 26px 22px 36px; background: var(--page); color: var(--ink);
  font: 400 15px/1.55 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1240px; margin: 0 auto; }

.bar { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 18px; margin-bottom: 18px; }
h1 { font-size: 22px; font-weight: 600; letter-spacing: -0.016em; margin: 0; }
.bar .live { font-size: 13px; color: var(--ink-2); }
.bar .prov { font-size: 12.5px; color: var(--ink-3); margin-left: auto; }

.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 13px; margin-bottom: 13px; }
.tile { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 15px 17px 14px; }
.tile .v { font-size: 30px; font-weight: 600; letter-spacing: -0.028em; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tile .u { font-size: 18px; font-weight: 500; }
.tile .k { font-size: 12.5px; color: var(--ink-2); margin-top: 5px; }
.tile .c { font-size: 12px; color: var(--ink-3); margin-top: 2px; font-variant-numeric: tabular-nums; }

.row { display: grid; grid-template-columns: 1.6fr 1fr; gap: 13px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px 12px; margin-bottom: 13px; }
.head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; margin-bottom: 10px; }
h2 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: -0.006em; }
.sub { font-size: 12.5px; color: var(--ink-3); }
.legend { display: flex; gap: 14px; font-size: 12.5px; color: var(--ink-2); margin-left: auto; }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.sw { width: 14px; height: 3px; border-radius: 2px; }

svg { display: block; width: 100%; height: auto; }
.tick { font-size: 11px; fill: var(--ink-3); }
.peak { font-size: 11.5px; font-weight: 600; fill: var(--ink-2); }
.grid { stroke: var(--grid-c); stroke-width: 1; }

table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { border-bottom: 1px solid var(--border); padding: 8px; text-align: left; }
th { color: var(--ink-3); font-weight: 500; font-size: 12.5px; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:last-child td { border-bottom: none; }
tr.total td { border-top: 1px solid var(--border); font-weight: 600; padding-top: 10px; }

details { margin-top: 11px; border-top: 1px solid var(--border); padding-top: 9px; }
summary { cursor: pointer; font-size: 13px; color: var(--ink-3); }
details[open] summary { margin-bottom: 9px; }
.scroll { max-height: 300px; overflow-y: auto; }
.muted { color: var(--ink-2); }
footer { color: var(--ink-3); font-size: 12.5px; margin-top: 6px; }
footer a { color: var(--ink-2); }

@media (max-width: 920px) {
  .tiles { grid-template-columns: repeat(2, 1fr); }
  .row { grid-template-columns: 1fr; }
  .bar .prov { margin-left: 0; flex-basis: 100%; }
}
"""

SHELL = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forecast accuracy — Queensland electricity demand</title>
<style>__CSS__</style>
</head><body><div class="wrap">__BODY__</div></body></html>"""


def render_dashboard() -> str:
    """Return the full dashboard page as HTML."""
    d = build_dashboard_data()

    if d["empty"]:
        body = ('<div class="bar"><h1>Forecast accuracy</h1></div>'
                '<div class="card"><p class="muted">No predictions scored yet. Each '
                'forecast is logged when made and compared once AEMO publishes the '
                'actual demand.</p></div>')
        return SHELL.replace("__CSS__", CSS).replace("__BODY__", body)

    o, m = d["overall"], d["model"]
    prov = (f'<span class="prov">Model trained {m["trained_at"]} · '
            f'held-out {m["holdout_mape"]:.2f}%</span>') if m else ""

    horizon_rows = "".join(
        f'<tr><td>{r["day"]} day{"s" if r["day"] > 1 else ""}</td>'
        f'<td class="num">{r["mape"]:.2f}%</td>'
        f'<td class="num">{r["mae"]:,.0f}</td>'
        f'<td class="num">{r["n"]:,}</td></tr>'
        for r in d["by_horizon"])
    horizon_rows += (f'<tr class="total"><td>All</td>'
                     f'<td class="num">{o["mape"]:.2f}%</td>'
                     f'<td class="num">{o["mae"]:,.0f}</td>'
                     f'<td class="num">{o["n"]:,}</td></tr>')

    s = d["series"]
    data_rows = "".join(
        f'<tr><td>{s["labels"][i]}</td>'
        f'<td class="num">{s["actual"][i]:,.0f}</td>'
        f'<td class="num">{s["predicted"][i]:,.0f}</td>'
        f'<td class="num">{s["predicted"][i] - s["actual"][i]:+,.0f}</td></tr>'
        for i in range(len(s["actual"]) - 1, -1, -1))

    worst = max(d["by_hour"], key=lambda r: r["mae"])

    body = f"""
<div class="bar">
  <h1>Forecast accuracy</h1>
  <span class="live">Scored after the fact · {o["first"]} to {o["last"]}</span>
  {prov}
</div>

<div class="tiles">
  <div class="tile">
    <div class="v">{o["mape"]:.2f}%</div>
    <div class="k">Mean absolute % error</div>
    <div class="c">Naive baseline {BASELINE_MAPE}%</div>
  </div>
  <div class="tile">
    <div class="v">{o["mae"]:,.0f}<span class="u"> MW</span></div>
    <div class="k">Mean absolute error</div>
    <div class="c">Naive baseline {BASELINE_MAE} MW</div>
  </div>
  <div class="tile">
    <div class="v">{o["n"]:,}</div>
    <div class="k">Predictions scored</div>
    <div class="c">1 to 72 hours ahead</div>
  </div>
  <div class="tile">
    <div class="v">{o["bias"]:+,.0f}<span class="u"> MW</span></div>
    <div class="k">Mean signed error</div>
    <div class="c">{"Under" if o["bias"] < 0 else "Over"}-predicts slightly</div>
  </div>
</div>

<div class="card">
  <div class="head">
    <h2>Predicted against actual</h2>
    <span class="sub">Last {CHART_DAYS} days, Brisbane time</span>
    <span class="legend">
      <span><i class="sw" style="background:var(--actual)"></i>Actual</span>
      <span><i class="sw" style="background:var(--predicted)"></i>Predicted</span>
    </span>
  </div>
  {line_chart(s)}
  <details>
    <summary>View as table</summary>
    <div class="scroll"><table>
      <thead><tr><th>Hour</th><th class="num">Actual</th>
      <th class="num">Predicted</th><th class="num">Error</th></tr></thead>
      <tbody>{data_rows}</tbody></table></div>
  </details>
</div>

<div class="row">
  <div class="card">
    <div class="head">
      <h2>Error by hour of day</h2>
      <span class="sub">Worst at {worst["hour"]:02d}:00 — rooftop solar makes midday volatile</span>
    </div>
    {bar_chart(d["by_hour"])}
  </div>

  <div class="card">
    <div class="head">
      <h2>By forecast horizon</h2>
      <span class="sub">Tracks weather forecast quality</span>
    </div>
    <table>
      <thead><tr><th>Ahead</th><th class="num">MAPE</th>
      <th class="num">MAE</th><th class="num">n</th></tr></thead>
      <tbody>{horizon_rows}</tbody>
    </table>
  </div>
</div>

<footer>
  Demand from AEMO · weather from <a href="https://open-meteo.com">Open-Meteo</a> (CC BY 4.0)
  · charts rendered server-side, no JavaScript
</footer>"""

    return SHELL.replace("__CSS__", CSS).replace("__BODY__", body)
