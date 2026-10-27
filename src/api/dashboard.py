"""Accuracy dashboard: logged predictions scored against observed demand.

Rendered entirely in Python. Charts are SVG built with string formatting, so the
page carries no JavaScript and no charting library.
"""

import math

import pandas as pd

from src.ingest.database import read_predictions_vs_actual, read_all
from src.models.train import load_model

CHART_DAYS = 7
TZ = "Australia/Brisbane"

# The benchmark anyone can beat you with: demand equals the same hour a week ago.
NAIVE_LAG_HOURS = 168


def naive_baseline(scored: pd.DataFrame) -> dict | None:
    """Score the naive benchmark on exactly the hours the model was scored on.

    Quoting a baseline measured over a different period is the mistake this
    project's own README warns about. So the benchmark is computed here, from
    the observations table, for the same rows and nothing else. Hours with no
    observation a week earlier are dropped from both sides, so the comparison
    stays like for like.
    """
    history = read_all()["demand_mw"]
    reference = scored["target_utc"] - pd.Timedelta(hours=NAIVE_LAG_HOURS)

    matched = scored.assign(naive=reference.map(history)).dropna(subset=["naive"])
    if matched.empty:
        return None

    naive_err = (matched["naive"] - matched["actual_mw"]).abs()
    model_err = matched["abs_err"]

    naive_mape = float((naive_err / matched["actual_mw"]).mean() * 100)
    model_mape = float((model_err / matched["actual_mw"]).mean() * 100)

    return {
        "n": int(len(matched)),
        "naive_mae": float(naive_err.mean()),
        "naive_mape": naive_mape,
        "model_mae": float(model_err.mean()),
        "model_mape": model_mape,
        "skill": (1 - model_mape / naive_mape) * 100 if naive_mape else 0.0,
    }


# ---------------------------------------------------------------- data

def _time_ticks(local, every: int):
    """Indices to label on a time axis, every N hours. Day name at midnight."""
    ticks = []
    for i, t in enumerate(local):
        if t.hour % every == 0 and t.minute == 0:
            ticks.append({"i": i,
                          "time": t.strftime("%H:%M"),
                          "day": t.strftime("%a %d %b") if t.hour == 0 else ""})
    return ticks


def build_dashboard_data() -> dict:
    """Everything the page needs, computed with pandas."""
    df = read_predictions_vs_actual()
    scored = df.dropna(subset=["actual_mw"]).copy()

    if scored.empty:
        return {"empty": True}

    scored["abs_err"] = scored["error_mw"].abs()
    scored["abs_pct"] = (scored["abs_err"] / scored["actual_mw"]) * 100
    scored["brisbane"] = scored["target_utc"].dt.tz_convert(TZ)

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

    baseline = naive_baseline(scored)

    # How many rows can prove they were issued before the hour they describe?
    if "lead_hours" in scored.columns:
        provable = int((scored["lead_hours"] > 0).sum())
    else:
        provable = 0

    hourly = scored.groupby(scored["brisbane"].dt.hour)["abs_err"].mean()
    by_hour = [{"hour": int(h), "mae": float(v)} for h, v in hourly.items()]

    # Same cut, split by how far ahead the forecast was made. Both in MW, so
    # they share one axis - bars for the near horizon, a line for the far one.
    far_day = int(scored["day"].max())
    near = scored[scored["day"] == 1].groupby(scored["brisbane"].dt.hour)["abs_err"].mean()
    far = scored[scored["day"] == far_day].groupby(scored["brisbane"].dt.hour)["abs_err"].mean()
    combo = {
        "far_day": far_day,
        "rows": [{"hour": int(h),
                  "near": float(near.get(h, float("nan"))),
                  "far": float(far.get(h, float("nan")))}
                 for h in range(24) if h in near.index or h in far.index],
    }

    latest = (scored.sort_values("made_at_utc")
                    .groupby("target_utc", as_index=False)
                    .last())
    cutoff = latest["target_utc"].max() - pd.Timedelta(days=CHART_DAYS)
    window = latest[latest["target_utc"] >= cutoff].sort_values("target_utc")
    local = list(window["target_utc"].dt.tz_convert(TZ))

    series = {
        "labels": [t.strftime("%a %d %b, %H:%M") for t in local],
        "ticks": _time_ticks(local, 12),
        "actual": [float(v) for v in window["actual_mw"]],
        "predicted": [float(v) for v in window["predicted_mw"]],
    }

    # ---- what the service currently expects to happen -------------------
    outlook = tomorrow = None
    future = df[df["actual_mw"].isna()].copy()
    if not future.empty:
        future = (future.sort_values("made_at_utc")
                        .groupby("target_utc", as_index=False)
                        .last()
                        .sort_values("target_utc"))
        flocal = list(future["target_utc"].dt.tz_convert(TZ))
        values = [float(v) for v in future["predicted_mw"]]

        tomorrow_date = (pd.Timestamp.now(tz=TZ) + pd.Timedelta(days=1)).date()
        band = [i for i, t in enumerate(flocal) if t.date() == tomorrow_date]

        outlook = {
            "labels": [t.strftime("%a %d %b, %H:%M") for t in flocal],
            "ticks": _time_ticks(flocal, 6),
            "values": values,
            "band": (band[0], band[-1]) if band else None,
            "days": len({t.date() for t in flocal}),
            "made_at": (future["made_at_utc"].max().tz_convert(TZ)
                        .strftime("%d %b, %H:%M")),
            "from": flocal[0].strftime("%a %d %b, %H:%M"),
            "to": flocal[-1].strftime("%a %d %b, %H:%M"),
        }

        if band:
            rows = [{"hour": flocal[i].hour, "mw": values[i]} for i in band]
            peak = max(rows, key=lambda r: r["mw"])
            trough = min(rows, key=lambda r: r["mw"])
            tomorrow = {
                "date": pd.Timestamp(tomorrow_date).strftime("%A %d %B"),
                "rows": rows,
                "peak": peak, "trough": trough,
                "mean": sum(r["mw"] for r in rows) / len(rows),
                "complete": len(rows) == 24,
            }

    try:
        bundle = load_model()
        model = {
            "trained_at": pd.Timestamp(bundle["trained_at"])
                            .tz_convert(TZ).strftime("%d %b %Y"),
            "holdout_mape": bundle["metrics"]["MAPE"],
        }
    except Exception:
        model = None

    return {"empty": False, "overall": overall, "by_horizon": by_horizon,
            "by_hour": by_hour, "combo": combo, "series": series,
            "outlook": outlook, "tomorrow": tomorrow, "model": model,
            "baseline": baseline, "provable": provable}


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


def _y_grid(parts, y, lo, hi, step, left, right, W):
    """Horizontal grid lines with value labels down the left."""
    v = lo
    while v <= hi + step / 2:
        gy = y(v)
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{W - right}" '
                     f'y2="{gy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{gy:.1f}" text-anchor="end" '
                     f'dominant-baseline="middle" class="tick">{v:,.0f}</text>')
        v += step


def _time_axis(parts, ticks, x, n, H, W, left, right):
    """Two-row time axis: clock time, with the date under each midnight."""
    for t in ticks:
        px = x(t["i"])
        anchor = "start" if px < left + 18 else "end" if px > W - right - 18 else "middle"
        parts.append(f'<line x1="{px:.1f}" y1="{H - 40}" x2="{px:.1f}" '
                     f'y2="{H - 36}" class="grid"/>')
        parts.append(f'<text x="{px:.1f}" y="{H - 25}" text-anchor="{anchor}" '
                     f'class="tick">{t["time"]}</text>')
        if t["day"]:
            parts.append(f'<text x="{px:.1f}" y="{H - 8}" text-anchor="{anchor}" '
                         f'class="tick day">{t["day"]}</text>')


def line_chart(series: dict, W: int = 1180, H: int = 320) -> str:
    """Predicted and actual demand over time."""
    actual, predicted = series["actual"], series["predicted"]
    n = len(actual)
    if n < 2:
        return '<p class="muted">Not enough scored predictions yet to plot.</p>'

    left, right, top, bottom = 56, 14, 14, 48
    both = actual + predicted
    lo, hi, step = _nice_bounds(min(both), max(both))

    def x(i):
        return left + (i / (n - 1)) * (W - left - right)

    def y(v):
        return top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom)

    parts = []
    _y_grid(parts, y, lo, hi, step, left, right, W)
    _time_axis(parts, series["ticks"], x, n, H, W, left, right)

    for values, colour in ((actual, "--actual"), (predicted, "--predicted")):
        pts = " ".join(f"{x(i):.1f},{y(val):.1f}" for i, val in enumerate(values))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var({colour})" '
                     f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Predicted and actual '
            f'demand over the last {CHART_DAYS} days">' + "".join(parts) + "</svg>")


def forecast_chart(outlook: dict, W: int = 1180, H: int = 300) -> str:
    """Predicted demand ahead of time, with tomorrow shaded."""
    values = outlook["values"]
    n = len(values)
    if n < 2:
        return '<p class="muted">No forward forecast stored yet.</p>'

    left, right, top, bottom = 56, 14, 22, 48
    lo, hi, step = _nice_bounds(min(values), max(values))

    def x(i):
        return left + (i / (n - 1)) * (W - left - right)

    def y(v):
        return top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom)

    parts = []

    band = outlook["band"]
    if band:
        x0, x1 = x(band[0]), x(band[1])
        parts.append(f'<rect x="{x0:.1f}" y="{top}" width="{x1 - x0:.1f}" '
                     f'height="{H - top - bottom:.1f}" fill="var(--actual)" '
                     f'opacity="0.075"/>')

    _y_grid(parts, y, lo, hi, step, left, right, W)
    _time_axis(parts, outlook["ticks"], x, n, H, W, left, right)

    if band:
        parts.append(f'<text x="{(x(band[0]) + x(band[1])) / 2:.1f}" y="{top - 6}" '
                     f'text-anchor="middle" class="peak">Tomorrow</text>')

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--predicted)" '
                 f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Forecast demand for the '
            f'next {outlook["days"]} days">' + "".join(parts) + "</svg>")


def _bars(parts, rows, key, y, baseline, left, band, width, colour, fmt):
    """Rounded data-end, square baseline, native tooltip per bar."""
    for i, row in enumerate(rows):
        v = row[key]
        if v != v:                                    # NaN
            continue
        cx = left + band * i + band / 2
        bx, by = cx - width / 2, y(v)
        height = baseline - by
        r = min(4, max(height, 0.1) / 2, width / 2)
        d = (f"M{bx:.1f},{baseline:.1f} L{bx:.1f},{by + r:.1f} "
             f"Q{bx:.1f},{by:.1f} {bx + r:.1f},{by:.1f} "
             f"L{bx + width - r:.1f},{by:.1f} "
             f"Q{bx + width:.1f},{by:.1f} {bx + width:.1f},{by + r:.1f} "
             f"L{bx + width:.1f},{baseline:.1f} Z")
        parts.append(f'<path d="{d}" fill="var({colour})">'
                     f'<title>{row["hour"]:02d}:00 — {fmt(v)}</title></path>')


def combo_chart(combo: dict, W: int = 720, H: int = 270) -> str:
    """Error by hour: bars for day-ahead, a line for the longest horizon.

    Both series are mean absolute error in MW, so they share one axis. A
    second y-scale would let any gap be drawn as any size.
    """
    rows = combo["rows"]
    if not rows:
        return ""

    values = [v for r in rows for v in (r["near"], r["far"]) if v == v]
    left, right, top, bottom = 50, 12, 14, 26
    _, hi, step = _nice_bounds(0, max(values))

    band = (W - left - right) / len(rows)
    width = min(20, band - 4)

    def x(i):
        return left + band * i + band / 2

    def y(v):
        return top + (1 - v / hi) * (H - top - bottom)

    parts = []
    _y_grid(parts, y, 0, hi, step, left, right, W)
    _bars(parts, rows, "near", y, y(0), left, band, width, "--actual",
          lambda v: f"{v:,.0f} MW, 1 day ahead")

    pts = [(x(i), y(r["far"])) for i, r in enumerate(rows) if r["far"] == r["far"]]
    if len(pts) > 1:
        parts.append('<polyline points="'
                     + " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
                     + '" fill="none" stroke="var(--predicted)" stroke-width="2" '
                       'stroke-linejoin="round" stroke-linecap="round"/>')
        for i, r in enumerate(rows):
            if r["far"] == r["far"]:
                parts.append(f'<circle cx="{x(i):.1f}" cy="{y(r["far"]):.1f}" r="4" '
                             f'fill="var(--predicted)" stroke="var(--card)" '
                             f'stroke-width="2"><title>{r["hour"]:02d}:00 — '
                             f'{r["far"]:,.0f} MW, {combo["far_day"]} days ahead</title></circle>')

    for i, r in enumerate(rows):
        if r["hour"] % 4 == 0:
            parts.append(f'<text x="{x(i):.1f}" y="{H - 7}" text-anchor="middle" '
                         f'class="tick">{r["hour"]:02d}</text>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean absolute error by '
            f'hour of day, one day ahead against {combo["far_day"]} days ahead">'
            + "".join(parts) + "</svg>")


def tomorrow_chart(tomorrow: dict, W: int = 720, H: int = 250) -> str:
    """Tomorrow's forecast, hour by hour, peak and trough labelled."""
    rows = tomorrow["rows"]
    if not rows:
        return ""

    values = [r["mw"] for r in rows]
    left, right, top, bottom = 52, 12, 26, 26
    lo, hi, step = _nice_bounds(min(values) * 0.97, max(values))

    band = (W - left - right) / len(rows)
    width = min(22, band - 3)

    def x(i):
        return left + band * i + band / 2

    def y(v):
        return top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom)

    parts = []
    _y_grid(parts, y, lo, hi, step, left, right, W)
    _bars(parts, rows, "mw", y, y(lo), left, band, width, "--predicted",
          lambda v: f"{v:,.0f} MW forecast")

    for row, cls in ((tomorrow["peak"], "peak"), (tomorrow["trough"], "peak")):
        i = rows.index(row)
        parts.append(f'<text x="{x(i):.1f}" y="{y(row["mw"]) - 9:.1f}" '
                     f'text-anchor="middle" class="{cls}">{row["mw"]:,.0f}</text>')

    for i, r in enumerate(rows):
        if r["hour"] % 4 == 0:
            parts.append(f'<text x="{x(i):.1f}" y="{H - 7}" text-anchor="middle" '
                         f'class="tick">{r["hour"]:02d}</text>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Forecast demand for '
            f'{tomorrow["date"]}, hour by hour">' + "".join(parts) + "</svg>")


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
.row.even { grid-template-columns: 1.6fr 1fr; }
.facts { display: grid; gap: 9px; }
.fact { display: flex; justify-content: space-between; align-items: baseline;
        gap: 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
.fact:last-child { border-bottom: none; }
.fact .l { font-size: 13px; color: var(--ink-2); }
.fact .r { font-size: 17px; font-weight: 600; font-variant-numeric: tabular-nums; }
.fact .r small { font-size: 12.5px; font-weight: 500; color: var(--ink-3); margin-left: 5px; }
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
.tick.day { font-size: 11.5px; fill: var(--ink-2); }
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

    b = d["baseline"]
    if b:
        base_mape = (f'vs {b["naive_mape"]:.2f}% same-hour-last-week '
                     f'&mdash; {b["skill"]:.0f}% better')
        base_mae = f'vs {b["naive_mae"]:,.0f} MW same-hour-last-week'
        like_for_like = (f'Benchmark measured on the same {b["n"]:,} hours, '
                         f'not a different window.')
    else:
        base_mape = base_mae = "no benchmark available yet"
        like_for_like = ""
    ol, tm = d["outlook"], d["tomorrow"]

    outlook_card = ""
    if ol:
        outlook_card = f"""
<div class="card">
  <div class="head">
    <h2>What the service expects next</h2>
    <span class="sub">{ol["from"]} to {ol["to"]} · forecast made {ol["made_at"]}</span>
    <span class="legend">
      <span><i class="sw" style="background:var(--predicted)"></i>Predicted</span>
    </span>
  </div>
  {forecast_chart(ol)}
</div>"""

    tomorrow_row = ""
    if tm:
        hours = "".join(
            f'<tr><td>{r["hour"]:02d}:00</td><td class="num">{r["mw"]:,.0f}</td></tr>'
            for r in tm["rows"])
        partial = "" if tm["complete"] else (
            f'<div class="fact"><span class="l">Hours covered</span>'
            f'<span class="r">{len(tm["rows"])}<small>of 24</small></span></div>')
        tomorrow_row = f"""
<div class="row">
  <div class="card">
    <div class="head">
      <h2>Tomorrow, hour by hour</h2>
      <span class="sub">{tm["date"]}, Brisbane time</span>
    </div>
    {tomorrow_chart(tm)}
  </div>

  <div class="card">
    <div class="head"><h2>Tomorrow at a glance</h2></div>
    <div class="facts">
      <div class="fact"><span class="l">Highest hour</span>
        <span class="r">{tm["peak"]["mw"]:,.0f}<small>MW at {tm["peak"]["hour"]:02d}:00</small></span></div>
      <div class="fact"><span class="l">Lowest hour</span>
        <span class="r">{tm["trough"]["mw"]:,.0f}<small>MW at {tm["trough"]["hour"]:02d}:00</small></span></div>
      <div class="fact"><span class="l">Daily mean</span>
        <span class="r">{tm["mean"]:,.0f}<small>MW</small></span></div>
      <div class="fact"><span class="l">Peak-to-trough swing</span>
        <span class="r">{tm["peak"]["mw"] - tm["trough"]["mw"]:,.0f}<small>MW</small></span></div>
      {partial}
    </div>
    <details>
      <summary>All {len(tm["rows"])} hours</summary>
      <div class="scroll"><table>
        <thead><tr><th>Hour</th><th class="num">Forecast MW</th></tr></thead>
        <tbody>{hours}</tbody></table></div>
    </details>
  </div>
</div>"""

    body = f"""
<div class="bar">
  <h1>Forecast accuracy</h1>
  <span class="live">Scored after the fact · {o["first"]} to {o["last"]} · {like_for_like}</span>
  {prov}
</div>

<div class="tiles">
  <div class="tile">
    <div class="v">{o["mape"]:.2f}%</div>
    <div class="k">Mean absolute % error</div>
    <div class="c">{base_mape}</div>
  </div>
  <div class="tile">
    <div class="v">{o["mae"]:,.0f}<span class="u"> MW</span></div>
    <div class="k">Mean absolute error</div>
    <div class="c">{base_mae}</div>
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
{outlook_card}
{tomorrow_row}
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
      <span class="legend">
        <span><i class="sw" style="background:var(--actual)"></i>1 day ahead</span>
        <span><i class="sw" style="background:var(--predicted)"></i>{d["combo"]["far_day"]} days ahead</span>
      </span>
    </div>
    {combo_chart(d["combo"])}
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
