"""Prediction API for Queensland electricity demand."""

import os
import threading
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from src.api.dashboard import render_dashboard
from src.features.build import build_features
from src.ingest.database import read_all, read_predictions_vs_actual, read_with_future, row_count
from src.models.horizon import recursive_forecast
from src.models.train import load_model


def measured_accuracy() -> dict:
    """Error by forecast horizon, read from the prediction log.

    The endpoint used to advertise a fixed "2.3-2.7%", copied from a backtest
    on observed weather. Live forecasts use forecast weather and do worse. A
    service that states an accuracy it does not achieve is worse than one that
    states none, so this reads the scored rows and reports what actually
    happened.
    """
    scored = read_predictions_vs_actual().dropna(subset=["actual_mw"])
    if scored.empty:
        return {"status": "no scored forecasts yet"}

    day = ((scored["hours_ahead"] - 1) // 24) + 1
    pct = (scored["error_mw"].abs() / scored["actual_mw"]) * 100

    return {
        "source": "measured on logged forecasts scored against observed demand",
        "forecasts_scored": int(len(scored)),
        "mape_overall": round(float(pct.mean()), 2),
        "mape_by_day_ahead": {
            f"day_{int(d)}": round(float(pct[day == d].mean()), 2)
            for d in sorted(day.unique())
        },
    }



REFRESH_HOURS = float(os.getenv("REFRESH_HOURS", "6"))
STARTUP_DELAY_SECONDS = float(os.getenv("STARTUP_DELAY_SECONDS", "30"))
REFRESH_ENABLED = os.getenv("REFRESH_ENABLED", "1") != "0"


def data_age_hours() -> float:
    """How old the newest observation is. Large if the database is unreadable."""
    try:
        return (pd.Timestamp.now(tz="UTC") - read_all().index.max()).total_seconds() / 3600
    except Exception:
        return float("inf")


def refresh_loop():
    """Keep the data current in the background.

    The old version slept for six hours before its first update, so a container
    that restarted served stale data for a quarter of a day. It now refreshes
    shortly after start-up, unless the database is already fresher than the
    refresh interval, which is the case right after the bootstrap.
    """
    from scripts.daily_update import main as update

    time.sleep(STARTUP_DELAY_SECONDS)          # let uvicorn bind its port first

    if data_age_hours() < REFRESH_HOURS:
        time.sleep(REFRESH_HOURS * 3600)

    while True:
        try:
            update()
        except Exception as exc:
            print(f"Background refresh failed: {exc}")
        time.sleep(REFRESH_HOURS * 3600)


@asynccontextmanager
async def lifespan(app):
    if REFRESH_ENABLED:
        threading.Thread(target=refresh_loop, daemon=True).start()
    yield


app = FastAPI(title="QLD Energy Demand Forecast", version="0.3.0", lifespan=lifespan)
BUNDLE = load_model()


@app.get("/")
def root():
    return {
        "service": "QLD electricity demand forecast",
        "model": "gradient boosting",
        "endpoints": [
            "/health",
            "/predict?timestamp=YYYY-MM-DDTHH:MM",
            "/forecast?hours=N",
            "/dashboard",
            "/export.csv",
            "/docs",
        ],
    }


@app.get("/health")
def health():
    """Service status, data freshness, and which model is live."""
    try:
        history = read_all()
        latest = history.index.max()
        age_hours = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 3600

        return {
            "status": "ok",
            "rows": row_count(),
            "latest_observation_utc": latest.isoformat(),
            "data_age_hours": round(age_hours, 1),
            "model_trained_at": BUNDLE["trained_at"],
            "model_mape": round(BUNDLE["metrics"]["MAPE"], 2),
            "model_trained_on": BUNDLE.get("trained_on", "not recorded"),
            "model_evaluated_on": BUNDLE.get("evaluated_on", "not recorded"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")


@app.get("/predict")
def predict_demand(timestamp: str):
    """Predict demand in MW for a UTC timestamp."""
    try:
        target = pd.Timestamp(timestamp, tz="UTC")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    features = build_features(read_all())

    if target not in features.index:
        raise HTTPException(
            status_code=404,
            detail=f"No features available for {target}. Data covers "
                   f"{features.index.min()} to {features.index.max()}",
        )

    row = features.loc[[target], BUNDLE["features"]]
    value = float(BUNDLE["model"].predict(row)[0])

    return {
        "timestamp_utc": target.isoformat(),
        "predicted_demand_mw": round(value, 1),
        "model": "gradient_boosting",
    }

@app.get("/forecast")
def forecast(hours: int = 48):
    """Forecast demand for the next N hours beyond the last observation.

    Predictions feed into their own lag features. Accuracy is not asserted
    here: `accuracy` reports what the logged forecasts actually scored. The
    hard limit on horizon is how far ahead weather is available.
    """
    if not 1 <= hours <= 168:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 168")

    df = read_with_future()
    series = recursive_forecast(df, hours, BUNDLE["model"], BUNDLE["features"])

    if series.empty:
        raise HTTPException(status_code=503, detail="No forecast weather available")

    local = series.tz_convert("Australia/Brisbane")

    return {
        "origin_utc": df["demand_mw"].last_valid_index().isoformat(),
        "hours_requested": hours,
        "hours_returned": len(series),
        "model": "gradient_boosting_recursive",
        "accuracy": measured_accuracy(),
        "forecast": [
            {"time_brisbane": ts.isoformat(), "demand_mw": round(v, 1)}
            for ts, v in local.items()
        ],
    }

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return render_dashboard()


@app.get("/export.csv")
def export_csv():
    import io
    df = read_predictions_vs_actual()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )