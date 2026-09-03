"""Prediction API for Queensland electricity demand."""

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features.build import build_features
from src.ingest.database import read_all, row_count, read_with_future
from src.models.train import load_model

from src.models.horizon import recursive_forecast



app = FastAPI(title="QLD Energy Demand Forecast", version="0.2.0")

BUNDLE = load_model()


@app.get("/")
def root():
    return {
        "service": "QLD electricity demand forecast",
        "model": "gradient boosting",
        "endpoints": ["/health", "/predict?timestamp=YYYY-MM-DDTHH:MM"],
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

    Predictions feed into their own lag features, so accuracy holds at roughly
    2.3-2.7% MAPE out to seven days. The real limit is weather availability.
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
        "expected_mape": "2.3-2.7%",
        "forecast": [
            {"time_brisbane": ts.isoformat(), "demand_mw": round(v, 1)}
            for ts, v in local.items()
        ],
    }