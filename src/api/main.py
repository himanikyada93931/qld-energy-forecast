"""Prediction API for Queensland electricity demand."""

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.ingest.database import read_all, row_count
from src.models.baseline import predict

app = FastAPI(title="QLD Energy Demand Forecast", version="0.1.0")


@app.get("/")
def root():
    """Basic service information."""
    return {
        "service": "QLD electricity demand forecast",
        "model": "seasonal naive (same hour last week)",
        "endpoints": ["/health", "/predict?timestamp=YYYY-MM-DDTHH:MM"],
    }


@app.get("/health")
def health():
    """Is the service alive, and how fresh is its data?"""
    try:
        history = read_all()
        latest = history.index.max()
        age_hours = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 3600

        return {
            "status": "ok",
            "rows": row_count(),
            "latest_observation_utc": latest.isoformat(),
            "data_age_hours": round(age_hours, 1),
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

    history = read_all()

    try:
        value = predict(history, target)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "timestamp_utc": target.isoformat(),
        "predicted_demand_mw": round(value, 1),
        "model": "seasonal_naive_168h",
    }