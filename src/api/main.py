"""Prediction API for Queensland electricity demand."""

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.features.build import build_features
from src.ingest.database import read_all, row_count
from src.models.train import load_model

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