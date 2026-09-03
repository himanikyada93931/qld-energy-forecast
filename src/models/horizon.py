"""Measure how forecast error grows when predictions feed into their own lags."""

import numpy as np
import pandas as pd

from src.features.build import build_features
from src.models.train import load_model

LAG_HOURS = 168  # longest lookback any feature needs


def recursive_forecast(history: pd.DataFrame, steps: int, model, features) -> pd.Series:
    """Predict `steps` hours ahead, feeding each prediction back in as history.

    `history` must contain demand and weather covering the forecast window.
    Demand beyond the origin is ignored — that is the point.
    """
    work = history.copy()
    origin = work["demand_mw"].last_valid_index()

    predictions = {}

    for step in range(1, steps + 1):
        target = origin + pd.Timedelta(hours=step)
        if target not in work.index:
            break

        recent = work.loc[target - pd.Timedelta(hours=LAG_HOURS + 24):target]
        built = build_features_keep_last(recent)

        if built is None:
            break

        value = float(model.predict(built[features])[0])
        predictions[target] = value

        # Treat the prediction as truth for the next iteration
        work.loc[target, "demand_mw"] = value

    return pd.Series(predictions)


def build_features_keep_last(df: pd.DataFrame):
    """Build features and return only the final row, or None if incomplete."""
    from src.features.build import add_time_features, add_weather_features, add_lag_features

    out = add_time_features(df.copy())
    out = add_weather_features(out)
    out = add_lag_features(out)

    last = out.iloc[[-1]]
    return None if last.drop(columns=["demand_mw", "price_aud_mwh"]).isna().any().any() else last

def horizon_error(df: pd.DataFrame, max_hours: int = 96, n_origins: int = 30) -> pd.DataFrame:
    """Backtest recursive forecasts from many past origins.

    At each origin, demand after that point is hidden, so the forecast must
    feed on its own predictions — the same as real use.
    """
    bundle = load_model()
    model, features = bundle["model"], bundle["features"]

    usable = df.iloc[LAG_HOURS + 48:-max_hours]
    origins = usable.index[np.linspace(0, len(usable) - 1, n_origins).astype(int)]

    records = []
    for origin in origins:
        window = df.loc[:origin + pd.Timedelta(hours=max_hours)].copy()
        actual = window["demand_mw"].copy()

        # Hide everything after the origin
        window.loc[window.index > origin, "demand_mw"] = np.nan

        predicted = recursive_forecast(window, max_hours, model, features)

        for ts, value in predicted.items():
            if pd.notna(actual.get(ts)):
                records.append({
                    "hours_ahead": int((ts - origin).total_seconds() // 3600),
                    "abs_error": abs(value - actual[ts]),
                    "actual": actual[ts],
                })

    r = pd.DataFrame(records)
    r["day"] = ((r["hours_ahead"] - 1) // 24) + 1

    return r.groupby("day").apply(
        lambda g: pd.Series({
            "n": len(g),
            "MAE": g["abs_error"].mean(),
            "MAPE": (g["abs_error"] / g["actual"]).mean() * 100,
        }),
        include_groups=False,
    )