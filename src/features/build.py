"""Turn raw observations into model features."""

import numpy as np
import pandas as pd

HEATING_BASE = 18.0
COOLING_BASE = 24.0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features, using Brisbane local time."""
    local = df.index.tz_convert("Australia/Brisbane")

    df["hour"] = local.hour
    df["dayofweek"] = local.dayofweek
    df["month"] = local.month
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)

    # Place cyclical values on a circle so 23:00 sits next to 00:00
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split temperature into heating and cooling need."""
    temp = df["temperature_2m"]

    df["heating_degrees"] = (HEATING_BASE - temp).clip(lower=0)
    df["cooling_degrees"] = (temp - COOLING_BASE).clip(lower=0)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past demand. Only values that would genuinely be known in advance."""
    for hours in (24, 48, 168):
        df[f"demand_lag_{hours}h"] = df["demand_mw"].shift(hours)

    df["demand_roll_24h"] = df["demand_mw"].shift(24).rolling(24).mean()
    df["demand_roll_168h"] = df["demand_mw"].shift(24).rolling(168).mean()

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline. Drops rows with incomplete lags."""
    df = df.copy()
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_lag_features(df)

    return df.dropna()

def build_forecast_features(df: pd.DataFrame, hours_ahead: int = 24) -> pd.DataFrame:
    """Build features for timestamps beyond the last demand observation.

    Only goes as far as the shortest lag allows: with a 24-hour lag, predictions
    can extend 24 hours past the last known demand value.
    """
    last_known = df["demand_mw"].last_valid_index()

    future_index = pd.date_range(
        last_known + pd.Timedelta(hours=1),
        last_known + pd.Timedelta(hours=hours_ahead),
        freq="1h",
    )

    combined = df.reindex(df.index.union(future_index))

    combined = add_time_features(combined)
    combined = add_weather_features(combined)
    combined = add_lag_features(combined)

    return combined.loc[future_index]