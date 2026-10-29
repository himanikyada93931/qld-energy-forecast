"""Turn raw observations into model features."""

from datetime import timedelta

import holidays
import numpy as np
import pandas as pd

HEATING_BASE = 18.0
COOLING_BASE = 24.0

# Queensland public holidays. The package works out any year on demand, so this
# keeps working as the service rolls forward instead of needing a hand-edited
# list of dates every January.
QLD_HOLIDAYS = holidays.Australia(subdiv="QLD")

# The Ekka is a Brisbane-only holiday: the rest of the state, and the industrial
# load that dominates it, works as normal, so state-wide demand barely dips.
# Backtesting every holiday in the data confirmed it: flagging the Ekka made
# those days 54% worse (166 -> 256 MW) while excluding it left the other
# holidays 36% better. Treated as an ordinary working day for that reason.
BRISBANE_ONLY = {"The Royal Queensland Show"}


def is_statewide_holiday(day) -> bool:
    """True on a public holiday that most of Queensland actually takes off."""
    return day in QLD_HOLIDAYS and QLD_HOLIDAYS.get(day) not in BRISBANE_ONLY


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


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag Queensland public holidays by Brisbane calendar date.

    Without this the model treats a public holiday as an ordinary weekday and
    over-predicts it. Holiday dates are fixed years in advance, so there is
    nothing to leak here: a forecast for next Monday can legitimately already
    know that next Monday is a holiday.
    """
    day = pd.Series(df.index.tz_convert("Australia/Brisbane").date, index=df.index)

    df["is_holiday"] = day.map(is_statewide_holiday).astype(int)
    df["is_day_after_holiday"] = day.map(
        lambda d: is_statewide_holiday(d - timedelta(days=1))
    ).astype(int)

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


def apply_feature_steps(df: pd.DataFrame) -> pd.DataFrame:
    """The one definition of the feature pipeline.

    Every caller goes through here - training, the forecast endpoint and the
    recursive backtest - so the live service can never be built on a different
    set of columns from the one the model was trained on.
    """
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = add_weather_features(df)
    df = add_lag_features(df)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline. Drops rows with incomplete lags."""
    return apply_feature_steps(df.copy()).dropna()

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

    combined = apply_feature_steps(df.reindex(df.index.union(future_index)))

    return combined.loc[future_index]