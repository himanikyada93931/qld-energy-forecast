"""The feature builder is where leakage hides. These tests pin its behaviour.

A leak does not raise an error. It produces a better score and a model that
fails in production, so every rule the builder relies on is asserted here.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build import add_lag_features, build_features, build_forecast_features


@pytest.fixture
def hourly():
    """Two months of hourly data, enough for a 168-hour lag plus a rolling window."""
    index = pd.date_range("2026-01-01", periods=24 * 60, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "demand_mw": np.linspace(5000, 6000, len(index)) + rng.normal(0, 50, len(index)),
            "price_aud_mwh": rng.normal(80, 20, len(index)),
            "temperature_2m": rng.normal(22, 5, len(index)),
            "apparent_temperature": rng.normal(22, 5, len(index)),
            "relative_humidity_2m": rng.uniform(40, 90, len(index)),
            "cloud_cover": rng.uniform(0, 100, len(index)),
            "shortwave_radiation": rng.uniform(0, 900, len(index)),
        },
        index=index,
    )


@pytest.mark.parametrize("hours", [24, 48, 168])
def test_lag_is_exactly_that_many_hours_back(hourly, hours):
    """demand_lag_24h at 6pm must be the demand at 6pm yesterday, not 5pm or 7pm."""
    out = add_lag_features(hourly.copy())
    t = hourly.index[400]

    assert out.loc[t, f"demand_lag_{hours}h"] == pytest.approx(
        hourly.loc[t - pd.Timedelta(hours=hours), "demand_mw"]
    )


def test_rolling_means_exclude_the_present(hourly):
    """The 24h mean must be shifted, or it averages in the value being predicted."""
    out = add_lag_features(hourly.copy())
    t = hourly.index[400]

    window = hourly["demand_mw"].loc[t - pd.Timedelta(hours=47): t - pd.Timedelta(hours=24)]
    assert len(window) == 24
    assert out.loc[t, "demand_roll_24h"] == pytest.approx(window.mean())


def test_no_one_hour_lag_feature(hourly):
    """The 1-hour lag correlates at 0.885 and is unknowable at forecast time."""
    out = build_features(hourly.copy())
    assert "demand_lag_1h" not in out.columns


def test_price_is_not_a_feature(hourly):
    """Tomorrow's price is not known when forecasting tomorrow's demand."""
    from src.models.train import EXCLUDE
    assert "price_aud_mwh" in EXCLUDE


def test_cyclical_hour_wraps(hourly):
    """23:00 and 00:00 must sit next to each other on the circle, not 23 apart."""
    out = build_features(hourly.copy())
    local = out.index.tz_convert("Australia/Brisbane")

    at_23 = out[local.hour == 23].iloc[0]
    at_00 = out[local.hour == 0].iloc[0]

    distance = np.hypot(at_23["hour_sin"] - at_00["hour_sin"],
                        at_23["hour_cos"] - at_00["hour_cos"])
    assert distance < 0.3


def test_build_features_leaves_no_gaps(hourly):
    """Incomplete lag rows must be dropped, not filled."""
    out = build_features(hourly.copy())
    assert not out.isna().any().any()
    assert len(out) < len(hourly)


def test_forecast_features_start_after_the_last_observation(hourly):
    """Forecast rows must describe hours we have not observed yet."""
    df = hourly.copy()
    df.loc[df.index[-10:], "demand_mw"] = np.nan

    future = build_forecast_features(df, hours_ahead=24)
    last_known = df["demand_mw"].last_valid_index()

    assert future.index.min() == last_known + pd.Timedelta(hours=1)
    assert len(future) == 24
