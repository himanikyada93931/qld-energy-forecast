"""Test setup that must run before the application is imported.

Two things happen here, both before `src.api.main` is loaded.

First, the background refresh thread is switched off. It fetches from AEMO and
Open-Meteo, which a test suite must never do: it is slow, it needs the network,
and it would write to the real database.

Second, if there is no database, a small synthetic one is built. The real
database is deliberately not committed to Git, so continuous integration starts
without one. Rather than skipping the API tests there, the endpoints are
exercised against generated data that has the same shape as the real thing.
"""

import os

os.environ["REFRESH_ENABLED"] = "0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import DB_PATH  # noqa: E402
from src.ingest.database import (  # noqa: E402
    log_predictions,
    row_count,
    upsert,
    upsert_future_weather,
)

WEATHER_COLUMNS = ["temperature_2m", "apparent_temperature",
                   "relative_humidity_2m", "cloud_cover", "shortwave_radiation"]

# Enough history for a 168-hour lag plus a 168-hour rolling window, and then some.
SEED_HOURS = 24 * 40
FORECAST_HOURS = 72


def _weather(index, rng):
    """Weather with a daily shape, so solar radiation peaks around midday."""
    hour = index.tz_convert("Australia/Brisbane").hour.to_numpy()
    daylight = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)

    return pd.DataFrame(
        {
            "temperature_2m": 22 + 6 * daylight + rng.normal(0, 1.5, len(index)),
            "apparent_temperature": 22 + 6 * daylight + rng.normal(0, 1.5, len(index)),
            "relative_humidity_2m": rng.uniform(40, 90, len(index)),
            "cloud_cover": rng.uniform(0, 100, len(index)),
            "shortwave_radiation": 850 * daylight + rng.normal(0, 20, len(index)),
        },
        index=index,
    )


def _seed_database():
    """Build a small database with the same shape as the real one."""
    rng = np.random.default_rng(42)
    end = pd.Timestamp.now(tz="UTC").floor("h")
    index = pd.date_range(end - pd.Timedelta(hours=SEED_HOURS - 1), end, freq="1h")

    hour = index.tz_convert("Australia/Brisbane").hour.to_numpy()
    demand = (5900
              + 1500 * np.sin((hour - 10) / 24 * 2 * np.pi)
              - 1100 * np.exp(-((hour - 11) ** 2) / 8)
              + rng.normal(0, 120, len(index)))

    observations = _weather(index, rng)
    observations.insert(0, "demand_mw", demand)
    observations.insert(1, "price_aud_mwh", rng.normal(80, 25, len(index)))
    observations.index.name = "timestamp_utc"
    upsert(observations)

    future = pd.date_range(end + pd.Timedelta(hours=1),
                           end + pd.Timedelta(hours=FORECAST_HOURS), freq="1h")
    future_weather = _weather(future, rng)
    future_weather.index.name = "timestamp_utc"
    upsert_future_weather(future_weather[WEATHER_COLUMNS])

    # A handful of forecasts for hours that have already happened, so the
    # dashboard and the measured-accuracy endpoint have something to score.
    scored = index[-72:]
    predictions = pd.Series(demand[-72:] + rng.normal(0, 150, 72), index=scored)
    log_predictions(predictions, scored[0] - pd.Timedelta(hours=1), "synthetic")


if not DB_PATH.exists() or row_count() < SEED_HOURS:
    _seed_database()
