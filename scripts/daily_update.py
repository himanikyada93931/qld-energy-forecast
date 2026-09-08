"""Fetch any new data and load it into the database. Safe to re-run."""

import logging
import sys
from datetime import date, timedelta
import pandas as pd

from src.models.train import load_model
from src.models.horizon import recursive_forecast

from src.config import LOGS
from src.ingest.aemo import fetch_month
from src.ingest.weather import fetch_weather, fetch_forecast
from src.ingest.transform import build_dataset, build_dataset_with_future
from src.ingest.database import (upsert, row_count, upsert_future_weather,log_predictions, read_with_future)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS / "daily_update.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

WEATHER_LAG_DAYS = 5


def main():
    log.info("=" * 50)
    log.info("Daily update starting")

    before = row_count()
    log.info("Rows before: %s", before)

    today = date.today()
    failures = []

    last_month = today.replace(day=1) - timedelta(days=1)
    for year, month in [(last_month.year, last_month.month), (today.year, today.month)]:
        try:
            fetch_month(year, month, overwrite=True)
            log.info("AEMO %s-%02d fetched", year, month)
        except Exception as exc:
            log.error("AEMO %s-%02d failed: %s", year, month, exc)
            failures.append(f"aemo {year}-{month:02d}")

    end = today - timedelta(days=WEATHER_LAG_DAYS)
    start = end - timedelta(days=30)
    try:
        fetch_weather(start.isoformat(), end.isoformat(), overwrite=True)
        log.info("Weather %s to %s fetched", start, end)
    except Exception as exc:
        log.error("Weather failed: %s", exc)
        failures.append("weather")

    try:
        fetch_forecast()
        log.info("Weather forecast fetched")
    except Exception as exc:
        log.error("Weather forecast failed: %s", exc)
        failures.append("forecast")

    if failures:
        log.error("Aborting, %d source(s) failed: %s", len(failures), ", ".join(failures))
        return 1

    try:
        df = build_dataset()
        upsert(df)
        after = row_count()

        # Store weather for hours that have no demand yet, so the API can forecast
        with_future = build_dataset_with_future()
        future_only = with_future[with_future["demand_mw"].isna()]
        weather_cols = ["temperature_2m", "apparent_temperature",
                        "relative_humidity_2m", "cloud_cover", "shortwave_radiation"]
        n_future = upsert_future_weather(future_only[weather_cols])
        log.info("Future weather rows stored: %s", n_future)
    except Exception as exc:
        log.exception("Load failed: %s", exc)
        return 1

    log.info("Rows after: %s (+%s new)", after, after - before)
    log.info("Latest timestamp: %s", df.index.max())
    try:
        bundle = load_model()
        data = read_with_future()
        origin = data["demand_mw"].last_valid_index()
        series = recursive_forecast(data, 72, bundle["model"], bundle["features"])
        n = log_predictions(series, origin, bundle["trained_at"])
        log.info("Predictions logged: %s", n)
    except Exception as exc:
        log.error("Prediction logging failed: %s", exc)
    log.info("Finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())