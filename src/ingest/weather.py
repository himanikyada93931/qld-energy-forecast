import json
import time
from pathlib import Path

import requests

from src.config import DATA_RAW

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

BRISBANE_LAT = -27.4698
BRISBANE_LON = 153.0251

VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
]


def raw_path(start_date: str, end_date: str) -> Path:
    """Where this weather response should be saved in data/raw/."""
    return DATA_RAW / f"weather_{start_date}_{end_date}.json"


def fetch_weather(start_date: str, end_date: str,
                  lat: float = BRISBANE_LAT, lon: float = BRISBANE_LON,
                  overwrite: bool = False) -> Path:
    """Download hourly weather for a date range. Dates are YYYY-MM-DD.

    Timestamps come back in UTC because the timezone parameter is omitted.
    """
    path = raw_path(start_date, end_date)

    if path.exists() and not overwrite:
        return path

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(VARIABLES),
    }

    for attempt in range(3):
        try:
            response = requests.get(BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if data.get("error"):
                raise RuntimeError(f"Open-Meteo error: {data.get('reason')}")

            path.write_text(json.dumps(data), encoding="utf-8")
            return path

        except requests.exceptions.HTTPError:
            raise

        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch weather after 3 attempts")


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def forecast_path() -> Path:
    """Where the rolling forecast response is saved."""
    return DATA_RAW / "weather_forecast.json"


def fetch_forecast(lat: float = BRISBANE_LAT, lon: float = BRISBANE_LON,
                   past_days: int = 7, forecast_days: int = 3) -> Path:
    """Fetch recent and upcoming weather to bridge the ERA5 archive delay.

    Always overwrites: this is a rolling window that must stay current.
    """
    path = forecast_path()

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    for attempt in range(3):
        try:
            response = requests.get(FORECAST_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if data.get("error"):
                raise RuntimeError(f"Open-Meteo error: {data.get('reason')}")

            path.write_text(json.dumps(data), encoding="utf-8")
            return path

        except requests.exceptions.HTTPError:
            raise

        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)

    raise RuntimeError("Failed to fetch weather forecast after 3 attempts")