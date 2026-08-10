import time
from pathlib import Path

import requests

from src.config import DATA_RAW

BASE_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand"


def filename_for(year: int, month: int, region: str = "QLD1") -> str:
    """Return the AEMO filename for one month of one region."""
    return f"PRICE_AND_DEMAND_{year}{month:02d}_{region}.csv"


def build_url(year: int, month: int, region: str = "QLD1") -> str:
    """Return the AEMO download URL for one month of one region."""
    return f"{BASE_URL}/{filename_for(year, month, region)}"


def raw_path(year: int, month: int, region: str = "QLD1") -> Path:
    """Where this month's file should be saved in data/raw/."""
    return DATA_RAW / filename_for(year, month, region)


def fetch_month(year: int, month: int, region: str = "QLD1",
                overwrite: bool = False) -> Path:
    """Download one month of AEMO data if not already present.

    Returns the saved path. Retries temporary failures, raises on permanent ones.
    """
    path = raw_path(year, month, region)

    if path.exists() and not overwrite:
        return path

    url = build_url(year, month, region)

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
            return path

        except requests.exceptions.HTTPError:
            raise

        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to fetch {url} after 3 attempts")

def fetch_range(start_year: int, start_month: int,
                end_year: int, end_month: int,
                region: str = "QLD1") -> list[Path]:
    """Fetch every month in the range. Continue if individual months fail."""
    paths = []
    failures = []

    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        try:
            paths.append(fetch_month(year, month, region))
        except Exception as exc:
            failures.append((year, month, exc))

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    if failures:
        print(f"{len(failures)} month(s) failed:")
        for y, m, exc in failures:
            print(f"  {y}-{m:02d}: {exc}")

    return paths