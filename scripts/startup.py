"""Build the database on first boot if the volume is empty, then start the API."""

import logging
import sys

from src.config import DB_PATH
from src.ingest.database import row_count

from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    MIN_ROWS = 10_000

    if DB_PATH.exists() and row_count() >= MIN_ROWS:
        log.info("Database has %s rows (minimum %s), skipping build",
                 row_count(), MIN_ROWS)
        return 0

    log.info("Empty database, building from source APIs. This takes a few minutes.")

    from scripts.daily_update import main as update
    from src.ingest.aemo import fetch_range
    from datetime import date

    today = date.today()
    start_year = today.year - 2
    fetch_range(start_year, today.month, today.year, today.month)

    from src.ingest.weather import fetch_weather

    start = date(start_year, today.month, 1)
    end = today - timedelta(days=6)
    fetch_weather(start.isoformat(), end.isoformat())
    log.info("Historical weather fetched: %s to %s", start, end)

    return update()


if __name__ == "__main__":
    sys.exit(main())