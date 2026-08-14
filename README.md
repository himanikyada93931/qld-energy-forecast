# Queensland Electricity Demand Forecasting

A live service that predicts electricity demand for Queensland using public market
and weather data.

**Live:** https://qld-energy-forecast-production.up.railway.app/health

## Status

Week 2 — baseline model deployed. Data pipeline running daily.

## What it does

- Fetches Queensland demand and price data from AEMO (5-minute intervals)
- Fetches hourly weather from Open-Meteo, including solar radiation
- Converts all timestamps to UTC and resamples demand to hourly
- Stores everything in SQLite with an upsert, so re-runs never duplicate
- Runs daily with logging, retry, and exponential backoff on failure
- Serves predictions over a FastAPI endpoint, containerised with Docker

Currently holding ~17,000 hourly observations from September 2024.

## Model

Seasonal naive baseline: demand equals the same hour one week earlier.

| Metric | Value |
|---|---|
| MAE | 434 MW |
| RMSE | 639 MW |
| MAPE | 7.36% |
| Bias | −4.6 MW |

This is deliberately simple. It is the number every future model has to beat —
without a baseline, an accuracy figure means nothing. A naive method already gets
within 7% because electricity demand is highly repetitive.

One week rather than one day, because weekday and weekend demand differ; a 168-hour
lag keeps the day type matching.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status, row count, and data freshness |
| `GET /predict?timestamp=YYYY-MM-DDTHH:MM` | Predicted demand in MW |
| `GET /docs` | Interactive API documentation |

## Running locally

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .
    pip install -r requirements.txt
    uvicorn src.api.main:app --reload

## Running in Docker

    docker build -t qld-energy .
    docker run -p 8000:8000 qld-energy

## Updating the data

    python scripts/daily_update.py

Safe to re-run at any time. Fetches only recent months, aborts without writing if
any source is unavailable, and exits non-zero so a scheduler can detect failure.

## Structure

    src/ingest/     fetchers, transforms, database
    src/models/     baseline model
    src/api/        FastAPI service
    scripts/        scheduled jobs
    notebooks/      exploratory analysis and data notes
    tests/          automated checks
    data/raw/       unmodified API responses (not in git)

## Data notes

Findings from the inspection notebook that shaped the design:

**Timestamps mark the end of each interval.** AEMO publishes in UTC+10 with no
daylight saving. All data is converted to UTC on ingest.

**Open-Meteo mixes conventions in a single response.** `shortwave_radiation` is a
preceding-hour mean; everything else is instantaneous.

**Demand is lowest around 11am, not overnight.** `TOTALDEMAND` measures draw from
the grid, and rooftop solar supplies homes directly at midday — the duck curve.
Price follows the same shape, collapsing to around $13/MWh at 10am against $113 at
5pm, and going negative 10% of the time. This is why solar radiation is a required
feature rather than an optional one.

**Cloud cover is a poor proxy for sunlight.** One sample day showed 77% cloud with
near clear-sky radiation, because total cloud includes high cirrus that blocks
little light. `shortwave_radiation` measures what actually reaches the ground.

## Known shortcuts

Deliberate and documented, not oversights.

- **The SQLite database is committed to Git** as a one-off snapshot so the container
  has data at build time. This is not good practice — a binary file that changes
  daily does not belong in version control, and the deployed copy goes stale. The
  proper fix is a startup job that builds the database from the APIs, or a
  persistent volume. Planned for week 5.
- **Python version mismatch.** Development runs 3.14, the container 3.12, because
  3.14 slim images are not yet reliable for all dependencies. These should match.
- **No automated tests yet.** Planned for week 5.

## Data sources

- AEMO aggregated price and demand data
- Weather data by [Open-Meteo.com](https://open-meteo.com), CC BY 4.0
  (Zippenfenig, P. 2023, DOI 10.5281/ZENODO.7970649)