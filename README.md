# Queensland Electricity Demand Forecasting

A live service that predicts electricity demand for Queensland using public market
and weather data.

**Live:** https://qld-energy-forecast-production.up.railway.app/health

## Status

Week 3 — gradient boosting model trained, validated, and deployed.

## What it does

- Fetches Queensland demand and price data from AEMO (5-minute intervals)
- Fetches hourly weather from Open-Meteo, including solar radiation
- Converts all timestamps to UTC and resamples demand to hourly
- Stores everything in SQLite with an upsert, so re-runs never duplicate
- Runs daily with logging, retry, and exponential backoff on failure
- Builds 20 features, trains a gradient boosting model, and serves it over FastAPI
- Containerised with Docker and deployed to Railway

Currently holding ~17,300 hourly observations from September 2024.

## Results

Tested on the most recent 25% of the data, with training restricted to the period
before it.

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal naive baseline | 434 MW | 639 MW | 7.36% |
| Linear regression | 215 MW | — | 3.74% |
| Random forest | 165 MW | — | 2.89% |
| **Gradient boosting (deployed)** | **145 MW** | **210 MW** | **2.55%** |

A 66% reduction in error against the baseline.

The baseline — demand equals the same hour one week earlier — was built first
deliberately. Electricity demand is highly repetitive, so a naive method already
reaches 7.36%. Without that number, an accuracy figure means nothing.

Linear regression more than halving the baseline error indicates the features are
doing most of the work, not the algorithm.

### Where the model fails

Mean absolute error by hour of day, Brisbane time:

| Time | Mean error |
|---|---|
| 2–5am | ~65 MW |
| 11am–3pm | ~270 MW |
| 6–7pm | ~105 MW |

Midday error is roughly four times the overnight error. This was predicted from the
exploratory analysis before the model was built: rooftop solar makes midday grid
demand volatile in a way the model cannot fully anticipate.

### Caveat

The model was trained and tested on *actual* weather. In production it would receive
*forecast* weather, which carries its own error, so real-world performance will be
somewhat worse than 2.55%. Weather accounts for only about 9% of feature importance,
so the effect is limited, but the figure should be read with this in mind.

## Features

| Group | Detail |
|---|---|
| Calendar | Hour, day of week, month, weekend flag |
| Cyclical | Hour and day-of-week as sine/cosine pairs |
| Weather | Temperature, apparent temperature, humidity, cloud cover, solar radiation |
| Degree days | Heating (below 18°C) and cooling (above 24°C) split into separate features |
| Lags | Demand at 24h, 48h and 168h earlier |
| Rolling | 24-hour and 168-hour means, shifted to exclude the present |

Two deliberate exclusions:

- **No 1-hour lag**, despite a 0.885 correlation. Predicting tomorrow at 6pm would
  require tomorrow at 5pm, which does not exist at prediction time.
- **Price is excluded.** Tomorrow's wholesale price is not known when forecasting
  tomorrow's demand.

Rolling windows are shifted 24 hours before averaging so they never include the
current value.

Top feature importances: `demand_lag_24h` (0.76), `demand_lag_168h` (0.06),
`shortwave_radiation` (0.04).

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, row count, data freshness, and which model is live |
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

Safe to re-run at any time. Fetches only recent months, aborts without writing if any
source is unavailable, and exits non-zero so a scheduler can detect failure.

## Retraining

    python -c "from src.ingest.database import read_all; from src.features.build import build_features; from src.models.train import train_and_save; print(train_and_save(build_features(read_all())))"

Saves the model together with its feature list, metrics and training timestamp, so
the `/health` endpoint reports what is actually deployed rather than relying on this
file being current.

## Structure

    src/ingest/     fetchers, transforms, database
    src/features/   feature engineering
    src/models/     baseline and trained models
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

**Demand is lowest around 11am, not overnight.** `TOTALDEMAND` measures draw from the
grid, and rooftop solar supplies homes directly at midday — the duck curve. Price
follows the same shape, collapsing to around $13/MWh at 10am against $113 at 5pm, and
going negative 10% of the time. This is why solar radiation is a required feature
rather than an optional one, and why demand versus temperature is a U shape rather
than a line.

**Cloud cover is a poor proxy for sunlight.** One sample day showed 77% cloud with
near clear-sky radiation, because total cloud includes high cirrus that blocks little
light. `shortwave_radiation` measures what actually reaches the ground.

## Known limitations

Deliberate and documented, not oversights.

- **No true forecasting yet.** `/predict` works only within the range of stored data,
  because lag features do not exist for future timestamps. Forecasting tomorrow
  requires either recursive prediction or a separate model trained on forecast
  weather.
- **Data is about five days behind.** The ERA5 weather archive lags by roughly five
  days, and the inner join trims demand to match.
- **The database and model file are committed to Git** so the container has them at
  build time. A binary that changes on every retrain does not belong in version
  control. The proper fix is a startup job or a persistent volume.
- **Python version mismatch.** Development runs 3.14, the container 3.12, because
  3.14 slim images are not yet reliable for all dependencies.
- **No automated tests yet.**

## Data sources

- AEMO aggregated price and demand data
- Weather data by [Open-Meteo.com](https://open-meteo.com), CC BY 4.0
  (Zippenfenig, P. 2023, DOI 10.5281/ZENODO.7970649)