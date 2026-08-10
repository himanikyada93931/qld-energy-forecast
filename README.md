# Queensland Electricity Demand Forecasting

A live service that predicts electricity demand for Queensland using public
market and weather data.

## Status

Week 1 complete - automated data pipeline running daily.

## What it does

- Fetches Queensland demand and price data from AEMO (5-minute intervals)
- Fetches hourly weather from Open-Meteo, including solar radiation
- Converts all timestamps to UTC and resamples demand to hourly
- Stores everything in SQLite with an upsert, so re-runs never duplicate
- Runs daily at 2am via Task Scheduler, with logging and retry on failure

Currently holding ~17,000 hourly observations from September 2024.

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .
    pip install -r requirements.txt

## Usage

    python scripts/daily_update.py

## Structure

    src/ingest/     fetchers, transforms, database
    scripts/        scheduled jobs
    notebooks/      exploratory analysis and data notes
    tests/          automated checks
    data/raw/       unmodified API responses (not in git)

## Data sources

- AEMO aggregated price and demand data
- Weather data by [Open-Meteo.com](https://open-meteo.com), CC BY 4.0

## Notes

Timestamps mark the **end** of each interval. AEMO publishes in UTC+10 with no
daylight saving; all data is converted to UTC on ingest.