# Queensland Electricity Demand Forecasting

A live service that predicts electricity demand for Queensland using public
market and weather data.

## Status

Week 1 — data ingestion.

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .
    pip install -r requirements.txt

## Structure

    src/         production code
    notebooks/   exploratory analysis
    tests/       automated checks
    data/raw/    unmodified API responses