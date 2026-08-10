import json
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW


def load_aemo() -> pd.DataFrame:
    """Read every AEMO CSV in data/raw/ into one hourly DataFrame in UTC."""
    files = sorted(DATA_RAW.glob("PRICE_AND_DEMAND_*.csv"))
    if not files:
        raise FileNotFoundError(f"No AEMO CSVs found in {DATA_RAW}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df["SETTLEMENTDATE"] = (
        pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
          .dt.tz_localize("Etc/GMT-10")
          .dt.tz_convert("UTC")
    )

    df = df.drop(columns=["REGION", "PERIODTYPE"])
    df = df.drop_duplicates(subset="SETTLEMENTDATE").sort_values("SETTLEMENTDATE")
    df = df.set_index("SETTLEMENTDATE")

    hourly = df.resample("1h", label="right", closed="right").mean()
    hourly = hourly.rename(columns={"TOTALDEMAND": "demand_mw", "RRP": "price_aud_mwh"})

    return hourly


def load_weather() -> pd.DataFrame:
    """Read the Open-Meteo JSON files in data/raw/ into one DataFrame in UTC."""
    files = sorted(DATA_RAW.glob("weather_*.json"))
    if not files:
        raise FileNotFoundError(f"No weather JSON found in {DATA_RAW}")

    frames = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        frames.append(pd.DataFrame(data["hourly"]))

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    df = df.drop_duplicates(subset="time").sort_values("time")

    return df.set_index("time")


def build_dataset() -> pd.DataFrame:
    """Join hourly demand and weather on the UTC timestamp."""
    demand = load_aemo()
    weather = load_weather()

    df = demand.join(weather, how="inner")
    df.index.name = "timestamp_utc"

    return df