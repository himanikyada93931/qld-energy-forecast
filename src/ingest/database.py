import sqlite3

import pandas as pd

from src.config import DB_PATH

WEATHER_TABLE = "weather_future"

WEATHER_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {WEATHER_TABLE} (
    timestamp_utc        TEXT PRIMARY KEY,
    temperature_2m       REAL,
    apparent_temperature REAL,
    relative_humidity_2m REAL,
    cloud_cover          REAL,
    shortwave_radiation  REAL
)
"""

TABLE = "observations"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    timestamp_utc        TEXT PRIMARY KEY,
    demand_mw            REAL,
    price_aud_mwh        REAL,
    temperature_2m       REAL,
    apparent_temperature REAL,
    relative_humidity_2m REAL,
    cloud_cover          REAL,
    shortwave_radiation  REAL
)
"""


def connect() -> sqlite3.Connection:
    """Open a connection to the project database, creating the table if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.execute(WEATHER_SCHEMA)
    conn.commit()
    return conn


def upsert(df: pd.DataFrame) -> int:
    """Insert rows, replacing any that already exist. Returns rows written."""
    records = df.reset_index()
    records["timestamp_utc"] = records["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    columns = list(records.columns)
    placeholders = ",".join("?" * len(columns))
    sql = f"INSERT OR REPLACE INTO {TABLE} ({','.join(columns)}) VALUES ({placeholders})"

    with connect() as conn:
        conn.executemany(sql, records.itertuples(index=False, name=None))
        conn.commit()

    return len(records)

def upsert_future_weather(df: pd.DataFrame) -> int:
    """Store weather for timestamps that have no demand yet."""
    records = df.reset_index()
    records["timestamp_utc"] = records["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    columns = list(records.columns)
    placeholders = ",".join("?" * len(columns))
    sql = f"INSERT OR REPLACE INTO {WEATHER_TABLE} ({','.join(columns)}) VALUES ({placeholders})"

    with connect() as conn:
        conn.executemany(sql, records.itertuples(index=False, name=None))
        conn.commit()

    return len(records)


def read_with_future() -> pd.DataFrame:
    """Observations plus future weather rows, for forecasting."""
    observed = read_all()

    with connect() as conn:
        future = pd.read_sql(f"SELECT * FROM {WEATHER_TABLE} ORDER BY timestamp_utc", conn)

    if future.empty:
        return observed

    future["timestamp_utc"] = pd.to_datetime(future["timestamp_utc"], format="ISO8601", utc=True)
    future = future.set_index("timestamp_utc")
    future = future[~future.index.isin(observed.index)]

    combined = pd.concat([observed, future]).sort_index()
    combined.index.name = "timestamp_utc"
    return combined


def read_all() -> pd.DataFrame:
    """Read the whole table back as a DataFrame indexed by UTC timestamp."""
    with connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {TABLE} ORDER BY timestamp_utc", conn)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], format="ISO8601", utc=True)
    return df.set_index("timestamp_utc")


def row_count() -> int:
    """How many rows are currently stored."""
    with connect() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]