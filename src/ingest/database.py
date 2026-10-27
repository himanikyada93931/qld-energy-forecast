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

PREDICTIONS_TABLE = "predictions"

PREDICTIONS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
    target_utc     TEXT NOT NULL,
    made_at_utc    TEXT NOT NULL,   -- last observed hour the forecast was built from
    issued_at_utc  TEXT,            -- wall-clock time the forecast was written
    hours_ahead    INTEGER,         -- target minus data origin
    lead_hours     INTEGER,         -- target minus issue time: the honest lead time
    predicted_mw   REAL,
    model_version  TEXT,
    PRIMARY KEY (target_utc, issued_at_utc)
)
"""


def _migrate_predictions(conn) -> None:
    """Add issue-time columns to an older predictions table, keeping every row.

    The original table keyed on (target, data origin) and used INSERT OR
    REPLACE, so each six-hourly refresh overwrote the earlier forecast for the
    same hour. That made "logged before the fact" impossible to prove. The new
    key includes the issue time, so refreshes append instead of overwriting.
    Rows written before this change keep a NULL issue time: their lead time is
    not known, and the dashboard says so rather than guessing.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({PREDICTIONS_TABLE})")}

    if not existing:
        conn.execute(PREDICTIONS_SCHEMA)
        return

    if "issued_at_utc" in existing:
        return

    conn.execute(f"ALTER TABLE {PREDICTIONS_TABLE} RENAME TO predictions_legacy")
    conn.execute(PREDICTIONS_SCHEMA)
    conn.execute(
        f"""INSERT OR IGNORE INTO {PREDICTIONS_TABLE}
            (target_utc, made_at_utc, issued_at_utc, hours_ahead,
             lead_hours, predicted_mw, model_version)
            SELECT target_utc, made_at_utc, NULL, hours_ahead,
                   NULL, predicted_mw, model_version
            FROM predictions_legacy"""
    )
    conn.commit()


def connect() -> sqlite3.Connection:
    """Open a connection to the project database, creating the table if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.execute(WEATHER_SCHEMA)
    _migrate_predictions(conn)
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

def log_predictions(series: pd.Series, made_at: pd.Timestamp,
                    model_version: str) -> int:
    """Record what was predicted, from which data, and at what clock time.

    `made_at` is the last observed hour the forecast was built from. That is
    not the same as when the forecast was issued: the data always lags the
    clock. `lead_hours` is measured from the issue time, so a row can only
    claim to be ahead of the fact if it genuinely was.

    Rows are appended, never replaced, so the history of what was believed and
    when survives every refresh.
    """
    issued_at = pd.Timestamp.now(tz="UTC").floor("s")

    rows = [
        (
            ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            made_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            issued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            int((ts - made_at).total_seconds() // 3600),
            int((ts - issued_at).total_seconds() // 3600),
            float(value),
            model_version,
        )
        for ts, value in series.items()
    ]

    sql = (f"INSERT OR IGNORE INTO {PREDICTIONS_TABLE} "
           "(target_utc, made_at_utc, issued_at_utc, hours_ahead, "
           "lead_hours, predicted_mw, model_version) "
           "VALUES (?,?,?,?,?,?,?)")

    with connect() as conn:
        conn.executemany(sql, rows)
        conn.commit()

    return len(rows)


def read_predictions_vs_actual() -> pd.DataFrame:
    """Join logged predictions against what actually happened."""
    with connect() as conn:
        df = pd.read_sql(
            f"""SELECT p.target_utc, p.made_at_utc, p.issued_at_utc,
                       p.hours_ahead, p.lead_hours,
                       p.predicted_mw, p.model_version, o.demand_mw AS actual_mw
                FROM {PREDICTIONS_TABLE} p
                LEFT JOIN {TABLE} o ON o.timestamp_utc = p.target_utc
                ORDER BY p.target_utc""",
            conn,
        )

    for col in ("target_utc", "made_at_utc", "issued_at_utc"):
        df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)

    df["error_mw"] = df["predicted_mw"] - df["actual_mw"]
    return df