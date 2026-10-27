"""The prediction log is the evidence behind every accuracy claim.

If a refresh can overwrite an earlier forecast, the log stops being proof of
what was believed and when. These tests hold that guarantee.
"""

import sqlite3
import time

import pandas as pd
import pytest

import src.config as config
import src.ingest.database as db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point the database module at a throwaway file."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def _forecast(hours=3):
    index = pd.date_range("2026-06-01", periods=hours, freq="1h", tz="UTC")
    return pd.Series([5000.0 + i for i in range(hours)], index=index)


def test_new_table_has_issue_time_columns(temp_db):
    with db.connect() as conn:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({db.PREDICTIONS_TABLE})")}
    assert {"issued_at_utc", "lead_hours"} <= cols


def test_refresh_appends_instead_of_overwriting(temp_db):
    """Two runs for the same hours must leave two rows, not one.

    The original schema keyed on (target, data origin) with INSERT OR REPLACE,
    so each six-hourly refresh destroyed the earlier forecast for that hour.
    """
    origin = pd.Timestamp("2026-05-31 14:00", tz="UTC")

    db.log_predictions(_forecast(), origin, "model-a")
    time.sleep(1.1)                           # so the second run has a distinct issue time
    db.log_predictions(_forecast(), origin, "model-b")

    rows = db.read_predictions_vs_actual()
    assert len(rows) == 6
    assert rows["issued_at_utc"].nunique() == 2


def test_lead_time_is_measured_from_the_issue_time(temp_db):
    """hours_ahead counts from the data; lead_hours counts from the clock.

    They are not the same, because the data always lags the clock. Only
    lead_hours can support a claim that a forecast preceded its outcome.
    """
    origin = pd.Timestamp("2026-05-31 14:00", tz="UTC")
    db.log_predictions(_forecast(1), origin, "model-a")

    row = db.read_predictions_vs_actual().iloc[0]
    expected = (row["target_utc"] - row["made_at_utc"]).total_seconds() // 3600

    assert row["hours_ahead"] == expected
    assert row["lead_hours"] != row["hours_ahead"]


def test_legacy_rows_survive_migration_without_a_guessed_issue_time(temp_db):
    """Old rows keep a NULL issue time rather than a fabricated one."""
    conn = sqlite3.connect(temp_db)
    conn.execute("""CREATE TABLE predictions (
        target_utc TEXT NOT NULL, made_at_utc TEXT NOT NULL, hours_ahead INTEGER,
        predicted_mw REAL, model_version TEXT,
        PRIMARY KEY (target_utc, made_at_utc))""")
    conn.execute("INSERT INTO predictions VALUES "
                 "('2026-01-01T00:00:00Z','2025-12-31T14:00:00Z',10,5000.0,'old')")
    conn.commit()
    conn.close()

    rows = db.read_predictions_vs_actual()
    assert len(rows) == 1
    assert pd.isna(rows.iloc[0]["issued_at_utc"])
