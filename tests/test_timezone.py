"""Time zones are the quiet killer in this project.

AEMO publishes in Queensland time with no daylight saving, Open-Meteo returns
UTC, and the dashboard shows Brisbane. A conversion that is silently wrong
shifts the midday solar dip to the early hours and nothing raises an error.
"""

import pandas as pd


def test_aemo_local_time_is_utc_plus_ten():
    """Etc/GMT-10 is UTC+10, despite the sign. Getting this backwards is a 20-hour error."""
    local = pd.Timestamp("2026-07-01 10:00:00").tz_localize("Etc/GMT-10")
    assert local.tz_convert("UTC") == pd.Timestamp("2026-07-01 00:00:00", tz="UTC")


def test_queensland_has_no_daylight_saving():
    """Summer and winter must both be UTC+10, or half the year is an hour out."""
    for month in ("01", "07"):
        offset = (pd.Timestamp(f"2026-{month}-15 12:00", tz="Australia/Brisbane")
                  .utcoffset().total_seconds() / 3600)
        assert offset == 10


def test_brisbane_display_matches_the_stored_utc_hour():
    """Midnight UTC is 10am Brisbane: the hour the duck curve dips."""
    utc = pd.Timestamp("2026-10-01 00:00", tz="UTC")
    assert utc.tz_convert("Australia/Brisbane").hour == 10


def test_stored_timestamps_are_timezone_aware():
    """A naive timestamp compares wrongly against an aware one, or raises."""
    from src.ingest.database import read_all
    df = read_all()
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"
