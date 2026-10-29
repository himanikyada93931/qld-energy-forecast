"""The API is the public face of the project, so its contract is tested.

Most important: the service must not advertise an accuracy it has not
measured. /forecast once returned a hard-coded "2.3-2.7%" while live
forecasts were scoring 3.3% to 4.2%.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_lists_every_endpoint_it_serves():
    body = client.get("/").json()
    for path in ("/health", "/forecast?hours=N", "/dashboard", "/export.csv"):
        assert path in body["endpoints"]


def test_health_reports_both_model_windows():
    """A reader must be able to see what the model was scored on and shipped on."""
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model_trained_on" in body
    assert "model_evaluated_on" in body


def test_health_training_window_is_not_shorter_than_the_evaluation_window():
    """The shipped model must see at least everything the scorer saw."""
    body = client.get("/health").json()
    trained, evaluated = body["model_trained_on"], body["model_evaluated_on"]
    if isinstance(trained, dict) and isinstance(evaluated, dict):
        assert trained["rows"] > evaluated["rows"]
        assert trained["end"] >= evaluated["end"]


@pytest.mark.parametrize("hours", [0, 169, -1])
def test_forecast_rejects_horizons_it_cannot_serve(hours):
    assert client.get(f"/forecast?hours={hours}").status_code == 400


def test_forecast_reports_measured_accuracy_not_a_fixed_claim():
    body = client.get("/forecast?hours=6").json()
    assert "expected_mape" not in body
    assert body["accuracy"]["source"].startswith("measured")


def test_dashboard_carries_no_javascript():
    """Charts are server-rendered SVG; a script tag means something regressed."""
    html = client.get("/dashboard").text
    assert "<script" not in html


def test_export_csv_keeps_the_columns_the_power_bi_report_reads():
    """The export is a contract, not an internal detail.

    A Power BI query pins the columns it expects when it is first written.
    Adding two columns to this endpoint once pushed actual_mw and error_mw
    past the end of that pin, and the report silently served stale data for
    days instead of failing. Any change to these names has to be deliberate.
    """
    response = client.get("/export.csv")
    assert response.status_code == 200

    header = response.text.splitlines()[0].split(",")

    required = {"target_utc", "made_at_utc", "issued_at_utc", "hours_ahead",
                "lead_hours", "predicted_mw", "model_version", "actual_mw", "error_mw"}
    missing = required - set(header)

    assert not missing, f"export.csv no longer provides: {sorted(missing)}"
