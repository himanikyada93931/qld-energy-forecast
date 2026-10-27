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
