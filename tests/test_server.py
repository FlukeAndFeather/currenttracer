"""Tests for the FastAPI server using a mocked OceanCurrents."""

import pytest
from fastapi.testclient import TestClient

import currenttracer.server as server_module
from currenttracer.server import app


class FakeCurrents:
    def velocity_at(self, lon, lat, t_hours):
        return 0.1, 0.05


@pytest.fixture(autouse=True)
def _patch_currents(monkeypatch):
    """Replace get_currents so tests don't need real NetCDF data."""
    monkeypatch.setattr(server_module, "get_currents", lambda: FakeCurrents())


@pytest.fixture()
def client():
    return TestClient(app)


class TestTraceEndpoint:
    def test_basic_request(self, client):
        resp = client.get("/trace", params={"lon": 0, "lat": 0, "duration_days": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert "trajectory" in body
        traj = body["trajectory"]
        assert len(traj) > 1
        # Each point is [lon, lat, t_hours]
        assert len(traj[0]) == 3

    def test_invalid_lon(self, client):
        resp = client.get("/trace", params={"lon": 999, "lat": 0})
        assert resp.status_code == 422

    def test_missing_params(self, client):
        resp = client.get("/trace")
        assert resp.status_code == 422

    def test_duration_capped(self, client):
        resp = client.get(
            "/trace", params={"lon": 0, "lat": 0, "duration_days": 365, "dt_hours": 6}
        )
        assert resp.status_code == 200
