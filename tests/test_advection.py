"""Tests for the advection module using synthetic current data."""

import numpy as np
import pytest

from currenttracer.advection import _velocity_deg_per_hour, trace
from currenttracer.data import OceanCurrents


class FakeCurrents:
    """A mock OceanCurrents that returns a constant velocity everywhere.

    Velocity is 0.1 m/s eastward, 0.0 m/s northward — a simple uniform
    zonal current that makes trajectory predictions easy to verify.
    """

    def velocity_at(self, lon: float, lat: float, t_hours: float):
        return 0.1, 0.0  # u=0.1 m/s east, v=0 m/s


class TestVelocityConversion:
    def test_equator(self):
        """At the equator, cos(lat)=1 so zonal conversion is straightforward."""
        currents = FakeCurrents()
        dpos = _velocity_deg_per_hour(currents, lon=0.0, lat=0.0, t_hours=0.0)
        # 0.1 m/s -> degrees/hour at equator
        expected_dlon = (0.1 / 111_320.0) * 3600.0
        assert dpos[0] == pytest.approx(expected_dlon, rel=1e-6)
        assert dpos[1] == pytest.approx(0.0)

    def test_high_latitude(self):
        """At 60°N, zonal displacement in degrees should be roughly doubled."""
        currents = FakeCurrents()
        dpos_eq = _velocity_deg_per_hour(currents, 0.0, 0.0, 0.0)
        dpos_60 = _velocity_deg_per_hour(currents, 0.0, 60.0, 0.0)
        ratio = dpos_60[0] / dpos_eq[0]
        # cos(60°) = 0.5, so ratio should be ~2
        assert ratio == pytest.approx(2.0, rel=1e-2)


class TestTrace:
    def test_stationary_in_zero_current(self):
        """A particle in zero current stays put."""

        class StillCurrents:
            def velocity_at(self, lon, lat, t_hours):
                return 0.0, 0.0

        traj = trace(StillCurrents(), lon=10.0, lat=20.0, duration_hours=24, dt_hours=1)
        for lon, lat, _ in traj:
            assert lon == pytest.approx(10.0)
            assert lat == pytest.approx(20.0)

    def test_eastward_drift(self):
        """Uniform eastward current should move the particle east."""
        traj = trace(FakeCurrents(), lon=0.0, lat=0.0, duration_hours=24, dt_hours=1)
        assert len(traj) == 25  # 24 steps + initial position
        final_lon = traj[-1][0]
        assert final_lon > 0.0  # moved east
        assert traj[-1][1] == pytest.approx(0.0, abs=1e-10)  # no north/south drift

    def test_trajectory_length(self):
        traj = trace(FakeCurrents(), lon=0.0, lat=0.0, duration_hours=48, dt_hours=6)
        assert len(traj) == 9  # 48/6 = 8 steps + initial


class TestOceanCurrentsInit:
    def test_missing_dir(self, tmp_path):
        """Should raise if no .nc files are found."""
        with pytest.raises(FileNotFoundError):
            OceanCurrents(tmp_path)
