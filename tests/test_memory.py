"""Drop 20 tracers across the world's oceans and verify memory stays under 1 GB."""

import tracemalloc

import pytest

from currenttracer.advection import trace
from currenttracer.data import OceanCurrents

from .conftest import TEST_DATA_DIR

# 20 ocean points spread across major basins.
OCEAN_POINTS = [
    # North Atlantic
    (-40, 40),
    (-30, 55),
    (-60, 25),
    # South Atlantic
    (-20, -15),
    (-30, -35),
    # North Pacific
    (-170, 35),
    (-140, 50),
    (160, 30),
    # South Pacific
    (-120, -25),
    (-160, -40),
    (170, -15),
    # Indian Ocean
    (70, -10),
    (80, -30),
    (55, 5),
    # Southern Ocean
    (-60, -55),
    (30, -50),
    (140, -55),
    # Arctic / sub-Arctic
    (-10, 65),
    (20, 70),
    # Mediterranean
    (18, 35),
]

DURATION_DAYS = 365
DT_HOURS = 6
MEMORY_LIMIT_MB = 1024


@pytest.fixture(scope="module")
def currents(synthetic_zarr):
    return OceanCurrents(TEST_DATA_DIR)


def test_20_tracers_under_1gb(currents):
    tracemalloc.start()

    peak_mb = 0
    for i, (lon, lat) in enumerate(OCEAN_POINTS):
        traj = trace(
            currents,
            lon=lon,
            lat=lat,
            duration_hours=DURATION_DAYS * 24,
            dt_hours=DT_HOURS,
            t0_hours=0,
        )
        _, current_peak = tracemalloc.get_traced_memory()
        peak_mb = current_peak / (1024 * 1024)
        print(
            f"[{i + 1}/20] ({lon:>5}, {lat:>3}) -> "
            f"{len(traj)} pts, peak {peak_mb:.0f} MB"
        )
        assert len(traj) > 1, f"Tracer at ({lon}, {lat}) produced empty trajectory"

    tracemalloc.stop()
    print(f"\nFinal peak memory: {peak_mb:.0f} MB")
    assert peak_mb < MEMORY_LIMIT_MB, (
        f"Peak memory {peak_mb:.0f} MB exceeds {MEMORY_LIMIT_MB} MB limit"
    )
