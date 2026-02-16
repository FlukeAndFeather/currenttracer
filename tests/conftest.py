"""Generate a synthetic global Zarr dataset for testing."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

TEST_DATA_DIR = Path(__file__).parent / "test_data"
ZARR_PATH = TEST_DATA_DIR / "currents.zarr"

# 5 mph ≈ 2.235 m/s
MAX_SPEED_MS = 5 * 0.44704


@pytest.fixture(scope="session", autouse=True)
def synthetic_zarr():
    """Create a global synthetic Zarr store once per test session."""
    if ZARR_PATH.exists():
        shutil.rmtree(ZARR_PATH)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Coarser resolution (1°) to keep test data small.
    lon = np.arange(-180, 180, 1.0)
    lat = np.arange(-80, 91, 1.0)
    time = np.arange(
        np.datetime64("2024-01-01"),
        np.datetime64("2025-01-01"),
        np.timedelta64(1, "D"),
    )

    rng = np.random.default_rng(42)
    shape = (len(time), len(lat), len(lon))

    # Random velocities up to MAX_SPEED_MS.
    uo = (rng.random(shape).astype(np.float32) * 2 - 1) * MAX_SPEED_MS
    vo = (rng.random(shape).astype(np.float32) * 2 - 1) * MAX_SPEED_MS

    ds = xr.Dataset(
        {
            "uo": (["time", "latitude", "longitude"], uo),
            "vo": (["time", "latitude", "longitude"], vo),
        },
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon,
        },
    )

    ds = ds.chunk({"time": 5, "latitude": 30, "longitude": 30})
    ds.to_zarr(ZARR_PATH, mode="w")

    yield ZARR_PATH

    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
