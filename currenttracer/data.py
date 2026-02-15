"""Load Copernicus ocean current NetCDF data and build velocity interpolators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


class OceanCurrents:
    """Provides interpolated surface current velocities from NetCDF data.

    Expects NetCDF files with variables ``uo`` (eastward) and ``vo``
    (northward) on a regular lon/lat grid, as produced by the Copernicus
    Global Ocean Physics Analysis and Forecast product.
    """

    def __init__(self, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No .nc files found in {data_dir}")

        ds = xr.open_mfdataset(files, combine="by_coords")

        # Surface layer only (depth index 0) if depth dimension exists.
        if "depth" in ds.dims:
            ds = ds.isel(depth=0)

        self.lon = ds["longitude"].values.astype(np.float64)
        self.lat = ds["latitude"].values.astype(np.float64)

        # Time as fractional hours since first timestep.
        self.time_ref = ds["time"].values[0]
        time_hours = (
            (ds["time"].values - self.time_ref) / np.timedelta64(1, "h")
        ).astype(np.float64)
        self.time_hours = time_hours

        # Velocity arrays: (time, lat, lon).  Fill NaN (land) with 0.
        self.uo = np.nan_to_num(ds["uo"].values.astype(np.float64))
        self.vo = np.nan_to_num(ds["vo"].values.astype(np.float64))

        ds.close()

        self._interp_uo = RegularGridInterpolator(
            (time_hours, self.lat, self.lon),
            self.uo,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self._interp_vo = RegularGridInterpolator(
            (time_hours, self.lat, self.lon),
            self.vo,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def velocity_at(
        self, lon: float, lat: float, t_hours: float
    ) -> tuple[float, float]:
        """Return (u, v) in m/s at the given position and time."""
        pt = np.array([[t_hours, lat, lon]])
        u = float(self._interp_uo(pt)[0])
        v = float(self._interp_vo(pt)[0])
        return u, v
