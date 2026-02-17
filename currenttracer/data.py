"""Load coarsened ocean current data and build velocity interpolators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


class OceanCurrents:
    """Ocean current data loaded entirely into memory at startup.

    Expects a small, coarsened Zarr store (e.g. 1/2°, 60 days, ~188 MB).
    Builds interpolators once; velocity lookups are instant.
    """

    def __init__(self, data_dir: str | Path) -> None:
        zarr_path = Path(data_dir) / "currents_coarse.zarr"
        ds = xr.open_zarr(zarr_path, consolidated=False)

        self.lon = ds["longitude"].values.astype(np.float64)
        self.lat = ds["latitude"].values.astype(np.float64)

        self.time_ref = ds["time"].values[0]
        self.time_hours = (
            (ds["time"].values - self.time_ref) / np.timedelta64(1, "h")
        ).astype(np.float64)

        uo = np.nan_to_num(ds["uo"].values.astype(np.float64))
        vo = np.nan_to_num(ds["vo"].values.astype(np.float64))
        ds.close()

        self._interp_uo = RegularGridInterpolator(
            (self.time_hours, self.lat, self.lon),
            uo,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self._interp_vo = RegularGridInterpolator(
            (self.time_hours, self.lat, self.lon),
            vo,
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
