"""Load Copernicus ocean current NetCDF data and build velocity interpolators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Maximum assumed current speed in km/h (5 mph ≈ 8.05 km/h).
MAX_SPEED_KMH = 5 * 1.609344

# Approximate km per degree of latitude.
KM_PER_DEG_LAT = 111.32


class OceanCurrents:
    """Lazy-loading ocean current data.

    Opens NetCDF files without loading velocity arrays into memory.
    Call :meth:`local_field` to load a small spatial/temporal subset
    for a specific trace request.
    """

    def __init__(self, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No .nc files found in {data_dir}")

        self._ds = xr.open_mfdataset(files, combine="by_coords")

        if "depth" in self._ds.dims:
            self._ds = self._ds.isel(depth=0)

        # Coordinate arrays are small — safe to load into memory.
        self.lon = self._ds["longitude"].values.astype(np.float64)
        self.lat = self._ds["latitude"].values.astype(np.float64)

        self.time_ref = self._ds["time"].values[0]
        self.time_hours = (
            (self._ds["time"].values - self.time_ref) / np.timedelta64(1, "h")
        ).astype(np.float64)

    def local_field(
        self,
        lon: float,
        lat: float,
        t0_hours: float,
        dt_hours: float,
    ) -> LocalField:
        """Load a chunk around (lon, lat) for a time window of dt_hours.

        The spatial extent (dx, dy) is computed from the maximum distance
        a tracer could drift at MAX_SPEED_KMH over dt_hours.
        """
        max_dist_km = MAX_SPEED_KMH * dt_hours
        dy = max_dist_km / KM_PER_DEG_LAT
        cos_lat = max(np.cos(np.radians(lat)), 1e-6)
        dx = max_dist_km / (KM_PER_DEG_LAT * cos_lat)

        lon_min = lon - dx
        lon_max = lon + dx
        lat_min = max(lat - dy, float(self.lat.min()))
        lat_max = min(lat + dy, float(self.lat.max()))
        t_end = t0_hours + dt_hours

        lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
        lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
        time_mask = (self.time_hours >= t0_hours) & (self.time_hours <= t_end)

        lon_idx = np.where(lon_mask)[0]
        lat_idx = np.where(lat_mask)[0]
        time_idx = np.where(time_mask)[0]

        # Expand by 1 on each side for interpolation edges.
        if len(lon_idx) > 0:
            lon_idx = np.arange(
                max(lon_idx[0] - 1, 0),
                min(lon_idx[-1] + 2, len(self.lon)),
            )
        if len(lat_idx) > 0:
            lat_idx = np.arange(
                max(lat_idx[0] - 1, 0),
                min(lat_idx[-1] + 2, len(self.lat)),
            )
        if len(time_idx) > 0:
            time_idx = np.arange(
                max(time_idx[0] - 1, 0),
                min(time_idx[-1] + 2, len(self.time_hours)),
            )

        sub = self._ds.isel(
            longitude=lon_idx, latitude=lat_idx, time=time_idx
        )

        sub_lon = sub["longitude"].values.astype(np.float64)
        sub_lat = sub["latitude"].values.astype(np.float64)
        sub_time = (
            (sub["time"].values - self.time_ref) / np.timedelta64(1, "h")
        ).astype(np.float64)

        uo = np.nan_to_num(sub["uo"].values.astype(np.float64))
        vo = np.nan_to_num(sub["vo"].values.astype(np.float64))

        return LocalField(sub_lon, sub_lat, sub_time, uo, vo)


class LocalField:
    """Interpolators for a spatial/temporal subset of ocean currents."""

    def __init__(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        time_hours: np.ndarray,
        uo: np.ndarray,
        vo: np.ndarray,
    ) -> None:
        self.lon = lon
        self.lat = lat
        self.time_hours = time_hours
        self._interp_uo = RegularGridInterpolator(
            (time_hours, lat, lon),
            uo,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self._interp_vo = RegularGridInterpolator(
            (time_hours, lat, lon),
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
