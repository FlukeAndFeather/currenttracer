"""Load ocean current data from a Zarr store and build velocity interpolators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Maximum assumed current speed in km/h (1.5 mph ≈ 2.41 km/h).
# Conservative estimate; advection reloads if tracer nears chunk edge.
MAX_SPEED_KMH = 1.5 * 1.609344

# Approximate km per degree of latitude.
KM_PER_DEG_LAT = 111.32


class OceanCurrents:
    """Lazy-loading ocean current data backed by a Zarr store.

    Opening the store is near-instant and uses minimal memory.
    Call :meth:`local_field` to load a small spatial/temporal subset.
    """

    def __init__(self, data_dir: str | Path) -> None:
        zarr_path = Path(data_dir) / "currents.zarr"
        self._ds = xr.open_zarr(zarr_path, consolidated=False)

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
        lat_min = lat - dy
        lat_max = lat + dy

        t_start = self.time_ref + np.timedelta64(int(t0_hours * 3600), "s")
        t_end = self.time_ref + np.timedelta64(int((t0_hours + dt_hours) * 3600), "s")

        sub = self._ds.sel(
            longitude=slice(lon_min, lon_max),
            latitude=slice(lat_min, lat_max),
            time=slice(t_start, t_end),
        )

        # If the subset is empty on any dimension (e.g. time beyond data
        # range, or location outside spatial coverage), return a zero field.
        if (
            sub.sizes["time"] < 2
            or sub.sizes["latitude"] < 2
            or sub.sizes["longitude"] < 2
        ):
            return _ZERO_FIELD

        sub_lon = sub["longitude"].values.astype(np.float64)
        sub_lat = sub["latitude"].values.astype(np.float64)
        sub_time = (
            (sub["time"].values - self.time_ref) / np.timedelta64(1, "h")
        ).astype(np.float64)

        uo = np.nan_to_num(sub["uo"].values.astype(np.float64))
        vo = np.nan_to_num(sub["vo"].values.astype(np.float64))

        return LocalField(sub_lon, sub_lat, sub_time, uo, vo)


class _ZeroField:
    """Sentinel field that always returns zero velocity."""

    def velocity_at(
        self, lon: float, lat: float, t_hours: float
    ) -> tuple[float, float]:
        return 0.0, 0.0


_ZERO_FIELD = _ZeroField()


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
