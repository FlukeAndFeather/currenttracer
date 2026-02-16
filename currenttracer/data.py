"""Load Copernicus ocean current NetCDF data and build velocity interpolators."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Maximum assumed current speed in km/h (5 mph ≈ 8.05 km/h).
MAX_SPEED_KMH = 5 * 1.609344

# Approximate km per degree of latitude.
KM_PER_DEG_LAT = 111.32


def _parse_bounds(filename: str) -> tuple[float, float, float, float] | None:
    """Extract (lon_min, lon_max, lat_min, lat_max) from a Copernicus filename.

    Example filename fragment: 80.00W-0.00E_20.00N-60.00N
    """
    def _parse_lon(s: str) -> float:
        if s.endswith("W"):
            return -float(s[:-1])
        return float(s[:-1])

    def _parse_lat(s: str) -> float:
        if s.endswith("S"):
            return -float(s[:-1])
        return float(s[:-1])

    # Match lon range and lat range patterns in the filename.
    m = re.search(
        r"(\d+\.\d+[WE])-(\d+\.\d+[WE])_(\d+\.\d+[NS])-(\d+\.\d+[NS])",
        filename,
    )
    if not m:
        return None
    return (
        _parse_lon(m.group(1)),
        _parse_lon(m.group(2)),
        _parse_lat(m.group(3)),
        _parse_lat(m.group(4)),
    )


class _FileEntry:
    """Index entry for one NetCDF file."""

    __slots__ = ("path", "lon_min", "lon_max", "lat_min", "lat_max")

    def __init__(
        self, path: Path,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
    ) -> None:
        self.path = path
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max

    def overlaps(
        self,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
    ) -> bool:
        return (
            self.lon_min <= lon_max and self.lon_max >= lon_min
            and self.lat_min <= lat_max and self.lat_max >= lat_min
        )


class OceanCurrents:
    """Lazy-loading ocean current data.

    Indexes NetCDF files by spatial bounds from their filenames.
    Only opens the files needed for each trace request.
    """

    def __init__(self, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        files = sorted(data_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No .nc files found in {data_dir}")

        self._index: list[_FileEntry] = []
        for f in files:
            bounds = _parse_bounds(f.name)
            if bounds is not None:
                self._index.append(_FileEntry(f, *bounds))

        if not self._index:
            raise FileNotFoundError(
                f"No .nc files with parseable bounds in {data_dir}"
            )

        # Read time coordinates from the first file (all tiles share
        # the same time range since data was downloaded with no time chunking).
        with xr.open_dataset(self._index[0].path) as ds:
            self.time_ref = ds["time"].values[0]
            self.time_hours = (
                (ds["time"].values - self.time_ref) / np.timedelta64(1, "h")
            ).astype(np.float64)

    def _find_files(
        self,
        lon_min: float, lon_max: float,
        lat_min: float, lat_max: float,
    ) -> list[Path]:
        """Return paths of files overlapping the given bounding box."""
        return [
            e.path for e in self._index
            if e.overlaps(lon_min, lon_max, lat_min, lat_max)
        ]

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

        paths = self._find_files(lon_min, lon_max, lat_min, lat_max)
        if not paths:
            raise ValueError(
                f"No data files cover region around ({lon}, {lat})"
            )

        # Open only the matching files, select the spatial/temporal subset.
        ds = xr.open_mfdataset(paths, combine="by_coords")
        if "depth" in ds.dims:
            ds = ds.isel(depth=0)

        # Spatial subset.
        ds = ds.sel(
            longitude=slice(lon_min, lon_max),
            latitude=slice(lat_min, lat_max),
        )

        # Temporal subset.
        t_start = self.time_ref + np.timedelta64(int(t0_hours * 3600), "s")
        t_end = self.time_ref + np.timedelta64(int((t0_hours + dt_hours) * 3600), "s")
        ds = ds.sel(time=slice(t_start, t_end))

        sub_lon = ds["longitude"].values.astype(np.float64)
        sub_lat = ds["latitude"].values.astype(np.float64)
        sub_time = (
            (ds["time"].values - self.time_ref) / np.timedelta64(1, "h")
        ).astype(np.float64)

        uo = np.nan_to_num(ds["uo"].values.astype(np.float64))
        vo = np.nan_to_num(ds["vo"].values.astype(np.float64))
        ds.close()

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
