"""Particle advection using RK4 integration on ocean current fields."""

from __future__ import annotations

import numpy as np

from currenttracer.data import LocalField, OceanCurrents, _ZeroField

# Metres per degree of latitude (approximate).
M_PER_DEG_LAT = 111_320.0

# Time window (hours) for each loaded chunk.
CHUNK_DT_HOURS = 10 * 24

# Reload when the tracer is within this fraction of the chunk edge.
EDGE_MARGIN = 0.2


def _velocity_deg_per_hour(
    field: LocalField, lon: float, lat: float, t_hours: float
) -> np.ndarray:
    """Convert m/s velocity to degrees/hour at the given position."""
    u, v = field.velocity_at(lon, lat, t_hours)
    cos_lat = np.cos(np.radians(lat))
    # Avoid division by zero near the poles.
    if cos_lat < 1e-6:
        dlon_dt = 0.0
    else:
        dlon_dt = (u / (M_PER_DEG_LAT * cos_lat)) * 3600.0
    dlat_dt = (v / M_PER_DEG_LAT) * 3600.0
    return np.array([dlon_dt, dlat_dt])


def _needs_reload(field, lon, lat, t_hours):
    """Check if the tracer is near the edge of the loaded field."""
    if isinstance(field, _ZeroField):
        return True

    lon_range = field.lon[-1] - field.lon[0]
    lat_range = field.lat[-1] - field.lat[0]
    t_range = field.time_hours[-1] - field.time_hours[0]

    lon_margin = lon_range * EDGE_MARGIN
    lat_margin = lat_range * EDGE_MARGIN
    t_margin = t_range * EDGE_MARGIN

    if lon < field.lon[0] + lon_margin or lon > field.lon[-1] - lon_margin:
        return True
    if lat < field.lat[0] + lat_margin or lat > field.lat[-1] - lat_margin:
        return True
    if t_hours > field.time_hours[-1] - t_margin:
        return True
    return False


def trace(
    currents: OceanCurrents,
    lon: float,
    lat: float,
    duration_hours: float,
    dt_hours: float = 1.0,
    t0_hours: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Trace a particle from (lon, lat) using RK4 integration.

    Loads a small spatial chunk and only reloads when the tracer
    approaches the edge of the loaded region or time window.

    Returns a list of (lon, lat, t_hours) tuples.
    """
    pos = np.array([lon, lat])
    t = t0_hours
    t_end = t0_hours + duration_hours
    trajectory: list[tuple[float, float, float]] = [(lon, lat, t)]

    field = None

    while t < t_end:
        # Load or reload chunk when needed.
        if field is None or _needs_reload(field, pos[0], pos[1], t):
            remaining = t_end - t
            chunk_dt = min(CHUNK_DT_HOURS, remaining)
            field = currents.local_field(pos[0], pos[1], t, chunk_dt)

        step = min(dt_hours, t_end - t)
        k1 = _velocity_deg_per_hour(field, pos[0], pos[1], t)
        k2 = _velocity_deg_per_hour(
            field,
            pos[0] + 0.5 * step * k1[0],
            pos[1] + 0.5 * step * k1[1],
            t + 0.5 * step,
        )
        k3 = _velocity_deg_per_hour(
            field,
            pos[0] + 0.5 * step * k2[0],
            pos[1] + 0.5 * step * k2[1],
            t + 0.5 * step,
        )
        k4 = _velocity_deg_per_hour(
            field,
            pos[0] + step * k3[0],
            pos[1] + step * k3[1],
            t + step,
        )
        pos = pos + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += step
        trajectory.append((float(pos[0]), float(pos[1]), float(t)))

    return trajectory
