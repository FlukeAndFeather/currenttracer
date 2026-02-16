"""Particle advection using RK4 integration on ocean current fields."""

from __future__ import annotations

import numpy as np

from currenttracer.data import LocalField, OceanCurrents

# Metres per degree of latitude (approximate).
M_PER_DEG_LAT = 111_320.0

# Time window (hours) for each loaded chunk. 5 days.
CHUNK_DT_HOURS = 5 * 24


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


def trace(
    currents: OceanCurrents,
    lon: float,
    lat: float,
    duration_hours: float,
    dt_hours: float = 1.0,
    t0_hours: float = 0.0,
) -> list[tuple[float, float, float]]:
    """Trace a particle from (lon, lat) using RK4 integration.

    Loads data in time windows of CHUNK_DT_HOURS, sized spatially by
    the maximum drift distance at assumed max current speed.

    Returns a list of (lon, lat, t_hours) tuples.
    """
    pos = np.array([lon, lat])
    t = t0_hours
    t_end = t0_hours + duration_hours
    trajectory: list[tuple[float, float, float]] = [(lon, lat, t)]

    while t < t_end:
        # Load a chunk centered on current position for the next window.
        chunk_end = min(t + CHUNK_DT_HOURS, t_end)
        chunk_dt = chunk_end - t
        field = currents.local_field(pos[0], pos[1], t, chunk_dt)

        # Step through this chunk with RK4.
        while t < chunk_end:
            step = min(dt_hours, chunk_end - t)
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
