"""Particle advection using RK4 integration on ocean current fields."""

from __future__ import annotations

import numpy as np

from currenttracer.data import OceanCurrents

# Metres per degree of latitude (approximate).
M_PER_DEG_LAT = 111_320.0


def _velocity_deg_per_hour(
    currents: OceanCurrents, lon: float, lat: float, t_hours: float
) -> np.ndarray:
    """Convert m/s velocity to degrees/hour at the given position."""
    u, v = currents.velocity_at(lon, lat, t_hours)
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

    Returns a list of (lon, lat, t_hours) tuples representing the
    trajectory.
    """
    pos = np.array([lon, lat])
    t = t0_hours
    n_steps = int(duration_hours / dt_hours)
    trajectory: list[tuple[float, float, float]] = [(lon, lat, t)]

    for _ in range(n_steps):
        k1 = _velocity_deg_per_hour(currents, pos[0], pos[1], t)
        k2 = _velocity_deg_per_hour(
            currents,
            pos[0] + 0.5 * dt_hours * k1[0],
            pos[1] + 0.5 * dt_hours * k1[1],
            t + 0.5 * dt_hours,
        )
        k3 = _velocity_deg_per_hour(
            currents,
            pos[0] + 0.5 * dt_hours * k2[0],
            pos[1] + 0.5 * dt_hours * k2[1],
            t + 0.5 * dt_hours,
        )
        k4 = _velocity_deg_per_hour(
            currents,
            pos[0] + dt_hours * k3[0],
            pos[1] + dt_hours * k3[1],
            t + dt_hours,
        )
        pos = pos + (dt_hours / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt_hours
        trajectory.append((float(pos[0]), float(pos[1]), float(t)))

    return trajectory
