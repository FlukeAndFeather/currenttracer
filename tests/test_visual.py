"""Visual tests that produce MP4 animations of particle trajectories.

Run with:
    pytest tests/test_visual.py -s

Outputs are saved to tests/output/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pytest

from currenttracer.advection import trace

OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture(autouse=True)
def _ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def _draw_current_field(ax, currents, xlim, ylim, grid_n=16):
    """Draw a quiver plot of the current field as semi-transparent arrows.

    Returns the quiver artist so it can be included in blit updates if needed.
    """
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(grid_n):
        for j in range(grid_n):
            u, v = currents.velocity_at(X[i, j], Y[i, j], 0.0)
            U[i, j] = u
            V[i, j] = v
    ax.quiver(
        X, Y, U, V,
        color="white", alpha=0.15, scale=5, width=0.004,
        headwidth=4, headlength=5, zorder=1,
    )


def _compute_limits(lons, lats, xlim=None, ylim=None):
    """Compute axis limits from trajectory or explicit bounds."""
    if xlim is None:
        pad = max((max(lons) - min(lons)) * 0.15, 1.0)
        xlim = (min(lons) - pad, max(lons) + pad)
    if ylim is None:
        pad = max((max(lats) - min(lats)) * 0.15, 1.0)
        ylim = (min(lats) - pad, max(lats) + pad)
    return xlim, ylim


def _setup_axes(fig, ax, title, xlim, ylim):
    """Apply common dark styling to axes."""
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_xlabel("Longitude", color="#aaa")
    ax.set_ylabel("Latitude", color="#aaa")
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title, color="#eee", fontsize=13)


def _make_mp4(
    trajectory: list[tuple[float, float, float]],
    currents,
    filename: str,
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    fps: int = 30,
    trail_length: int = 40,
):
    """Animate a trajectory and save as MP4.

    Shows a glowing head with a fading trail over a quiver plot of the
    current field.
    """
    lons = [p[0] for p in trajectory]
    lats = [p[1] for p in trajectory]
    hours = [p[2] for p in trajectory]

    xlim, ylim = _compute_limits(lons, lats, xlim, ylim)

    fig, ax = plt.subplots(figsize=(8, 5))
    _setup_axes(fig, ax, title, xlim, ylim)

    # Current field arrows (static background).
    _draw_current_field(ax, currents, xlim, ylim)

    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, color="#aaa", fontsize=10,
        verticalalignment="top",
    )

    # Start marker.
    ax.plot(lons[0], lats[0], "o", color="#555", markersize=6, zorder=2)

    # Trail segments and head dot (will be updated each frame).
    (trail_line,) = ax.plot([], [], "-", color="cyan", alpha=0.4, linewidth=2, zorder=3)
    (head_dot,) = ax.plot([], [], "o", color="cyan", markersize=8, zorder=5)
    (glow_dot,) = ax.plot([], [], "o", color="cyan", alpha=0.25, markersize=16, zorder=4)

    def update(frame):
        start = max(0, frame - trail_length)
        trail_line.set_data(lons[start : frame + 1], lats[start : frame + 1])
        head_dot.set_data([lons[frame]], [lats[frame]])
        glow_dot.set_data([lons[frame]], [lats[frame]])
        days = hours[frame] / 24.0
        time_text.set_text(f"Day {days:.1f}")
        return trail_line, head_dot, glow_dot, time_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(lons), interval=1000 // fps, blit=True,
    )
    out_path = OUTPUT_DIR / filename
    anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Mock current fields
# ---------------------------------------------------------------------------

class UniformEast:
    """Constant 0.5 m/s eastward current."""
    def velocity_at(self, lon, lat, t_hours):
        return 0.5, 0.0


class CircularGyre:
    """A simple clockwise gyre centred at (0, 0).

    Velocity is tangential to circles around the origin, producing a
    circular trajectory that's easy to verify visually.
    """
    def velocity_at(self, lon, lat, t_hours):
        r = np.sqrt(lon**2 + lat**2)
        if r < 1e-8:
            return 0.0, 0.0
        speed = 0.3  # m/s
        # tangent to the circle, clockwise
        u = speed * (-lat / r)
        v = speed * (lon / r)
        return u, v


class MeanderingJet:
    """A jet that flows east with a sinusoidal north-south wobble.

    Mimics a simplified Gulf-Stream-like current.
    """
    def velocity_at(self, lon, lat, t_hours):
        u = 0.8
        v = 0.3 * np.sin(np.radians(lon) * 4)
        return u, v


# ---------------------------------------------------------------------------
# Tests — each one produces an MP4
# ---------------------------------------------------------------------------

class TestVisual:
    def test_uniform_eastward(self):
        """Particle should drift in a straight line to the east."""
        currents = UniformEast()
        traj = trace(currents, lon=0, lat=0, duration_hours=24 * 30, dt_hours=3)
        _make_mp4(
            traj, currents, "uniform_east.mp4",
            title="Uniform eastward current (0.5 m/s)",
        )

    def test_circular_gyre(self):
        """Particle should orbit in an approximately circular path."""
        currents = CircularGyre()
        traj = trace(currents, lon=5, lat=0, duration_hours=24 * 60, dt_hours=3)
        _make_mp4(
            traj, currents, "circular_gyre.mp4",
            title="Clockwise circular gyre",
            xlim=(-8, 8), ylim=(-8, 8),
        )

    def test_meandering_jet(self):
        """Particle should follow a sinusoidal eastward path."""
        currents = MeanderingJet()
        traj = trace(currents, lon=-20, lat=0, duration_hours=24 * 30, dt_hours=3)
        _make_mp4(
            traj, currents, "meandering_jet.mp4",
            title="Meandering jet (Gulf Stream-like)",
        )

    def test_multiple_start_points(self):
        """Several particles released in the gyre — should all orbit."""
        gyre = CircularGyre()
        xlim, ylim = (-10, 10), (-10, 10)

        fig, ax = plt.subplots(figsize=(8, 8))
        _setup_axes(fig, ax, "Multiple particles in a gyre", xlim, ylim)
        _draw_current_field(ax, gyre, xlim, ylim)

        colors = ["cyan", "#ff6b6b", "#ffd93d", "#6bff6b"]
        starts = [(5, 0), (0, 5), (-5, 0), (0, -5)]
        trajs = [
            trace(gyre, lon=s[0], lat=s[1], duration_hours=24 * 60, dt_hours=3)
            for s in starts
        ]

        lines = []
        heads = []
        for c in colors:
            (line,) = ax.plot([], [], "-", color=c, alpha=0.4, linewidth=2)
            (head,) = ax.plot([], [], "o", color=c, markersize=7)
            lines.append(line)
            heads.append(head)

        trail_length = 40

        def update(frame):
            for i, traj in enumerate(trajs):
                start = max(0, frame - trail_length)
                xs = [p[0] for p in traj[start : frame + 1]]
                ys = [p[1] for p in traj[start : frame + 1]]
                lines[i].set_data(xs, ys)
                heads[i].set_data([traj[frame][0]], [traj[frame][1]])
            return lines + heads

        n_frames = min(len(t) for t in trajs)
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames, interval=33, blit=True,
        )
        out_path = OUTPUT_DIR / "multi_gyre.mp4"
        anim.save(str(out_path), writer="ffmpeg", fps=30, dpi=100)
        plt.close(fig)
        print(f"  Saved: {out_path}")
