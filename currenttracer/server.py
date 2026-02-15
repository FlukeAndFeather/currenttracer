"""FastAPI application serving the globe UI and /trace endpoint."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from currenttracer.advection import trace
from currenttracer.data import OceanCurrents

DATA_DIR = Path(os.environ.get("CURRENTTRACER_DATA", "data"))
CESIUM_TOKEN = os.environ.get("CESIUM_TOKEN", "")

# Rate limiting: 5 trace requests per minute per IP.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Current Tracer")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Rate limit exceeded. Try again shortly.", status_code=429)


# --- data loading (deferred until first request) ---

_currents: OceanCurrents | None = None


def get_currents() -> OceanCurrents:
    global _currents
    if _currents is None:
        _currents = OceanCurrents(DATA_DIR)
    return _currents


# --- routes ---


@app.get("/config")
async def config():
    """Serve client configuration and data time range."""
    currents = get_currents()
    return JSONResponse(content={
        "cesium_token": CESIUM_TOKEN,
        "max_hours": float(currents.time_hours[-1]),
    })

MAX_DURATION_HOURS = 365 * 24  # 1 year
MIN_DT_HOURS = 1.0


@app.get("/trace")
@limiter.limit("60/minute")
async def trace_endpoint(
    request: Request,
    lon: float = Query(..., ge=-180, le=180),
    lat: float = Query(..., ge=-90, le=90),
    duration_days: float = Query(default=30, gt=0, le=365),
    dt_hours: float = Query(default=6, ge=MIN_DT_HOURS),
    t0_hours: float = Query(default=0, ge=0),
):
    """Compute a particle trajectory from the given starting point."""
    currents = get_currents()
    duration_hours = duration_days * 24
    trajectory = trace(
        currents,
        lon=lon,
        lat=lat,
        duration_hours=min(duration_hours, MAX_DURATION_HOURS),
        dt_hours=max(dt_hours, MIN_DT_HOURS),
        t0_hours=t0_hours,
    )
    return JSONResponse(
        content={"trajectory": trajectory},
    )


# Serve static frontend files (index.html, style.css).
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main():
    """Entry point for ``currenttracer`` console script."""
    import uvicorn

    uvicorn.run(
        "currenttracer.server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
