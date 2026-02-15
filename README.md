# Current Tracer

Interactive web app for tracing ocean surface current trajectories on a 3D globe. Click anywhere on the ocean to drop a particle and watch it drift with the currents over time.

Built with [CesiumJS](https://cesium.com/cesiumjs/) for the globe, [FastAPI](https://fastapi.tiangolo.com/) for the backend, and [Copernicus Marine](https://marine.copernicus.eu/) ocean current data.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate currenttracer
pip install -e . --no-deps
```

### 2. Download ocean current data

Requires a free [Copernicus Marine](https://marine.copernicus.eu/) account.

```bash
# One month, North Atlantic (good for testing)
python download_data.py --start 2024-06-01 --end 2024-06-30 \
    --min-lon -80 --max-lon 0 --min-lat 20 --max-lat 60

# Full year, global
python download_data.py --start 2024-01-01 --end 2024-12-31
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
CESIUM_TOKEN=your-cesium-ion-token
CURRENTTRACER_DATA=data
```

Get a free Cesium Ion token at https://ion.cesium.com/.

### 4. Run the server

```bash
./run.sh
```

Then open http://localhost:8000.

## Usage

- Click the ocean to drop a tracer particle
- Multiple tracers share a global timeline
- Use the speed slider to control animation speed
- Click Reset to clear all tracers

## Project structure

```
currenttracer/
    __init__.py
    data.py            # NetCDF loading and velocity interpolation
    advection.py       # RK4 particle tracer
    server.py          # FastAPI app and /trace endpoint
    static/
        index.html     # CesiumJS globe and UI
        style.css
data/                  # Downloaded NetCDF files (not tracked)
tests/
    test_advection.py  # Unit tests with mock currents
    test_server.py     # API endpoint tests
    test_visual.py     # MP4 animation tests
download_data.py       # Copernicus Marine data download script
environment.yml        # Conda environment
```

---

Built with [Claude Code](https://claude.ai/claude-code).
