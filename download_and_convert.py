"""Download ocean current data and build a Zarr store, one chunk at a time.

For each spatial tile:
  1. Download from Copernicus into memory (no intermediate files)
  2. Write directly into the Zarr store
  3. Move on to the next tile

    python download_and_convert.py [--output data/currents.zarr]
"""

import argparse
import json
from pathlib import Path

import copernicusmarine
import numpy as np
import xarray as xr
import zarr

# Zarr chunk sizes matching data.py expectations.
TIME_CHUNK = 5
LAT_CHUNK = 120
LON_CHUNK = 120

# Download tile size (degrees).
TILE_LON = 30
TILE_LAT = 30

# Known grid: 1/12 degree, -180 to 180 lon, -80 to 90 lat, daily 2024.
RESOLUTION = 1 / 12
FULL_LON = np.arange(-180, 180, RESOLUTION)
FULL_LAT = np.arange(-80, 90 + RESOLUTION / 2, RESOLUTION)
FULL_TIME = np.arange(
    np.datetime64("2024-01-01"),
    np.datetime64("2025-01-01"),
    np.timedelta64(1, "D"),
)

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


def _tile_edges(lo, hi, step):
    edges = []
    val = lo
    while val < hi:
        edges.append(val)
        val += step
    edges.append(hi)
    return edges


def _download_tile(lo0, lo1, la0, la1):
    """Download a tile and return it as an xarray Dataset in memory."""
    return copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        variables=["uo", "vo"],
        minimum_longitude=lo0,
        maximum_longitude=lo1,
        minimum_latitude=la0,
        maximum_latitude=la1,
        minimum_depth=0,
        maximum_depth=1,
        start_datetime="2024-01-01T00:00:00",
        end_datetime="2024-12-31T23:59:59",
    )


def _write_zarr_array(store_path, name, data, chunks, fill_value=0.0, attrs=None):
    """Write a zarr v2 array using filesystem operations."""
    arr_dir = store_path / name
    arr_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "zarr_format": 2,
        "shape": list(data.shape),
        "chunks": list(chunks),
        "dtype": data.dtype.str,
        "compressor": None,
        "fill_value": None if np.isnan(fill_value) else fill_value,
        "order": "C",
        "filters": None,
    }
    (arr_dir / ".zarray").write_text(json.dumps(meta))
    if attrs:
        (arr_dir / ".zattrs").write_text(json.dumps(attrs))

    if data.ndim == 1:
        for ci in range(0, data.shape[0], chunks[0]):
            chunk_data = data[ci:ci + chunks[0]]
            (arr_dir / str(ci // chunks[0])).write_bytes(chunk_data.tobytes())


def _create_empty_zarr_array(store_path, name, shape, chunks, dtype,
                              fill_value, attrs=None):
    """Create an empty zarr v2 array on disk (metadata only)."""
    arr_dir = store_path / name
    arr_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "zarr_format": 2,
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": dtype,
        "compressor": {
            "id": "zlib",
            "level": 4,
        },
        "fill_value": fill_value,
        "order": "C",
        "filters": None,
    }
    (arr_dir / ".zarray").write_text(json.dumps(meta))
    if attrs:
        (arr_dir / ".zattrs").write_text(json.dumps(attrs))


def _load_progress(path):
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_progress(path, completed):
    path.write_text(json.dumps(sorted(completed)))


def _nearest_idx(arr, val):
    """Index of nearest value in a sorted array."""
    return int(np.searchsorted(arr, val, side="left"))


def _create_store(output):
    """Create the empty Zarr store with known grid dimensions."""
    nt, nlat, nlon = len(FULL_TIME), len(FULL_LAT), len(FULL_LON)
    print(f"Grid: time={nt}, lat={nlat}, lon={nlon}")

    output.mkdir(parents=True, exist_ok=True)
    (output / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

    dims = lambda d: {"_ARRAY_DIMENSIONS": d}

    _write_zarr_array(output, "time",
                      FULL_TIME.astype("datetime64[ns]"),
                      chunks=(nt,), fill_value=0, attrs=dims(["time"]))
    _write_zarr_array(output, "latitude", FULL_LAT.astype(np.float64),
                      chunks=(nlat,), fill_value=float("nan"),
                      attrs=dims(["latitude"]))
    _write_zarr_array(output, "longitude", FULL_LON.astype(np.float64),
                      chunks=(nlon,), fill_value=float("nan"),
                      attrs=dims(["longitude"]))

    for var in ("uo", "vo"):
        _create_empty_zarr_array(
            output, var,
            shape=(nt, nlat, nlon),
            chunks=(TIME_CHUNK, LAT_CHUNK, LON_CHUNK),
            dtype="<f4", fill_value=0.0,
            attrs=dims(["time", "latitude", "longitude"]),
        )

    print("Zarr store created.")


def _write_tile(store, ds):
    """Write an in-memory xarray Dataset tile into the opened Zarr store."""
    if "depth" in ds.dims:
        ds = ds.isel(depth=0).drop_vars("depth", errors="ignore")

    file_lats = ds["latitude"].values
    flip_lat = file_lats[0] > file_lats[-1]

    sorted_lons = np.sort(ds["longitude"].values)
    sorted_lats = np.sort(file_lats)

    lon_start = _nearest_idx(FULL_LON, sorted_lons[0])
    lon_end = lon_start + len(sorted_lons)
    lat_start = _nearest_idx(FULL_LAT, sorted_lats[0])
    lat_end = lat_start + len(sorted_lats)

    for var in ("uo", "vo"):
        # Load one variable at a time to keep memory low.
        data = ds[var].values.astype(np.float32)
        if flip_lat:
            data = data[:, ::-1, :]
        np.nan_to_num(data, nan=0.0, copy=False)
        store[var][:, lat_start:lat_end, lon_start:lon_end] = data
        del data


def main():
    parser = argparse.ArgumentParser(
        description="Download ocean currents and build a Zarr store."
    )
    parser.add_argument("--output", default="data/currents.zarr")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    progress_path = output.parent / f"{output.name}.progress.json"

    lon_edges = _tile_edges(-180, 180, TILE_LON)
    lat_edges = _tile_edges(-80, 90, TILE_LAT)
    total = (len(lon_edges) - 1) * (len(lat_edges) - 1)

    completed = _load_progress(progress_path)
    print(f"Tiles: {total} total, {len(completed)} already done")

    if not (output.exists() and (output / ".zgroup").exists()):
        _create_store(output)

    store = zarr.open(str(output), mode="r+")

    count = len(completed)
    for li in range(len(lat_edges) - 1):
        for lo in range(len(lon_edges) - 1):
            lo0, lo1 = lon_edges[lo], lon_edges[lo + 1]
            la0, la1 = lat_edges[li], lat_edges[li + 1]
            tile_key = f"{lo0},{la0}"

            if tile_key in completed:
                continue

            count += 1
            print(f"[{count}/{total}] lon [{lo0}, {lo1}] lat [{la0}, {la1}]...")

            ds = _download_tile(lo0, lo1, la0, la1)
            _write_tile(store, ds)
            ds.close()

            completed.add(tile_key)
            _save_progress(progress_path, completed)
            print(f"  Done. ({count}/{total})")

    if progress_path.exists():
        progress_path.unlink()

    print("\nVerifying...")
    result = xr.open_zarr(output, consolidated=False)
    print(f"Zarr store: time={result.sizes['time']}, "
          f"lat={result.sizes['latitude']}, lon={result.sizes['longitude']}")
    result.close()
    print("Done!")


if __name__ == "__main__":
    main()
