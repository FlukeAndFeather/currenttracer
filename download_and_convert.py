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

# Global extent.
MIN_LON = -180
MAX_LON = 180
MIN_LAT = -80
MAX_LAT = 90

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"


def _tile_edges(lo, hi, step):
    edges = []
    val = lo
    while val < hi:
        edges.append(val)
        val += step
    edges.append(hi)
    return edges


def _download_tile(lo0, lo1, la0, la1, start_date, end_date):
    """Download a tile and return it as an xarray Dataset in memory."""
    ds = copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        variables=["uo", "vo"],
        minimum_longitude=lo0,
        maximum_longitude=lo1,
        minimum_latitude=la0,
        maximum_latitude=la1,
        minimum_depth=0,
        maximum_depth=1,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
    )
    return ds


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
        "compressor": None,
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


def main():
    parser = argparse.ArgumentParser(
        description="Download ocean currents and build a Zarr store."
    )
    parser.add_argument("--output", default="data/currents.zarr")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    progress_path = output.parent / f"{output.name}.progress.json"

    lon_edges = _tile_edges(MIN_LON, MAX_LON, TILE_LON)
    lat_edges = _tile_edges(MIN_LAT, MAX_LAT, TILE_LAT)
    total = (len(lon_edges) - 1) * (len(lat_edges) - 1)

    completed = _load_progress(progress_path)
    print(f"Tiles: {total} total, {len(completed)} already done")

    store_exists = output.exists() and (output / ".zgroup").exists()

    # --- Step 1: Download first tile to learn grid ---
    if not store_exists:
        print("Downloading first tile to determine grid...")
        ds = _download_tile(
            lon_edges[0], lon_edges[1], lat_edges[0], lat_edges[1],
            args.start, args.end,
        )

        if "depth" in ds.dims:
            ds = ds.isel(depth=0).drop_vars("depth", errors="ignore")

        tile_lon = np.sort(ds["longitude"].values)
        resolution = round(float(tile_lon[1] - tile_lon[0]), 6)
        time_arr = ds["time"].values
        print(f"Resolution: {resolution}°")

        # Build full global grid.
        full_lon = np.arange(MIN_LON, MAX_LON, resolution).astype(np.float64)
        full_lat = np.arange(MIN_LAT, MAX_LAT + resolution / 2, resolution).astype(np.float64)
        nt, nlat, nlon = len(time_arr), len(full_lat), len(full_lon)
        print(f"Global grid: time={nt}, lat={nlat}, lon={nlon}")

        # --- Step 2: Create empty Zarr store ---
        output.mkdir(parents=True, exist_ok=True)
        (output / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

        dims = lambda d: {"_ARRAY_DIMENSIONS": d}

        _write_zarr_array(output, "time", time_arr.astype("datetime64[ns]"),
                          chunks=(nt,), fill_value=0, attrs=dims(["time"]))
        _write_zarr_array(output, "latitude", full_lat,
                          chunks=(nlat,), fill_value=float("nan"),
                          attrs=dims(["latitude"]))
        _write_zarr_array(output, "longitude", full_lon,
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

        # Write first tile.
        _write_tile(output, ds, full_lon, full_lat)
        tile_key = f"{lon_edges[0]},{lat_edges[0]}"
        completed.add(tile_key)
        _save_progress(progress_path, completed)
        ds.close()
        print(f"[1/{total}] Done")
    else:
        check = xr.open_zarr(output, consolidated=False)
        full_lon = check["longitude"].values.astype(np.float64)
        full_lat = check["latitude"].values.astype(np.float64)
        check.close()
        print("Resuming with existing Zarr store.")

    # --- Step 3: Download and write remaining tiles ---
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

            ds = _download_tile(lo0, lo1, la0, la1, args.start, args.end)
            if "depth" in ds.dims:
                ds = ds.isel(depth=0).drop_vars("depth", errors="ignore")

            _write_tile(output, ds, full_lon, full_lat)
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


def _write_tile(store_path, ds, full_lon, full_lat):
    """Write an in-memory xarray Dataset tile into the Zarr store."""
    file_lons = ds["longitude"].values
    file_lats = ds["latitude"].values

    lon_to_idx = {round(v, 4): i for i, v in enumerate(full_lon)}
    lat_to_idx = {round(v, 4): i for i, v in enumerate(full_lat)}

    sorted_lons = np.sort(file_lons)
    sorted_lats = np.sort(file_lats)

    lon_start = lon_to_idx[round(float(sorted_lons[0]), 4)]
    lon_end = lon_to_idx[round(float(sorted_lons[-1]), 4)] + 1
    lat_start = lat_to_idx[round(float(sorted_lats[0]), 4)]
    lat_end = lat_to_idx[round(float(sorted_lats[-1]), 4)] + 1

    flip_lat = file_lats[0] > file_lats[-1]

    store = zarr.open(str(store_path), mode="r+")

    for var in ("uo", "vo"):
        data = ds[var].values
        if flip_lat:
            data = data[:, ::-1, :]
        data = np.nan_to_num(data, nan=0.0).astype(np.float32)
        store[var][:, lat_start:lat_end, lon_start:lon_end] = data
        del data


if __name__ == "__main__":
    main()
