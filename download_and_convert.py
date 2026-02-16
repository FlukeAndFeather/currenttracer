"""Download ocean current tiles and stream them directly into a Zarr store.

Downloads one 30x30 degree tile at a time from Copernicus Marine Service,
writes it into the Zarr store, then deletes the NetCDF file. This keeps
peak disk usage to ~200 MB above the final Zarr store size.

Supports resume: if interrupted, re-run the same command and it will skip
tiles that were already written.

    python download_and_convert.py [--output data/currents.zarr]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import copernicusmarine
import numpy as np
import xarray as xr

# Zarr chunk sizes matching data.py expectations.
TIME_CHUNK = 5
LAT_CHUNK = 120
LON_CHUNK = 120

# Spatial tile size for downloads (degrees).
TILE_LON = 30
TILE_LAT = 30

# Global extent.
MIN_LON = -180
MAX_LON = 180
MIN_LAT = -80
MAX_LAT = 90


def _write_zarr_array(store_path: Path, name: str, data: np.ndarray,
                      chunks: tuple, fill_value=0.0, attrs: dict | None = None):
    """Write a zarr v2 array using filesystem operations (no zarr API)."""
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

    # Write 1-D coordinate chunks.
    if data.ndim == 1:
        for ci in range(0, data.shape[0], chunks[0]):
            chunk_data = data[ci:ci + chunks[0]]
            chunk_key = str(ci // chunks[0])
            (arr_dir / chunk_key).write_bytes(chunk_data.tobytes())


def _create_empty_zarr_array(store_path: Path, name: str, shape: tuple,
                             chunks: tuple, dtype: str, fill_value: float,
                             attrs: dict | None = None):
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


def _tile_edges(lo, hi, step):
    """Return bin edges from lo to hi with given step."""
    edges = []
    val = lo
    while val < hi:
        edges.append(val)
        val += step
    edges.append(hi)
    return edges


def _download_tile(lo0, lo1, la0, la1, start_date, end_date, tmp_dir):
    """Download a single tile from Copernicus. Returns path to .nc file."""
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=lo0,
        maximum_longitude=lo1,
        minimum_latitude=la0,
        maximum_latitude=la1,
        minimum_depth=0,
        maximum_depth=1,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
        output_directory=str(tmp_dir),
        file_format="netcdf",
    )
    # Find the downloaded file.
    nc_files = list(tmp_dir.glob("*.nc"))
    if not nc_files:
        raise RuntimeError(f"No .nc file found after download for tile ({lo0},{la0})-({lo1},{la1})")
    return nc_files[0]


def _load_progress(progress_path):
    """Load set of completed tile keys from progress file."""
    if progress_path.exists():
        return set(json.loads(progress_path.read_text()))
    return set()


def _save_progress(progress_path, completed):
    """Save set of completed tile keys to progress file."""
    progress_path.write_text(json.dumps(sorted(completed)))


def main():
    parser = argparse.ArgumentParser(
        description="Download ocean currents and convert to Zarr in one pass."
    )
    parser.add_argument("--output", default="data/currents.zarr",
                        help="Output Zarr store path. Default: data/currents.zarr")
    parser.add_argument("--start", default="2024-01-01",
                        help="Start date (YYYY-MM-DD). Default: 2024-01-01")
    parser.add_argument("--end", default="2024-12-31",
                        help="End date (YYYY-MM-DD). Default: 2024-12-31")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    tmp_dir = output_path.parent / "_tmp_tile"
    progress_path = output_path.parent / f"{output_path.name}.progress.json"

    lon_edges = _tile_edges(MIN_LON, MAX_LON, TILE_LON)
    lat_edges = _tile_edges(MIN_LAT, MAX_LAT, TILE_LAT)
    total_tiles = (len(lon_edges) - 1) * (len(lat_edges) - 1)

    completed = _load_progress(progress_path)
    print(f"Tiles: {total_tiles} total, {len(completed)} already done")

    # --- Step 1: Download first tile to learn grid resolution ---
    store_exists = output_path.exists() and (output_path / ".zgroup").exists()

    if not store_exists:
        print("Downloading first tile to determine grid resolution...")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        nc_path = _download_tile(
            lon_edges[0], lon_edges[1], lat_edges[0], lat_edges[1],
            args.start, args.end, tmp_dir,
        )

        with xr.open_dataset(nc_path) as ds:
            # Get resolution from the first tile's coordinates.
            tile_lon = np.sort(ds["longitude"].values)
            tile_lat = np.sort(ds["latitude"].values)
            time_arr = ds["time"].values

            resolution = round(float(tile_lon[1] - tile_lon[0]), 6)
            print(f"Resolution: {resolution}°")

        # Build the full global grid at this resolution.
        full_lon = np.arange(MIN_LON, MAX_LON, resolution).astype(np.float64)
        full_lat = np.arange(MIN_LAT, MAX_LAT + resolution / 2, resolution).astype(np.float64)
        nt = len(time_arr)
        nlat = len(full_lat)
        nlon = len(full_lon)

        print(f"Global grid: time={nt}, lat={nlat}, lon={nlon}")

        # --- Step 2: Create empty Zarr store ---
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))

        dims_attr = lambda dims: {"_ARRAY_DIMENSIONS": dims}

        time_data = time_arr.astype("datetime64[ns]")
        _write_zarr_array(output_path, "time", time_data,
                          chunks=(nt,), fill_value=0,
                          attrs=dims_attr(["time"]))

        _write_zarr_array(output_path, "latitude", full_lat,
                          chunks=(nlat,), fill_value=float("nan"),
                          attrs=dims_attr(["latitude"]))

        _write_zarr_array(output_path, "longitude", full_lon,
                          chunks=(nlon,), fill_value=float("nan"),
                          attrs=dims_attr(["longitude"]))

        _create_empty_zarr_array(
            output_path, "uo",
            shape=(nt, nlat, nlon),
            chunks=(TIME_CHUNK, LAT_CHUNK, LON_CHUNK),
            dtype="<f4", fill_value=0.0,
            attrs=dims_attr(["time", "latitude", "longitude"]),
        )
        _create_empty_zarr_array(
            output_path, "vo",
            shape=(nt, nlat, nlon),
            chunks=(TIME_CHUNK, LAT_CHUNK, LON_CHUNK),
            dtype="<f4", fill_value=0.0,
            attrs=dims_attr(["time", "latitude", "longitude"]),
        )

        print("Empty Zarr store created.")

        # Process the first tile we already downloaded.
        _process_tile(output_path, nc_path, lon_edges[0], lon_edges[1],
                      lat_edges[0], lat_edges[1], full_lon, full_lat)
        tile_key = f"{lon_edges[0]},{lat_edges[0]}"
        completed.add(tile_key)
        _save_progress(progress_path, completed)

        # Clean up.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[1/{total_tiles}] Done: lon [{lon_edges[0]}, {lon_edges[1]}] "
              f"lat [{lat_edges[0]}, {lat_edges[1]}]")
    else:
        # Store already exists — read back coordinate arrays for index mapping.
        ds_check = xr.open_zarr(output_path, consolidated=False)
        full_lon = ds_check["longitude"].values.astype(np.float64)
        full_lat = ds_check["latitude"].values.astype(np.float64)
        ds_check.close()
        print("Resuming with existing Zarr store.")

    # --- Step 3: Process remaining tiles ---
    import zarr

    count = len(completed)
    for li in range(len(lat_edges) - 1):
        for lo in range(len(lon_edges) - 1):
            lo0, lo1 = lon_edges[lo], lon_edges[lo + 1]
            la0, la1 = lat_edges[li], lat_edges[li + 1]
            tile_key = f"{lo0},{la0}"

            if tile_key in completed:
                continue

            count += 1
            print(f"[{count}/{total_tiles}] Downloading lon [{lo0}, {lo1}] lat [{la0}, {la1}]...")

            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                nc_path = _download_tile(lo0, lo1, la0, la1,
                                         args.start, args.end, tmp_dir)
                _process_tile(output_path, nc_path, lo0, lo1, la0, la1,
                              full_lon, full_lat)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            completed.add(tile_key)
            _save_progress(progress_path, completed)
            print(f"  Done. ({count}/{total_tiles})")

    # Clean up progress file.
    if progress_path.exists():
        progress_path.unlink()

    # Verify.
    print("\nVerifying...")
    result = xr.open_zarr(output_path, consolidated=False)
    print(f"Zarr store: time={result.sizes['time']}, "
          f"lat={result.sizes['latitude']}, lon={result.sizes['longitude']}")
    result.close()
    print("Done!")


def _process_tile(store_path, nc_path, lo0, lo1, la0, la1, full_lon, full_lat):
    """Read a NetCDF tile and write its data into the Zarr store."""
    import zarr

    ds = xr.open_dataset(nc_path)

    if "depth" in ds.dims:
        ds = ds.isel(depth=0).drop_vars("depth", errors="ignore")

    file_lons = ds["longitude"].values
    file_lats = ds["latitude"].values

    # Build index lookups.
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

    # Write uo.
    uo = ds["uo"].values
    if flip_lat:
        uo = uo[:, ::-1, :]
    uo = np.nan_to_num(uo, nan=0.0).astype(np.float32)
    store["uo"][:, lat_start:lat_end, lon_start:lon_end] = uo
    del uo

    # Write vo.
    vo = ds["vo"].values
    if flip_lat:
        vo = vo[:, ::-1, :]
    vo = np.nan_to_num(vo, nan=0.0).astype(np.float32)
    store["vo"][:, lat_start:lat_end, lon_start:lon_end] = vo
    del vo

    ds.close()


if __name__ == "__main__":
    main()
