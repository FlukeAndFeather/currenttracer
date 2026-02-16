"""Convert NetCDF ocean current tiles into a single Zarr store.

Run once after downloading data:

    python convert_to_zarr.py [--input data] [--output data/currents.zarr]

Processes one NetCDF file at a time to stay within 1 GB RAM.
The resulting Zarr store is chunked for efficient partial reads by the
web server (small spatial/temporal subsets per trace request).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

# Target chunk sizes: ~10° spatial, 5 days temporal.
TIME_CHUNK = 5
LAT_CHUNK = 120
LON_CHUNK = 120


def convert(input_dir: str = "data", output_path: str = "data/currents.zarr"):
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    files = sorted(input_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {input_dir}")

    print(f"Found {len(files)} NetCDF files.")

    # --- Pass 1: scan all files to build the full global grid ---
    print("Scanning coordinates...")
    all_lons = set()
    all_lats = set()
    time_arr = None

    for f in files:
        with xr.open_dataset(f) as ds:
            # Round to avoid floating-point near-duplicates across tiles.
            all_lons.update(np.round(ds["longitude"].values, 4).tolist())
            all_lats.update(np.round(ds["latitude"].values, 4).tolist())
            if time_arr is None:
                time_arr = ds["time"].values

    full_lon = np.array(sorted(all_lons), dtype=np.float64)
    full_lat = np.array(sorted(all_lats), dtype=np.float64)  # ascending
    nt = len(time_arr)
    nlat = len(full_lat)
    nlon = len(full_lon)

    print(f"Global grid: time={nt}, lat={nlat}, lon={nlon}")

    # --- Create empty Zarr store with full dimensions ---
    if output_path.exists():
        shutil.rmtree(output_path)

    store = zarr.open(str(output_path), mode="w")

    # Coordinate arrays — use NaN fill_value so xarray doesn't mask real 0.0 values.
    store.create_dataset("time", data=time_arr.astype("datetime64[ns]"),
                         chunks=(nt,), overwrite=True)
    store.create_dataset("latitude", data=full_lat,
                         chunks=(nlat,), fill_value=np.nan, overwrite=True)
    store.create_dataset("longitude", data=full_lon,
                         chunks=(nlon,), fill_value=np.nan, overwrite=True)

    # Data arrays — created empty with fill_value=0, written per-tile below.
    store.zeros(
        "uo",
        shape=(nt, nlat, nlon),
        chunks=(TIME_CHUNK, LAT_CHUNK, LON_CHUNK),
        dtype="float32",
    )
    store.zeros(
        "vo",
        shape=(nt, nlat, nlon),
        chunks=(TIME_CHUNK, LAT_CHUNK, LON_CHUNK),
        dtype="float32",
    )

    # xarray needs _ARRAY_DIMENSIONS to interpret Zarr as a Dataset.
    store["uo"].attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]
    store["vo"].attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]
    store["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
    store["latitude"].attrs["_ARRAY_DIMENSIONS"] = ["latitude"]
    store["longitude"].attrs["_ARRAY_DIMENSIONS"] = ["longitude"]

    # Build index lookups for fast coordinate mapping.
    lon_to_idx = {round(v, 4): i for i, v in enumerate(full_lon)}
    lat_to_idx = {round(v, 4): i for i, v in enumerate(full_lat)}

    # --- Pass 2: write each file's data into the correct region ---
    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {f.name}")

        ds = xr.open_dataset(f)

        if "depth" in ds.dims:
            ds = ds.isel(depth=0)

        file_lons = ds["longitude"].values
        file_lats = ds["latitude"].values

        # Find where this tile's coordinates sit in the global grid.
        lon_start = lon_to_idx[round(float(file_lons[0]), 4)]
        lon_end = lon_to_idx[round(float(file_lons[-1]), 4)] + 1

        sorted_lats = np.sort(file_lats)
        lat_start = lat_to_idx[round(float(sorted_lats[0]), 4)]
        lat_end = lat_to_idx[round(float(sorted_lats[-1]), 4)] + 1

        flip_lat = file_lats[0] > file_lats[-1]

        # Load, fix, and write uo.
        uo = ds["uo"].values
        if flip_lat:
            uo = uo[:, ::-1, :]
        uo = np.nan_to_num(uo, nan=0.0).astype(np.float32)
        store["uo"][:, lat_start:lat_end, lon_start:lon_end] = uo
        del uo

        # Load, fix, and write vo.
        vo = ds["vo"].values
        if flip_lat:
            vo = vo[:, ::-1, :]
        vo = np.nan_to_num(vo, nan=0.0).astype(np.float32)
        store["vo"][:, lat_start:lat_end, lon_start:lon_end] = vo
        del vo

        ds.close()

    # Verify the result.
    print("Verifying...")
    result = xr.open_zarr(output_path, consolidated=False)
    print(f"Zarr store shape: time={result.sizes['time']}, "
          f"lat={result.sizes['latitude']}, lon={result.sizes['longitude']}")
    result.close()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NetCDF ocean current tiles to a single Zarr store."
    )
    parser.add_argument(
        "--input", default="data",
        help="Directory containing .nc files. Default: data",
    )
    parser.add_argument(
        "--output", default="data/currents.zarr",
        help="Output Zarr store path. Default: data/currents.zarr",
    )
    args = parser.parse_args()
    convert(args.input, args.output)
