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
CHUNKS = {"time": 5, "latitude": 120, "longitude": 120}


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
            all_lons.update(ds["longitude"].values.tolist())
            all_lats.update(ds["latitude"].values.tolist())
            if time_arr is None:
                time_arr = ds["time"].values

    full_lon = np.array(sorted(all_lons))
    full_lat = np.array(sorted(all_lats))  # ascending
    nt = len(time_arr)
    nlat = len(full_lat)
    nlon = len(full_lon)

    print(f"Global grid: time={nt}, lat={nlat}, lon={nlon}")

    # --- Create empty Zarr store with full dimensions ---
    if output_path.exists():
        shutil.rmtree(output_path)

    # Build an empty dataset with the full shape.
    ds_empty = xr.Dataset(
        {
            "uo": (["time", "latitude", "longitude"],
                   zarr.zeros((nt, nlat, nlon), chunks=(CHUNKS["time"], CHUNKS["latitude"], CHUNKS["longitude"]), dtype="float32")),
            "vo": (["time", "latitude", "longitude"],
                   zarr.zeros((nt, nlat, nlon), chunks=(CHUNKS["time"], CHUNKS["latitude"], CHUNKS["longitude"]), dtype="float32")),
        },
        coords={
            "time": time_arr,
            "latitude": full_lat,
            "longitude": full_lon,
        },
    )
    ds_empty.to_zarr(output_path, mode="w", compute=False)
    ds_empty.close()

    # Re-open the Zarr store for direct writing.
    store = zarr.open(str(output_path), mode="r+")

    # Build index lookups for fast coordinate mapping.
    lon_to_idx = {v: i for i, v in enumerate(full_lon)}
    lat_to_idx = {v: i for i, v in enumerate(full_lat)}

    # --- Pass 2: write each file's data into the correct region ---
    for i, f in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {f.name}")

        ds = xr.open_dataset(f)

        if "depth" in ds.dims:
            ds = ds.isel(depth=0)

        file_lons = ds["longitude"].values
        file_lats = ds["latitude"].values

        # Find where this tile's coordinates sit in the global grid.
        lon_start = lon_to_idx[file_lons[0]]
        lon_end = lon_to_idx[file_lons[-1]] + 1
        lat_indices = np.array([lat_to_idx[v] for v in sorted(file_lats)])
        lat_start = lat_indices[0]
        lat_end = lat_indices[-1] + 1

        # Load and write uo.
        uo = ds["uo"].values
        if file_lats[0] > file_lats[-1]:
            uo = uo[:, ::-1, :]  # flip lat to ascending
        uo = np.nan_to_num(uo, nan=0.0).astype(np.float32)
        store["uo"][:, lat_start:lat_end, lon_start:lon_end] = uo

        # Load and write vo.
        vo = ds["vo"].values
        if file_lats[0] > file_lats[-1]:
            vo = vo[:, ::-1, :]
        vo = np.nan_to_num(vo, nan=0.0).astype(np.float32)
        store["vo"][:, lat_start:lat_end, lon_start:lon_end] = vo

        ds.close()

    # Verify the result.
    print("Verifying...")
    result = xr.open_zarr(output_path)
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
