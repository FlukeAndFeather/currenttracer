"""Convert NetCDF ocean current tiles into a single Zarr store.

Run once after downloading data:

    python convert_to_zarr.py [--input data] [--output data/currents.zarr]

The resulting Zarr store is chunked for efficient partial reads by the
web server (small spatial/temporal subsets per trace request).
"""

import argparse
from pathlib import Path

import xarray as xr

# Target chunk sizes: ~10° spatial, 5 days temporal.
CHUNKS = {"time": 5, "latitude": 120, "longitude": 120}


def convert(input_dir: str = "data", output_path: str = "data/currents.zarr"):
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    files = sorted(input_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {input_dir}")

    print(f"Opening {len(files)} NetCDF files...")
    ds = xr.open_mfdataset(files, combine="by_coords")

    # Surface layer only.
    if "depth" in ds.dims:
        ds = ds.isel(depth=0).drop_vars("depth", errors="ignore")

    # Ensure latitude is ascending (Copernicus data may be descending).
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")

    # Keep only the variables we need.
    ds = ds[["uo", "vo"]]

    print(f"Dataset shape: time={ds.sizes['time']}, "
          f"lat={ds.sizes['latitude']}, lon={ds.sizes['longitude']}")
    print(f"Re-chunking to {CHUNKS}...")

    ds = ds.chunk(CHUNKS)

    print(f"Writing to {output_path}...")
    ds.to_zarr(output_path, mode="w")

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
