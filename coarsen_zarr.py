"""Coarsen the full-resolution Zarr store for in-memory use.

Reads the 1/12° daily ocean current data, slices to the first 60 days,
coarsens spatially by 6× (to 1/2°), and writes a small Zarr store that
can be loaded entirely into memory (~188 MB).

    python coarsen_zarr.py [--input data/currents.zarr] [--output data/currents_coarse.zarr]
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description="Coarsen ocean current Zarr store.")
    parser.add_argument("--input", default="data/currents.zarr")
    parser.add_argument("--output", default="data/currents_coarse.zarr")
    parser.add_argument("--days", type=int, default=60, help="Number of days to keep.")
    parser.add_argument("--factor", type=int, default=6, help="Spatial coarsening factor.")
    args = parser.parse_args()

    print(f"Opening {args.input}...")
    ds = xr.open_zarr(args.input, consolidated=False)
    print(f"  Full shape: time={ds.sizes['time']}, "
          f"lat={ds.sizes['latitude']}, lon={ds.sizes['longitude']}")

    # Slice to first N days.
    ds = ds.isel(time=slice(0, args.days))
    print(f"  After time slice: {ds.sizes['time']} days")

    # Coarsen spatially.
    print(f"  Coarsening by {args.factor}x...")
    ds = ds.coarsen(latitude=args.factor, longitude=args.factor, boundary="trim").mean()
    print(f"  Coarsened shape: lat={ds.sizes['latitude']}, lon={ds.sizes['longitude']}")

    # Replace NaN with 0 and convert to float32.
    ds["uo"] = ds["uo"].fillna(0.0).astype(np.float32)
    ds["vo"] = ds["vo"].fillna(0.0).astype(np.float32)

    # Write to zarr.
    output = Path(args.output)
    print(f"Writing to {output}...")
    ds.to_zarr(str(output), mode="w")

    # Verify.
    result = xr.open_zarr(str(output), consolidated=False)
    print(f"Done! Shape: time={result.sizes['time']}, "
          f"lat={result.sizes['latitude']}, lon={result.sizes['longitude']}")
    uo_mb = result["uo"].nbytes / 1e6
    vo_mb = result["vo"].nbytes / 1e6
    print(f"  uo: {uo_mb:.0f} MB, vo: {vo_mb:.0f} MB, total: {uo_mb + vo_mb:.0f} MB")


if __name__ == "__main__":
    main()
