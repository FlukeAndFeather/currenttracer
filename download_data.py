"""Download surface current data from Copernicus Marine Service.

Requires a free Copernicus Marine account. On first run, the toolbox
will prompt for your username and password (then caches credentials).

    conda install -c conda-forge copernicusmarine
    python download_data.py

Adjust the date range, bounding box, and output directory as needed.
"""

import argparse
from datetime import datetime

import copernicusmarine


def download(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    min_lon: float = -180,
    max_lon: float = 180,
    min_lat: float = -90,
    max_lat: float = 90,
    output_dir: str = "data",
):
    """Download daily surface current (uo, vo) from the global ocean
    physics multiyear reanalysis product.

    Dataset: cmems_mod_glo_phy_my_0.083deg_P1D-m
    Variables: uo (eastward velocity), vo (northward velocity)
    Depth: surface only (0–1 m)
    """
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["uo", "vo"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        minimum_depth=0,
        maximum_depth=1,
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
        output_directory=output_dir,
        file_format="netcdf",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ocean surface current data from Copernicus Marine Service."
    )
    parser.add_argument(
        "--start", default="2024-01-01",
        help="Start date (YYYY-MM-DD). Default: 2024-01-01",
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="End date (YYYY-MM-DD). Default: 2024-12-31",
    )
    parser.add_argument(
        "--min-lon", type=float, default=-180, help="Minimum longitude. Default: -180",
    )
    parser.add_argument(
        "--max-lon", type=float, default=180, help="Maximum longitude. Default: 180",
    )
    parser.add_argument(
        "--min-lat", type=float, default=-90, help="Minimum latitude. Default: -90",
    )
    parser.add_argument(
        "--max-lat", type=float, default=90, help="Maximum latitude. Default: 90",
    )
    parser.add_argument(
        "--output", default="data", help="Output directory. Default: data",
    )
    args = parser.parse_args()

    download(
        start_date=args.start,
        end_date=args.end,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        output_dir=args.output,
    )
