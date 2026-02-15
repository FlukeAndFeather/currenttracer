"""Download surface current data from Copernicus Marine Service.

Requires a free Copernicus Marine account. On first run, the toolbox
will prompt for your username and password (then caches credentials).

    conda install -c conda-forge copernicusmarine
    python download_data.py

Adjust the date range, bounding box, and output directory as needed.
"""

import argparse
import json
from datetime import datetime, timedelta

import copernicusmarine


def download(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    min_lon: float = -180,
    max_lon: float = 180,
    min_lat: float = -80,
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


def download_chunked(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    min_lon: float = -180,
    max_lon: float = 180,
    min_lat: float = -80,
    max_lat: float = 90,
    output_dir: str = "data",
    chunk_lon: float | None = None,
    chunk_lat: float | None = None,
    chunk_days: float | None = None,
):
    """Download in spatial/temporal chunks to avoid memory issues.

    Any chunk dimension set to None downloads that full range in one go.
    """
    lon_edges = _chunk_edges(min_lon, max_lon, chunk_lon)
    lat_edges = _chunk_edges(min_lat, max_lat, chunk_lat)
    time_edges = _chunk_time_edges(start_date, end_date, chunk_days)

    total = (len(lon_edges) - 1) * (len(lat_edges) - 1) * (len(time_edges) - 1)
    count = 0

    for ti in range(len(time_edges) - 1):
        t0 = time_edges[ti]
        t1 = time_edges[ti + 1]
        for li in range(len(lat_edges) - 1):
            for lo in range(len(lon_edges) - 1):
                count += 1
                lo0, lo1 = lon_edges[lo], lon_edges[lo + 1]
                la0, la1 = lat_edges[li], lat_edges[li + 1]
                print(f"[{count}/{total}] lon [{lo0}, {lo1}] lat [{la0}, {la1}] time [{t0}, {t1}]")
                download(
                    start_date=t0,
                    end_date=t1,
                    min_lon=lo0,
                    max_lon=lo1,
                    min_lat=la0,
                    max_lat=la1,
                    output_dir=output_dir,
                )


def _chunk_edges(lo: float, hi: float, step: float | None) -> list[float]:
    """Return bin edges from lo to hi with given step size."""
    if step is None:
        return [lo, hi]
    edges = []
    val = lo
    while val < hi:
        edges.append(val)
        val += step
    edges.append(hi)
    return edges


def _chunk_time_edges(start: str, end: str, step_days: float | None) -> list[str]:
    """Return date-string edges from start to end with given step in days."""
    if step_days is None:
        return [start, end]
    dt = timedelta(days=step_days)
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    edges = []
    cur = s
    while cur < e:
        edges.append(cur.strftime("%Y-%m-%d"))
        cur += dt
    edges.append(e.strftime("%Y-%m-%d"))
    return edges


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
        "--min-lat", type=float, default=-80, help="Minimum latitude. Default: -80",
    )
    parser.add_argument(
        "--max-lat", type=float, default=90, help="Maximum latitude. Default: 90",
    )
    parser.add_argument(
        "--output", default="data", help="Output directory. Default: data",
    )
    parser.add_argument(
        "--chunks", type=json.loads, default=None, metavar="[LON,LAT,DAYS]",
        help='Chunk sizes as JSON list, e.g. \'[10, 10, null]\'. '
             'null means no chunking on that dimension.',
    )
    args = parser.parse_args()

    if args.chunks:
        clon, clat, cdays = args.chunks
        download_chunked(
            start_date=args.start,
            end_date=args.end,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            output_dir=args.output,
            chunk_lon=clon,
            chunk_lat=clat,
            chunk_days=cdays,
        )
    else:
        download(
            start_date=args.start,
            end_date=args.end,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            output_dir=args.output,
        )
