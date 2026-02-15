## Data

Download ocean surface current NetCDF files into this directory using the download script:

```bash
# Full year, global
python download_data.py --start 2024-01-01 --end 2024-12-31

# One month, North Atlantic (smaller, good for testing)
python download_data.py --start 2024-06-01 --end 2024-06-30 --min-lon -80 --max-lon 0 --min-lat 20 --max-lat 60
```

Requires a free [Copernicus Marine](https://marine.copernicus.eu/) account.
