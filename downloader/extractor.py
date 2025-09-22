# downloader/extractor.py
"""
Extract metadata and a short textual summary from a netCDF file using xarray.
Returns a dict with keys:
  variables, lat_min, lat_max, lon_min, lon_max, time_start, time_end, year, summary
"""

from datetime import datetime
import numpy as np
import xarray as xr
import os

def _safe_min_max(arr):
    try:
        a = np.array(arr)
        if a.size == 0:
            return None, None
        return float(np.nanmin(a)), float(np.nanmax(a))
    except Exception:
        return None, None

def _try_coords(ds, names):
    for n in names:
        if n in ds.coords:
            try:
                vals = ds.coords[n].values
                return _safe_min_max(vals)
            except Exception:
                continue
        if n in ds.data_vars:
            try:
                vals = ds[n].values
                return _safe_min_max(vals)
            except Exception:
                continue
    return None, None

def summarize_netcdf(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        ds = xr.open_dataset(path, mask_and_scale=False)
    except Exception as e:
        # try with decode_times=False for problematic files
        ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)

    # variables (limit to a few for summary)
    vars_ = list(ds.data_vars.keys())

    # lat/lon
    lat_min, lat_max = _try_coords(ds, ["lat", "latitude", "LATITUDE", "y"])
    lon_min, lon_max = _try_coords(ds, ["lon", "longitude", "LONGITUDE", "x"])

    # time
    time_start = time_end = None
    for tname in ("time", "TIME", "Time"):
        if tname in ds.coords or tname in ds.data_vars:
            try:
                tvals = ds.get(tname)
                arr = np.array(tvals)
                if arr.size:
                    # handle numpy datetime64 or numbers
                    if np.issubdtype(arr.dtype, np.datetime64):
                        tmin = arr.min()
                        tmax = arr.max()
                        # convert to ISO strings
                        time_start = np.datetime_as_string(tmin, unit='s')
                        time_end = np.datetime_as_string(tmax, unit='s')
                    else:
                        # non-datetime times: just record first/last as-is
                        time_start = str(arr.flat[0])
                        time_end = str(arr.flat[-1])
                break
            except Exception:
                continue

    # year detection (from time_start if possible)
    year = None
    if time_start:
        try:
            year = int(str(time_start)[:4])
        except Exception:
            year = None

    # build summary text
    parts = []
    if vars_:
        parts.append(f"Variables: {', '.join(vars_[:6])}{'...' if len(vars_)>6 else ''}")
    if lat_min is not None and lat_max is not None:
        parts.append(f"Lat: {lat_min:.3f} to {lat_max:.3f}")
    if lon_min is not None and lon_max is not None:
        parts.append(f"Lon: {lon_min:.3f} to {lon_max:.3f}")
    if time_start and time_end:
        parts.append(f"Time: {time_start} to {time_end}")

    summary = " | ".join(parts) if parts else f"File: {os.path.basename(path)} (no readable metadata)"

    # close dataset to free file handles
    try:
        ds.close()
    except Exception:
        pass

    return {
        "variables": vars_,
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
        "time_start": time_start, "time_end": time_end,
        "year": year,
        "summary": summary
    }
