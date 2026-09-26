"""
load.py

Related task: Week 1 Computational - "lead the data-adequacy count and get the raw dataset loading".

The Global Drifter Program (GDP) hourly dataset is an xarray.Dataset in clouddrift's
"ragged array" layout: every drifter's readings are concatenated into one long list
(the "obs" dimension), and "rowsize" says how many readings each drifter contributed.

Streaming the full ~16GB store takes minutes, so load_region() pulls region R's
readings once and caches them in data/interim/ (gitignored). The raw data is never
modified; deleting the cache just forces a fresh extract.
"""
import os
import numpy as np
import clouddrift as cd

from src.common.config import INTERIM_DIR, load_region_bounds

# Per-reading variables kept for region R (EDA and the transition matrix need nothing else)
EXTRA_VARS = ["time", "ve", "vn", "drogue_status", "gap"]
BOX_KEYS = ["lat_min", "lat_max", "lon_min", "lon_max"]


def load_gdp():
    """Open the GDP hourly dataset from clouddrift's cloud-hosted copy (lazy, nothing downloaded yet).

    Returns:
        xarray.Dataset: the full GDP hourly dataset.
    """
    return cd.datasets.gdp1h()


def in_box(lon, lat, region):
    """True/False for every reading: is it inside region R's box?

    Longitudes are mapped onto -180..180 first so the comparison works whatever
    convention the data uses. The else-branch handles a box that crosses the dateline.
    """
    lon_norm = ((lon + 180) % 360) - 180
    lon_min, lon_max = region["lon_min"], region["lon_max"]
    if lon_min <= lon_max:
        lon_ok = (lon_norm >= lon_min) & (lon_norm <= lon_max)
    else:
        lon_ok = (lon_norm >= lon_min) | (lon_norm <= lon_max)
    return lon_ok & (lat >= region["lat_min"]) & (lat <= region["lat_max"])


def extract_region(ds, region):
    """Pull every reading inside region R out of the full dataset.

    Inputs:
        ds: the full GDP dataset from load_gdp().
        region: dict from config/region.yaml.
    Returns:
        dict of 1-D arrays, one entry per in-region reading: lon, lat, drifter_id
        (GDP buoy ID) and EXTRA_VARS.
    """
    lon = ds["lon"].values
    lat = ds["lat"].values
    rowsize = ds["rowsize"].values

    # The GDP product has no missing positions, but zarr silently fills a chunk that
    # failed to download with NaN. That gave partial, run-to-run different totals, so fail loudly.
    if np.isnan(lon).any() or np.isnan(lat).any() or rowsize.sum() != lon.size:
        raise RuntimeError("Incomplete download (missing lon/lat chunks). Rerun on a stable connection.")

    idx = np.flatnonzero(in_box(lon, lat, region))

    # rowsize -> which drifter owns each reading (reading i belongs to the first drifter whose
    # cumulative row count exceeds i)
    owner = np.searchsorted(np.cumsum(rowsize), idx, side="right")

    data = {"lon": lon[idx], "lat": lat[idx], "drifter_id": ds["id"].values[owner]}
    del lon, lat  # ~1.6GB each; free before pulling the rest
    for var in EXTRA_VARS:
        data[var] = ds[var].values[idx]
    return data


def load_region(region=None):
    """Region R's readings, from the local cache if it matches the configured box.

    Inputs:
        region: dict from config/region.yaml (read from disk if not given).
    Returns:
        dict of 1-D arrays as described in extract_region().
    """
    region = region or load_region_bounds()
    path = os.path.join(INTERIM_DIR, f"{region['name']}.npz")

    if os.path.exists(path):
        cached = dict(np.load(path))
        # The box is stored with the cache so an edited region.yaml can't silently reuse old data
        if all(float(cached[k]) == region[k] for k in BOX_KEYS):
            return cached

    print("Extracting region from the cloud dataset (one-off, roughly 10 minutes)...")
    data = extract_region(load_gdp(), region)
    os.makedirs(INTERIM_DIR, exist_ok=True)
    np.savez(path, **data, **{k: region[k] for k in BOX_KEYS})
    return data
