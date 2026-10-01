"""
drifter_records.py

Related task: Week 3 Computational, step 1 - per-drifter information the transition table
needs to decide what happened to a drifter that is not in R tau days after a start reading.

The region cache from load.py holds only readings inside R, so on its own it can't tell
"left the box" apart from "stopped reporting". This module fetches, for every drifter that
visits R, what its whole record looks like (in or out of R), and caches it separately in
data/interim/<region>_records.npz, so the region cache doesn't need rebuilding.

Two tables:
- traj_*: GDP metadata, one entry per drifter (id, end_date, typedeath).
- seg_*: the continuous stretches of each drifter's hourly record. A new stretch starts
  wherever consecutive readings are more than 1 hour apart, so every hour from seg_start
  to seg_end has a reading. For R's drifters, 49% of records have at least one such gap
  (8,751 gaps, median 20 hours), so gaps can't be ignored.

Why stretches rather than end_date: end_date is the DAC's "last good date", which matched
the last hourly reading (within 1 hour) for only 7% of R's drifters and is missing for some.
It is kept for reference; the stretches decide whether a drifter was still reporting.

Run: python3 -m src.ingest.drifter_records  (about a minute the first time)
"""
import os
import numpy as np

from src.common.config import INTERIM_DIR, load_region_bounds
from src.ingest.load import BOX_KEYS, load_gdp, load_region


def record_segments(time, rowsize, traj):
    """Continuous stretches of hourly readings for the chosen drifters.

    Inputs:
        time: the full "time" array (every reading of every drifter, ragged-array order).
        rowsize: readings per drifter, as in the ragged array.
        traj: indices of the drifters to process.
    Returns:
        (seg_traj, seg_start, seg_end): for every stretch, the drifter index and the times
        of its first and last reading.
    """
    stops = np.cumsum(rowsize)
    starts = stops - rowsize
    seg_traj, seg_start, seg_end = [], [], []
    for k in traj:
        t = time[starts[k]:stops[k]]
        # a jump of more than 1 hour between consecutive readings ends one stretch and starts the next
        breaks = np.flatnonzero(np.diff(t) > np.timedelta64(1, "h"))
        first = np.r_[0, breaks + 1]
        last = np.r_[breaks, t.size - 1]
        seg_traj.append(np.full(first.size, k))
        seg_start.append(t[first])
        seg_end.append(t[last])
    return np.concatenate(seg_traj), np.concatenate(seg_start), np.concatenate(seg_end)


def extract_records(ds, drifter_ids):
    """Per-drifter metadata and record stretches for the given drifters.

    Inputs:
        ds: the full GDP dataset from load_gdp().
        drifter_ids: GDP buoy IDs to keep (the drifters that visit R).
    Returns:
        dict of 1-D arrays: traj_id, traj_end_date, traj_typedeath (one per drifter) and
        seg_id, seg_start, seg_end (one per continuous stretch).
    """
    ids = ds["id"].values
    traj = np.flatnonzero(np.isin(ids, drifter_ids))
    # The small per-drifter variables are fetched before the big time array: fetching them
    # straight after it failed repeatedly with "Server disconnected"
    records = {
        "traj_id": ids[traj],
        "traj_end_date": ds["end_date"].values[traj],
        "traj_typedeath": ds["typedeath"].values[traj],  # 1 = ran aground
    }
    rowsize = ds["rowsize"].values

    time = ds["time"].values  # every reading, ~1.6GB; only needed long enough to find the gaps
    # zarr fills a chunk that failed to download with NaT, which would look like a gap
    if np.isnat(time).any():
        raise RuntimeError("Incomplete download (missing time chunks). Rerun on a stable connection.")
    seg_traj, records["seg_start"], records["seg_end"] = record_segments(time, rowsize, traj)
    records["seg_id"] = ids[seg_traj]
    return records


def load_records(region=None):
    """Record stretches and metadata for R's drifters, from the local cache if it matches the box.

    Inputs:
        region: dict from config/region.yaml (read from disk if not given).
    Returns:
        dict of 1-D arrays as described in extract_records().
    """
    region = region or load_region_bounds()
    path = os.path.join(INTERIM_DIR, f"{region['name']}_records.npz")

    if os.path.exists(path):
        cached = dict(np.load(path))
        # The box is stored with the cache so an edited region.yaml can't silently reuse old data
        if all(float(cached[k]) == region[k] for k in BOX_KEYS):
            return cached

    drifter_ids = np.unique(load_region(region)["drifter_id"])
    print("Fetching drifter records from the cloud dataset (one-off, about a minute)...")
    records = extract_records(load_gdp(), drifter_ids)
    os.makedirs(INTERIM_DIR, exist_ok=True)
    np.savez(path, **records, **{k: region[k] for k in BOX_KEYS})
    return records


if __name__ == "__main__":
    records = load_records()
    n_drifters = records["traj_id"].size
    n_gaps = records["seg_id"].size - n_drifters
    print(f"Drifters visiting R: {n_drifters:,}; continuous stretches: {records['seg_id'].size:,} ({n_gaps:,} gaps)")
