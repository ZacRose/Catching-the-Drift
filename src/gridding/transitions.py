"""
transitions.py

Related task: Week 3 Computational - count how drifters move between grid cells over a
fixed time lag tau, the raw material for the transition matrix P.

Two steps, kept separate so one table serves every later analysis:
1. build_transitions(): one row per drifter per day in R, with where that drifter was
   tau later. Positions (not cell ids) are stored, so the same table can be binned onto any
   grid size, and filtered by drogue status or year, without being rebuilt.
2. transition_counts(): bin a table onto a grid and count start cell -> end cell moves,
   plus one extra state meaning "left R".

Each row's "outcome" says what was found tau after the start:
    0 IN_R    the drifter's reading at start + tau is inside R: an ordinary cell-to-cell move.
    1 LEFT_R  the drifter was still reporting at start + tau, but not from inside R: it left the box.
    2 ENDED   the drifter's record stopped before start + tau: destination unknown.
    3 GAP     start + tau falls in a gap in the drifter's record: destination unknown.
Only 0 and 1 are counted. 2 and 3 stay in the table rather than being dropped, so their
share can be reported, with typedeath (1 = ran aground) for groundings. Week 6 measured these at
the coast (src/gridding/coast.py): too few drifters run aground there to estimate a separate
coastline state, so their share is reported instead.

tau defaults to 15 days: the time a median-speed drogued drifter takes to cross a 2 degree
cell north to south (data_quality.py). The same tau is used at every grid size so that
resolution comparisons aren't confounded by the lag.

Loading the saved table (tau_days is a single number, not a column, so take it out before
filtering rows):
    table = dict(np.load("data/processed/transitions_tau15d.npz"))
    tau_days = table.pop("tau_days")

Run: python3 -m src.gridding.transitions
"""
import os
import numpy as np

from src.common.config import PROCESSED_DIR, load_region_bounds
from src.ingest.load import BOX_KEYS, load_region
from src.ingest.drifter_records import load_records
from src.gridding.grid import grid_edges, cell_index, per_cell_counts

TAU_DAYS = 15
BASELINE_CELL_SIZE = 4.0  # the coarse baseline grid (project_plan.txt, decision 5)
IN_R, LEFT_R, ENDED, GAP = 0, 1, 2, 3
OUTCOME_NAMES = {IN_R: "stayed in R", LEFT_R: "left R", ENDED: "record ended", GAP: "gap in record"}


def _hours(time):
    """Whole hours since 1970 as int64 (GDP hourly times all fall exactly on the hour)."""
    return time.astype("datetime64[h]").astype(np.int64)


def _key(drifter_rank, hours):
    """One sortable int64 per (drifter, hour) pair: drifter in the high 32 bits, hour in the low 32.

    Sorting or searching these keys is the same as sorting by drifter, then by time, which
    lets "this drifter's reading at hour h" be found with a single np.searchsorted.
    """
    return (drifter_rank.astype(np.int64) << 32) | hours


def build_transitions(data, records, tau_days=TAU_DAYS):
    """One row per drifter-day start in R, with the same drifter's position tau days later.

    Inputs:
        data: region readings from load_region().
        records: per-drifter stretches and metadata from load_records().
        tau_days: lag in whole days.
    Returns:
        dict of 1-D arrays, one entry per start: drifter_id, start_time, start_lon, start_lat,
        end_lon, end_lat (NaN unless outcome is IN_R), drogued (drogue attached at the start),
        outcome (see module docstring) and typedeath (how the drifter's record ended).
    """
    # Rank drifters 0..n-1 (IDs are too large to pack into a key directly)
    ids, rank = np.unique(data["drifter_id"], return_inverse=True)
    # Drifter ranks are looked up in records by ID, so both caches must hold the same drifters
    if not np.array_equal(ids, np.unique(records["traj_id"])):
        raise ValueError("Region cache and records cache list different drifters; "
                         "delete data/interim/*_records.npz and rerun.")
    hours = _hours(data["time"])
    keys = _key(rank, hours)
    order = np.argsort(keys, kind="stable")  # the cache is already in this order; don't rely on it
    sorted_keys = keys[order]

    # Start readings: one per drifter per day, at 00:00 UTC. Hourly starts from the same
    # drifter are near-copies of each other and would inflate counts 24x without adding information.
    start = np.flatnonzero(hours % 24 == 0)
    target_hours = hours[start] + 24 * tau_days
    target_keys = _key(rank[start], target_hours)

    # Destination: that drifter's in-R reading at exactly start + tau, if one exists
    pos = np.minimum(np.searchsorted(sorted_keys, target_keys), sorted_keys.size - 1)
    found = sorted_keys[pos] == target_keys
    end = order[pos]  # index of the destination reading in data (only meaningful where found)

    outcome = np.full(start.size, IN_R, dtype=np.int8)
    outcome[~found] = _missing_destination_outcome(records, ids, rank[start[~found]], target_hours[~found])

    end_lon = np.where(found, data["lon"][end], np.nan).astype(np.float32)
    end_lat = np.where(found, data["lat"][end], np.nan).astype(np.float32)

    # typedeath is per drifter; spread it onto each row via the drifter rank
    typedeath_by_rank = np.zeros(ids.size, dtype=np.int8)
    typedeath_by_rank[np.searchsorted(ids, records["traj_id"])] = records["traj_typedeath"]

    return {
        "drifter_id": data["drifter_id"][start],
        "start_time": data["time"][start],
        "start_lon": data["lon"][start].astype(np.float32),
        "start_lat": data["lat"][start].astype(np.float32),
        "end_lon": end_lon,
        "end_lat": end_lat,
        "drogued": data["drogue_status"][start],
        "outcome": outcome,
        "typedeath": typedeath_by_rank[rank[start]],
    }


def _missing_destination_outcome(records, ids, drifter_rank, target_hours):
    """LEFT_R, ENDED or GAP for starts whose drifter has no in-R reading at start + tau.

    Inputs:
        records: from load_records().
        ids: sorted drifter IDs, whose positions are the ranks used in build_transitions().
        drifter_rank, target_hours: the drifter and the hour being looked up, per start.
    """
    seg_rank = np.searchsorted(ids, records["seg_id"])
    seg_start = _hours(records["seg_start"])
    seg_end = _hours(records["seg_end"])
    seg_keys = _key(seg_rank, seg_start)
    seg_order = np.argsort(seg_keys)
    seg_keys = seg_keys[seg_order]

    # The stretch that could contain the target is the last one starting at or before it
    i = np.searchsorted(seg_keys, _key(drifter_rank, target_hours), side="right") - 1
    i_clipped = np.maximum(i, 0)
    same_drifter = (i >= 0) & (seg_rank[seg_order][i_clipped] == drifter_rank)
    reporting = same_drifter & (target_hours <= seg_end[seg_order][i_clipped])

    # Not reporting: after the drifter's last stretch means the record ended, otherwise a gap
    last_end = np.full(ids.size, np.iinfo(np.int64).min)
    np.maximum.at(last_end, seg_rank, seg_end)
    ended = target_hours > last_end[drifter_rank]

    # Reporting, and not from inside R (the caller already looked there), so it left the box
    return np.where(reporting, LEFT_R, np.where(ended, ENDED, GAP)).astype(np.int8)


def transition_counts(table, region, cell_size=None):
    """Count start cell -> end cell transitions on a grid, with "left R" as an extra state.

    Inputs:
        table: from build_transitions() (or loaded as in the module docstring), optionally
               filtered first, e.g. drogued rows only:
               {k: v[table["drogued"]] for k, v in table.items()}.
        region: dict from config/region.yaml.
        cell_size: cell width in degrees (defaults to the working grid in region.yaml).
    Returns:
        (counts, start_drifters):
        counts is (n_cells + 1) x (n_cells + 1); counts[i, j] = transitions from cell i to
        cell j, and the last row/column (index n_cells) is "left R". Cells are numbered as
        in grid.cell_index(). The "left R" row is all zeros, as no start lies outside R.
        start_drifters (length n_cells) is the number of distinct drifters starting a
        counted transition in each cell.
    """
    known = table["outcome"] <= LEFT_R  # ENDED and GAP rows have no destination to count
    start_cell, n_lat_cells, n_lon_cells = cell_index(
        table["start_lon"][known], table["start_lat"][known], region, cell_size
    )
    n_cells = n_lat_cells * n_lon_cells

    end_cell = np.full(start_cell.size, n_cells)  # default: the "left R" state
    in_r = table["outcome"][known] == IN_R
    end_lon = table["end_lon"][known][in_r]
    end_lat = table["end_lat"][known][in_r]
    end_cell_in_r, _, _ = cell_index(end_lon, end_lat, region, cell_size)
    end_cell[in_r] = end_cell_in_r

    # Count each (start, end) pair by flattening it to one index
    n_states = n_cells + 1
    pair = start_cell * n_states + end_cell
    counts = np.bincount(pair, minlength=n_states ** 2).reshape(n_states, n_states)
    _, start_drifters = per_cell_counts(start_cell, table["drifter_id"][known], n_cells)
    return counts, start_drifters


def row_normalise(counts):
    """Naive (maximum-likelihood) transition matrix P from counts: each row divided by its total.

    Rows with no transitions (land, or cells never sampled) are left as all zeros, so they
    are easy to spot and mask. "Left R" is made absorbing (P = 1 of staying there), since
    drifters aren't followed once they leave the box.

    Caution for powers of P: a few counted moves end in a cell that has no row of its own
    (39 of them at 4 degrees, tau = 15 days), so rows of P^n sum to slightly less than 1.
    """
    totals = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    P[-1, -1] = 1.0
    return P


def save_grid_result(path, region, cell_size, tau_days, **arrays):
    """Save arrays from one grid size with what's needed to map cell ids back to positions."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    n_lat_cells, n_lon_cells = len(lat_edges) - 1, len(lon_edges) - 1
    np.savez_compressed(
        path, **arrays, cell_size_deg=cell_size, n_lat_cells=n_lat_cells, n_lon_cells=n_lon_cells,
        tau_days=tau_days, **{k: region[k] for k in BOX_KEYS},
    )


if __name__ == "__main__":
    region = load_region_bounds()
    table = build_transitions(load_region(region), load_records(region), TAU_DAYS)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    table_path = os.path.join(PROCESSED_DIR, f"transitions_tau{TAU_DAYS}d.npz")
    np.savez_compressed(table_path, **table, tau_days=TAU_DAYS)

    print(f"Transition table, tau = {TAU_DAYS} days: {table['outcome'].size:,} starts "
          f"from {np.unique(table['drifter_id']).size:,} drifters")
    for code, name in OUTCOME_NAMES.items():
        print(f"  {name}: {np.mean(table['outcome'] == code):.1%}")
    print(f"Saved {table_path}")

    # Counts on the working grid (input to the Week 6 shrinkage model)
    fine = region["cell_size_deg"]
    counts_fine, drifters_fine = transition_counts(table, region, fine)
    path_fine = os.path.join(PROCESSED_DIR, f"transition_counts_{fine:g}deg_tau{TAU_DAYS}d.npz")
    save_grid_result(path_fine, region, fine, TAU_DAYS, counts=counts_fine, start_drifters=drifters_fine)
    print(f"{fine:g} deg: {counts_fine.sum():,} transitions counted; saved {path_fine}")

    # Naive P on the coarse baseline grid
    coarse = BASELINE_CELL_SIZE
    counts_coarse, drifters_coarse = transition_counts(table, region, coarse)
    P_coarse = row_normalise(counts_coarse)
    path_coarse = os.path.join(PROCESSED_DIR, f"naive_P_{coarse:g}deg_tau{TAU_DAYS}d.npz")
    save_grid_result(path_coarse, region, coarse, TAU_DAYS,
                     P=P_coarse, counts=counts_coarse, start_drifters=drifters_coarse)
    sampled = counts_coarse[:-1].sum(axis=1) > 0
    stay = np.diag(P_coarse)[:-1][sampled]
    print(f"{coarse:g} deg: {sampled.sum()} cells with transitions; median chance of still being in "
          f"the same cell after {TAU_DAYS} days {np.median(stay):.2f}; saved {path_coarse}")
