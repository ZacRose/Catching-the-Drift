"""
transition_data.py

Related task: Week 3 Quantitative - shared loader for the transition table that
src/gridding/transitions.py (Computational) writes to data/processed/. Every Week 3
quant script reads the table through here, so it is loaded and checked the same way everywhere.

The table (src/gridding/transitions.py) has one row per drifter-day start, with the
same drifter's position exactly tau later and an "outcome" code for each row:
stayed in R, left R, record ended, or gap in record. Only the first two have a known
destination, so this loader keeps those and drops the rest (project_plan.txt, Week 3
step 2), reporting how many it dropped. The outcome codes are imported from
transitions.py itself, so they stay in step if Computational ever renumbers them.

States: cells are numbered by grid.cell_index (0 is the south-west corner, counting east
along each row of cells, then the next row north), and one extra state, index n_cells,
means "left R".
"""
import os
import numpy as np

from src.common.config import PROCESSED_DIR
from src.ingest.load import in_box
from src.gridding.grid import cell_index, grid_edges
from src.gridding.transitions import IN_R, LEFT_R, OUTCOME_NAMES, TAU_DAYS

TRANSITIONS_PATH = os.path.join(PROCESSED_DIR, f"transitions_tau{TAU_DAYS}d.npz")
COLUMNS = ["drifter_id", "start_time", "start_lon", "start_lat", "end_lon", "end_lat", "drogued", "outcome"]


def load_transitions(region, path=TRANSITIONS_PATH, verbose=True):
    """The usable transitions as a dict of 1-D arrays (one entry per transition).

    Keeps rows whose destination is known ("stayed in R" or "left R") and adds a
    boolean "exited" (True = left R). Rows where the record ended or had a gap at
    start + tau are dropped, since where those drifters went is unknown.

    Inputs:
        region: dict from config/region.yaml (used to sanity-check end positions).
        path: the .npz written by src/gridding/transitions.py.
    Returns:
        dict with the COLUMNS names plus "exited"; start_time as datetime64.
    """
    return usable_transitions(load_raw_transitions(path), region, verbose)


def load_raw_transitions(path=TRANSITIONS_PATH):
    """Every row of a saved transition table (all four outcomes), as a dict of the COLUMNS arrays."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Pull the latest data/processed/ from git, or run "
            "python3 -m src.gridding.transitions to build it."
        )
    stored = np.load(path)
    missing = [c for c in COLUMNS if c not in stored.files]
    if missing:
        raise KeyError(f"{os.path.basename(path)} has no {missing} (it has {stored.files}).")
    return {c: stored[c] for c in COLUMNS}


def usable_transitions(table, region, verbose=True):
    """The rows of a raw table with a known destination, cleaned as load_transitions() describes.

    Split out of load_transitions() in Week 7 so a table built in memory (e.g. the 30-day moves
    chained in markov_check.py) is cleaned exactly the same way as one read from disk.
    """
    outcome = table["outcome"]
    if verbose:
        print(f"Transition table: {outcome.size:,} rows")
        for code, name in OUTCOME_NAMES.items():
            kept = "kept" if code in (IN_R, LEFT_R) else "dropped"
            print(f"  {name}: {np.sum(outcome == code):,} ({kept})")
    table = subset(table, np.isin(outcome, [IN_R, LEFT_R]))
    table["exited"] = table["outcome"] == LEFT_R

    if not np.issubdtype(table["start_time"].dtype, np.datetime64):
        table["start_time"] = table["start_time"].astype("datetime64[s]")  # GDP times are seconds since 1970
    table["drogued"] = table["drogued"].astype(bool)
    for k in ("start_lon", "start_lat", "end_lon", "end_lat"):
        table[k] = table[k].astype(np.float64)

    stay = ~table["exited"]
    if not np.all(in_box(table["end_lon"][stay], table["end_lat"][stay], region)):
        raise ValueError("Some 'stayed in R' rows end outside R - check transitions.py.")
    return table


def subset(table, mask):
    """The rows of a transition table where mask is True."""
    return {k: v[mask] for k, v in table.items()}


def to_states(table, region, cell_size):
    """Start and end state of every transition on a given grid.

    grid.cell_index clips out-of-box positions into edge cells, so exits are
    assigned the "left R" state here before gridding, never by cell_index.

    Returns:
        (start_state, end_state, n_cells). The exit state is n_cells.
    """
    if not np.all(in_box(table["start_lon"], table["start_lat"], region)):
        raise ValueError("Some transitions start outside R - check transitions.py.")
    start, n_lat, n_lon = cell_index(table["start_lon"], table["start_lat"], region, cell_size)
    n_cells = n_lat * n_lon

    end = np.full(start.shape, n_cells)
    stay = ~table["exited"]
    end[stay], _, _ = cell_index(table["end_lon"][stay], table["end_lat"][stay], region, cell_size)
    return start, end, n_cells


def parent_states(region, fine_size, coarse_size):
    """For every state of the fine grid, the coarse-grid state that contains it.

    Both grids start at the box's south-west corner (grid_edges), so with an integer
    ratio k a coarse cell is exactly a k x k block of fine cells - e.g. a 4 deg cell is
    the 2x2 block of 2 deg cells used as its shrinkage prior. Exit maps to exit.
    """
    k = coarse_size / fine_size
    if abs(k - round(k)) > 1e-9:
        raise ValueError(f"{coarse_size} deg cells don't nest exactly in {fine_size} deg cells")
    k = int(round(k))
    fine_lat, fine_lon = (len(e) - 1 for e in grid_edges(region, fine_size))
    coarse_lat, coarse_lon = (len(e) - 1 for e in grid_edges(region, coarse_size))

    ids = np.arange(fine_lat * fine_lon)
    row, col = ids // fine_lon, ids % fine_lon
    parent = (row // k) * coarse_lon + (col // k)
    return np.append(parent, coarse_lat * coarse_lon)
