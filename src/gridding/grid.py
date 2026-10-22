"""
grid.py

Spatial grid for region R: cell edges, which cell each reading falls in, and
per-cell counts. Shared by the coverage map, the adequacy check, the EDA and
(from Week 3) the transition-matrix counts, so every stage bins identically.

Cells are regular in degrees, so their area shrinks with cos(latitude): per-area results
(e.g. the quasi-stationary distribution read as a density) must divide by cell area. Downstream
code should use only the flat cell id, so a different cell shape (e.g. hexagons) would
need only a new cell_index().
"""
import numpy as np


def grid_edges(region, cell_size=None):
    """Latitude and longitude cell edges for region R.

    Inputs:
        region: dict from config/region.yaml.
        cell_size: cell width in degrees (defaults to the working grid in region.yaml).
    Returns:
        (lat_edges, lon_edges) as increasing arrays. For a box crossing the dateline,
        longitudes run past 180 so the edges stay increasing. If the box isn't a whole
        number of cells wide, the last row/column overhangs the box and is only partly filled.
    """
    step = cell_size or region["cell_size_deg"]
    lon_max = region["lon_max"] + 360 if region["lon_min"] > region["lon_max"] else region["lon_max"]
    lat_edges = np.arange(region["lat_min"], region["lat_max"] + step, step)
    lon_edges = np.arange(region["lon_min"], lon_max + step, step)
    return lat_edges, lon_edges


def cell_index(lon, lat, region, cell_size=None):
    """Flat cell id for every reading (row-major: cell = lat_row * n_lon_cells + lon_col).

    Inputs:
        lon, lat: arrays of in-region positions.
        region, cell_size: as in grid_edges().
    Returns:
        (cell_id, n_lat_cells, n_lon_cells).
    """
    lon_norm = ((lon + 180) % 360) - 180
    if region["lon_min"] > region["lon_max"]:
        lon_norm = np.where(lon_norm < 0, lon_norm + 360, lon_norm)

    lat_edges, lon_edges = grid_edges(region, cell_size)
    n_lat_cells, n_lon_cells = len(lat_edges) - 1, len(lon_edges) - 1

    # clip puts readings sitting exactly on the northern/eastern edge into the last cell
    lat_idx = np.clip(np.digitize(lat, lat_edges) - 1, 0, n_lat_cells - 1)
    lon_idx = np.clip(np.digitize(lon_norm, lon_edges) - 1, 0, n_lon_cells - 1)
    return lat_idx * n_lon_cells + lon_idx, n_lat_cells, n_lon_cells


def per_cell_counts(cell, drifter_id, n_cells):
    """Observations and distinct drifters in every cell.

    Inputs:
        cell: cell id per reading (from cell_index).
        drifter_id: drifter per reading.
        n_cells: total cells in the grid (empty cells get 0).
    Returns:
        (obs_counts, drifter_counts), each an array of length n_cells.
    """
    obs_counts = np.bincount(cell, minlength=n_cells)
    cell_drifter_pairs = np.unique(np.column_stack([cell, drifter_id]), axis=0)
    drifter_counts = np.bincount(cell_drifter_pairs[:, 0], minlength=n_cells)
    return obs_counts, drifter_counts
