"""
Checks for the region/grid helpers every stage depends on. Synthetic data only, no download.
Run: python3 -m pytest tests
"""
import numpy as np

from src.ingest.load import in_box
from src.gridding.grid import grid_edges, cell_index, per_cell_counts

BOX = {"lat_min": -42, "lat_max": -4, "lon_min": -60, "lon_max": 20, "cell_size_deg": 2.0}
DATELINE_BOX = {"lat_min": -10, "lat_max": 10, "lon_min": 170, "lon_max": -170, "cell_size_deg": 5.0}


def test_in_box_handles_both_longitude_conventions():
    lon = np.array([-59.0, 301.0, 299.0, 21.0])  # 301E = 59W (inside), 299E = 61W (outside)
    lat = np.full(4, -20.0)
    assert in_box(lon, lat, BOX).tolist() == [True, True, False, False]


def test_in_box_across_dateline():
    lon = np.array([175.0, -175.0, 0.0])
    lat = np.zeros(3)
    assert in_box(lon, lat, DATELINE_BOX).tolist() == [True, True, False]


def test_grid_edges_cover_box_exactly():
    lat_edges, lon_edges = grid_edges(BOX)
    assert (lat_edges[0], lat_edges[-1], len(lat_edges) - 1) == (-42, -4, 19)
    assert (lon_edges[0], lon_edges[-1], len(lon_edges) - 1) == (-60, 20, 40)


def test_cell_index_corners_and_edges():
    # south-west corner, a point inside the second row/column, and the north-east corner
    lon = np.array([-60.0, -57.0, 20.0])
    lat = np.array([-42.0, -39.0, -4.0])
    cell, n_lat, n_lon = cell_index(lon, lat, BOX)
    assert (n_lat, n_lon) == (19, 40)
    assert cell.tolist() == [0, 1 * 40 + 1, n_lat * n_lon - 1]


def test_cell_index_across_dateline_keeps_cells_contiguous():
    lon = np.array([171.0, 176.0, -174.0])
    cell, _, n_lon = cell_index(lon, np.zeros(3), DATELINE_BOX)
    assert n_lon == 4
    assert (cell % n_lon).tolist() == [0, 1, 3]


def test_per_cell_counts_counts_each_drifter_once():
    cell = np.array([0, 0, 0, 2, 2])
    drifter = np.array([7, 7, 8, 7, 7])
    obs, drifters = per_cell_counts(cell, drifter, n_cells=3)
    assert obs.tolist() == [3, 0, 2]
    assert drifters.tolist() == [2, 0, 1]
