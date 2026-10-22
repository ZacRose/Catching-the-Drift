"""
Checks for the poster trajectory map helpers. Synthetic data only, no download.
Run: python3 -m pytest tests
"""
import numpy as np

from src.gridding.trajectory_map import split_tracks


def test_split_tracks_breaks_at_new_drifter_and_long_gap():
    days = np.array(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-10", "2000-01-11",
                     "2000-01-01", "2000-01-02"], dtype="datetime64[D]")
    drifter_id = np.array([1, 1, 1, 1, 1, 2, 2])
    lon = np.arange(7.0)
    pieces = split_tracks(lon, np.zeros(7), drifter_id, days)
    assert [p[:, 0].tolist() for p in pieces] == [[0, 1, 2], [3, 4], [5, 6]]


def test_split_tracks_drops_single_readings():
    days = np.array(["2000-01-01", "2000-01-05", "2000-01-06"], dtype="datetime64[D]")
    pieces = split_tracks(np.arange(3.0), np.zeros(3), np.array([1, 1, 1]), days)
    assert [p[:, 0].tolist() for p in pieces] == [[1, 2]]
