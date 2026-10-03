"""
Checks for the Week 3 quant code (split, states, estimators, displacement). Synthetic data only.
Run: python3 -m pytest tests
"""
import os
import tempfile
import numpy as np

from src.inference.transition_data import to_states, parent_states
from src.inference.split import split_drifters, set_mask, save_split, load_split
from src.inference.estimators import (
    count_matrix, naive_P, persistence_P, prior_rows, shrinkage_P, log_lik,
)
from src.inference.displacement_eda import displacement_km

BOX = {"lat_min": -42, "lat_max": -4, "lon_min": -60, "lon_max": 20, "cell_size_deg": 2.0}


def make_table(start_lon, start_lat, end_lon, end_lat, exited, drifter_id=None):
    n = len(start_lon)
    return {
        "drifter_id": np.arange(n) if drifter_id is None else np.asarray(drifter_id),
        "start_time": np.full(n, np.datetime64("2015-01-01")),
        "start_lon": np.asarray(start_lon, float), "start_lat": np.asarray(start_lat, float),
        "end_lon": np.asarray(end_lon, float), "end_lat": np.asarray(end_lat, float),
        "drogued": np.zeros(n, bool), "exited": np.asarray(exited, bool),
    }


def test_split_is_by_drifter_disjoint_and_reproducible():
    ids = np.repeat(np.arange(500), 7)  # 7 transitions per drifter
    a, b = split_drifters(ids), split_drifters(ids)
    assert a == b
    sets = {s: {d for d, v in a.items() if v == s} for s in ("train", "valid", "test")}
    assert sum(len(s) for s in sets.values()) == 500
    assert not (sets["train"] & sets["test"]) and not (sets["valid"] & sets["test"])
    assert len(sets["test"]) == 100 and len(sets["valid"]) == 80
    # every transition of a test drifter is in the test mask
    mask = set_mask(ids, a, "test")
    assert mask.sum() == 7 * 100


def test_load_split_refuses_a_table_with_unassigned_drifters():
    path = os.path.join(tempfile.mkdtemp(), "split.csv")
    save_split(split_drifters(np.arange(10)), path)
    assert len(load_split(np.arange(10), path)) == 10
    try:
        load_split(np.arange(11), path)  # drifter 10 is new, e.g. after a rebuild
    except ValueError:
        pass
    else:
        raise AssertionError("a drifter missing from the split should raise")


def test_states_stay_move_exit():
    # stay in the SW cell; move one cell east; exit (NaN end, flagged)
    table = make_table([-59, -59, -59], [-41, -41, -41], [-59, -57, np.nan], [-41, -41, np.nan], [0, 0, 1])
    start, end, n_cells = to_states(table, BOX, 2.0)
    assert n_cells == 19 * 40
    assert start.tolist() == [0, 0, 0]
    assert end.tolist() == [0, 1, n_cells]


def test_parent_of_2deg_is_its_4deg_block():
    parent = parent_states(BOX, 2.0, 4.0)
    n_lon_fine, n_lon_coarse = 40, 20
    # fine cells (row 0, cols 0-1) and (row 1, cols 0-1) all sit in coarse cell 0
    assert {parent[0], parent[1], parent[n_lon_fine], parent[n_lon_fine + 1]} == {0}
    assert parent[2] == 1 and parent[2 * n_lon_fine] == n_lon_coarse
    assert parent[-1] == 10 * 20  # exit -> coarse exit (4 deg grid has 10 x 20 cells)


def test_naive_gives_zero_to_unseen_and_shrinkage_does_not():
    start, end = np.array([0, 0, 1]), np.array([0, 1, 1])
    n_states = 5
    C = count_matrix(start, end, n_states)
    P = naive_P(C)
    assert np.allclose(P[0], [0.5, 0.5, 0, 0, 0])
    assert np.isneginf(log_lik(P, np.array([0]), np.array([2])))[0]

    parent = np.array([0, 0, 1, 1, 2])  # 4 fine cells in 2 coarse cells, plus exit
    S = shrinkage_P(C, prior_rows(C, parent), alpha=10)
    assert np.allclose(S.sum(axis=1), 1)
    assert np.all(S[:4] > 0)
    assert S[-1, -1] == 1  # exit absorbing


def test_shrinkage_tends_to_naive_with_lots_of_data_and_to_prior_with_none():
    rng = np.random.default_rng(0)
    start = rng.integers(0, 4, 20000)
    end = (start + rng.integers(0, 2, 20000)) % 4
    C = count_matrix(start, end, 5)
    prior = prior_rows(C, np.array([0, 0, 1, 1, 2]))
    assert np.allclose(shrinkage_P(C, prior, 1.0)[:4], naive_P(C)[:4], atol=1e-3)
    assert np.allclose(shrinkage_P(np.zeros_like(C), prior, 50.0), prior)


def test_persistence_rows_sum_to_one():
    P = persistence_P(6, 0.1)
    assert np.allclose(P.sum(axis=1), 1) and P[0, 0] == 0.9 and P[-1, -1] == 1


def test_displacement_one_degree_north_is_about_111_km():
    table = make_table([0, 0], [-20, -20], [0, np.nan], [-19, np.nan], [0, 1])
    east, north, dist = displacement_km(table)
    assert abs(north[0] - 111.2) < 0.1 and abs(east[0]) < 1e-9
    assert np.isnan(dist[1])
