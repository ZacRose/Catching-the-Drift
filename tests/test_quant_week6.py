"""
Checks for the Week 6 quant code (neighbour-only baseline, effective sample size). Synthetic data only.
Run: python3 -m pytest tests
"""
import numpy as np

from src.inference.estimators import count_matrix, neighbour_P
from src.inference.evaluate import seen_cells
from src.inference.shrinkage_fit import effective_drifters, moves_per_drifter


def test_neighbour_rows_sum_to_one_and_exit_is_absorbing():
    P = neighbour_P(3, 4, np.ones(12, dtype=bool), stay=0.5, leak=0.05)
    assert P.shape == (13, 13)
    assert np.allclose(P.sum(axis=1), 1)
    assert P[-1, -1] == 1 and P[-1, :-1].sum() == 0
    assert np.all(P[:-1] > 0)  # the leak reaches every state, so no held-out move scores -inf


def test_neighbour_shares_move_among_touching_ocean_cells_only():
    # 3 x 3 grid, centre cell 4; cell 0 (a corner) is land
    ocean = np.ones(9, dtype=bool)
    ocean[0] = False
    P = neighbour_P(3, 3, ocean, stay=0.4, leak=0.1)
    assert P[4, 4] == 0.4
    neighbours = [1, 2, 3, 5, 6, 7, 8]  # cell 0 touches 4 but is land, so it only gets leak
    assert np.allclose(P[4, neighbours], 0.5 / 7)
    assert np.isclose(P[4, 0], 0.1 / 2)  # the leak is shared by land cell 0 and the exit
    assert np.isclose(P[4, 9], 0.1 / 2)


def test_neighbour_cell_without_ocean_neighbours_keeps_the_share():
    ocean = np.array([True, False, False, False])  # 2 x 2 grid, only cell 0 is ocean
    P = neighbour_P(2, 2, ocean, stay=0.3, leak=0.1)
    assert np.isclose(P[0, 0], 0.9)
    assert np.allclose(P.sum(axis=1), 1)


def test_seen_cells_counts_starts_and_arrivals_but_not_exits():
    counts = count_matrix(np.array([0, 0]), np.array([2, 3]), 4)  # 0 -> 2, and 0 -> exit (state 3)
    assert seen_cells(counts).tolist() == [True, False, True]


def test_effective_drifters_adds_the_prior_in_drifters():
    counts = np.zeros((4, 4))
    counts[0, 0], counts[1, 1] = 30, 150  # cell 0: 30 moves by 2 drifters; cell 1: 150 by 10
    drifters = np.array([2, 10, 0])
    per_drifter = moves_per_drifter(counts, drifters)
    assert per_drifter == 15
    assert np.allclose(effective_drifters(drifters, 60, per_drifter), [6, 14, 4])
