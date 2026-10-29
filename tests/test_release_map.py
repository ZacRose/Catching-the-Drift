"""
Checks for the release map's matrix powers on a 2-cell toy matrix (no data needed).
Run: python3 -m pytest tests
"""
import numpy as np

from src.linalg.release_map import propagate

# States: cell 0, cell 1, "left R". Cell 0 keeps half and sends half to cell 1;
# cell 1 sends everything out of R.
P = np.array([
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
])


def test_propagate_matches_hand_calculation():
    dists = propagate(P, start_cell=0, steps=(1, 2, 3))
    assert np.allclose(dists[1], [0.5, 0.5, 0.0])
    assert np.allclose(dists[2], [0.25, 0.25, 0.5])
    assert np.allclose(dists[3], [0.125, 0.125, 0.75])


def test_left_r_is_absorbing_and_mass_is_kept():
    dists = propagate(P, start_cell=1, steps=(1, 10))
    assert np.allclose(dists[1], [0.0, 0.0, 1.0])
    assert np.isclose(dists[10].sum(), 1.0)


def test_rowless_cell_loses_mass():
    # A move into a cell with no row (all zeros) drops out, as the docstring warns
    P_gap = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.isclose(propagate(P_gap, start_cell=0, steps=(2,))[2].sum(), 0.0)
