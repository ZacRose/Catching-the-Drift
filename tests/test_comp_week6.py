"""
Checks for the Week 6 Computational code (drifter bootstrap, coastal cells, coastal release).
Synthetic data only.
Run: python3 -m pytest tests
"""
import numpy as np
from matplotlib.path import Path

from src.gridding.coast import coastal_cells, land_fraction, outcome_shares
from src.gridding.transitions import ENDED, GAP, IN_R, LEFT_R
from src.inference.bootstrap import (
    drifter_pairs, drifter_totals, matrix_intervals, percentile_interval, replicate_count_matrix,
    replicate_matrices, resample_counts, resampled_means,
)
from src.inference.estimators import count_matrix, naive_P
from src.linalg.release_map import propagate

# Three drifters on a 3-state chain (2 cells + "left R"): 10 and 20 start in cell 0, 30 in cell 1.
START = np.array([0, 0, 0, 0, 1, 1, 1])
END = np.array([1, 1, 0, 1, 2, 1, 2])
DRIFTER = np.array([10, 10, 20, 20, 30, 30, 30])


def test_resample_counts_draws_every_drifter_count_with_replacement():
    draws = resample_counts(5, n_boot=200, seed=1)
    assert draws.shape == (200, 5)
    assert np.all(draws.sum(axis=1) == 5)
    assert np.array_equal(draws, resample_counts(5, n_boot=200, seed=1))  # fixed seed, same replicates
    assert (draws == 0).any() and (draws >= 2).any()  # some drifters left out, some drawn twice


def test_replicate_counts_match_the_data_when_each_drifter_is_drawn_once():
    pairs = drifter_pairs(START, END, DRIFTER, 3)
    assert np.array_equal(replicate_count_matrix(pairs, np.ones(3, dtype=int)), count_matrix(START, END, 3))


def test_replicate_counts_repeat_a_drifter_drawn_twice_and_drop_one_not_drawn():
    pairs = drifter_pairs(START, END, DRIFTER, 3)
    counts = replicate_count_matrix(pairs, np.array([2, 0, 1]))  # drifter 10 twice, 20 never, 30 once
    expected = 2 * count_matrix(START[:2], END[:2], 3) + count_matrix(START[4:], END[4:], 3)
    assert np.array_equal(counts, expected)


def test_matrix_intervals_are_the_same_in_one_block_or_many():
    P_hat = naive_P(count_matrix(START, END, 3))
    one = matrix_intervals(START, END, DRIFTER, 3, naive_P, P_hat, n_boot=300, seed=2)
    many = matrix_intervals(START, END, DRIFTER, 3, naive_P, P_hat, n_boot=300, seed=2, max_block_bytes=1)
    for key in ("se", "ci_low", "ci_high", "row_tv", "row_empty_share"):
        assert np.allclose(one[key], many[key], equal_nan=True)


def test_matrix_intervals_agree_with_the_replicates_and_skip_empty_rows():
    P_hat = naive_P(count_matrix(START, END, 3))
    result = matrix_intervals(START, END, DRIFTER, 3, naive_P, P_hat, n_boot=400, seed=3)
    stack = np.array(list(replicate_matrices(START, END, DRIFTER, 3, naive_P, n_boot=400, seed=3)))
    # cell 1 has one drifter: replicates that leave it out give naive P an empty row, which is skipped
    empty = stack[:, 1].sum(axis=1) == 0
    assert np.isclose(result["row_empty_share"][1], empty.mean()) and 0 < empty.mean() < 1
    assert np.isclose(result["se"][1, 1], stack[~empty, 1, 1].std(ddof=1))
    empty0 = stack[:, 0].sum(axis=1) == 0  # cell 0 is empty only when drifter 30 is drawn three times
    assert np.isclose(result["se"][0, 1], stack[~empty0, 0, 1].std(ddof=1))
    # cell 1's row is identical whenever its only drifter is drawn: no spread (the documented limitation)
    assert np.isclose(np.nanmax(result["row_tv"][:, 1]), 0)
    assert result["se"][2, 2] == 0  # "left R" is absorbing in every replicate


def test_resampled_means_reweight_drifters():
    values = np.array([1.0, 3.0, 5.0, 2.0, 2.0, 2.0, 2.0])
    totals, moves = drifter_totals(values, DRIFTER)
    assert np.array_equal(totals, [4.0, 7.0, 6.0]) and np.array_equal(moves, [2, 2, 3])
    draws = np.array([[1, 1, 1], [3, 0, 0], [0, 1, 2]])
    assert np.allclose(resampled_means(totals, moves, draws), [values.mean(), 2.0, 19 / 8])
    # a mask keeps only some moves (e.g. one cell group)
    totals_m, moves_m = drifter_totals(values, DRIFTER, mask=START == 0)
    assert np.array_equal(moves_m, [2, 2, 0]) and np.array_equal(totals_m, [4.0, 7.0, 0.0])


def test_percentile_interval_ignores_nan():
    low, high = percentile_interval(np.array([np.nan, *np.arange(101.0)]), level=0.9)
    assert np.isclose(low, 5) and np.isclose(high, 95)


# A 4 x 4 degree box of 2 deg cells (0 = south-west, 1 = south-east, 2, 3 = the row above). Land
# east of lon 3.05, so the eastern cells are part land; edges avoid lattice points.
REGION = {"lat_min": 0, "lat_max": 4, "lon_min": 0, "lon_max": 4, "cell_size_deg": 2.0,
          "coastal_zone": {"lat_min": 0, "lat_max": 4, "lon_min": 0, "lon_max": 4}}
COAST = [Path(np.array([[3.05, -1.05], [10.05, -1.05], [10.05, 5.05], [3.05, 5.05], [3.05, -1.05]]))]


def test_land_fraction_and_coastal_cells():
    land = land_fraction(REGION, 2.0, COAST)
    assert np.allclose(land[[0, 2]], 0)
    assert np.all((land[[1, 3]] > 0.4) & (land[[1, 3]] < 0.6))
    assert np.array_equal(coastal_cells(REGION, 2.0, land), [1, 3])
    # the band limits which coastal cells count
    south_only = {**REGION, "coastal_zone": {"lat_min": 0, "lat_max": 2, "lon_min": 0, "lon_max": 4}}
    assert np.array_equal(coastal_cells(south_only, 2.0, land), [1])
    # a cell that is all land is not coastal
    all_land = [Path(np.array([[-1.05, -1.05], [10.05, -1.05], [10.05, 5.05], [-1.05, 5.05], [-1.05, -1.05]]))]
    assert coastal_cells(REGION, 2.0, land_fraction(REGION, 2.0, all_land)).size == 0


def test_outcome_shares_split_coastal_starts_from_the_rest():
    table = {"start_lon": np.array([3.0, 3.0, 3.0, 3.0, 1.0, 1.0]),
             "start_lat": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
             "outcome": np.array([IN_R, ENDED, ENDED, GAP, LEFT_R, ENDED]),
             "typedeath": np.array([1, 1, 3, 1, 0, 1]),
             "drifter_id": np.array([1, 1, 2, 2, 3, 3])}
    coastal, rest = outcome_shares(table, REGION, 2.0, np.array([1]))
    assert coastal["starts"] == 4 and coastal["drifters"] == 2
    assert coastal["pct_record_ended"] == 50 and coastal["pct_gap_in_record"] == 25
    assert coastal["pct_ended_and_ran_aground"] == 25 and coastal["drifters_ended_and_ran_aground"] == 1
    assert rest["starts"] == 2 and rest["pct_left_R"] == 50 and rest["pct_ended_and_ran_aground"] == 50


def test_release_over_several_cells_is_spread_evenly():
    P = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 0.0, 1.0]])
    assert np.allclose(propagate(P, [0, 1], steps=(1,))[1], [0.5, 0.0, 0.5])
    assert np.allclose(propagate(P, 0, steps=(1,))[1], [1.0, 0.0, 0.0])  # a single cell works as before
