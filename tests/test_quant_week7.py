"""
Checks for the Week 7 quant code (30-day chaining, Markov check, quasi-stationary distribution,
mixing time, cell areas). Synthetic data only.
Run: python3 -m pytest tests
"""
import numpy as np

from src.gridding.transitions import ENDED, GAP, IN_R, LEFT_R
from src.inference.markov_check import chain_transitions, reentry_summary, row_tv
from src.linalg.long_run import cell_areas_km2, mixing_steps, quasi_stationary

BOX = {"lat_min": -42, "lat_max": -4, "lon_min": -60, "lon_max": 20, "cell_size_deg": 2.0}
DAY0 = np.datetime64("2015-01-01T00:00", "ns")


def raw_table(rows):
    """rows: (drifter, start day, start lon, start lat, end lon, end lat, outcome). Every drifter
    in R at the end of a row has a row starting then, as in the real table (rows starting at
    day 30 end their record, so chaining never needs a day-45 row)."""
    d, day, slon, slat, elon, elat, out = (np.array(c) for c in zip(*rows))
    return {
        "drifter_id": d.astype(np.int64),
        "start_time": DAY0 + day.astype("timedelta64[D]"),
        "start_lon": slon.astype(np.float32), "start_lat": slat.astype(np.float32),
        "end_lon": elon.astype(np.float32), "end_lat": elat.astype(np.float32),
        "drogued": np.zeros(d.size, bool), "outcome": out.astype(np.int8),
    }


NAN = np.nan
TABLE = raw_table([
    # drifter 1: stays in R throughout -> 30-day end is the day-15 row's end
    (1, 0, -10, -30, -11, -31, IN_R), (1, 15, -11, -31, -13, -32, IN_R), (1, 30, -13, -32, NAN, NAN, ENDED),
    # drifter 2: leaves R by day 15, back inside at day 30
    (2, 0, 19, -20, NAN, NAN, LEFT_R), (2, 30, 18, -21, NAN, NAN, ENDED),
    # drifter 3: leaves R by day 15 and is never seen inside again
    (3, 0, 19, -10, NAN, NAN, LEFT_R),
    # drifter 4: in R at day 15, record ends before day 30
    (4, 0, -30, -20, -31, -20, IN_R), (4, 15, -31, -20, NAN, NAN, ENDED),
    # drifter 5: gap at day 15, back in R at day 30
    (5, 0, -40, -35, NAN, NAN, GAP), (5, 30, -41, -36, NAN, NAN, ENDED),
])


def test_chain_follows_drifters_still_in_r():
    out = chain_transitions(TABLE, tau_days=15, steps=2)
    assert out["outcome"][0] == IN_R
    assert (out["end_lon"][0], out["end_lat"][0]) == (-13, -32)
    assert out["outcome"][6] == ENDED  # in R at day 15, record ended before day 30
    assert out["outcome"][8] == IN_R and out["end_lon"][8] == -41  # gap at 15, row at 30


def test_chain_exit_conventions():
    absorbing = chain_transitions(TABLE, tau_days=15, steps=2, reentry="absorbing")
    observed = chain_transitions(TABLE, tau_days=15, steps=2, reentry="observed")
    assert absorbing["outcome"][3] == LEFT_R  # came back, but P^2 can't show that
    assert observed["outcome"][3] == IN_R and observed["end_lon"][3] == 18
    assert absorbing["outcome"][5] == observed["outcome"][5] == LEFT_R  # never came back


def test_reentry_summary_counts_returns():
    summary = reentry_summary(TABLE, tau_days=15, steps=2)
    assert summary == {"left_after_one_step": 2, "back_in_R_at_end": 1, "pct_back": 50.0}


def test_row_tv_is_zero_for_identical_rows_and_one_for_disjoint():
    A = np.array([[1.0, 0.0], [0.5, 0.5]])
    B = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert row_tv(A, B).tolist() == [1.0, 0.0]


def test_quasi_stationary_known_answer():
    Q = np.array([[0.90, 0.05], [0.05, 0.90]])  # symmetric, 5% lost per step from each cell
    lam1, pi, lam2 = quasi_stationary(Q)
    assert np.isclose(lam1, 0.95) and np.allclose(pi, [0.5, 0.5]) and np.isclose(lam2, 0.85)
    Q = np.array([[0.6, 0.3], [0.1, 0.85]])  # asymmetric: check the eigen-equation directly
    lam1, pi, _ = quasi_stationary(Q)
    assert np.allclose(pi @ Q, lam1 * pi) and np.isclose(pi.sum(), 1) and np.all(pi > 0)


def test_mixing_is_one_step_when_every_row_already_matches():
    Q = 0.9 * np.tile([0.2, 0.3, 0.5], (3, 1))
    _, pi, _ = quasi_stationary(Q)
    assert np.allclose(pi, [0.2, 0.3, 0.5])
    assert mixing_steps(Q, pi).tolist() == [1, 1, 1]


def test_cell_areas_add_up_and_shrink_polewards():
    areas = cell_areas_km2(BOX, 2.0).reshape(19, 40)
    R = 6371.0
    box = R ** 2 * np.radians(80) * (np.sin(np.radians(-4)) - np.sin(np.radians(-42)))
    assert np.isclose(areas.sum(), box)
    assert areas[0, 0] < areas[-1, 0]  # southern row smaller than northern row
    # 4 deg grid overhangs the box by half a row: only the part inside counts
    areas4 = cell_areas_km2(BOX, 4.0).reshape(10, 20)
    assert np.isclose(areas4.sum(), box)
