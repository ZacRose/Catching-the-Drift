"""Week 8 Quantitative: robustness settings and the region-swap table, on hand-made data."""
import numpy as np
import pytest

from src.gridding.transitions import ENDED, GAP, IN_R, LEFT_R
from src.inference.region_swap import sub_box_table
from src.inference.robustness import YEAR, apply_filter, coarse_of, headline, settings, to_coarse

REGION = {"lat_min": -4.0, "lat_max": 4.0, "lon_min": 0.0, "lon_max": 8.0, "cell_size_deg": 2.0}


def test_sub_box_relabels_moves_that_end_outside():
    box = {"lat_min": -4.0, "lat_max": 0.0, "lon_min": 0.0, "lon_max": 4.0}
    stored = {
        "drifter_id": np.arange(6), "start_lon": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 6.0]),
        "start_lat": np.full(6, -1.0), "end_lon": np.array([2.0, 6.0, np.nan, np.nan, np.nan, 1.0]),
        "end_lat": np.array([-2.0, -1.0, np.nan, np.nan, np.nan, -1.0]),
        "outcome": np.array([IN_R, IN_R, LEFT_R, ENDED, GAP, IN_R]), "tau_days": 15,
    }
    out = sub_box_table(stored, box)
    assert out["drifter_id"].tolist() == [0, 1, 2, 3, 4]  # the start outside the box is dropped
    assert out["outcome"].tolist() == [IN_R, LEFT_R, LEFT_R, ENDED, GAP]  # ends outside -> left the box
    assert np.isnan(out["end_lon"][1]) and out["end_lon"][0] == 2.0
    assert stored["outcome"][1] == IN_R  # the input isn't modified
    assert out["tau_days"] == 15


def test_filters_and_settings():
    table = {"drogued": np.array([True, False, True]),
             "start_time": np.array(["2005-01-01", "2010-01-01", "2015-06-01"], dtype="datetime64[s]"),
             "x": np.arange(3)}
    assert apply_filter(table, "undrogued")["x"].tolist() == [1]
    assert apply_filter(table, "drogued")["x"].tolist() == [0, 2]
    assert apply_filter(table, "2010+")["x"].tolist() == [1, 2]
    assert apply_filter(table, "all") is table
    with pytest.raises(ValueError):
        apply_filter(table, "winter")
    assert YEAR == np.datetime64("2010-01-01")
    s = settings()
    assert s[0][0] == "base" and s[0][2:6] == (15, 2.0, "shrinkage", "all")
    baseline = (15, 4.0, "naive", "all")  # the 4 deg settings change one thing from the 4 deg baseline instead
    for x in [x for x in s[1:] if x[2:] != baseline]:
        assert min(sum(a != b for a, b in zip(x[2:], ref)) for ref in (s[0][2:], baseline)) == 1, x
    assert coarse_of(1.0) == 2.0 and coarse_of(2.0) == 4.0


def test_to_coarse_adds_children():
    keep = np.array([0, 1, 4, 5, 2])  # 2 deg cells on a 4 x 4 grid; 0, 1, 4, 5 make up 4 deg cell 0
    q = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
    out = to_coarse(q, keep, REGION, 2.0)
    assert out.size == 4 and np.isclose(out[0], 0.7) and np.isclose(out[1], 0.3) and np.isclose(out.sum(), 1)


def test_headline_skips_horizons_that_are_not_whole_steps():
    n = 16  # 2 deg cells of REGION
    P = np.zeros((n + 1, n + 1))
    P[np.arange(n), np.arange(n)] = 0.9
    P[np.arange(n), (np.arange(n) + 1) % n] = 0.09
    P[:n, n] = 0.01
    P[n, n] = 1.0
    keep = np.arange(n)
    out, q = headline(P, 60, keep, np.array([0]), np.array([0, 1]), np.zeros(n, dtype=bool), np.ones(n),
                      REGION, 2.0)
    assert np.isnan(out["pct_left_30d"]) and np.isnan(out["pct_left_90d"])  # 30 and 90 aren't multiples of 60
    assert np.isclose(out["pct_left_360d"], 100 * (1 - 0.99 ** 6))  # the same 1% leaves every step
    assert np.isclose(out["years_in_box"], 60 / 365.25 / 0.01)
    assert np.allclose(q, 1 / n)
