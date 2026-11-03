"""
Checks for the Business seasonal-forecast code (src/forecasting/seasonal_forecast.py). Synthetic data only.
Run: python3 -m pytest tests
"""
import os
import tempfile
import numpy as np

from src.forecasting.seasonal_forecast import (
    BENCH, METHODS, empirical_P, load_table, propagate_cycle, rolling_forecasts, season_index,
    shrink, skill_with_ci,
)

BOX = {"lat_min": -42, "lat_max": -4, "lon_min": -60, "lon_max": 20, "cell_size_deg": 2.0,
       "coastal_zone": {"lat_min": -34, "lat_max": -18, "lon_min": 0, "lon_max": 20}}


def test_seasons_are_three_consecutive_months_with_december_in_djf():
    months = np.arange(1, 13)
    assert list(season_index(months)) == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0]


def test_december_belongs_to_next_years_djf():
    n = 3
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.npz")
        np.savez(path, drifter_id=np.array([1, 1, 2]),
                 start_time=np.array(["2010-12-31", "2011-01-01", "2011-03-01"], dtype="datetime64[ns]"),
                 start_lon=np.full(n, 5.0), start_lat=np.full(n, -25.0),
                 end_lon=np.full(n, 4.0), end_lat=np.full(n, -24.0),
                 outcome=np.array([0, 0, 2]))                      # third row has no destination: dropped
        t = load_table(path, BOX)
    assert t["start"].size == 2
    assert list(t["syear"]) == [2011, 2011] and list(t["season"]) == [0, 0]


def test_forecast_uses_only_earlier_years():
    series = {(y, s): float(y + 10 * s) for y in range(2000, 2013) for s in range(4)}
    base = {r["season_year"]: r for r in rolling_forecasts(series, first_test_year=2008) if r["season"] == 1}
    changed = dict(series)
    for k in series:
        if k[0] >= 2010:
            changed[k] += 1000.0                                  # the future must not leak into earlier forecasts
    after = {r["season_year"]: r for r in rolling_forecasts(changed, first_test_year=2008) if r["season"] == 1}
    for y in (2008, 2009, 2010):
        for m in METHODS:
            assert base[y][m] == after[y][m]
    prior = [v for (yy, _), v in series.items() if yy < 2008]
    assert np.isclose(base[2008][BENCH], np.mean(prior))
    assert base[2008]["Last year's season (naive)"] == series[(2007, 1)]


def test_shrinkage_limits_and_rows_sum_to_one():
    counts = np.array([[8.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    prior = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8], [0.0, 0.0, 1.0]])
    assert np.allclose(shrink(counts, prior, 1e-9)[0], [0.8, 0.2, 0.0], atol=1e-6)
    assert np.allclose(shrink(counts, prior, 1e9)[0], prior[0], atol=1e-6)
    assert np.allclose(shrink(counts, prior, 5.0)[1], prior[1])    # no data: all prior
    assert np.allclose(shrink(counts, prior, 5.0).sum(axis=1), 1.0)


def test_empty_rows_stay_put_so_exit_is_absorbing():
    P = empirical_P(np.array([[0.0, 3.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(P[0], [0.0, 0.75, 0.25]) and np.allclose(P[1], [0, 1, 0]) and np.allclose(P[2], [0, 0, 1])


def test_release_switches_season_matrix_after_six_steps():
    to_1 = np.array([[0, 1.0, 0], [0, 1.0, 0], [0, 0, 1.0]])       # season 0: state 0 -> 1
    to_2 = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 0, 1.0]])       # season 1: state 1 -> 2
    stay = np.eye(3)
    kept = propagate_cycle([to_1, to_2, stay, stay], 0, np.array([0]), steps=(6, 7))
    assert np.allclose(kept[6], [0, 1, 0])                          # six steps are all in season 0
    assert np.allclose(kept[7], [0, 0, 1])                          # the seventh uses season 1


def test_skill_is_one_for_a_perfect_forecast():
    rows = [{"season_year": 2008 + i // 4, "season": i % 4, "truth": 1.0 + i, BENCH: 5.0,
             "Seasonal mean": 1.0 + i, "Recency-weighted seasonal mean": 5.0, "Last year's season (naive)": 5.0}
            for i in range(12)]
    out = {r["method"]: r for r in skill_with_ci(rows, 50, np.random.default_rng(0))}
    assert np.isclose(out["Seasonal mean"]["skill"], 1.0)
    assert np.isclose(out["Recency-weighted seasonal mean"]["skill"], 0.0)
