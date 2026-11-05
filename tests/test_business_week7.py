"""
Checks for the Business seasonal-forecast code (src/forecasting/seasonal_forecast.py). Synthetic data only.
Run: python3 -m pytest tests
"""
import os
import tempfile
import numpy as np

from src.forecasting.seasonal_forecast import (
    BENCH, KM_PER_DEG, METHODS, broken_at_gaps, empirical_P, group_rows, load_table, matrix_skill, propagate_cycle,
    release_run, rolling_forecasts, season_differences, season_index, season_year_series, shrink, skill_with_ci,
    target_values,
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


def test_moves_weight_the_means_so_a_thin_early_season_counts_less():
    series = {(2000, s): 9.0 for s in range(4)}
    series.update({(y, s): 1.0 for y in (2001, 2002) for s in range(4)})
    moves = {k: (10 if k[0] == 2000 else 990) for k in series}
    r = next(x for x in rolling_forecasts(series, first_test_year=2002, moves=moves) if x["season"] == 0)
    assert np.isclose(r[BENCH], (10 * 9 + 990 * 1) / 1000)
    assert np.isclose(r["Seasonal mean"], (10 * 9 + 990 * 1) / 1000)
    w_old, w_new = 10 * 0.5 ** (2 / 3), 990 * 0.5 ** (1 / 3)
    assert np.isclose(r["Recency-weighted seasonal mean"], (w_old * 9 + w_new * 1) / (w_old + w_new))
    assert r["Last year's season (naive)"] == 1.0
    equal = next(x for x in rolling_forecasts(series, first_test_year=2002) if x["season"] == 0)
    assert np.isclose(equal[BENCH], 5.0) and np.isclose(equal["Seasonal mean"], 5.0)


def synthetic_coastal_table():
    """20 drifters per season of 2000-2003, in the coastal box. DJF moves end north of it, the others don't."""
    ids, times, end_lat = [], [], []
    for y in range(2000, 2004):
        for month, north in (("01", 2.0), ("04", 0.5), ("07", 0.5), ("10", 0.5)):
            for i in range(20):
                ids.append(len(ids))
                times.append(f"{y}-{month}-15")
                end_lat.append(-19.0 + north)
    n = len(ids)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.npz")
        np.savez(path, drifter_id=np.array(ids), start_time=np.array(times, dtype="datetime64[ns]"),
                 start_lon=np.full(n, 10.0), start_lat=np.full(n, -19.0), end_lon=np.full(n, 10.0),
                 end_lat=np.array(end_lat), outcome=np.zeros(n, dtype=int))
        return load_table(path, BOX)


def test_leaving_the_box_northward_is_measured_without_leaving_R():
    t = synthetic_coastal_table()
    values, mask = target_values(t)["leave_north_pct"]
    series, moves = season_year_series(values, mask, t, group_rows(t))
    assert not t["exited"].any()
    assert series[(2000, 0)] == 100.0 and series[(2000, 2)] == 0.0 and moves[(2000, 0)] == 20


def test_season_differences_compare_pooled_means_and_count_years():
    t = synthetic_coastal_table()
    rows = season_differences(t, group_rows(t), 50, np.random.default_rng(0))
    north = next(r for r in rows if r["quantity"] == "north_km" and r["season"] == "DJF" and r["minus"] == "JJA")
    assert np.isclose(north["difference"], 1.5 * KM_PER_DEG, atol=1e-3)
    assert np.isclose(north["ci_low"], north["difference"], atol=1e-3)     # every year is the same, so no spread
    assert north["years_higher"] == 4 and north["years_compared"] == 4


def test_release_alpha_sets_how_far_a_season_moves_from_the_all_season_matrix():
    counts = [np.array([[8.0, 2.0, 0.0], [1.0, 8.0, 1.0], [0.0, 0.0, 0.0]]),
              np.array([[2.0, 8.0, 0.0], [1.0, 8.0, 1.0], [0.0, 0.0, 0.0]])] * 2
    release = np.array([0])
    heavy = release_run(counts, release, None, None, alpha=1e9)
    light = release_run(counts, release, None, None, alpha=1.0)
    assert np.allclose(heavy["DJF"][6], heavy["All seasons"][6], atol=1e-6)
    assert not np.allclose(light["DJF"][6], light["All seasons"][6], atol=1e-3)


def seasonal_chain_table(seasons_differ, n_per_season=600, seed=0):
    """Two states and "left": in season 0 moves go to state 1, otherwise to state 0 (if the seasons differ)."""
    rng = np.random.default_rng(seed)
    rows = {"start": [], "end": [], "rank": [], "syear": [], "season": []}
    for year in range(2003, 2011):
        for s in range(4):
            to_one = 0.9 if (s == 0 or not seasons_differ) else 0.1
            end = (rng.random(n_per_season) < to_one).astype(int)
            rows["start"].append(np.zeros(n_per_season, dtype=int))
            rows["end"].append(end)
            rows["rank"].append(rng.integers(0, 40, n_per_season))
            rows["syear"].append(np.full(n_per_season, year))
            rows["season"].append(np.full(n_per_season, s))
    t = {k: np.concatenate(v) for k, v in rows.items()}
    t["n_drifters"], t["n_cells"] = 40, 2
    return t


def test_matrix_skill_finds_a_real_season_and_no_gain_where_there_is_none():
    t = seasonal_chain_table(True)
    rows = matrix_skill(t, group_rows(t), (10,), 50, np.random.default_rng(0))
    assert next(r for r in rows if r["method"] == "Seasonal")["gain_per_move"] > 0.2
    t = seasonal_chain_table(False)
    rows = matrix_skill(t, group_rows(t), (10,), 50, np.random.default_rng(0))
    assert abs(next(r for r in rows if r["method"] == "Seasonal")["gain_per_move"]) < 0.01


def test_matrix_skill_uses_only_earlier_years():
    """Moves from a later season, too small to be scored itself, would change the scores if any forecast saw them.
    The same moves placed earlier must change them, or this test couldn't catch a leak."""
    t = seasonal_chain_table(True)
    base = matrix_skill(t, group_rows(t), (10,), 20, np.random.default_rng(0))

    def with_extra_moves(year):
        out = {k: (np.concatenate([v, v[:400]]) if isinstance(v, np.ndarray) else v) for k, v in t.items()}
        out["syear"][-400:] = year
        out["end"][-400:] = 1 - out["end"][-400:]
        return matrix_skill(out, group_rows(out), (10,), 20, np.random.default_rng(0))

    assert with_extra_moves(2012) == base
    assert with_extra_moves(2002) != base


def test_gaps_break_the_line_only_where_seasons_are_missing():
    xs, ys = broken_at_gaps([2008.0, 2008.25, 2010.0], [1, 2, 3])
    assert xs[:2] == [2008.0, 2008.25] and np.isnan(xs[2]) and xs[3] == 2010.0
    assert ys[:2] == [1, 2] and np.isnan(ys[2]) and ys[3] == 3
