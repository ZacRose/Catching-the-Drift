"""Week 9: the undrogued comparison (src/inference/drogue_comparison.py) and the seasonal matrices
(src/forecasting/seasonal_matrices.py), and the parts of the app's exporter that carry them. Synthetic data only.
Run: python3 -m pytest tests
"""
import csv
import numpy as np

from src.app.check_intervals import read_off
from src.app.export_app_data import (
    LOG16, REACH, drogue_checks, from_log16, functionals, log16, quantities, season_checks, season_walk, start_matrix, step_back,
)
from src.forecasting.seasonal_forecast import season_index
from src.gridding.transitions import TAU_DAYS
from src.forecasting.seasonal_matrices import (
    chain_scores, chain_season, leave_year_out, season_and_year, season_counts, season_matrices, season_replicates, step_seasons,
    westward_by_season,
)
from src.inference.bootstrap import drifter_pairs, replicate_count_matrix, replicate_matrices, resample_counts
from src.inference.drogue_comparison import heldout_rows, paired_replicates, track, undrogued_only
from src.inference.estimators import count_matrix, naive_P
from src.linalg.connectivity import reached, start_matrix as source_matrix


def smoothed(counts):
    """naive P that keeps every row a distribution, with "left" absorbing."""
    c = counts + 0.01
    c[-1] = 0
    c[-1, -1] = 1
    return naive_P(c)


def random_chain(n=9, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((n + 1, n + 1))
    P[:n, n] *= 0.05
    P /= P.sum(axis=1, keepdims=True)
    P[n] = 0
    P[n, n] = 1
    return P


def moves(n_moves=800, n_cells=6, n_drifters=30, seed=2):
    rng = np.random.default_rng(seed)
    start = rng.integers(0, n_cells, n_moves)
    end = rng.integers(0, n_cells + 1, n_moves)
    ids = rng.integers(0, n_drifters, n_moves) * 5 + 2
    season = rng.integers(0, 4, n_moves)
    return start, end, ids, season, n_cells + 1


# ---------- all drifters against undrogued only
def test_undrogued_only_keeps_the_moves_that_start_without_a_drogue():
    table = {"drifter_id": np.array([1, 1, 2]), "drogued": np.array([True, False, False])}
    out = undrogued_only(table)
    assert out["drifter_id"].tolist() == [1, 2] and not out["drogued"].any()


def test_paired_replicates_share_one_draw_of_the_drifters():
    start, end, ids, _, k = moves()
    undrogued = np.random.default_rng(3).random(ids.size) < 0.6
    undrogued[ids == 2] = False  # a drifter with no undrogued moves still takes part in the draw
    paired = list(paired_replicates(start, end, ids, undrogued, k, smoothed, smoothed, n_boot=5, seed=9))
    plain = list(replicate_matrices(start, end, ids, k, smoothed, 5, 9))
    draws = resample_counts(np.unique(ids).size, 5, 9)
    in_order = np.unique(ids)
    for b, (P_all, P_sub) in enumerate(paired):
        assert np.array_equal(P_all, plain[b])  # the first matrix is the ordinary bootstrap's
        weights = draws[b][np.searchsorted(in_order, ids)] * undrogued  # the slow way: weight each subset move by its drifter
        counts = np.bincount(start * k + end, weights=weights, minlength=k * k).reshape(k, k)
        assert np.allclose(P_sub, smoothed(counts))


def test_paired_replicates_with_the_same_moves_give_no_gap():
    start, end, ids, _, k = moves()
    for P_a, P_b in paired_replicates(start, end, ids, np.ones(ids.size, dtype=bool), k, smoothed, smoothed, n_boot=3):
        assert np.allclose(P_a, P_b)


def test_track_matches_stepping_forward_and_first_passage():
    P = random_chain()
    lat, lon = np.linspace(-3, 3, 9), np.linspace(1, 9, 9)
    coast = np.array([2, 5])
    S = source_matrix([np.array([0]), np.array([3, 4])], P.shape[0])
    out = track(S, P, coast, lat, lon, n_steps=6)
    x = S @ np.linalg.matrix_power(P, 6)
    assert np.allclose(out["pct_left"][5], 100 * x[:, 9])
    assert np.allclose(out["pct_coastal"][5], 100 * x[:, coast].sum(axis=1))
    assert np.allclose(out["centre_lon"][5], x[:, :9] @ lon / x[:, :9].sum(axis=1))
    assert np.allclose(out["pct_reached_coast"][5], 100 * reached(S, P, coast, steps=(6,))[6])
    assert np.all(out["pct_reached_coast"][5] >= out["pct_coastal"][5])  # having been there includes being there
    assert np.all(np.diff(out["pct_reached_coast"], axis=0) >= -1e-12)  # and it can only grow


def test_heldout_rows_prefer_the_matrix_that_generated_the_moves():
    """Undrogued moves follow one matrix and drogued moves another, so each matrix must win on its own moves, with an interval."""
    region = {"lat_min": 0.0, "lat_max": 4.0, "lon_min": 0.0, "lon_max": 4.0, "cell_size_deg": 2.0}  # 4 cells
    rng = np.random.default_rng(0)
    P_u = np.array([[.1, .7, .1, .05, .05], [.1, .7, .1, .05, .05], [.1, .7, .1, .05, .05], [.1, .7, .1, .05, .05], [0, 0, 0, 0, 1]])
    P_d = np.array([[.7, .1, .1, .05, .05], [.7, .1, .1, .05, .05], [.7, .1, .1, .05, .05], [.7, .1, .1, .05, .05], [0, 0, 0, 0, 1]])
    centres = {0: (1.0, 1.0), 1: (3.0, 1.0), 2: (1.0, 3.0), 3: (3.0, 3.0)}  # lon, lat of cells numbered from the south-west
    ids, lons, lats, drogued, end_lon, end_lat, exited = [], [], [], [], [], [], []
    for d in range(60):
        is_drogued = d % 2 == 0
        P = P_d if is_drogued else P_u
        for _ in range(20):
            s = int(rng.integers(0, 4))
            e = int(rng.choice(5, p=P[s]))
            ids.append(d); lons.append(centres[s][0]); lats.append(centres[s][1]); drogued.append(is_drogued)
            end_lon.append(centres[e][0] if e < 4 else np.nan); end_lat.append(centres[e][1] if e < 4 else np.nan); exited.append(e == 4)
    table = {"drifter_id": np.array(ids), "start_lon": np.array(lons), "start_lat": np.array(lats), "end_lon": np.array(end_lon),
             "end_lat": np.array(end_lat), "drogued": np.array(drogued), "exited": np.array(exited)}
    assignment = {int(d): "test" for d in np.unique(ids)}
    rows = {r["cells"]: r for r in heldout_rows({"all": P_d, "undrogued": P_u}, table, assignment, region, 2.0, 1, 15, n_boot=200)}
    assert rows["undrogued test moves"]["diff_low"] > 0  # the undrogued matrix wins on undrogued moves
    assert rows["drogued test moves"]["diff_high"] < 0
    assert rows["undrogued test moves"]["moves"] + rows["drogued test moves"]["moves"] == len(ids)


# ---------- seasonal matrices
def test_season_and_year_read_the_start_of_the_move():
    season, year = season_and_year(np.array(["2010-12-31", "2011-01-01", "2011-06-15"], dtype="datetime64[ns]"))
    assert season.tolist() == [0, 0, 2] and year.tolist() == [2010, 2011, 2011]


def test_season_counts_add_up_to_the_all_year_counts():
    start, end, _, season, k = moves()
    counts = season_counts(start, end, season, k)
    assert counts.shape == (4, k, k)
    assert np.array_equal(counts.sum(axis=0), count_matrix(start, end, k))
    assert counts[1].sum() == np.sum(season == 1)


def test_season_matrices_run_from_the_counts_to_the_all_year_row():
    P = random_chain(n=3)
    counts = np.zeros((4, 4, 4))
    counts[0, 0] = [8, 2, 0, 0]      # cell 0 in season 0
    seasons = season_matrices(P, counts, 1e-9)
    assert np.allclose(seasons[0, 0], [0.8, 0.2, 0, 0], atol=1e-6)            # no shrinkage: the season's own counts
    assert np.allclose(season_matrices(P, counts, 1e9)[0, 0], P[0], atol=1e-6)  # huge shrinkage: the all-year row
    seasons = season_matrices(P, counts, 10.0)
    assert np.allclose(seasons[0, 0], (counts[0, 0] + 10 * P[0]) / 20)
    assert np.allclose(seasons[1], P)                                         # a season with no moves is the all-year matrix
    assert np.allclose(seasons.sum(axis=2), 1)
    assert np.allclose(seasons[:, 3], [0, 0, 0, 1])                           # "left" stays absorbing


def test_season_matrices_leave_the_chance_of_running_aground_alone():
    n_water = 4                                                               # 3 cells, left, then 2 coasts
    P = np.zeros((6, 6))
    P[:4, :4] = random_chain(n=3)
    P[0, :4] *= 0.9
    P[0, 4:] = [0.06, 0.04]
    P[4, 4] = P[5, 5] = 1
    counts = np.zeros((4, 6, 6))
    counts[0, 0, 1] = 50
    counts[0, 0, 5] = 7                                                        # groundings don't count towards the water rows
    seasons = season_matrices(P, counts, 10.0, n_water)
    assert np.allclose(seasons[0, 0, 4:], [0.06, 0.04])
    assert np.isclose(seasons[0, 0, :4].sum(), 0.9) and np.allclose(seasons.sum(axis=2), 1)
    assert np.allclose(seasons[0, 0, :4], 0.9 * (counts[0, 0, :4] + 10 * P[0, :4] / 0.9) / 60)
    assert np.allclose(seasons[0, 4:], P[4:])


def test_chains_change_season_every_90_days_and_go_round_the_year():
    assert [chain_season(0, s) for s in (0, 5, 6, 11, 12, 18, 23, 24)] == [0, 0, 1, 1, 2, 3, 3, 0]
    assert chain_season(3, 6) == 0  # spring runs into summer


def test_season_replicates_follow_the_ordinary_bootstrap():
    start, end, ids, season, k = moves()
    plain = list(replicate_matrices(start, end, ids, k, smoothed, 4, 11))
    pairs = drifter_pairs(start, end, ids, k)
    draws = resample_counts(pairs["drifter_ids"].size, 4, 11)
    for b, (P, counts) in enumerate(season_replicates(start, end, season, ids, k, smoothed, 4, 11)):
        assert np.allclose(P, plain[b])
        assert np.array_equal(counts.sum(axis=0), replicate_count_matrix(pairs, draws[b]))
    in_season = season == 2  # one season's counts are the draw's weights on that season's moves alone
    weights = draws[3][np.searchsorted(pairs["drifter_ids"], ids)] * in_season
    assert np.array_equal(counts[2], np.bincount(start * k + end, weights=weights, minlength=k * k).reshape(k, k))


def test_leave_year_out_never_fits_the_year_it_scores():
    start, end, ids, season, k = moves(n_moves=1200)
    year = 2000 + np.random.default_rng(5).integers(0, 4, start.size)
    scored = year == 2001
    scored[np.flatnonzero(scored)[::2]] = False       # half of 2001 is scored; the rest could only get in through a leak
    fit_rows = ~scored
    base = leave_year_out(start, end, season, year, fit_rows, scored, smoothed, (5.0,), k)

    def change(which):
        changed = end.copy()
        changed[which] = (changed[which] + 1) % k
        return leave_year_out(start, changed, season, year, fit_rows, scored, smoothed, (5.0,), k)

    same_year = change(~scored & (year == 2001))
    assert np.array_equal(same_year[0], base[0]) and np.array_equal(same_year[1], base[1])
    other_year = change(year == 2002)                 # the same change in another year must show, or this couldn't catch a leak
    assert not np.array_equal(other_year[0], base[0]) and not np.array_equal(other_year[1], base[1])


def test_leave_year_out_finds_a_season_that_recurs_and_none_where_there_is_none():
    rng = np.random.default_rng(1)
    n = 8000
    start = np.zeros(n, dtype=int)
    season, year = rng.integers(0, 4, n), 2000 + rng.integers(0, 6, n)
    k = 3
    def score(to_one):
        end = (rng.random(n) < to_one).astype(int)
        annual, seasonal = leave_year_out(start, end, season, year, np.ones(n, dtype=bool), np.ones(n, dtype=bool), smoothed, (50.0,), k)
        return (seasonal[0] - annual).mean()
    assert score(np.where(season == 0, 0.9, 0.1)) > 0.1                        # the seasons differ, every year
    assert abs(score(np.full(n, 0.5))) < 0.005                                 # the seasons are the same
    by_year = rng.random(6)[year - 2000]                                       # each year differs, but no season recurs
    assert score(by_year) < 0.005


def test_steps_change_season_by_the_date():
    steps = step_seasons(np.array(["2010-02-20", "2010-11-20", "2010-07-01"], dtype="datetime64[ns]"), 3)
    assert steps[0].tolist() == [0, 1, 1]   # 20 Feb, 7 Mar, 22 Mar
    assert steps[1].tolist() == [3, 0, 0]   # 20 Nov, 5 Dec, 20 Dec
    assert steps[2].tolist() == [2, 2, 2]
    assert steps.shape == (3, 3) and season_index(np.array([12]))[0] == 0


def test_chain_scores_switch_only_when_a_chain_crosses_seasons():
    P = random_chain(n=3, seed=1)
    seasons = np.stack([random_chain(n=3, seed=s + 10) for s in range(4)])
    start, end = np.array([0, 2]), np.array([1, 3])
    when = np.array(["2010-02-20", "2010-07-01"], dtype="datetime64[ns]")   # crosses from summer to autumn; stays in winter
    out = chain_scores(P, seasons, start, end, when, 2)
    assert np.isclose(out["all year"][0], np.log((P @ P)[0, 1]))
    assert np.isclose(out["stuck"][0], np.log((seasons[0] @ seasons[0])[0, 1]))
    assert np.isclose(out["switching"][0], np.log((seasons[0] @ seasons[1])[0, 1]))
    assert np.isclose(out["switching"][1], out["stuck"][1])
    assert np.isclose(out["switching"][1], np.log((seasons[2] @ seasons[2])[2, 3]))


def test_westward_drift_reads_the_row_it_is_given():
    lat_c, lon_c = np.zeros(3), np.array([0.0, 1.0, 2.0])
    row = np.zeros((4, 4, 3))
    row[:, 2] = [1.0, 0.0, 0.0]                  # from cell 2 everything goes two cells west
    row[2, 2] = [0.0, 0.0, 1.0]                  # except in winter, when it stays
    west = westward_by_season(row, np.ones((4, 3)), np.array([2]), lat_c, lon_c)
    assert np.isclose(west[0], 2 * 111.195) and np.isclose(west[2], 0)


# ---------- the exporter's parts
def test_season_walk_matches_stepping_forward_through_the_seasons():
    n, n_shore = 9, 2
    k = n + 1 + n_shore
    seasons = np.zeros((4, k, k))
    for s in range(4):
        seasons[s, :n + 1, :n + 1] = random_chain(n=n, seed=s)
        seasons[s, :n, :n + 1] *= 0.95
        seasons[s, :n, n + 1] = 0.05
        seasons[s, n + 1:, n + 1:] = np.eye(n_shore)
    lat, lon = np.linspace(-3, 3, n), np.linspace(1, 9, n)
    benguela, coast = np.array([0, 1, 5]), np.array([1, 2])
    F0 = functionals(lat, lon, benguela, coast, n_shore)
    S = start_matrix([3, -1], np.array([4, 7]), k)
    steps = [1, 2, 6, 7, 13, 24, 30, 36]
    found = {(first, m): F.copy() for first, m, F in season_walk(seasons, F0, steps)}
    assert set(found) == {(first, m) for first in range(4) for m in steps}
    for (first, m), F in found.items():
        X, H = S.copy(), S.copy()
        for step in range(m):
            P = seasons[chain_season(first, step)]
            Q = P.copy()
            Q[coast] = 0
            Q[coast, coast] = 1
            X, H = X @ P, H @ Q
        got = quantities(S @ F, np.zeros(2), np.zeros(2))
        inside = X[:, :n].sum(axis=1)
        assert np.allclose(got[0], 100 * X[:, n]) and np.allclose(got[2], 100 * X[:, coast].sum(axis=1))
        assert np.allclose(got[3], X[:, :n] @ lat / inside) and np.allclose(got[6], 100 * X[:, n + 1:].sum(axis=1))
        assert np.allclose(got[7], 100 * H[:, coast].sum(axis=1))


def test_step_back_counts_material_that_has_reached_the_coast_only_once():
    P = random_chain()
    coast = np.array([2, 5])
    F0 = functionals(np.linspace(-3, 3, 9), np.linspace(1, 9, 9), np.array([0]), coast)
    S = start_matrix([0, 3], np.array([1]), P.shape[0])
    for m, F in enumerate(step_back(P, F0, 6), start=1):
        assert np.allclose((S @ F)[:, REACH], reached(S, P, coast, steps=(m,))[m])
        assert np.allclose(F[coast, REACH], 1)   # already there


def test_log16_keeps_small_numbers_and_says_which_are_missing():
    values = np.array([0.0, 1e-7, 2e-6, 0.0123, 1.0, 99.5, 12500.0, np.nan])
    back = from_log16(log16(values))
    assert back[0] == 0 and back[1] == 0              # too small to matter
    assert np.isnan(back[-1]) and not np.isnan(back[:-1]).any()
    assert np.allclose(back[2:-1], values[2:-1], rtol=4e-4)
    assert 10 ** LOG16[1] > 12500


def test_the_page_reads_steps_between_stored_ones_off_a_log_line():
    stored = [1, 2, 4, 8]
    values = np.array([1.0, 10.0, 1000.0, 0.0])
    assert read_off(values, stored, 2) == 10.0
    assert np.isclose(read_off(values, stored, 3), 100.0)   # half way on a log scale
    assert np.isclose(read_off(values, stored, 6), 500.0)   # a zero falls back on a straight line


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def test_the_page_quotes_the_undrogued_results(tmp_path):
    assert drogue_checks(str(tmp_path)) == {}
    rows = [{"check": "likely area", "matrix": m, "days": d, "level_pct": "90.0", "observed_pct": v} for m, d, v in
            (("Undrogued only", "15", "91.6"), ("Undrogued only", "90", "90.7"), ("Undrogued only", "360", "90.0"), ("All drifters", "360", "84.7"))]
    rows += [{"check": "likely area", "matrix": "Undrogued only", "days": "360", "level_pct": "50.0", "observed_pct": "1"}]  # not the 90% line
    rows += [{"check": "having left the box", "matrix": "Undrogued only", "days": d, "cells": "all starts", "predicted_pct": p, "observed_pct": o} for d, p, o in (("90", "5.77", "4.79"), ("360", "13.1", "9.7"))]
    rows += [{"check": "held-out score", "days": d, "cells": "undrogued test moves", "difference": v} for d, v in (("15", "-0.0099"), ("90", "0.0633"), ("360", "0.1588"))]
    rows += [{"check": "held-out score", "days": "15", "cells": "drogued test moves", "difference": "-0.36"}]
    write_rows(tmp_path / f"drogue_comparison_tau{TAU_DAYS}d.csv", rows)
    q = drogue_checks(str(tmp_path))["undrogued"]
    assert q["line_90"] == ["92", "91", "90"]
    assert q["left_box"][1] == "13.1% predicted against 9.7% observed"
    assert "-0.010" in q["score"] and "+0.159" in q["score"] and "90%" in q["score"] and "85%" in q["score"]
    assert "coverage" not in q
    write_rows(tmp_path / f"interval_coverage_undrogued_tau{TAU_DAYS}d.csv", [{"starts": s, "pct_covered": v} for s, v in
               (("single cells, 100+ drifters", "90"), ("single cells, 10-99 drifters", "88"), ("single cells, under 10 drifters", "80"), ("whole Benguela coast", "70"))])
    assert drogue_checks(str(tmp_path))["undrogued"]["coverage"] == ["90", "88", "80", "70"]


def test_the_page_quotes_the_seasonal_results(tmp_path):
    assert season_checks(str(tmp_path)) == {}
    rows = [{"check": "signal kept", "set": "all", "pct_of_gap_kept": "34.0"}, {"check": "signal kept", "set": "undrogued", "pct_of_gap_kept": "21.0"},
            {"check": "held-out score", "set": "all", "gain_per_move": "0.0041"}, {"check": "held-out score", "set": "undrogued", "gain_per_move": "0.0057"}]
    rows += [{"check": "crossing seasons", "set": "all", "compare": c, "days": d, "gain_per_move": v} for c, d, v in
             (("switching minus all year", "180", "0.0178"), ("switching minus all year", "360", "0.0102"), ("stuck minus all year", "180", "-0.0087"))]
    write_rows(tmp_path / f"seasonal_matrices_tau{TAU_DAYS}d.csv", rows)
    q = season_checks(str(tmp_path))["seasons"]
    assert q["kept"] == {"all": "34", "undrogued": "21"}
    assert "0.4%" in q["detail"] and "+0.018" in q["detail"] and "-0.009" in q["detail"]
