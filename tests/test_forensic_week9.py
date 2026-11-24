"""Week 9 forensic review: the new checks on hand-made data with answers worked out by hand, plus modules that had
no test of their own (drifter_records, adequacy, coastal_release) and the app's Seasons data."""
import csv
import json
import os
import numpy as np
import pytest

from src.app.export_app_data import PATTERN_PATH, trust_checks
from src.common.config import load_region_bounds
from src.forecasting.seasonal_robustness import (
    app_block, fixed_mix_gaps, season_year_sums, shuffled_gap, subset_table, years_higher,
)
from src.gridding.transitions import ENDED, GAP, IN_R, LEFT_R, TAU_DAYS
from src.inference.adequacy import adequacy_summary
from src.inference.bootstrap import drifter_pairs, replicate_count_matrix
from src.inference.calibration import back_in_box, coverage, first_exit_counts, left_box, likely_sets, reliability
from src.inference.data_forensics import (
    decision8_evidence, drogue_at_end, drogue_regained, flag_moves, hourly_pairs, nearest_edge, run_ids, window_gaps,
)
from src.inference.interval_coverage import block_ids, functionals, numbers, simulate
from src.inference.year_intervals import holdout_sets, start_years
from src.ingest.drifter_records import record_segments
from src.linalg.coastal_release import release_summary
from src.linalg.connectivity import (
    absorbing, arrival_days, edge_shares, in_destination, left_by_edge, reached, start_matrix, subregion_map,
)
from src.linalg.product import load_product
from src.run_all import steps

BOX = {"lat_min": -4.0, "lat_max": 4.0, "lon_min": 0.0, "lon_max": 8.0, "cell_size_deg": 2.0}  # 4 x 4 cells
T0 = np.datetime64("2000-01-01T00", "ns")


def hours(*h):
    return T0 + np.array(h) * np.timedelta64(1, "h")


# ---------- drifter records (decides "left the box" against "stopped reporting") ----------
def test_record_segments_split_at_gaps_of_more_than_an_hour():
    time = hours(0, 1, 2, 10, 11, 0, 1)
    seg_traj, first, last = record_segments(time, np.array([5, 2]), np.array([0, 1]))
    assert seg_traj.tolist() == [0, 0, 1]
    assert first.tolist() == hours(0, 10, 0).tolist() and last.tolist() == hours(2, 11, 1).tolist()


def test_adequacy_summary_counts_only_cells_with_readings():
    drifters = np.array([0, 50, 100, 200])
    row = adequacy_summary(drifters, np.array([0, 10, 10, 10]), 2.0)
    assert row["ocean_cells"] == 3 and row["median_drifters"] == 100
    assert row["pct_cells_adequate"] == pytest.approx(66.7, abs=0.05)


def test_release_summary_reports_lost_mass_and_left_box():
    region = {"lat_min": 0.0, "lat_max": 2.0, "lon_min": 0.0, "lon_max": 4.0}  # two 2 deg cells, then "left"
    dist = np.array([0.5, 0.2, 0.2])
    r = release_summary(dist, np.array([1]), np.array([False, True]), region, 2.0)
    assert r["pct_left_R"] == pytest.approx(20) and r["pct_in_coastal_zone"] == pytest.approx(20)
    assert r["pct_in_cells_without_drifters"] == pytest.approx(20) and r["pct_lost_to_rowless_cells"] == pytest.approx(10)
    assert r["centre_lat"] == pytest.approx(1) and r["centre_lon"] == pytest.approx((0.5 * 1 + 0.2 * 3) / 0.7)


# ---------- positions against velocities ----------
def track(lon_step=0.01, n=400):
    """One drifter moving steadily east along the equator: its reported velocity agrees with its positions."""
    lon = 1.0 + lon_step * np.arange(n)
    speed = lon_step * 111.195 * 1000 / 3600
    return lon, np.zeros(n), np.full(n, 7), hours(*range(n)), np.full(n, speed), np.zeros(n)


def test_hourly_pairs_finds_an_impossible_jump_and_skips_gaps():
    lon, lat, did, tm, _, _ = track(n=6)
    lon = lon.copy()
    lon[3:] += 1.0  # 111 km in one hour
    first, east, north = hourly_pairs(lon, lat, did, tm)
    assert first.size == 5 and np.argmax(east) == 2
    assert east[2] == pytest.approx((0.01 + 1.0) * 111195 / 3600, rel=1e-6) and np.allclose(north, 0)
    assert hourly_pairs(lon, lat, did, hours(0, 1, 2, 9, 10, 11))[0].tolist() == [0, 1, 3, 4]


def test_window_gaps_are_zero_when_velocity_matches_and_catch_a_shift():
    lon, lat, did, tm, ve, vn = track()
    start, gap, moved = window_gaps(lon, lat, ve, vn, did, tm)
    assert start.tolist() == [0, 24]  # the 00:00 starts with 360 unbroken hours ahead (400 readings: hours 0 and 24)
    assert gap.max() < 1e-6 and moved[0] == pytest.approx(0.01 * 360 * 111.195, rel=1e-6)
    shifted = lon.copy()
    shifted[200:] += 1.0
    assert window_gaps(shifted, lat, ve, vn, did, tm)[1][0] == pytest.approx(111.195, rel=1e-3)
    assert run_ids(did, tm).max() == 0


def test_flag_moves_marks_windows_spanning_a_jump_and_named_windows():
    table = {"drifter_id": np.array([1, 1, 2]), "start_time": hours(0, 480, 0)}
    jump_id, jump_time = np.array([1]), hours(100)
    flagged = flag_moves(table, jump_id, jump_time, np.array([2]), hours(0))
    assert flagged.tolist() == [True, False, True]  # a 15-day window from hour 0 holds hour 100; one from hour 480 doesn't


def test_drogue_flags_regained_and_status_at_the_end_of_a_move():
    did, drogued = np.array([1, 1, 1, 2, 2]), np.array([True, False, True, True, False])
    assert drogue_regained(did, drogued) == (1, 1)
    tm = hours(0, 360, 720, 0, 360)
    table = {"outcome": np.array([IN_R, IN_R, LEFT_R]), "drifter_id": np.array([1, 2, 2]), "start_time": hours(0, 0, 360)}
    rows, end = drogue_at_end(table, did, tm, drogued)
    assert rows.tolist() == [0, 1] and end.tolist() == [False, False]


def test_decision8_counts_the_rule_it_rejected():
    last = np.datetime64("2000-03-01T00", "ns")
    records = {"traj_id": np.array([1, 2]), "traj_end_date": np.array([last, np.datetime64("NaT", "ns")]),
               "seg_id": np.array([1, 1, 2]), "seg_start": np.array([T0, T0 + np.timedelta64(100, "h"), T0]),
               "seg_end": np.array([T0 + np.timedelta64(80, "h"), last, last])}
    tau = np.timedelta64(TAU_DAYS, "D")
    table = {"drifter_id": np.array([1, 1, 2]), "start_time": np.array([last - tau, T0, T0]),
             "outcome": np.array([GAP, IN_R, LEFT_R])}
    e = decision8_evidence(records, table, None, None)
    assert e["drifters"] == 2 and e["drifters_with_no_end_date"] == 1 and e["gaps"] == 1 and e["median_gap_hours"] == 20
    assert e["pct_end_date_within_1h_of_last_reading"] == 50 and e["pct_drifters_with_gaps"] == 50
    assert e["rows_end_date_rule_calls_exits"] == 1 and e["of_which_really_gaps"] == 1
    assert e["real_exits_it_misses"] == 1  # drifter 2 has no end_date, so the rule can't see its exit


def test_nearest_edge_uses_km_not_degrees():
    edge, distance = nearest_edge(np.array([-3.5, 3.0]), np.array([4.0, 7.5]), BOX)
    assert edge.tolist() == [0, 3] and distance[0] == pytest.approx(0.5)


# ---------- connectivity ----------
def two_box_region():
    region = {**BOX, "subregions": [{"name": "south", "lat_min": -4, "lat_max": 0, "lon_min": 0, "lon_max": 8},
                                    {"name": "north", "lat_min": 0, "lat_max": 4, "lon_min": 0, "lon_max": 8}]}
    return region


def test_subregions_cover_the_box_once_and_the_coast_is_its_own():
    names, label = subregion_map(two_box_region(), 2.0, np.array([0, 15]))
    assert names == ["Benguela coast", "south", "north"]
    assert label.tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0]
    gap = two_box_region()
    gap["subregions"][1]["lat_min"] = 2
    with pytest.raises(ValueError, match="in no subregion"):
        subregion_map(gap, 2.0, np.array([0]))
    both = two_box_region()
    both["subregions"][1]["lat_min"] = -2
    with pytest.raises(ValueError, match="overlaps"):
        subregion_map(both, 2.0, np.array([0]))


def test_committed_subregions_partition_the_real_box():
    region = load_region_bounds()
    names, label = subregion_map(region, region["cell_size_deg"], np.array([0]))
    assert len(names) == len(region["subregions"]) + 1 and label.min() >= 0


def chain():
    """Cells 0, 1, 2 and "left": 0 -> 1 -> 2 in steps, 2 leaks half to "left" and keeps half."""
    return np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.5, 0.5], [0, 0, 0, 1.0]])


def test_in_destination_adds_up_and_reached_counts_a_visit_that_has_passed():
    P, S = chain(), start_matrix([np.array([0])], 4)
    label = np.array([0, 1, 2])
    snapshot = in_destination(S, P, label, 3, steps=(1, 3))
    assert snapshot[1].tolist() == [[0, 1, 0, 0]] and snapshot[3].tolist() == [[0, 0, 0.5, 0.5]]
    assert np.allclose(snapshot[3].sum(axis=1), 1)
    # After 3 steps nothing is in cell 1 any more, but all of it has been through
    assert reached(S, P, np.array([1]), steps=(1, 3))[3][0] == pytest.approx(1)
    assert absorbing(P, np.array([1]))[1].tolist() == [0, 1, 0, 0]
    assert reached(S, P, np.array([2]), steps=(1, 2))[1][0] == 0 and reached(S, P, np.array([2]), steps=(1, 2))[2][0] == 1


def test_arrival_day_is_the_first_step_past_the_level_and_nan_if_never():
    P, S = chain(), start_matrix([np.array([0])], 4)
    assert arrival_days(S, P, np.array([2]), levels=(0.5,), max_steps=5, tau=15).tolist() == [[30.0]]
    Q = np.eye(4)
    assert np.isnan(arrival_days(S, Q, np.array([2]), levels=(0.5,), max_steps=5)).all()


def test_exit_edges_split_by_where_drifters_crossed_and_fall_back_to_the_nearest():
    shares = edge_shares(np.array([[3, 1, 0, 0], [0, 0, 0, 0]]), np.array([0, 3]))
    assert shares.tolist() == [[0.75, 0.25, 0, 0], [0, 0, 0, 1]]
    P = np.array([[0.5, 0, 0.5], [0, 1.0, 0], [0, 0, 1.0]])
    by = left_by_edge(start_matrix([np.array([0])], 3), P, shares, steps=(1, 2))
    assert by[1].tolist() == [[0.375, 0.125, 0, 0]]
    assert by[2].sum() == pytest.approx(0.75)  # half left in step 1, a quarter in step 2


# ---------- whole-year resampling and later years ----------
def test_a_group_of_years_is_resampled_whole():
    start, end, year = np.array([0, 0, 1]), np.array([1, 1, 0]), np.array([2000, 2000, 2001])
    pairs = drifter_pairs(start, end, year, 2)
    assert replicate_count_matrix(pairs, np.array([2, 0])).tolist() == [[0, 4], [0, 0]]  # both 2000 moves, twice; no 2001


def test_later_years_test_drifters_are_never_in_either_training_set():
    ids = np.repeat(np.arange(40), 4)
    first_year = np.where(np.arange(40) < 20, "2010", "2016")
    start = np.array([np.datetime64(f"{first_year[d]}-03-{k + 1:02d}T00", "ns") for d in range(40) for k in range(4)])
    table = {"drifter_id": ids, "start_time": start}
    test, past, control = holdout_sets(table, split_year=2015)
    assert np.unique(ids[test]).size == 10 and start_years(table)[test].min() == 2016
    assert not np.any(test & past) and not np.any(test & control)
    assert start_years(table)[past].max() == 2010 and past.sum() == 80
    assert abs(int(control.sum()) - int(past.sum())) <= 4 and start_years(table)[control].max() == 2016


# ---------- calibration ----------
def test_reliability_matches_a_chain_it_was_sampled_from():
    rng = np.random.default_rng(0)
    P = np.array([[0.6, 0.3, 0.1, 0.0], [0.2, 0.5, 0.25, 0.05], [0.1, 0.1, 0.7, 0.1], [0, 0, 0, 1.0]])
    start = rng.integers(0, 3, 30000)
    end = (P[start].cumsum(axis=1) < rng.random(30000)[:, None]).sum(axis=1)
    rows = reliability(P, start, end, rng.integers(0, 500, 30000), bin_edges=(0.0, 0.3, 0.55, 1.0000001), n_boot=50)
    for r in rows:
        assert r["observed_pct"] == pytest.approx(r["predicted_pct"], abs=1.5)


def test_likely_set_is_the_smallest_holding_the_level_of_material_still_in_the_box():
    P = np.array([[0.1, 0.5, 0.2, 0.2], [0.4, 0.4, 0.0, 0.2], [0, 0, 0, 1.0], [0, 0, 0, 1.0]])
    mask, held = likely_sets(P, 3, 0.5)
    assert mask[0].tolist() == [False, True, False] and held[0] == pytest.approx(0.5 / 0.8)
    assert mask[1].tolist() == [True, False, False] and held[1] == pytest.approx(0.5)
    mask, held = likely_sets(P, 3, 0.9)
    assert mask[0].tolist() == [True, True, True] and held[0] == pytest.approx(1)


def test_coverage_counts_moves_ending_inside_and_skips_exits():
    P = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0, 0, 1.0]])
    start, end = np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 2])
    rows = coverage(P, 2, start, end, np.arange(5), {"all": np.array([True, True])}, levels=(0.5,), n_boot=20)
    assert rows[0]["moves"] == 4 and rows[0]["observed_pct"] == 75


def test_exit_counts_stop_at_the_first_exit_or_unknown_position():
    # Rows are checkpoints, columns starts. Start 0 leaves at checkpoint 2 and comes back: still one exit. Start 1
    # has no known position at checkpoint 1, so it drops out even though it reappears. Start 2 leaves at once.
    grid = np.array([[IN_R, ENDED, LEFT_R], [LEFT_R, IN_R, IN_R], [IN_R, IN_R, LEFT_R]])
    at_risk, left = first_exit_counts(grid)
    assert at_risk.tolist() == [[True, False, True], [True, False, False], [False, False, False]]
    assert left.tolist() == [[False, False, True], [True, False, False], [False, False, False]]


def test_back_in_box_counts_leavers_inside_on_the_last_day():
    # Start 0 leaves and returns; start 1 leaves and stays out; start 2 never leaves; start 3 leaves, then its record ends
    outcomes = np.array([[LEFT_R, LEFT_R, IN_R, LEFT_R], [IN_R, LEFT_R, IN_R, ENDED]])
    assert back_in_box(outcomes) == {"moves": 2, "observed_pct": 50.0}


def test_left_box_is_kaplan_meier_with_censoring():
    """Four starts: one leaves at 1, one is censored at 2, one leaves at 2, one stays. Worked by hand:
    S = (1 - 1/4) x (1 - 1/2) = 0.375, so 62.5% have left, where counting only known rows would say 2/3."""
    P = np.array([[0.5, 0.5], [0, 1.0]])
    outcomes = np.array([[LEFT_R, IN_R, IN_R, IN_R], [IN_R, ENDED, LEFT_R, IN_R]])
    rows = left_box(np.linalg.matrix_power(P, 2), 1, np.zeros(4, dtype=int), outcomes, np.arange(4), n_boot=10)
    assert rows[0]["cells"] == "all starts" and rows[0]["predicted_pct"] == pytest.approx(75)
    assert rows[0]["observed_pct"] == pytest.approx(62.5)


# ---------- known-truth simulation ----------
def test_simulation_keeps_the_design_and_blocks_share_a_destination():
    day = np.datetime64("1970-01-01", "ns") + np.arange(30) * np.timedelta64(1, "D")  # block edges fall on multiples of 15
    start = np.array([0] * 20 + [1] * 10)
    blocks = block_ids(start, np.ones(30, dtype=int), day, block=15)
    assert np.unique(blocks).size == 3  # days 0-14 and 15-19 in cell 0 (a new block number), days 20-29 in cell 1
    P = np.array([[0.2, 0.3, 0.5], [0.5, 0.5, 0.0], [0, 0, 1.0]])
    end = simulate(P, start, blocks, np.random.default_rng(1))
    assert end.size == 30
    for b in np.unique(blocks):
        assert np.unique(end[blocks == b]).size == 1
    assert end[start == 1].max() <= 1  # cell 1 can't reach state 2


def test_simulated_frequencies_match_the_true_row():
    P = np.array([[0.2, 0.3, 0.5], [0, 0, 1.0], [0, 0, 1.0]])
    n = 20000
    end = simulate(P, np.zeros(n, dtype=int), np.arange(n), np.random.default_rng(2))
    assert np.allclose(np.bincount(end, minlength=3) / n, P[0], atol=0.015)


def test_simulation_numbers_step_backwards_like_forwards():
    P = np.array([[0.5, 0.25, 0.25], [0.5, 0.5, 0], [0, 0, 1.0]])
    F0 = functionals(2, np.array([-1.0, 1.0]), np.array([2.0, 4.0]), np.array([1]))
    S = np.array([[1.0, 0, 0]])
    out = numbers(P, F0, S, steps=(2,))
    x = S @ np.linalg.matrix_power(P, 2)
    assert out[("pct_left_box", 2)][0] == pytest.approx(100 * x[0, 2])
    assert out[("pct_in_coastal_zone", 2)][0] == pytest.approx(100 * x[0, 1])
    assert out[("centre_lon", 2)][0] == pytest.approx((x[0, 0] * 2 + x[0, 1] * 4) / x[0, :2].sum())


# ---------- seasonal stress tests ----------
def test_subset_table_renumbers_drifters():
    t = {"rank": np.array([0, 1, 2]), "drifter_id": np.array([10, 20, 30]), "n_drifters": 3, "n_cells": 4}
    out = subset_table(t, np.array([False, True, True]))
    assert out["rank"].tolist() == [0, 1] and out["n_drifters"] == 2 and out["n_cells"] == 4


def test_season_year_sums_and_years_higher():
    sums, counts = season_year_sums(np.array([1.0, 3.0, 5.0]), np.array([2000, 2000, 2001]), np.array([0, 0, 2]))
    assert sums[0, 0] == 4 and counts[0, 0] == 2 and sums[1, 2] == 5
    series = {(2000, 0): 2.0, (2000, 2): 1.0, (2001, 0): 1.0, (2001, 2): 3.0, (2002, 0): 9.0}
    assert years_higher(series, 0, 2) == (1, 2)


def test_fixed_mix_removes_a_difference_that_is_only_the_mix_of_drifters():
    # Drogued moves go 10, undrogued 20, in every season; the seasons differ only in how many of each they hold
    drogued = np.array(([True] * 8 + [False] * 2) * 2 + ([True] * 2 + [False] * 8) * 2)
    season = np.repeat(np.arange(4), 10)
    values = np.where(drogued, 10.0, 20.0)
    assert values[season == 0].mean() != pytest.approx(values[season == 2].mean())
    mix = fixed_mix_gaps(values, np.ones(40, dtype=bool), drogued, season)
    assert np.allclose(mix, 15.0)  # half of each overall


def test_shuffling_seasons_can_not_make_a_real_gap_look_small_every_time():
    rng = np.random.default_rng(0)
    syear = np.repeat(np.arange(2000, 2020), 4)
    season = np.tile(np.arange(4), 20)
    values = np.where(season == 0, 100.0, 50.0) + rng.normal(0, 1, 80)
    t = {"syear": syear, "season": season}
    observed, shuffled = shuffled_gap(values, np.ones(80, dtype=bool), t, 200, rng)
    assert observed == pytest.approx(50, abs=2) and np.quantile(shuffled, 0.95) < observed


def signal(**kw):
    base = {"DJF": 120.0, "MAM": 90.0, "JJA": 70.0, "SON": 120.0, "DJF_low": 110, "DJF_high": 130, "MAM_low": 80, "MAM_high": 100,
            "JJA_low": 60, "JJA_high": 80, "SON_low": 110, "SON_high": 130, "years_djf_higher": 8, "years_djf_compared": 8,
            "djf_minus_jja": 50.0, "djf_jja_low": 30.0, "djf_jja_high": 70.0}
    return {**base, **kw}


def check_block(block):
    assert set(block) == {"summary", "notes"} and len(block["notes"]) == 2  # renderInfo shows both
    assert "forecast for a particular date" in block["notes"][1]


def test_seasons_block_has_the_shape_the_page_shows():
    check_block(app_block(signal(), signal(djf_minus_jja=83.0), signal(djf_minus_jja=24.0), {"mae_annual_km": 33.0, "mae_seasonal_km": 26.0}))


@pytest.mark.skipif(not os.path.exists(PATTERN_PATH), reason="run python3 -m src.forecasting.seasonal_robustness first")
def test_saved_seasons_block_has_the_shape_the_page_shows():
    with open(PATTERN_PATH) as f:
        check_block(json.load(f))


def test_trust_checks_quote_the_results(tmp_path):
    assert trust_checks(str(tmp_path)) == {}
    rows = [{"check": "likely area", "days": d, "cells": "all with drifters", "level_pct": "90.0", "observed_pct": v, "predicted_pct": "", "bin_low": ""} for d, v in (("15", "91.7"), ("90", "89.0"), ("360", "85.4"))]
    rows += [{"check": "having left the box", "days": d, "cells": "all starts", "level_pct": "", "predicted_pct": p, "observed_pct": o, "bin_low": ""} for d, p, o in (("90", "6.6", "5.2"), ("360", "15.4", "10.4"))]
    rows += [{"check": "reliability", "days": "360", "cells": "", "level_pct": "", "predicted_pct": "1.37", "observed_pct": "1.33", "bin_low": "0.01"}]
    with open(tmp_path / f"calibration_tau{TAU_DAYS}d.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(tmp_path / f"year_intervals_tau{TAU_DAYS}d.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["quantity", "days", "width_ratio"])
        w.writeheader()
        w.writerow({"quantity": "centre_lon", "days": "360", "width_ratio": "1.78"})
    q = trust_checks(str(tmp_path))
    assert q["line_90"] == ["92", "89", "85"] and q["year_width"] == "1.8" and q["cell_1y"] == ["1.4", "1.3"]
    assert q["left_box"][1] == "15.4% predicted against 10.4% observed"
    assert "coverage" not in q  # the simulation is optional
    with open(tmp_path / f"interval_coverage_tau{TAU_DAYS}d.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["starts", "pct_covered"])
        w.writeheader()
        for starts, v in (("single cells, 100+ drifters", "90"), ("single cells, 100+ drifters", "94"), ("single cells, 10-99 drifters", "89"),
                          ("single cells, under 10 drifters", "83"), ("whole Benguela coast", "75")):
            w.writerow({"starts": starts, "pct_covered": v})
    assert trust_checks(str(tmp_path))["coverage"] == ["92", "89", "83", "75"]


# ---------- the product and the pipeline ----------
def test_both_grids_load_as_objective_o1_promises():
    fine, coarse = load_product(), load_product(cell_size=4.0)
    assert fine.P.shape == (761, 761) and fine.alpha > 0
    assert coarse.P.shape == (201, 201) and coarse.alpha == 0.0
    assert coarse.se.shape == coarse.P.shape and coarse.cell_of(-28.0, 14.0) != fine.cell_of(-28.0, 14.0)
    assert np.isclose(coarse.forecast(lat=-28.0, lon=14.0, days=15).sum(), 1)
    assert coarse.summary(coarse.forecast(lat=-28.0, lon=14.0, days=360))["pct_left_box"] == pytest.approx(1.3, abs=0.1)  # proposal Figure 1


def test_every_step_of_the_pipeline_is_a_real_module_in_a_workable_order():
    from src.common.config import REPO_ROOT
    order = steps()
    assert len(order) == len(set(order))
    for module in order:
        assert os.path.exists(os.path.join(REPO_ROOT, *module.split(".")) + ".py"), module
    for before, after in (("src.gridding.transitions", "src.inference.split"), ("src.inference.shrinkage_fit", "src.inference.bootstrap"),
                          ("src.inference.bootstrap", "src.linalg.coastal_release"), ("src.gridding.lags", "src.inference.markov_check"),
                          ("src.inference.data_forensics", "src.linalg.connectivity"), ("src.linalg.long_run", "src.inference.year_intervals"),
                          ("src.inference.calibration", "src.app.export_app_data"), ("src.forecasting.seasonal_robustness", "src.app.export_app_data"),
                          ("src.inference.drogue_comparison", "src.forecasting.seasonal_matrices"), ("src.forecasting.seasonal_matrices", "src.app.export_app_data"),
                          ("src.inference.drogue_comparison", "src.inference.interval_coverage"), ("src.app.export_app_data", "src.app.check_intervals")):
        assert order.index(before) < order.index(after), (before, after)
    assert set(steps(from_tables=True)) < set(order)
