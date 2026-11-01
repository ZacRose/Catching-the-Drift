"""
Checks for the Week 7 Computational code (transition tables at other lags, src/gridding/lags.py).
Hand-made hourly tracks; tau is 1 or 2 days so the tracks stay short.
Run: python3 -m pytest tests
"""
import numpy as np

from src.gridding.lags import brute_force_check, composition_check, lag_summary
from src.gridding.transitions import ENDED, GAP, IN_R, LEFT_R, build_transitions

BOX = {"lat_min": -42, "lat_max": -4, "lon_min": -60, "lon_max": 20, "cell_size_deg": 2.0}
T0 = np.datetime64("2000-01-01T00", "h")
SW_CORNER = (-59.0, -41.0)  # 2 deg cell 0
NEXT_CELL = (-57.0, -39.0)  # 2 deg cell 41


def make_inputs():
    """Three drifters. Returns (data, records) shaped like load_region() / load_records().

    1: in R for hours 0-48, at the SW corner until hour 47, then in the next cell at hour 48;
       its record ends there.
    2: in R for hours 0-12; its record carries on outside R until hour 72.
    3: in R at the SW corner, with a gap in its record: readings at hours 0-10 and 30-72 only.

    With tau = 1 day the seven starts (one per drifter per 00:00 reading) are:
        drifter 1, hour 0   stays in its cell         drifter 2, hour 0   left R
        drifter 1, hour 24  moves to the next cell    drifter 3, hour 0   gap (hour 24 missing)
        drifter 1, hour 48  record ended              drifter 3, hour 48  stays in its cell
                                                      drifter 3, hour 72  record ended
    """
    tracks = [  # (drifter id, hours in R, position at each of those hours)
        (1, np.arange(49), [SW_CORNER] * 48 + [NEXT_CELL]),
        (2, np.arange(13), [SW_CORNER] * 13),
        (3, np.r_[0:11, 30:73], [SW_CORNER] * 54),
    ]
    segments = [(1, 0, 48), (2, 0, 72), (3, 0, 10), (3, 30, 72)]  # (id, first hour, last hour)
    data = {
        "drifter_id": np.concatenate([np.full(len(h), i) for i, h, _ in tracks]),
        "time": np.concatenate([T0 + h for _, h, _ in tracks]).astype("datetime64[ns]"),
        "lon": np.concatenate([[p[0] for p in pos] for _, _, pos in tracks]),
        "lat": np.concatenate([[p[1] for p in pos] for _, _, pos in tracks]),
    }
    data["drogue_status"] = np.ones(data["lon"].size, dtype=bool)
    records = {
        "traj_id": np.array([1, 2, 3]),
        "traj_typedeath": np.zeros(3, dtype=np.int8),
        "seg_id": np.array([s[0] for s in segments]),
        "seg_start": np.array([T0 + s[1] for s in segments]).astype("datetime64[ns]"),
        "seg_end": np.array([T0 + s[2] for s in segments]).astype("datetime64[ns]"),
    }
    return data, records


def test_tables_at_different_lags_share_their_starts():
    data, records = make_inputs()
    one, two = build_transitions(data, records, 1), build_transitions(data, records, 2)
    assert one["outcome"].size == 7
    assert np.array_equal(one["drifter_id"], two["drifter_id"])
    assert np.array_equal(one["start_time"], two["start_time"])


def test_one_day_then_one_day_matches_the_two_day_table():
    data, records = make_inputs()
    one, two = build_transitions(data, records, 1), build_transitions(data, records, 2)
    result = composition_check(one, one, two, first_days=1)
    # only the three starts still in R after one day can be followed (drifter 1 at hours 0 and 24,
    # drifter 3 at hour 48)
    assert result == {"rows_checked": 3, "mismatches": 0}


def test_composition_check_catches_a_wrong_end_position():
    data, records = make_inputs()
    one, two = build_transitions(data, records, 1), build_transitions(data, records, 2)
    first_row = np.flatnonzero((two["drifter_id"] == 1) & (two["outcome"] == IN_R))[0]
    two["end_lon"][first_row] += 1.0
    assert composition_check(one, one, two, first_days=1)["mismatches"] == 1


def test_brute_force_lookup_agrees_on_every_outcome_and_catches_an_error():
    data, records = make_inputs()
    table = build_transitions(data, records, 1)
    assert sorted(set(table["outcome"].tolist())) == [IN_R, LEFT_R, ENDED, GAP]
    assert brute_force_check(data, records, table, 1) == {"rows_checked": 7, "mismatches": 0}

    table["outcome"][table["outcome"] == GAP] = ENDED  # call the gap a record end
    assert brute_force_check(data, records, table, 1)["mismatches"] == 1


def test_lag_summary_counts_outcomes_same_cell_moves_and_the_end_of_the_data():
    data, records = make_inputs()
    table = build_transitions(data, records, 1)
    ocean = {2.0: np.ones(19 * 40, dtype=bool)}
    # Pretend the dataset stops at hour 72: drifter 3's hour-72 start would end after it
    row = lag_summary(table, 1, BOX, ocean, T0 + np.timedelta64(72, "h"), cell_sizes=(2.0,))
    assert row["starts"] == 7 and row["drifters"] == 3 and row["counted_moves"] == 4
    assert row["pct_stayed_in_R"] == round(100 * 3 / 7, 2)
    assert row["pct_left_R"] == round(100 / 7, 2)
    assert row["pct_record_ended"] == round(100 * 2 / 7, 2)
    assert row["pct_gap_in_record"] == round(100 / 7, 2)
    assert row["pct_past_end_of_data"] == round(100 / 7, 2)
    assert row["pct_same_cell_2deg"] == round(100 * 2 / 3, 1)  # two of the three moves within R stay put
    assert row["pct_adequate_2deg"] == 0.0  # three drifters, far below N_MIN
    assert row["median_km"] == 0.0
