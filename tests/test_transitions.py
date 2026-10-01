"""
Checks for the transition table and counts, on hand-made hourly tracks (no download).
tau is 1 day here so the tracks stay short; the logic is the same at 15 days.
Run: python3 -m pytest tests
"""
import numpy as np

from src.gridding.transitions import (
    IN_R, LEFT_R, ENDED, GAP, build_transitions, transition_counts, row_normalise,
)

BOX = {"lat_min": -42, "lat_max": -4, "lon_min": -60, "lon_max": 20, "cell_size_deg": 2.0}
N_CELLS = 19 * 40  # 2 degree grid over BOX; the "left R" state is index N_CELLS
T0 = np.datetime64("2000-01-01T00", "h")
SW_CORNER = (-59.0, -41.0)  # cell 0
NEXT_CELL = (-57.0, -39.0)  # one row up and one column right: cell 41


def make_inputs():
    """Four drifters, one per case. Returns (data, records) shaped like load_region() / load_records().

    1 "stay":  in R at the SW corner for hours 0-48.
    2 "move":  at the SW corner for hours 0-23, then in the next cell at hour 24; record ends there.
    3 "exit":  record runs hours 0-48, but only hours 0-12 are inside R.
    4 "gap":   in R, record has a gap: readings at hours 0-10 and 30-60 only.
    """
    tracks = [  # (drifter id, hours in R, position at each of those hours)
        (1, np.arange(49), [SW_CORNER] * 49),
        (2, np.arange(25), [SW_CORNER] * 24 + [NEXT_CELL]),
        (3, np.arange(13), [SW_CORNER] * 13),
        (4, np.r_[0:11, 30:61], [SW_CORNER] * 42),
    ]
    segments = [(1, 0, 48), (2, 0, 24), (3, 0, 48), (4, 0, 10), (4, 30, 60)]  # (id, first hour, last hour)

    data = {
        "drifter_id": np.concatenate([np.full(len(h), i) for i, h, _ in tracks]),
        "time": np.concatenate([T0 + h for _, h, _ in tracks]).astype("datetime64[ns]"),
        "lon": np.concatenate([[p[0] for p in pos] for _, _, pos in tracks]),
        "lat": np.concatenate([[p[1] for p in pos] for _, _, pos in tracks]),
    }
    data["drogue_status"] = data["drifter_id"] != 2  # drifter 2 has lost its drogue
    records = {
        "traj_id": np.array([1, 2, 3, 4]),
        "traj_typedeath": np.array([3, 1, 2, 3], dtype=np.int8),  # drifter 2 ran aground
        "seg_id": np.array([s[0] for s in segments]),
        "seg_start": np.array([T0 + s[1] for s in segments]).astype("datetime64[ns]"),
        "seg_end": np.array([T0 + s[2] for s in segments]).astype("datetime64[ns]"),
    }
    return data, records


def outcomes_by_drifter(table):
    return {int(i): table["outcome"][table["drifter_id"] == i].tolist() for i in np.unique(table["drifter_id"])}


def test_one_start_per_drifter_per_day_at_midnight():
    table = build_transitions(*make_inputs(), tau_days=1)
    hours = table["start_time"].astype("datetime64[h]").astype(np.int64)
    assert np.all(hours % 24 == 0)
    # drifter 1 is in R at hours 0, 24, 48; drifter 2 at 0, 24; drifter 3 at 0; drifter 4 at 0, 48
    assert {i: len(o) for i, o in outcomes_by_drifter(table).items()} == {1: 3, 2: 2, 3: 1, 4: 2}


def test_outcomes_stay_move_exit_end_and_gap():
    table = build_transitions(*make_inputs(), tau_days=1)
    assert outcomes_by_drifter(table) == {
        1: [IN_R, IN_R, ENDED],  # stays; the start at hour 48 has no record at hour 72
        2: [IN_R, ENDED],        # moves one cell; record ends at hour 24
        3: [LEFT_R],             # still reporting at hour 24, but outside R
        4: [GAP, ENDED],         # hour 24 falls in the 11-29 gap; hour 72 is after the record ends
    }


def test_destination_positions_and_per_row_fields():
    table = build_transitions(*make_inputs(), tau_days=1)
    move = (table["drifter_id"] == 2) & (table["outcome"] == IN_R)
    assert (table["end_lon"][move][0], table["end_lat"][move][0]) == NEXT_CELL
    assert np.all(np.isnan(table["end_lon"][table["outcome"] != IN_R]))
    assert np.all(table["typedeath"][table["drifter_id"] == 2] == 1)
    assert not table["drogued"][table["drifter_id"] == 2].any()


def test_counts_include_exit_state_and_skip_unknown_destinations():
    table = build_transitions(*make_inputs(), tau_days=1)
    counts, start_drifters = transition_counts(table, BOX)
    assert counts.shape == (N_CELLS + 1, N_CELLS + 1)
    assert counts[0, 0] == 2          # drifter 1, twice
    assert counts[0, 41] == 1         # drifter 2 moving to the next cell
    assert counts[0, N_CELLS] == 1    # drifter 3 leaving R
    assert counts.sum() == 4          # ENDED and GAP rows are not counted
    assert counts[N_CELLS].sum() == 0  # nothing starts outside R
    assert start_drifters[0] == 3     # drifters 1, 2 and 3 start counted transitions in cell 0


def test_naive_P_rows_sum_to_one():
    counts, _ = transition_counts(build_transitions(*make_inputs(), tau_days=1), BOX)
    P = row_normalise(counts)
    sampled = counts.sum(axis=1) > 0
    assert np.allclose(P[sampled].sum(axis=1), 1)
    assert P[0, 0] == 0.5
    assert P[N_CELLS, N_CELLS] == 1                 # left R is absorbing
    assert np.all(P[:N_CELLS][~sampled[:N_CELLS]] == 0)  # cells without data stay all zero
