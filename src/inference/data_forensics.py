"""
data_forensics.py

Related task: Week 9 forensic review - checks on the data itself that nothing else in the pipeline makes.
data_quality.py judged speed outliers from the reported velocities only. These checks also ask whether
the positions agree with those velocities, because every result rests on positions.

1. Positions against velocities. The speed implied by consecutive hourly positions is compared with the
   reported velocity, and over 15 days (the step every result uses) the position change is compared with
   the reported velocity added up hour by hour. Moves that span an impossible jump (over MAX_REAL_SPEED,
   above the fastest reported speed) or disagree by over MAX_GAP_KM are then removed, and the headline
   numbers recomputed, so the size of the effect is measured rather than argued.
2. Drogue flags. A drifter can lose its drogue but not regain it, so the flag should only ever go from
   drogued to undrogued. Drogue status is recorded at the start of each move (open issue e), so the share
   of drogued starts that have lost the drogue by the end of the move sizes that limitation.
3. Decision 8. The evidence for not using end_date lived only in the plan and the AI log (open issue f).
   It is recomputed here from the records cache.
4. Where drifters leave. A "left R" row has no end position (issue e), but the drifter's last reading in
   the box before it left does, so the edge each exit crossed can be read off. These counts per start cell
   are saved for connectivity.py. The edge nearest the start cell is compared with the edge really crossed,
   to show what guessing from position alone would get wrong.
5. Basic sanity: starts in cells that are all land, and repeated readings.

Outputs: outputs/results/data_forensics.csv and data/processed/exit_edges_2deg_tau15d.npz
Run: python3 -m src.inference.data_forensics   (under a minute; needs the region and records caches)
"""
import csv
import os
import numpy as np

from src.common.config import PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import cell_index, grid_edges, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, ENDED, GAP, IN_R, LEFT_R, TAU_DAYS, _hours, _key, save_grid_result
from src.ingest.drifter_records import load_records
from src.ingest.load import load_region
from src.inference.bootstrap import shrinkage_fit
from src.inference.estimators import count_matrix
from src.inference.transition_data import TRANSITIONS_PATH, load_transitions, parent_states, to_states
from src.linalg.coastal_release import release_summary
from src.linalg.release_map import propagate

KM_PER_DEG = 111.195
MAX_REAL_SPEED = 4.0  # m/s; the fastest reported speed in R is 3.37 m/s
MAX_GAP_KM = 50.0  # over 15 days; a third of the median move (153 km)
EDGES = ("south", "north", "west", "east")
EXIT_EDGES_PATH = os.path.join(PROCESSED_DIR, f"exit_edges_2deg_tau{TAU_DAYS}d.npz")


def sorted_order(drifter_id, time):
    """Indices that sort readings by drifter, then time."""
    return np.lexsort((time, drifter_id))


def hourly_pairs(lon, lat, drifter_id, time):
    """Consecutive readings of one drifter exactly an hour apart, in sorted order.

    Returns:
        (first, east, north): index of each pair's first reading in the sorted arrays, and the eastward and
        northward speed (m/s) implied by the two positions.
    """
    same = drifter_id[1:] == drifter_id[:-1]
    one_hour = np.diff(_hours(time)) == 1
    first = np.flatnonzero(same & one_hour)
    mid = np.radians(0.5 * (lat[first] + lat[first + 1]))
    east = (lon[first + 1] - lon[first]) * KM_PER_DEG * 1000 * np.cos(mid) / 3600.0
    north = (lat[first + 1] - lat[first]) * KM_PER_DEG * 1000 / 3600.0
    return first, east, north


def run_ids(drifter_id, time):
    """A number per reading, the same along each unbroken hourly stretch of one drifter (sorted arrays)."""
    same = drifter_id[1:] == drifter_id[:-1]
    one_hour = np.diff(_hours(time)) == 1
    return np.concatenate(([0], np.cumsum(~(same & one_hour))))


def window_gaps(lon, lat, ve, vn, drifter_id, time, hours=24 * TAU_DAYS):
    """Position change against reported velocity added up, for windows starting at 00:00 UTC.

    Windows need unbroken hourly readings in R. Arrays must be sorted by drifter, then time.
    Returns:
        (start, gap_km, moved_km): each window's first reading, the distance between the two displacements,
        and how far the position moved.
    """
    run = run_ids(drifter_id, time)
    start = np.flatnonzero(_hours(time) % 24 == 0)
    start = start[start + hours < lon.size]
    start = start[run[start] == run[start + hours]]
    end = start + hours
    east_v = np.cumsum(ve.astype(float) * 3.6)  # km added per hour
    north_v = np.cumsum(vn.astype(float) * 3.6)
    mid = np.radians(0.5 * (lat[start] + lat[end]))
    east_p = (lon[end] - lon[start]) * KM_PER_DEG * np.cos(mid)
    north_p = (lat[end] - lat[start]) * KM_PER_DEG
    gap = np.hypot(east_p - (east_v[end] - east_v[start]), north_p - (north_v[end] - north_v[start]))
    return start, gap, np.hypot(east_p, north_p)


def flag_moves(table, jump_id, jump_time, gap_id, gap_time, tau_days=TAU_DAYS):
    """True for moves that span a position jump, or whose own window disagreed with the velocities."""
    ids, rank = np.unique(table["drifter_id"], return_inverse=True)
    flagged = np.zeros(rank.size, dtype=bool)
    hours = _hours(table["start_time"])
    for d, t in zip(jump_id, _hours(jump_time)):
        k = np.searchsorted(ids, d)
        if k < ids.size and ids[k] == d:
            flagged |= (rank == k) & (hours <= t) & (t < hours + 24 * tau_days)
    keep = np.isin(gap_id, ids)
    bad = _key(np.searchsorted(ids, gap_id[keep]), _hours(gap_time[keep]))
    return flagged | np.isin(_key(rank, hours), bad)


def drogue_regained(drifter_id, drogued):
    """Drifters whose flag goes from undrogued back to drogued, and how many such steps (sorted arrays)."""
    same = drifter_id[1:] == drifter_id[:-1]
    back = same & ~drogued[:-1] & drogued[1:]
    return int(np.unique(drifter_id[:-1][back]).size), int(back.sum())


def drogue_at_end(table, drifter_id, time, drogued, tau_days=TAU_DAYS):
    """For moves ending in R: drogue status at start + tau. Arrays must be sorted by drifter, then time.

    Returns:
        (rows, end_status): the table rows used, and the flag at the end of each.
    """
    ids, rank = np.unique(drifter_id, return_inverse=True)
    keys = _key(rank, _hours(time))
    rows = np.flatnonzero((table["outcome"] == IN_R) & np.isin(table["drifter_id"], ids))
    target = _key(np.searchsorted(ids, table["drifter_id"][rows]), _hours(table["start_time"][rows]) + 24 * tau_days)
    pos = np.minimum(np.searchsorted(keys, target), keys.size - 1)
    found = keys[pos] == target
    return rows[found], drogued[pos[found]]


def decision8_evidence(records, table, drifter_id, time):
    """The plan's case against end_date (decision 8), from the caches and the table.

    Inputs:
        records: load_records(); table: the table as saved; drifter_id, time: the region readings.
    Returns:
        dict of the numbers the plan quotes.
    """
    ids = records["traj_id"]
    end_date = records["traj_end_date"]
    if np.issubdtype(end_date.dtype, np.datetime64):
        missing = np.isnat(end_date)
    else:
        missing = ~np.isfinite(end_date)
    seg_end = _hours(records["seg_end"])
    rank = np.searchsorted(np.sort(ids), records["seg_id"])
    last = np.full(ids.size, np.iinfo(np.int64).min)
    np.maximum.at(last, rank, seg_end)
    order = np.argsort(ids)
    end_hours = np.full(ids.size, np.iinfo(np.int64).min)
    end_hours[~missing[order]] = _hours(end_date[order][~missing[order]])
    match = (~missing[order]) & (np.abs(end_hours - last) <= 1)

    stretches = np.bincount(rank, minlength=ids.size)
    seg_start = _hours(records["seg_start"])
    seg = np.lexsort((seg_start, rank))
    new_gap = rank[seg][1:] == rank[seg][:-1]
    gaps = (seg_start[seg][1:] - seg_end[seg][:-1])[new_gap]

    # The rule decision 8 rejected: no reading in R at start + tau, and end_date says it was still transmitting
    target = _hours(table["start_time"]) + 24 * TAU_DAYS
    row_end = end_hours[np.searchsorted(np.sort(ids), table["drifter_id"])]
    no_reading = table["outcome"] != IN_R
    says_exit = no_reading & (row_end >= target)
    return {
        "drifters": int(ids.size),
        "pct_end_date_within_1h_of_last_reading": 100 * float(match.mean()),
        "drifters_with_no_end_date": int(missing.sum()),
        "pct_drifters_with_gaps": 100 * float(np.mean(stretches > 1)),
        "gaps": int(gaps.size),
        "median_gap_hours": float(np.median(gaps)),
        "rows_end_date_rule_calls_exits": int(says_exit.sum()),
        "of_which_really_gaps": int(np.sum(says_exit & (table["outcome"] == GAP))),
        "of_which_really_ended": int(np.sum(says_exit & (table["outcome"] == ENDED))),
        "real_exits_it_misses": int(np.sum((table["outcome"] == LEFT_R) & ~says_exit)),
        "real_exits": int(np.sum(table["outcome"] == LEFT_R)),
    }


def nearest_edge(lat, lon, region):
    """Index into EDGES of the box edge nearest each position, measured in km (longitude shrinks with latitude)."""
    scale = np.cos(np.radians(lat))
    distance = np.stack([lat - region["lat_min"], region["lat_max"] - lat,
                         (lon - region["lon_min"]) * scale, (region["lon_max"] - lon) * scale])
    return np.argmin(distance, axis=0), np.min(distance, axis=0)


def exit_edges(table, region, cell_size, drifter_id, time, lon, lat):
    """Which edge each exit really crossed, and how often that is the edge nearest its start cell (sorted arrays).

    The crossing is taken from the drifter's last reading before its unbroken stretch in R ends. Exits whose
    stretch ends away from the box edge (a gap in the record first) can't be attributed and are left out.
    Returns:
        (summary, counts): summary is a dict of exits, how many could be attributed, the share of them that
        agree and the share by edge; counts[cell, k] is the number of exits from a start cell through EDGES[k].
    """
    exits = np.flatnonzero(table["outcome"] == LEFT_R)
    ids, rank = np.unique(drifter_id, return_inverse=True)
    keys = _key(rank, _hours(time))
    run = run_ids(drifter_id, time)
    last_of_run = np.searchsorted(run, run, side="right") - 1
    found = _key(np.searchsorted(ids, table["drifter_id"][exits]), _hours(table["start_time"][exits]))
    pos = np.minimum(np.searchsorted(keys, found), keys.size - 1)
    last = last_of_run[pos]
    true_edge, off_edge = nearest_edge(lat[last], lon[last], region)
    usable = off_edge <= 0.5  # degrees
    cell, n_lat, n_lon = cell_index(table["start_lon"][exits].astype(float), table["start_lat"][exits].astype(float),
                                    region, cell_size)
    lat_edges, lon_edges = grid_edges(region, cell_size)
    row, col = np.divmod(cell, n_lon)
    start_edge, _ = nearest_edge(0.5 * (lat_edges[:-1] + lat_edges[1:])[row], 0.5 * (lon_edges[:-1] + lon_edges[1:])[col], region)
    out = {"exits": int(exits.size), "attributed": int(usable.sum()),
           "pct_nearest_edge_is_true_edge": 100 * float(np.mean(start_edge[usable] == true_edge[usable]))}
    for k, name in enumerate(EDGES):
        out[f"pct_exits_through_{name}"] = 100 * float(np.mean(true_edge[usable] == k))
    counts = np.zeros((n_lat * n_lon, len(EDGES)), dtype=int)
    np.add.at(counts, (cell[usable], true_edge[usable]), 1)
    return out, counts


def headline_numbers(table, region, fine, coarse, alpha, zone):
    """Coastal release after 15 days, 3 months and 1 year from a 2 deg shrinkage P fitted to these rows."""
    start, end, n_cells = to_states(table, region, fine)
    P = shrinkage_fit(parent_states(region, fine, coarse), alpha)(count_matrix(start, end, n_cells + 1))
    _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)
    release = zone[drifters[zone] > 0]
    dists = propagate(P, release)
    out = {}
    for step, d in dists.items():
        r = release_summary(d, zone, drifters == 0, region, fine)
        out.update({f"{step * TAU_DAYS}d_pct_left_box": r["pct_left_R"], f"{step * TAU_DAYS}d_pct_in_coastal_zone": r["pct_in_coastal_zone"],
                    f"{step * TAU_DAYS}d_centre_lat": r["centre_lat"], f"{step * TAU_DAYS}d_centre_lon": r["centre_lon"]})
    return out


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    data = load_region(region)
    order = sorted_order(data["drifter_id"], data["time"])
    lon, lat, did, tm = (data[k][order] for k in ("lon", "lat", "drifter_id", "time"))
    ve, vn, drogued = data["ve"][order], data["vn"][order], data["drogue_status"][order]
    stored = dict(np.load(TRANSITIONS_PATH))
    stored.pop("tau_days")
    records = load_records(region)
    rows = []

    def record(check, quantity, value, note=""):
        value = round(float(value), 4) if isinstance(value, (float, np.floating)) else value
        rows.append({"check": check, "quantity": quantity, "value": value, "note": note})
        print(f"{check}: {quantity} = {value} {note}")

    # 1. Positions against velocities
    first, east, north = hourly_pairs(lon, lat, did, tm)
    speed = np.hypot(east, north)
    reported = np.hypot(0.5 * (ve[first] + ve[first + 1]), 0.5 * (vn[first] + vn[first + 1]))
    jumps = first[speed > MAX_REAL_SPEED]
    record("positions", "hourly pairs", int(first.size))
    record("positions", "fastest reported speed (m/s)", float(np.hypot(ve, vn).max()))
    record("positions", "fastest speed implied by positions (m/s)", float(speed.max()))
    record("positions", f"hourly pairs faster than {MAX_REAL_SPEED:g} m/s", int(jumps.size), "impossible: above any reported speed")
    record("positions", "drifters with such a jump", int(np.unique(did[jumps]).size))
    record("positions", "largest jump in one hour (km)", float(speed.max() * 3.6))
    record("positions", "median jump (km)", float(np.median(speed[speed > MAX_REAL_SPEED]) * 3.6))
    record("positions", "median reported speed at a jump (m/s)", float(np.median(reported[speed > MAX_REAL_SPEED])))

    start, gap, moved = window_gaps(lon, lat, ve, vn, did, tm)
    record("positions", "15-day windows with unbroken hourly data", int(start.size))
    record("positions", "median move in those windows (km)", float(np.median(moved)))
    record("positions", "median gap, position change vs velocity added up (km)", float(np.median(gap)))
    record("positions", "99.9th percentile of that gap (km)", float(np.quantile(gap, 0.999)))
    record("positions", "largest gap (km)", float(gap.max()))
    record("positions", f"windows with a gap over {MAX_GAP_KM:g} km", int(np.sum(gap > MAX_GAP_KM)))

    table = load_transitions(region, verbose=False)
    flagged = flag_moves(table, did[jumps], tm[jumps], did[start][gap > MAX_GAP_KM], tm[start][gap > MAX_GAP_KM])
    record("positions", "counted moves affected", int(flagged.sum()), f"of {flagged.size:,}")
    record("positions", "% of counted moves affected", 100 * float(flagged.mean()))
    alpha = float(np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))["alpha"])
    zone = coastal_cells(region, fine, land_fraction(region, fine, land_paths()))
    before = headline_numbers(table, region, fine, coarse, alpha, zone)
    after = headline_numbers({k: v[~flagged] for k, v in table.items()}, region, fine, coarse, alpha, zone)
    for k in before:
        record("positions, effect", f"coastal release {k}: all moves", before[k])
        record("positions, effect", f"coastal release {k}: affected moves removed", after[k])

    # 2. Drogue flags
    regained, steps = drogue_regained(did, drogued)
    record("drogue", "drifters whose flag goes back from undrogued to drogued", regained, f"{steps} such steps")
    used, end_flag = drogue_at_end(stored, did, tm, drogued)
    at_start = stored["drogued"][used]
    record("drogue", "moves ending in R with a drogued start", int(at_start.sum()))
    record("drogue", "% of them undrogued by the end of the move", 100 * float(np.mean(~end_flag[at_start])), "issue e")
    record("drogue", "% of all moves ending in R whose status changes", 100 * float(np.mean(end_flag != at_start)))

    # 3. Decision 8
    for k, v in decision8_evidence(records, stored, did, tm).items():
        record("decision 8", k, v)

    # 4. Where drifters leave
    summary, edge_counts = exit_edges(stored, region, fine, did, tm, lon, lat)
    for k, v in summary.items():
        record("exits", k, v)
    save_grid_result(EXIT_EDGES_PATH, region, fine, TAU_DAYS, counts=edge_counts, edges=np.array(EDGES))

    # 5. Sanity
    land = land_fraction(region, fine, land_paths())
    cell, _, _ = cell_index(stored["start_lon"].astype(float), stored["start_lat"].astype(float), region, fine)
    record("sanity", "starts in a 2 deg cell that is all land on the 1:110m outline", int(np.sum(land[cell] >= 1)))
    same_time = (did[1:] == did[:-1]) & (tm[1:] == tm[:-1])
    record("sanity", "repeated readings (same drifter, same time)", int(same_time.sum()))
    known = stored["outcome"] == IN_R
    record("sanity", "moves ending in R with no end position", int(np.isnan(stored["end_lon"][known]).sum()))
    record("sanity", "other rows with an end position", int(np.isfinite(stored["end_lon"][~known]).sum()))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "data_forensics.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")
