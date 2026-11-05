"""
export_app_data.py

Related task: Week 9 - the Drift Explorer app (app/index.html). A user picks a start position and a
timeframe and sees where material from there is likely to be, with the chance it is in the Benguela
zone, near the Benguela coast, has left the box or (optionally) has run aground, each with a 95%
interval. This packs the matrices and their drifter bootstraps into one JavaScript file the page loads, so
the app runs in any browser with no Python, server or internet.

Which matrix the page uses depends on three switches, and every combination is shipped:
- drifters: "all" is the product, the 2 deg shrinkage P (761 x 761), what the report quotes. "undrogued" is the same
  fit on moves that start without a drogue (src.inference.drogue_comparison).
- groundings: water is the matrix itself. Starts whose record ended are left out (decision 10), so material only
  drifts or leaves the box. shore adds running aground. GDP marks a drifter that ran aground (typedeath 1), and its
  starts in the last 15 days of its record are "record ended" rows. Those starts become moves into an absorbing
  "aground" state for the stretch of coast where the drifter came ashore (groundings()). For start cell i, with n_i
  known-destination moves and g_ik groundings on coast k, the grounding chance is shrunk towards the cell's 4 deg
  parent exactly as the product's rows are (the same alpha, empirical Bayes; estimators.py):
      P_shore[i, k] = (g_ik + alpha * q_k) / (n_i + g_i + alpha),  P_shore[i, j] = P[i, j] * (1 - sum_k P_shore[i, k])
  where q_k is the parent's share of moves that ran aground on coast k (its four cells pooled). So the
  water part is P scaled by the chance of staying afloat for 15 days. A cell whose few
  records all end aground no longer sends everything ashore, and a cell with no groundings of its own
  gets its neighbourhood's (small) chance. The bootstrap intervals show how rough the estimates are.
- release season: all year, or the first day of a season, with the matrix changing every 90 days as the seasons pass
  (src.forecasting.seasonal_matrices, which explains and tests this). The seasons' grounding chance stays the all-year one.

What goes in app/data/app_data.js (window.DRIFT_DATA):
- sets.all and sets.undrogued, each with:
  - P (float32, base64); the shore model's grounding rows as [cell, coast, probability] triples. The page
    steps any model forward itself (x @ P per 15-day step, as Product.forecast).
  - seasonal: beta and each season's counts. P^s is (n^s + beta P) / (n^s_i + beta), so the page rebuilds the seasons'
    matrices from these and P, instead of carrying four more dense matrices.
  - intervals per model: 95% bootstrap intervals for SHIPPED, from every start cell with drifters and from the whole
    Benguela coast release. All year: at every 15-day step up to MAX_STEPS, 1000 replicates with bootstrap.py's seed,
    so for the product the coast release reproduces coastal_release_tau15d.csv (checked below). Seasons: at
    SEASON_STEPS only, N_BOOT_SEASON replicates, as there are 16 sets of them; the page interpolates the steps between.
    Stored as 16 bits on a log scale (under 0.04% rounding). The estimates themselves are computed by the page.
    src/app/check_intervals.py re-derives a sample of them by brute force, and measures the interpolation.
- the grid, which cells are all land (for the page's contours), the box, the Benguela zone
  (coastal_zone in config/region.yaml) and its coastal cells, the subregions and named releases (connectivity.py),
  drifters per cell, the land outline.
- forecast: the Seasons view's block if FORECAST_PATH exists (src.forecasting.seasonal_robustness), else null.
- meta.checks: the numbers behind the page's "How far to trust it" notes (trust_checks), read from the results.

Every quantity is linear in the distribution, or (the centre) a ratio of linear ones, so for all start
cells at once they come from stepping P backwards on eight vectors, F_n = P F_(n-1): row i of F_n is
start cell i's value after n steps. 8 matrix-vector products per step instead of 596 forward runs. Having reached the
coast isn't linear in this way, but its vector is first-passage's own recursion: F_n = P F_(n-1), then 1 at coastal
cells. With seasons, a chain's steps aren't all the same matrix, so each is walked back from its last step (season_walk).

The Seasons view (Business, RQ4): src.forecasting.seasonal_robustness saves a JSON file at FORECAST_PATH, and
rerunning this script puts it in the page, which draws it as a chart with intervals. The expected shape is
    {"title": str, "summary": str, "quantity": str, "unit": str,
     "series": [{"label": str, "x": [str, ...], "y": [float, ...],
                 "low": [float, ...] (optional), "high": [float, ...] (optional), "kind": "observed" | "forecast"}],
     "notes": [str, ...]}

Run: python3 -m src.app.export_app_data   (about 25 minutes; needs data/processed/transitions_tau15d.npz,
shrinkage_P_2deg_tau15d.npz from src.inference.shrinkage_fit, the undrogued matrix from src.inference.drogue_comparison,
seasonal_counts_2deg_tau15d.npz from src.forecasting.seasonal_matrices and data/external/ne_110m_land.geojson)
"""
import base64
import csv
import datetime
import json
import os
import numpy as np

from src.common.config import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR, load_region_bounds
from src.forecasting.seasonal_forecast import SEASONS, STEPS_PER_SEASON
from src.forecasting.seasonal_matrices import COUNTS_PATH, N_BOOT_SEASON, SEASON_NAMES, season_and_year, season_counts, season_matrices, \
    season_replicates
from src.gridding.coast import RAN_AGROUND, coastal_cells, land_fraction, land_paths
from src.gridding.grid import cell_index, grid_edges, per_cell_counts
from src.gridding.trajectory_map import LAND_PATH, MAP_EXTENT
from src.gridding.transitions import BASELINE_CELL_SIZE, ENDED
from src.inference.bootstrap import LEVEL, MAX_BLOCK_BYTES, N_BOOT, SEED, percentile_interval, replicate_matrices, \
    shrinkage_fit
from src.inference.drogue_comparison import UNDROGUED_STEM, undrogued_only
from src.inference.estimators import count_matrix
from src.inference.transition_data import TAU_DAYS, TRANSITIONS_PATH, load_transitions, parent_states, to_states
from src.linalg.connectivity import box_cells, subregion_map

APP_DATA_PATH = os.path.join(REPO_ROOT, "app", "data", "app_data.js")
FORECAST_PATH = os.path.join(PROCESSED_DIR, "seasonal_forecast_app.json")
MAX_STEPS = 72  # 1,080 days, about 3 years: past the 2.7 years after which the coastal start is forgotten
QUANTITIES = ("pct_left_box", "pct_in_benguela", "pct_in_coastal_zone", "centre_lat", "centre_lon", "centre_km",
              "pct_aground", "pct_reached_coast")
SHIPPED = {"water": ("pct_left_box", "pct_in_benguela", "pct_in_coastal_zone", "centre_km", "pct_reached_coast"),
           "shore": ("pct_left_box", "pct_in_benguela", "pct_in_coastal_zone", "centre_km", "pct_reached_coast", "pct_aground")}
REACH = 7  # the column of functionals() that is first passage, not a share
SEASON_STEPS = [*range(1, 7), *range(8, 25, 2), *range(30, MAX_STEPS + 1, 6)]  # every 15 days to 3 months, 30 to a year, 90 to 3 years
LOG16 = (-6.0, 4.4)  # the range of log10(value) in 16-bit intervals: under 0.000001 is stored as 0, and 65535 means no value
SETS = ("all", "undrogued")
EARTH_RADIUS_KM = 6371.0


def cell_centres(region, cell_size):
    """(lat, lon) of every cell centre, flat in cell_index order."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat_c, lon_c = np.meshgrid(0.5 * (lat_edges[:-1] + lat_edges[1:]), 0.5 * (lon_edges[:-1] + lon_edges[1:]),
                               indexing="ij")
    return lat_c.ravel(), lon_c.ravel()


def zone_cells(region, cell_size, zone):
    """Flat ids of the cells lying wholly inside a lat/lon box (as coast.coastal_cells, without the land rule)."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat0, lon0 = (a.ravel() for a in np.meshgrid(lat_edges[:-1], lon_edges[:-1], indexing="ij"))
    tol = 1e-9
    return np.flatnonzero((lat0 >= zone["lat_min"] - tol) & (lat0 + cell_size <= zone["lat_max"] + tol)
                          & (lon0 >= zone["lon_min"] - tol) & (lon0 + cell_size <= zone["lon_max"] + tol))


def groundings(stored, region, cell_size, land):
    """Starts whose drifter ran aground within tau, and the cell where it came ashore.

    Where it came ashore: the drifter's last start in R (its last 00:00 UTC position in the box, a median
    9 km from its last reading on the 2026-11-02 check against the region cache), moved to the nearest
    cell the coastline passes through (part land, part sea) within one cell of it. If the coastline only
    runs along the cell's edge, the cell itself. Drifters further from the mapped coast are left out:
    small islands missing from the 1:110m outline (Ascension, Trindade, Fernando de Noronha) or likely
    metadata errors.

    Inputs:
        stored: the transition table as saved (all outcomes, with typedeath).
        land: land_fraction() for this grid.
    Returns:
        dict with start_lon, start_lat, drifter_id and shore (cell id) per kept start, plus the numbers
        of drifters kept and dropped.
    """
    rows = (stored["outcome"] == ENDED) & (stored["typedeath"] == RAN_AGROUND)
    did, t = stored["drifter_id"], stored["start_time"].astype("int64")
    order = np.lexsort((t, did))
    last = order[np.r_[np.flatnonzero(np.diff(did[order])), order.size - 1]]  # each drifter's last start
    last_by_id = dict(zip(did[last].tolist(), last.tolist()))

    lat_edges, lon_edges = grid_edges(region, cell_size)
    n_lat, n_lon = len(lat_edges) - 1, len(lon_edges) - 1
    lat_c, lon_c = cell_centres(region, cell_size)
    grid_land = land.reshape(n_lat, n_lon)
    shore_of = {}
    for d in np.unique(did[rows]).tolist():
        k = last_by_id[d]
        lon, lat = float(stored["start_lon"][k]), float(stored["start_lat"][k])
        cell = int(cell_index(np.array([lon]), np.array([lat]), region, cell_size)[0][0])
        r, q = divmod(cell, n_lon)
        near = [(r + a) * n_lon + q + b for a in (-1, 0, 1) for b in (-1, 0, 1)
                if 0 <= r + a < n_lat and 0 <= q + b < n_lon]
        coast = [c for c in near if 0 < land[c] < 1]
        if coast:
            dist = [np.hypot(lat_c[c] - lat, (lon_c[c] - lon) * np.cos(np.radians(lat))) for c in coast]
            shore_of[d] = coast[int(np.argmin(dist))]
        elif grid_land[max(r - 1, 0):r + 2, max(q - 1, 0):q + 2].max() > 0:
            shore_of[d] = cell
    keep = rows & np.isin(did, list(shore_of))
    return {"start_lon": stored["start_lon"][keep].astype(float), "start_lat": stored["start_lat"][keep].astype(float),
            "start_time": stored["start_time"][keep],
            "drifter_id": did[keep], "shore": np.array([shore_of[d] for d in did[keep].tolist()], dtype=int),
            "n_kept": len(shore_of), "n_dropped": int(np.unique(did[rows]).size - len(shore_of))}


def shore_fit(water_fit, n_cells, n_shore, parent, alpha):
    """The shore model as a function of an extended count matrix: states are the cells, left the box,
    then one aground state per coast. The water block is fitted exactly as the product; the grounding
    chance is shrunk towards the 4 deg parent's with weight alpha (module docstring).

    Inputs:
        parent: the coarse cell of every fine cell (transition_data.parent_states; only cells are used).
    """
    n = n_cells + 1
    par = np.asarray(parent[:n_cells])

    def fit(counts):
        P = np.zeros((n + n_shore, n + n_shore))
        P[:n, :n] = water_fit(counts[:n, :n])
        known = counts[:n_cells, :n].sum(axis=1)
        ground = counts[:n_cells, n:]
        n_par = np.bincount(par, weights=known, minlength=par.max() + 1)
        g_par = np.zeros((n_par.size, n_shore))
        np.add.at(g_par, par, ground)
        tot_par = n_par + g_par.sum(axis=1)
        q = np.divide(g_par, tot_par[:, None], out=np.zeros_like(g_par), where=tot_par[:, None] > 0)[par]
        share = (ground + alpha * q) / (known + ground.sum(axis=1) + alpha)[:, None]
        P[:n_cells, :n] *= (1 - share.sum(axis=1))[:, None]
        P[:n_cells, n:] = share
        P[n:, n:] = np.eye(n_shore)
        return P
    return fit


def functionals(lat_c, lon_c, benguela, coast, n_shore=0):
    """The eight vectors QUANTITIES are built from, one column each (rows = states: cells, left the box,
    then any aground states): left box, in the Benguela zone, in its coastal cells, still in the water
    inside the box, lat / lon weighted by that, aground, and having reached the coastal cells."""
    n_cells = lat_c.size
    F = np.zeros((n_cells + 1 + n_shore, 8))
    F[n_cells, 0] = 1.0
    F[benguela, 1] = 1.0
    F[coast, 2] = 1.0
    F[:n_cells, 3] = 1.0
    F[:n_cells, 4] = lat_c
    F[:n_cells, 5] = lon_c
    F[n_cells + 1:, 6] = 1.0
    F[coast, REACH] = 1.0
    return F


def step_back(P, F, n_steps):
    """Yield P^n F for n = 1..n_steps: row i is the expected value of each column after n steps from state i.
    The last column is first passage: material that has reached the coastal cells stays counted."""
    arrived = F[:, REACH] > 0
    for _ in range(n_steps):
        F = P @ F
        F[arrived, REACH] = 1.0
        yield F


def season_walk(seasons, F0, steps=SEASON_STEPS, per_season=STEPS_PER_SEASON):
    """Yield (first season, n, P_s0 P_s1 ... P_s(n-1) F0) for a release at the first day of every season, at every step asked for.

    A chain's matrices differ along it, so backward steps can't be shared as in step_back, except between chains
    ending at the same point of the year. So each walk starts from a chain's last step, at each of the year's 24 positions.
    """
    arrived = F0[:, REACH] > 0
    cycle = 4 * per_season
    for last in range(cycle):
        needed = {n for n in steps if (last - n + 1) % per_season == 0}
        F = F0
        for n in range(1, max(needed, default=0) + 1):
            position = (last - n + 1) % cycle
            F = seasons[position // per_season] @ F
            F[arrived, REACH] = 1.0
            if n in needed:
                yield position // per_season, n, F


def great_circle_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km."""
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def quantities(G, start_lat, start_lon):
    """QUANTITIES from functionals already applied to the start distributions (G = starts @ P^n F).
    Percentages for the shares; the centre is over material still in the water inside the box
    (release_map.centre_of_mass). The centre is NaN in a replicate that leaves a cell with no row (no drifters
    drawn there or in its 4 deg cell); percentile_interval skips those, as in bootstrap.py."""
    inside = G[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        lat, lon = G[:, 4] / inside, G[:, 5] / inside
    return np.stack([100 * G[:, 0], 100 * G[:, 1], 100 * G[:, 2], lat, lon,
                     great_circle_km(start_lat, start_lon, lat, lon), 100 * G[:, 6], 100 * G[:, REACH]])


def start_matrix(rows, release, n_states):
    """One start distribution per row: all material in that cell, or (row -1) spread evenly over the release cells."""
    S = np.zeros((len(rows), n_states))
    for k, cell in enumerate(rows):
        if cell < 0:
            S[k, release] = 1.0 / release.size
        else:
            S[k, cell] = 1.0
    return S


def bootstrap_intervals(start, end, drifter_id, n_states, fit, S, F0, start_lat, start_lon, n_steps=MAX_STEPS,
                        n_boot=N_BOOT, seed=SEED, level=LEVEL, max_block_bytes=MAX_BLOCK_BYTES):
    """Percentile interval of every quantity, start row and step over the drifter bootstrap.

    Rows are done in blocks so the replicate array stays under max_block_bytes; each block is one pass over
    the same replicates (fixed seed), so the blocks agree with a single pass.
    Returns:
        (low, high), each (len(QUANTITIES), n_steps, rows) float32.
    """
    n_rows = S.shape[0]
    per_row = len(QUANTITIES) * n_boot * n_steps * 4
    block = max(1, int(max_block_bytes // per_row))
    low = np.empty((len(QUANTITIES), n_steps, n_rows), dtype=np.float32)
    high = np.empty_like(low)
    for r0 in range(0, n_rows, block):
        rows = slice(r0, min(r0 + block, n_rows))
        reps = np.empty((len(QUANTITIES), n_boot, n_steps, rows.stop - r0), dtype=np.float32)
        for b, P_b in enumerate(replicate_matrices(start, end, drifter_id, n_states, fit, n_boot, seed)):
            for n, F in enumerate(step_back(P_b, F0, n_steps)):
                reps[:, b, n] = quantities(S[rows] @ F, start_lat[rows], start_lon[rows])
        with np.errstate(invalid="ignore"):
            lo, hi = percentile_interval(reps, level, axis=1)
        low[:, :, rows], high[:, :, rows] = lo, hi
        print(f"  intervals for start rows {r0}-{rows.stop - 1} of {n_rows} done")
    return low, high


def season_intervals(model, S, F0, start_lat, start_lon, beta, steps=SEASON_STEPS, n_boot=N_BOOT_SEASON, seed=SEED, level=LEVEL):
    """Percentile interval of every quantity, start row, stored step and release season over the drifter bootstrap.

    Returns:
        (low, high), each (4 seasons, len(QUANTITIES), len(steps), rows) float32.
    """
    at = {n: k for k, n in enumerate(steps)}
    reps = np.empty((4, len(QUANTITIES), n_boot, len(steps), S.shape[0]), dtype=np.float32)
    for b, (P_b, counts_b) in enumerate(season_replicates(model["start"], model["end"], model["season"], model["drifter_id"], model["n_states"], model["fit"], n_boot, seed)):
        seasons = season_matrices(P_b, counts_b, beta, model["n_water"])
        for first, n, F in season_walk(seasons, F0, steps):
            reps[first, :, b, at[n]] = quantities(S @ F, start_lat, start_lon)
    with np.errstate(invalid="ignore"):
        low, high = percentile_interval(reps, level, axis=2)
    return low.astype(np.float32), high.astype(np.float32)


def land_outline(path=LAND_PATH, extent=MAP_EXTENT, decimals=2):
    """Outer rings of the land polygons that reach the map extent, rounded (the map is 2 deg cells)."""
    lon_min, lon_max, lat_min, lat_max = extent
    with open(path) as f:
        features = json.load(f)["features"]
    rings = []
    for feature in features:
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for polygon in polygons:
            ring = np.asarray(polygon[0])
            if (ring[:, 0].max() < lon_min - 20 or ring[:, 0].min() > lon_max + 20
                    or ring[:, 1].max() < lat_min - 20 or ring[:, 1].min() > lat_max + 20):
                continue
            rings.append(np.round(ring, decimals).tolist())
    return rings


def check_coast_release(low, high, estimates, rows_path=os.path.join(RESULTS_DIR, f"coastal_release_tau{TAU_DAYS}d.csv")):
    """Compare the water model's coast release row with coastal_release.py's saved numbers (2 deg
    shrinkage), if saved. Returns the largest absolute difference over the quantities both have."""
    if not os.path.exists(rows_path):
        print(f"  {rows_path} not found: coast release not cross-checked")
        return None
    names = {"pct_left_R": "pct_left_box", "pct_in_coastal_zone": "pct_in_coastal_zone",
             "centre_lat": "centre_lat", "centre_lon": "centre_lon"}
    worst = 0.0
    with open(rows_path) as f:
        for row in csv.DictReader(f):
            if not row["model"].startswith(f"{2.0:g} deg shrinkage") or row["quantity"] not in names:
                continue
            q, n = QUANTITIES.index(names[row["quantity"]]), int(row["steps"]) - 1
            for saved, ours in ((row["estimate"], estimates[q, n, -1]), (row["ci_low"], low[q, n, -1]),
                                (row["ci_high"], high[q, n, -1])):
                worst = max(worst, abs(float(saved) - float(ours)))
    return worst


def b64_float32(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def log16(a):
    """Non-negative numbers as 16 bits on a log scale, base64 (decoded by model.js and check_intervals.from_log16)."""
    a = np.asarray(a, dtype=float)
    span = LOG16[1] - LOG16[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        code = np.round(np.clip((np.log10(a) - LOG16[0]) / span, 0, 1) * 65534)
    code = np.where(a * 10.0 ** -LOG16[0] < 1, 0, code)  # 0 stands for anything too small to matter
    return base64.b64encode(np.where(np.isnan(a), 65535, code).astype("<u2").tobytes()).decode("ascii")


def from_log16(text):
    """The numbers log16() stored (NaN where there was no value)."""
    code = np.frombuffer(base64.b64decode(text), dtype="<u2").astype(float)
    out = np.where(code == 0, 0.0, 10.0 ** (LOG16[0] + (code / 65534) * (LOG16[1] - LOG16[0])))
    return np.where(code == 65535, np.nan, out)


def load_forecast(path=FORECAST_PATH):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def trust_checks(results_dir=RESULTS_DIR):
    """What the page's "How far to trust it" notes quote, from calibration.py and year_intervals.py, so the
    wording can't drift from the checks. Empty if they haven't been run (the page then speaks more generally)."""
    cal_path = os.path.join(results_dir, f"calibration_tau{TAU_DAYS}d.csv")
    year_path = os.path.join(results_dir, f"year_intervals_tau{TAU_DAYS}d.csv")
    if not (os.path.exists(cal_path) and os.path.exists(year_path)):
        return {}
    with open(cal_path) as f:
        cal = list(csv.DictReader(f))
    with open(year_path) as f:
        year = list(csv.DictReader(f))
    line = {r["days"]: float(r["observed_pct"]) for r in cal if r["check"] == "likely area"
            and r["cells"] == "all with drifters" and float(r["level_pct"]) == 90}
    left = {r["days"]: (float(r["predicted_pct"]), float(r["observed_pct"])) for r in cal
            if r["check"] == "having left the box" and r["cells"] == "all starts"}
    cell = next(r for r in cal if r["check"] == "reliability" and r["days"] == "360" and float(r["bin_low"]) == 0.01)
    width = next(r for r in year if r["quantity"] == "centre_lon" and r["days"] == "360")
    out = {"line_90": [f"{line[d]:.0f}" for d in ("15", "90", "360")],
           "left_box": [f"{left[d][0]:.1f}% predicted against {left[d][1]:.1f}% observed" for d in ("90", "360")],
           "cell_1y": [f"{float(cell['predicted_pct']):.1f}", f"{float(cell['observed_pct']):.1f}"],
           "year_width": f"{float(width['width_ratio']):.1f}"}
    coverage_path = os.path.join(results_dir, f"interval_coverage_tau{TAU_DAYS}d.csv")
    if os.path.exists(coverage_path):
        out["coverage"] = coverage_medians(coverage_path)
    out |= drogue_checks(results_dir) | season_checks(results_dir)
    return out


def coverage_medians(path):
    """The median coverage of 95% intervals by data flag, then for the whole coast (interval_coverage.py)."""
    with open(path) as f:
        cover = list(csv.DictReader(f))
    median = lambda word: float(np.median([float(r["pct_covered"]) for r in cover if word in r["starts"]]))
    return [f"{median(word):.0f}" for word in ("100+", "10-99", "under 10", "whole")]


def drogue_checks(results_dir=RESULTS_DIR):
    """What the page says about the undrogued matrix, from drogue_comparison.py and interval_coverage.py."""
    path = os.path.join(results_dir, f"drogue_comparison_tau{TAU_DAYS}d.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    line = {(r["matrix"], r["days"]): r["observed_pct"] for r in rows if r["check"] == "likely area" and float(r["level_pct"]) == 90}
    left = {r["days"]: r for r in rows if r["check"] == "having left the box" and r["matrix"] == "Undrogued only" and r["cells"] == "all starts"}
    score = {r["days"]: float(r["difference"]) for r in rows if r["check"] == "held-out score" and r["cells"] == "undrogued test moves"}
    und = "Undrogued only"
    out = {"line_90": [f"{float(line[(und, d)]):.0f}" for d in ("15", "90", "360")],
           "score": (f"On undrogued drifters it never saw, this matrix predicts no better than the all-drifter one at 15 days ({score['15']:+.3f} in log-likelihood per move), "
                     f"but better at 3 months ({score['90']:+.3f}) and a year ({score['360']:+.3f}). Its 90% line held {float(line[(und, '360')]):.0f}% of them after a year, against "
                     f"{float(line[('All drifters', '360')]):.0f}% for the all-drifter matrix. For drifters that kept their drogue it is much worse, so choose the one that suits your material.")}
    if left:
        out["left_box"] = [f"{float(left[d]['predicted_pct']):.1f}% predicted against {float(left[d]['observed_pct']):.1f}% observed" for d in ("90", "360")]
    coverage_path = os.path.join(results_dir, f"interval_coverage_undrogued_tau{TAU_DAYS}d.csv")
    if os.path.exists(coverage_path):
        out["coverage"] = coverage_medians(coverage_path)
    return {"undrogued": out}


def season_checks(results_dir=RESULTS_DIR):
    """What the page says about the seasons, from seasonal_matrices.py."""
    path = os.path.join(results_dir, f"seasonal_matrices_tau{TAU_DAYS}d.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    kept = {r["set"]: f"{float(r['pct_of_gap_kept']):.0f}" for r in rows if r["check"] == "signal kept"}
    gain = next(float(r["gain_per_move"]) for r in rows if r["check"] == "held-out score" and r["set"] == "all")
    chain = {(r["compare"], r["days"]): float(r["gain_per_move"]) for r in rows if r["check"] == "crossing seasons" and r["set"] == "all"}
    if ("switching minus all year", "360") not in chain:
        return {"seasons": {"kept": kept, "detail": ""}}
    return {"seasons": {"kept": kept, "detail": (
        f"Tested on drifters and years the fit never saw. Knowing the season is a small gain ({100 * gain:.1f}% better likelihood per 15-day move). Changing matrix as the seasons pass "
        f"beats the all-year matrix at every horizon from 30 days to a year ({chain[('switching minus all year', '180')]:+.3f} in log-likelihood per move at 180 days). Keeping the release season's "
        f"matrix throughout does worse than the all-year matrix from about 6 months ({chain[('stuck minus all year', '180')]:+.3f}).")}}


def build_models(region, table, stored, alpha, land, shores=None):
    """Both models' states, fit, data and start rows (see the module docstring).

    Inputs:
        shores: the stretches of coast that get an "aground" state. By default the ones this table's groundings reach.
    """
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    parent = parent_states(region, fine, coarse)
    water_fit = shrinkage_fit(parent, alpha)
    start, end, n_cells = to_states(table, region, fine)
    _, start_drifters = per_cell_counts(start, table["drifter_id"], n_cells)

    g = groundings(stored, region, fine, land)
    shores = np.unique(g["shore"]) if shores is None else shores
    g_start = cell_index(g["start_lon"], g["start_lat"], region, fine)[0]
    g_end = n_cells + 1 + np.searchsorted(shores, g["shore"])
    season, _ = season_and_year(table["start_time"])
    g_season, _ = season_and_year(g["start_time"])
    models = {
        "water": {"start": start, "end": end, "drifter_id": table["drifter_id"], "n_states": n_cells + 1, "season": season,
                  "fit": water_fit, "shores": np.array([], dtype=int), "n_water": None},
        "shore": {"start": np.concatenate([start, g_start]), "end": np.concatenate([end, g_end]), "season": np.concatenate([season, g_season]),
                  "drifter_id": np.concatenate([table["drifter_id"], g["drifter_id"]]), "n_water": n_cells + 1,
                  "n_states": n_cells + 1 + shores.size, "fit": shore_fit(water_fit, n_cells, shores.size, parent, alpha),
                  "shores": shores},
    }
    for m in models.values():
        m["P"] = m["fit"](count_matrix(m["start"], m["end"], m["n_states"]))
    return models, n_cells, start_drifters, g


def saved_season_counts(key, n_states, path=COUNTS_PATH):
    """A set's season counts and beta, as src.forecasting.seasonal_matrices saved them."""
    z = np.load(path)
    counts = np.zeros((4, n_states, n_states))
    counts[z[f"{key}_season"], z[f"{key}_start"], z[f"{key}_end"]] = z[f"{key}_n"]
    return counts, float(z[f"{key}_beta"])


def ship(low, high, quantities_kept):
    """The interval arrays cut to what the page shows, in QUANTITIES' order (quantity, step, row) per array."""
    idx = [QUANTITIES.index(q) for q in quantities_kept]
    return log16(low[idx]), log16(high[idx])


def check_undrogued(low, high, estimates, rows_path=os.path.join(RESULTS_DIR, f"drogue_comparison_tau{TAU_DAYS}d.csv")):
    """Compare the undrogued coast release with drogue_comparison.py's saved numbers, if saved. Returns the largest gap."""
    if not os.path.exists(rows_path):
        print(f"  {rows_path} not found: undrogued coast release not cross-checked")
        return None
    names = {"pct_left": "pct_left_box", "pct_coastal": "pct_in_coastal_zone", "centre_lat": "centre_lat", "centre_lon": "centre_lon"}
    worst = 0.0
    with open(rows_path) as f:
        for row in csv.DictReader(f):
            if row["check"] != "answers" or row["source"] != "Benguela coast" or row["quantity"] not in names:
                continue
            q, n = QUANTITIES.index(names[row["quantity"]]), int(row["days"]) // TAU_DAYS - 1
            for saved, ours in ((row["undrogued_only"], estimates[q, n, -1]), (row["undrogued_low"], low[q, n, -1]), (row["undrogued_high"], high[q, n, -1])):
                worst = max(worst, abs(float(saved) - float(ours)))
    return worst


def export_set(key, region, table, stored, product, land, lat_c, lon_c, coast, benguela, shores=None):
    """One drifter set's block of the page's data. Returns (block, the stretches of coast used, the shore model's summary)."""
    fine = region["cell_size_deg"]
    alpha = float(product["alpha"])
    models, n_cells, start_drifters, g = build_models(region, table, stored, alpha, land, shores)
    if not np.allclose(models["water"]["P"], product["P"]):
        raise ValueError(f"Refitted {key} P differs from the saved matrix; rerun the script that saved it.")
    shore = models["shore"]
    n = n_cells + 1
    p_ground = shore["P"][:n_cells, n:].sum(axis=1)
    if not (np.allclose(shore["P"].sum(axis=1), 1)
            and np.allclose(shore["P"][:n_cells, :n], product["P"][:n_cells] * (1 - p_ground)[:, None])):
        raise ValueError("The shore model's water block isn't the matrix scaled by the chance of staying afloat.")
    print(f"{key}: groundings: {g['n_kept']} drifters on {np.unique(g['shore']).size} stretches of coast ({g['start_lon'].size:,} "
          f"starts); {g['n_dropped']} drifters left out, too far from the mapped coast. Largest 15-day grounding "
          f"chance in a cell with drifters {100 * p_ground[start_drifters > 0].max():.1f}%; {np.sum(p_ground > 0)} cells have any.")

    release = coast[start_drifters[coast] > 0]
    sampled = np.flatnonzero(start_drifters > 0)
    rows = np.append(sampled, -1)  # every cell with drifters, then the whole-coast release
    start_lat = np.append(lat_c[sampled], lat_c[release].mean())
    start_lon = np.append(lon_c[sampled], lon_c[release].mean())
    print(f"{key}: {fine:g} deg shrinkage (alpha {alpha:.1f}): {sampled.size} start cells with drifters, coast release over "
          f"{release.size} cells, {benguela.size} cells in the Benguela zone, {MAX_STEPS} steps of {TAU_DAYS} days", flush=True)

    counts, beta = saved_season_counts(key, n)
    if not np.array_equal(counts, season_counts(models["water"]["start"], models["water"]["end"], models["water"]["season"], n)):
        raise ValueError("Saved season counts differ from the table; rerun src.forecasting.seasonal_matrices.")

    intervals, season_iv = {}, {}
    for name, m in models.items():
        F0 = functionals(lat_c, lon_c, benguela, coast, m["shores"].size)
        S = start_matrix(rows, release, m["n_states"])
        estimates = np.stack([quantities(S @ F, start_lat, start_lon) for F in step_back(m["P"], F0, MAX_STEPS)], axis=1)
        x = S[-1].copy()  # the backward route must agree with stepping forward from a start (Product.forecast)
        for _ in range(MAX_STEPS):
            x = x @ m["P"]
        if not (np.isclose(100 * x[n_cells], estimates[0, -1, -1]) and np.isclose(100 * x[coast].sum(), estimates[2, -1, -1])
                and np.isclose(100 * x[n:].sum(), estimates[6, -1, -1])):
            raise ValueError("Backward functionals disagree with forward propagation.")
        print(f"{key}, {name} model: bootstrap, {N_BOOT} replicates of the drifters (seed {SEED}), {100 * LEVEL:g}% intervals", flush=True)
        low, high = bootstrap_intervals(m["start"], m["end"], m["drifter_id"], m["n_states"], m["fit"], S, F0, start_lat, start_lon)
        if name == "water":
            worst = check_coast_release(low, high, estimates) if key == "all" else check_undrogued(low, high, estimates)
            if worst is not None:
                print(f"  coast release vs the saved results: largest difference {worst:.2g}")
                if worst > 1e-3:
                    raise ValueError(f"The {key} coast release doesn't reproduce the saved results; check both use the same matrix.")
        lo, hi = ship(low, high, SHIPPED[name])
        intervals[name] = {"quantities": list(SHIPPED[name]), "rows": rows.tolist(), "steps": list(range(1, MAX_STEPS + 1)), "low": lo, "high": hi}

        print(f"{key}, {name} model: seasons, {N_BOOT_SEASON} replicates", flush=True)
        low, high = season_intervals(m, S, F0, start_lat, start_lon, beta)
        packed = [ship(low[s], high[s], SHIPPED[name]) for s in range(4)]
        season_iv[name] = {"quantities": list(SHIPPED[name]), "rows": rows.tolist(), "steps": SEASON_STEPS,
                           "low": [p[0] for p in packed], "high": [p[1] for p in packed]}

    # The shore model is shipped as its grounding rows, and the seasons as their counts; the page rebuilds the rest from P.
    ground_rows = [[int(i), int(k), round(float(shore["P"][i, n + k]), 7)] for i, k in zip(*np.nonzero(shore["P"][:n_cells, n:]))]
    season_pairs = {}
    for s in range(4):  # [flat index, count, flat index, count, ...]
        flat = counts[s].ravel()
        where = np.flatnonzero(flat)
        season_pairs[SEASONS[s]] = np.column_stack([where, flat[where]]).astype(int).ravel().tolist()
    block = {"alpha": alpha, "n_drifters": int(np.unique(table["drifter_id"]).size), "n_moves": int(table["drifter_id"].size),
             "start_drifters": start_drifters.astype(int).tolist(), "release_cells": release.tolist(), "P": b64_float32(product["P"]),
             "ground_rows": ground_rows, "n_grounded": g["n_kept"], "n_grounded_dropped": g["n_dropped"], "intervals": intervals,
             "seasonal": {"beta": beta, "counts": season_pairs, "intervals": season_iv}}
    return block, shore["shores"]


if __name__ == "__main__":
    region = load_region_bounds()
    fine = region["cell_size_deg"]
    base = load_transitions(region)
    with np.load(TRANSITIONS_PATH) as z:
        stored_all = {k: z[k] for k in z.files}
    undrogued_rows = ~stored_all["drogued"].astype(bool)
    stems = {"all": f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d", "undrogued": UNDROGUED_STEM}
    product_all = np.load(os.path.join(PROCESSED_DIR, f"{stems['all']}.npz"))
    land = land_fraction(region, fine, land_paths())
    lat_c, lon_c = cell_centres(region, fine)
    coast = coastal_cells(region, fine, land)
    benguela = zone_cells(region, fine, region["coastal_zone"])

    sets, shores = {}, None
    for key in SETS:
        table = base if key == "all" else undrogued_only(base)
        stored = stored_all if key == "all" else {k: (v[undrogued_rows] if v.shape == undrogued_rows.shape else v) for k, v in stored_all.items()}
        sets[key], shores = export_set(key, region, table, stored, np.load(os.path.join(PROCESSED_DIR, f"{stems[key]}.npz")), land, lat_c, lon_c, coast, benguela, shores)

    zone = region["coastal_zone"]
    years = base["start_time"].astype("datetime64[Y]").astype(int) + 1970
    names, label = subregion_map(region, fine, coast)
    data = {
        "meta": {"built": datetime.date.today().isoformat(), "product": f"{stems['all']}.npz", "n_boot": N_BOOT, "n_boot_season": N_BOOT_SEASON,
                 "level": LEVEL, "years": [int(years.min()), int(years.max())], "checks": trust_checks()},
        "region": {k: float(region[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")} | {"label": region["label"]},
        "benguela": {k: float(zone[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")} | {"label": zone["label"]},
        "cell_size": fine, "n_lat": int(product_all["n_lat_cells"]), "n_lon": int(product_all["n_lon_cells"]),
        "tau_days": TAU_DAYS, "max_steps": MAX_STEPS,
        "log16": list(LOG16),
        "seasons": {"codes": list(SEASONS), "names": list(SEASON_NAMES), "per_season": STEPS_PER_SEASON},
        "subregions": {"names": names, "label": label.tolist()},
        "named": [{"name": b["name"], "cells": box_cells(region, fine, b).tolist()} for b in region.get("named_releases", [])],
        "all_land": np.flatnonzero(land >= 1).tolist(),
        "coastal_cells": coast.tolist(), "benguela_cells": benguela.tolist(),
        "shore": {"cells": shores.tolist()},
        "sets": sets,
        "land": land_outline(),
        "forecast": load_forecast(),
    }
    os.makedirs(os.path.dirname(APP_DATA_PATH), exist_ok=True)
    with open(APP_DATA_PATH, "w") as f:
        f.write("// Made by src/app/export_app_data.py - don't edit by hand; rerun the script instead.\n")
        f.write("window.DRIFT_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"Saved {APP_DATA_PATH} ({os.path.getsize(APP_DATA_PATH) / 1e6:.1f} MB); forecast block: "
          f"{'included' if data['forecast'] else 'none yet'}")
