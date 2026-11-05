"""
export_app_data.py

Related task: Week 9 - the Drift Explorer app (app/index.html). A user picks a start position and a
timeframe and sees where material from there is likely to be, with the chance it is in the Benguela
zone, near the Benguela coast, has left the box or (optionally) has run aground, each with a 95%
interval. This packs the product (src/linalg/product.py) and its drifter bootstrap into one JavaScript
file the page loads, so the app runs in any browser with no Python, server or internet.

Two models, both shipped:
- water: the product itself, the 2 deg shrinkage P (761 x 761). Starts whose record ended are left out
  (decision 10), so material only drifts or leaves the box. This is what the report quotes.
- shore: the product plus running aground (the page's "Groundings" switch). GDP marks a drifter that
  ran aground (typedeath 1), and its starts in the last 15 days of its record are "record ended" rows.
  Those starts become moves into an absorbing "aground" state for the stretch of coast where the
  drifter came ashore (groundings()). For start cell i, with n_i known-destination moves and g_ik
  groundings on coast k, the grounding chance is shrunk towards the cell's 4 deg parent exactly as the
  product's rows are (the same alpha, empirical Bayes; estimators.py):
      P_shore[i, k] = (g_ik + alpha * q_k) / (n_i + g_i + alpha),  P_shore[i, j] = P[i, j] * (1 - sum_k P_shore[i, k])
  where q_k is the parent's share of moves that ran aground on coast k (its four cells pooled). So the
  water part is the product scaled by the chance of staying afloat for 15 days. A cell whose few
  records all end aground no longer sends everything ashore, and a cell with no groundings of its own
  gets its neighbourhood's (small) chance. The bootstrap intervals show how rough the estimates are.

What goes in app/data/app_data.js (window.DRIFT_DATA):
- P (float32, base64); the shore model's grounding rows as [cell, coast, probability] triples. The page
  steps either model forward itself (x @ P per 15-day step, as Product.forecast).
- intervals per model: 95% bootstrap intervals for QUANTITIES, from every start cell with drifters and
  from the whole Benguela coast release, at every 15-day step up to MAX_STEPS. 1000 replicates of the
  drifters with bootstrap.py's seed; for the water model these are bootstrap.py's own replicates, so the
  coast release reproduces coastal_release_tau15d.csv (checked below). The estimates themselves are
  computed by the page. src/app/check_intervals.py re-derives a sample of them by brute force.
- the grid, which cells are all land (for the page's contours), the box, the Benguela zone
  (coastal_zone in config/region.yaml) and its coastal cells, drifters per cell, the land outline.
- forecast: the seasonal forecast block if FORECAST_PATH exists, else null.

Every quantity is linear in the distribution, or (the centre) a ratio of linear ones, so for all start
cells at once they come from stepping P backwards on seven vectors, F_n = P F_(n-1): row i of F_n is
start cell i's value after n steps. 7 matrix-vector products per step instead of 596 forward runs.

Seasonal forecast hook (Business, RQ4): when the forecast is done, save a JSON file at FORECAST_PATH
and rerun this script. The app draws it as a chart with intervals; the expected shape is
    {"title": str, "summary": str, "quantity": str, "unit": "%",
     "series": [{"label": str, "x": [str, ...], "y": [float, ...],
                 "low": [float, ...] (optional), "high": [float, ...] (optional), "kind": "observed" | "forecast"}],
     "notes": [str, ...]}
Anything richer (e.g. a seasonal P to step forward) needs a matching change in app/js/app.js.

Run: python3 -m src.app.export_app_data   (about 5 minutes; needs data/processed/transitions_tau15d.npz,
shrinkage_P_2deg_tau15d.npz from src.inference.shrinkage_fit and data/external/ne_110m_land.geojson)
"""
import base64
import csv
import datetime
import json
import os
import numpy as np

from src.common.config import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR, load_region_bounds
from src.gridding.coast import RAN_AGROUND, coastal_cells, land_fraction, land_paths
from src.gridding.grid import cell_index, grid_edges, per_cell_counts
from src.gridding.trajectory_map import LAND_PATH, MAP_EXTENT
from src.gridding.transitions import BASELINE_CELL_SIZE, ENDED
from src.inference.bootstrap import LEVEL, MAX_BLOCK_BYTES, N_BOOT, SEED, percentile_interval, replicate_matrices, \
    shrinkage_fit
from src.inference.estimators import count_matrix
from src.inference.transition_data import TAU_DAYS, TRANSITIONS_PATH, load_transitions, parent_states, to_states

APP_DATA_PATH = os.path.join(REPO_ROOT, "app", "data", "app_data.js")
FORECAST_PATH = os.path.join(PROCESSED_DIR, "seasonal_forecast_app.json")
MAX_STEPS = 72  # 1,080 days, about 3 years: past the 2.7 years after which the coastal start is forgotten
QUANTITIES = ("pct_left_box", "pct_in_benguela", "pct_in_coastal_zone", "centre_lat", "centre_lon", "centre_km",
              "pct_aground")
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
    """The seven vectors QUANTITIES are built from, one column each (rows = states: cells, left the box,
    then any aground states): left box, in the Benguela zone, in its coastal cells, still in the water
    inside the box, lat / lon weighted by that, and aground."""
    n_cells = lat_c.size
    F = np.zeros((n_cells + 1 + n_shore, 7))
    F[n_cells, 0] = 1.0
    F[benguela, 1] = 1.0
    F[coast, 2] = 1.0
    F[:n_cells, 3] = 1.0
    F[:n_cells, 4] = lat_c
    F[:n_cells, 5] = lon_c
    F[n_cells + 1:, 6] = 1.0
    return F


def step_back(P, F, n_steps):
    """Yield P^n F for n = 1..n_steps: row i is the expected value of each column after n steps from state i."""
    for _ in range(n_steps):
        F = P @ F
        yield F


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
                     great_circle_km(start_lat, start_lon, lat, lon), 100 * G[:, 6]])


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


def load_forecast(path=FORECAST_PATH):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_models(region, table, stored, alpha, land):
    """Both models' states, fit, data and start rows (see the module docstring)."""
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    parent = parent_states(region, fine, coarse)
    water_fit = shrinkage_fit(parent, alpha)
    start, end, n_cells = to_states(table, region, fine)
    _, start_drifters = per_cell_counts(start, table["drifter_id"], n_cells)

    g = groundings(stored, region, fine, land)
    shores = np.unique(g["shore"])
    g_start = cell_index(g["start_lon"], g["start_lat"], region, fine)[0]
    g_end = n_cells + 1 + np.searchsorted(shores, g["shore"])
    models = {
        "water": {"start": start, "end": end, "drifter_id": table["drifter_id"], "n_states": n_cells + 1,
                  "fit": water_fit, "shores": np.array([], dtype=int)},
        "shore": {"start": np.concatenate([start, g_start]), "end": np.concatenate([end, g_end]),
                  "drifter_id": np.concatenate([table["drifter_id"], g["drifter_id"]]),
                  "n_states": n_cells + 1 + shores.size, "fit": shore_fit(water_fit, n_cells, shores.size, parent, alpha),
                  "shores": shores},
    }
    for m in models.values():
        m["P"] = m["fit"](count_matrix(m["start"], m["end"], m["n_states"]))
    return models, n_cells, start_drifters, g


if __name__ == "__main__":
    region = load_region_bounds()
    fine = region["cell_size_deg"]
    table = load_transitions(region)
    stored = np.load(TRANSITIONS_PATH)
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    alpha = float(product["alpha"])
    land = land_fraction(region, fine, land_paths())

    models, n_cells, start_drifters, g = build_models(region, table, stored, alpha, land)
    if not np.allclose(models["water"]["P"], product["P"]):
        raise ValueError("Refitted shrinkage P differs from the saved product; rerun src.inference.shrinkage_fit.")
    shore = models["shore"]
    n = n_cells + 1
    p_ground = shore["P"][:n_cells, n:].sum(axis=1)
    if not (np.allclose(shore["P"].sum(axis=1), 1)
            and np.allclose(shore["P"][:n_cells, :n], product["P"][:n_cells] * (1 - p_ground)[:, None])):
        raise ValueError("The shore model's water block isn't the product scaled by the chance of staying afloat.")
    print(f"Groundings: {g['n_kept']} drifters on {shore['shores'].size} stretches of coast ({g['start_lon'].size:,} "
          f"starts); {g['n_dropped']} drifters left out, too far from the mapped coast. Largest 15-day grounding "
          f"chance in a cell with drifters {100 * p_ground[start_drifters > 0].max():.1f}%; {np.sum(p_ground > 0)} cells have any.")

    lat_c, lon_c = cell_centres(region, fine)
    coast = coastal_cells(region, fine, land)
    release = coast[start_drifters[coast] > 0]
    benguela = zone_cells(region, fine, region["coastal_zone"])
    sampled = np.flatnonzero(start_drifters > 0)
    rows = np.append(sampled, -1)  # every cell with drifters, then the whole-coast release
    start_lat = np.append(lat_c[sampled], lat_c[release].mean())
    start_lon = np.append(lon_c[sampled], lon_c[release].mean())
    print(f"{fine:g} deg shrinkage (alpha {alpha:.1f}): {sampled.size} start cells with drifters, coast release over "
          f"{release.size} cells, {benguela.size} cells in the Benguela zone, {MAX_STEPS} steps of {TAU_DAYS} days")

    out = {}
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
        print(f"{name} model: bootstrap, {N_BOOT} replicates of the drifters (seed {SEED}), {100 * LEVEL:g}% intervals")
        low, high = bootstrap_intervals(m["start"], m["end"], m["drifter_id"], m["n_states"], m["fit"], S, F0,
                                        start_lat, start_lon)
        if name == "water":
            worst = check_coast_release(low, high, estimates)
            if worst is not None:
                print(f"  coast release vs coastal_release_tau{TAU_DAYS}d.csv: largest difference {worst:.2g}")
                if worst > 1e-3:
                    raise ValueError("Coast release doesn't reproduce coastal_release.py; check both use the same product.")
        out[name] = {"quantities": list(QUANTITIES), "rows": rows.tolist(), "steps": MAX_STEPS,
                     "low": b64_float32(low), "high": b64_float32(high)}

    # The shore model is shipped as its grounding rows; the page rebuilds it from P.
    ground_rows = [[int(i), int(k), round(float(shore["P"][i, n + k]), 7)]
                   for i, k in zip(*np.nonzero(shore["P"][:n_cells, n:]))]
    zone = region["coastal_zone"]
    years = table["start_time"].astype("datetime64[Y]").astype(int) + 1970
    data = {
        "meta": {"built": datetime.date.today().isoformat(), "product": f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz",
                 "alpha": alpha, "n_boot": N_BOOT, "level": LEVEL, "n_drifters": int(np.unique(table["drifter_id"]).size),
                 "n_moves": int(table["drifter_id"].size), "years": [int(years.min()), int(years.max())],
                 "n_grounded": g["n_kept"], "n_grounded_dropped": g["n_dropped"]},
        "region": {k: float(region[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")} | {"label": region["label"]},
        "benguela": {k: float(zone[k]) for k in ("lat_min", "lat_max", "lon_min", "lon_max")} | {"label": zone["label"]},
        "cell_size": fine, "n_lat": int(product["n_lat_cells"]), "n_lon": int(product["n_lon_cells"]),
        "tau_days": TAU_DAYS, "max_steps": MAX_STEPS,
        "start_drifters": start_drifters.astype(int).tolist(),
        "all_land": np.flatnonzero(land >= 1).tolist(),
        "coastal_cells": coast.tolist(), "release_cells": release.tolist(), "benguela_cells": benguela.tolist(),
        "P": b64_float32(product["P"]),
        "shore": {"cells": shore["shores"].tolist(), "rows": ground_rows},
        "intervals": out,
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
