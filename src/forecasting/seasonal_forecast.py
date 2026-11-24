"""
seasonal_forecast.py

Related task: Week 6-8 Business - RQ4 seasonality (ECON2209 forecasting). Does the pattern of
connections change with the season, and can a season's pattern be forecast from earlier years?

Everything starts from the Week 3 transition table (data/processed/transitions_tau15d.npz), as the
project plan says: start_time is all the seasonal information needed. Grid: the 4 deg baseline,
because a season holds only about a quarter of the moves, and the 2 deg grid is already thin
(decision 3, 5). Seasons are austral: DJF (summer), MAM, JJA, SON; December belongs to the DJF of
the following year, so each season is three consecutive months.

Four parts, in the order they answer the question:

A. Is there a seasonal signal? Per season, with drifter-bootstrap intervals (whole drifters are
   resampled, as in src/inference/bootstrap.py):
     - offshore_km: mean westward (offshore) distance of a 15-day move starting in the Benguela
       coastal box (coastal_zone in config/region.yaml) and staying in R.
     - north_km: mean northward distance of the same moves.
     - exit_pct: share of all 15-day moves that leave R.
     - leave_north_pct: share of the same coastal moves that end north of the box. This is the
       scoping note's "share leaving the coastal cells northward within tau", measured on the box.
       It is rare, so it is in the tables and not the figures.
   (Quantities per 15-day move.) Pairs of seasons are then compared by resampling whole season-years
   (seasonal_signal_differences.csv), because drifters in the same year share that year's ocean.

B. Can those quantities be forecast? Rolling origin (expanding window): the value for each
   season-year from FIRST_TEST_YEAR on is forecast using only earlier years, by four methods:
   annual mean (no season, the benchmark), seasonal mean, recency-weighted seasonal mean, and last
   year's same season. The means pool earlier season-years by their moves, so a thin 1990s season
   does not count as much as a season with 10,000 moves. Scored by MAE; skill = 1 - MAE / MAE of the
   benchmark, with an interval from resampling whole test years (a year's four seasons are not
   independent). A season-year only counts if at least MIN_DRIFTERS distinct drifters contribute
   (the adequacy idea, decision 3).

C. Can the whole 4 deg matrix be forecast? Same rolling origin, scored by the log-likelihood the
   forecast matrix gives to the observed moves of the test season. The seasonal matrix is the
   same season's earlier counts shrunk towards the annual matrix,
   p_ij = (n_ij + alpha * q_ij) / (n_i + alpha), the Dirichlet-multinomial form of
   src/inference/estimators.py. Reported for several alpha, with drifter-bootstrap intervals.
   A probability floor EPS keeps log 0 from appearing, so this is a harsher test than the
   Quant evaluation (which scores held-out drifters, not future years).

D. What does the season change for the Benguela coast? Release material evenly over the coastal
   cells that have drifters (outputs/results/coastal_cells.csv, 4 deg rows) at the start of each
   season, apply that season's matrix for the first 6 steps (about 3 months = one season), then
   the next seasons' matrices in turn out to 24 steps (about a year). Compared with the
   all-season matrix. Bootstrap intervals refit every matrix on drifters resampled with replacement.
   Shrinkage sets how far a season's matrix can move from the all-season one, so the release is
   repeated at other alphas (seasonal_release_alpha_sensitivity.csv).

Outputs (outputs/results/ and outputs/figures/): seasonal_coverage.csv, seasonal_signal.csv,
seasonal_signal_differences.csv, seasonal_forecast_scalar.csv, seasonal_forecast_scalar_detail.csv,
seasonal_matrix_skill.csv, seasonal_release_tau15d.csv, seasonal_release_alpha_sensitivity.csv;
seasonal_signal.png, seasonal_forecast_scalar.png, seasonal_matrix_skill.png,
seasonal_release_3months.png.

Run: python3 -m src.forecasting.seasonal_forecast            (about 1-2 minutes)
     python3 -m src.forecasting.seasonal_forecast --n-boot 200   (quicker, for a first look)
Needs only numpy, matplotlib and pyyaml (no clouddrift): it reads the saved table directly.
"""
import argparse
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Polygon

from src.common.config import FIGURES_DIR, PROCESSED_DIR, REPO_ROOT, RESULTS_DIR, load_region_bounds
from src.gridding.grid import cell_index, grid_edges

TABLE_PATH = os.path.join(PROCESSED_DIR, "transitions_tau15d.npz")
COASTAL_CELLS_PATH = os.path.join(RESULTS_DIR, "coastal_cells.csv")
LAND_PATH = os.path.join(REPO_ROOT, "data", "external", "ne_110m_land.geojson")

IN_R, LEFT_R = 0, 1            # outcome codes, as in src/gridding/transitions.py (2, 3 have no destination)
CELL_SIZE = 4.0                # deg; the baseline grid
SEASONS = ("DJF", "MAM", "JJA", "SON")
STEPS_PER_SEASON = 6           # 6 x 15 days = 90 days, one season
RELEASE_STEPS = (6, 24)        # about 3 months and about a year
KM_PER_DEG = 111.195

FIRST_TEST_YEAR = 2008         # data is thin before 2003 (coverage_map.png); 2003-2007 trains the first forecast
MIN_DRIFTERS = 15              # distinct drifters needed for a season-year value to count
MIN_SAME_SEASON = 2            # earlier years of the same season needed before forecasting it
HALFLIFE = 3.0                 # years, recency weighting
MIN_TEST_MOVES = 500           # a test season with fewer moves is skipped (matrix scoring)
ALPHAS = (10, 50, 250, 1000, 2000, 5000)   # pseudo-moves of annual prior, matrix forecast (the gain peaks inside this range)
ALPHA_RELEASE = 20             # light shrinkage for the descriptive release maps (all years, no forecasting)
ALPHA_RELEASE_CHECKS = (250, 1000)   # the release repeated with heavier shrinkage
SEASON_PAIRS = ((0, 1), (0, 2), (3, 0), (1, 2), (3, 1), (3, 2))   # (a, b) for "a minus b", as indices into SEASONS
EPS = 1e-3                     # probability floor mixed into forecast matrices before taking logs
N_BOOT = 1000
SEED = 3001
LEVEL = 95

BENCH = "Annual mean (benchmark)"
METHODS = (BENCH, "Seasonal mean", "Recency-weighted seasonal mean", "Last year's season (naive)")
TARGETS = {                    # the plotted targets; leave_north_pct is in the tables only
    "offshore_km": "Coastal offshore (westward) km per 15 d",
    "north_km": "Coastal northward km per 15 d",
    "exit_pct": "Moves leaving R (%), whole box",
}
BLUE, ORANGE = "#1f6aa5", "#d9822b"


# ----------------------------------------------------------------------------- data
def season_index(month):
    """0 DJF, 1 MAM, 2 JJA, 3 SON from calendar month 1-12 (December counts as DJF)."""
    return (np.asarray(month) % 12) // 3


def load_table(path=TABLE_PATH, region=None):
    """Usable transitions (stayed in R or left R) with season, season-year, km displacement and 4 deg states.

    States are numbered as in grid.cell_index; the extra state n_cells means "left R".
    """
    region = region or load_region_bounds()
    z = np.load(path)
    keep = z["outcome"] <= LEFT_R
    start_time = z["start_time"][keep]
    if not np.issubdtype(start_time.dtype, np.datetime64):
        start_time = start_time.astype("datetime64[s]")
    months = start_time.astype("datetime64[M]")
    month = months.astype(int) % 12 + 1
    year = months.astype("datetime64[Y]").astype(int) + 1970

    t = {k: z[k][keep].astype(np.float64) for k in ("start_lon", "start_lat", "end_lon", "end_lat")}
    t["drifter_id"] = z["drifter_id"][keep]
    t["exited"] = z["outcome"][keep] == LEFT_R
    t["season"] = season_index(month)
    t["syear"] = year + (month == 12)            # December belongs to next year's DJF
    t["rank"] = np.unique(t["drifter_id"], return_inverse=True)[1]
    t["n_drifters"] = int(t["rank"].max()) + 1

    lat_mid = np.radians(0.5 * (t["start_lat"] + t["end_lat"]))
    t["east_km"] = (t["end_lon"] - t["start_lon"]) * KM_PER_DEG * np.cos(lat_mid)   # NaN for exits
    t["north_km"] = (t["end_lat"] - t["start_lat"]) * KM_PER_DEG
    zone = region["coastal_zone"]
    t["in_band"] = ((t["start_lat"] >= zone["lat_min"]) & (t["start_lat"] <= zone["lat_max"])
                    & (t["start_lon"] >= zone["lon_min"]) & (t["start_lon"] <= zone["lon_max"]))
    t["north_of_band"] = t["in_band"] & (t["end_lat"] > zone["lat_max"])   # False for exits (no end position)

    t["start"], n_lat, n_lon = cell_index(t["start_lon"], t["start_lat"], region, CELL_SIZE)
    t["n_cells"] = n_lat * n_lon
    t["end"] = np.full(t["start"].shape, t["n_cells"])
    stay = ~t["exited"]
    t["end"][stay], _, _ = cell_index(t["end_lon"][stay], t["end_lat"][stay], region, CELL_SIZE)
    return t


# ----------------------------------------------------------------------------- bootstrap helpers
def drifter_sums(values, mask, t):
    """Per-drifter sum and count of `values` over rows in mask (rows with NaN must be masked out)."""
    r = t["rank"][mask]
    sums = np.bincount(r, weights=values[mask], minlength=t["n_drifters"])
    counts = np.bincount(r, minlength=t["n_drifters"]).astype(float)
    return sums, counts


def boot_ratio(sums, counts, n_boot, rng):
    """(estimate, lo, hi) of sum/count, resampling whole drifters that contribute at least one move."""
    keep = counts > 0
    s, c = sums[keep], counts[keep]
    draws = rng.integers(0, s.size, (n_boot, s.size))
    est = s[draws].sum(axis=1) / c[draws].sum(axis=1)
    lo, hi = np.percentile(est, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return s.sum() / c.sum(), lo, hi


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))   # union, so rows may differ (benchmark has no skill)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _round(row, nd=4):
    return {k: (round(float(v), nd) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}


# ----------------------------------------------------------------------------- targets
def target_values(t):
    """For each target: (value per row, mask of rows that count). Values are only read where mask is True."""
    band_stay = t["in_band"] & ~t["exited"]
    return {
        "offshore_km": (-t["east_km"], band_stay),
        "north_km": (t["north_km"], band_stay),
        "exit_pct": (100.0 * t["exited"], np.ones_like(t["exited"])),
        "leave_north_pct": (100.0 * t["north_of_band"], band_stay),
    }


def group_rows(t):
    """Row indices for every (season-year, season) present."""
    key = t["syear"] * 4 + t["season"]
    order = np.argsort(key, kind="stable")
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    return {(int(key[g[0]] // 4), int(key[g[0]] % 4)): g for g in np.split(order, bounds)}


def season_year_series(values, mask, t, groups, min_drifters=MIN_DRIFTERS):
    """({(season-year, season): mean value}, {same keys: moves}) for season-years with enough distinct drifters."""
    out, moves = {}, {}
    for k, rows in groups.items():
        rows = rows[mask[rows]]
        if rows.size and np.unique(t["rank"][rows]).size >= min_drifters:
            out[k] = float(values[rows].mean())
            moves[k] = int(rows.size)
    return out, moves


# ----------------------------------------------------------------------------- part A
def seasonal_signal(t, n_boot, rng):
    rows = []
    for name, (values, mask) in target_values(t).items():
        for s, label in enumerate(SEASONS):
            m = mask & (t["season"] == s)
            est, lo, hi = boot_ratio(*drifter_sums(values, m, t), n_boot, rng)
            rows.append(_round({"quantity": name, "season": label, "estimate": est, "ci_low": lo, "ci_high": hi,
                                "n_moves": int(m.sum()), "n_drifters": int(np.unique(t["rank"][m]).size)}))
    return rows


def coverage_table(t, groups):
    rows = []
    for (y, s), idx in sorted(groups.items()):
        band = idx[t["in_band"][idx]]
        rows.append({"season_year": y, "season": SEASONS[s], "moves": idx.size,
                     "drifters": int(np.unique(t["rank"][idx]).size), "coastal_moves": band.size,
                     "coastal_drifters": int(np.unique(t["rank"][band]).size),
                     "exits": int(t["exited"][idx].sum())})
    return rows


def season_differences(t, groups, n_boot, rng):
    """Differences between seasons' pooled means, with intervals from resampling whole season-years.

    Wider than the per-season drifter bootstrap, because it includes how much a season changes from year to year.
    years_higher counts the years the first season is higher, of the years where both pass MIN_DRIFTERS.
    """
    years = np.unique(t["syear"])
    draws = rng.integers(0, years.size, (n_boot, years.size))
    rows = []
    for name, (values, mask) in target_values(t).items():
        at = (np.searchsorted(years, t["syear"][mask]), t["season"][mask])
        sums, counts = np.zeros((years.size, 4)), np.zeros((years.size, 4))
        np.add.at(sums, at, values[mask])
        np.add.at(counts, at, 1)
        pooled = sums.sum(axis=0) / counts.sum(axis=0)
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        series, _ = season_year_series(values, mask, t, groups)
        for a, b in SEASON_PAIRS:
            lo, hi = np.percentile(boot[:, a] - boot[:, b], [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
            both = [y for y, s in series if s == a and (y, b) in series]
            rows.append(_round({"quantity": name, "season": SEASONS[a], "minus": SEASONS[b],
                                "difference": pooled[a] - pooled[b], "ci_low": lo, "ci_high": hi,
                                "years_higher": sum(series[(y, a)] > series[(y, b)] for y in both),
                                "years_compared": len(both)}))
    return rows


# ----------------------------------------------------------------------------- part B
def rolling_forecasts(series, first_test_year=FIRST_TEST_YEAR, halflife=HALFLIFE, moves=None):
    """One row per forecastable season-year: truth and each method's forecast, using only earlier years.

    The means weight earlier season-years by their moves; with moves=None each season-year counts equally.
    """
    rows = []
    for (year, s), truth in sorted(series.items()):
        if year < first_test_year:
            continue
        prior = {k: v for k, v in series.items() if k[0] < year}
        same = sorted((y, v) for (y, ss), v in prior.items() if ss == s)
        if len(same) < MIN_SAME_SEASON:
            continue
        years_same = np.array([y for y, _ in same], float)
        vals_same = np.array([v for _, v in same])
        w_prior = [1.0 if moves is None else moves[k] for k in prior]
        w_same = np.array([1.0 if moves is None else moves[(y, s)] for y, _ in same])
        rows.append({
            "season_year": year, "season": s, "truth": truth,
            BENCH: float(np.average(list(prior.values()), weights=w_prior)),
            "Seasonal mean": float(np.average(vals_same, weights=w_same)),
            "Recency-weighted seasonal mean": float(np.average(vals_same, weights=w_same * 0.5 ** ((year - years_same) / halflife))),
            "Last year's season (naive)": float(vals_same[-1]),   # the latest earlier year that has this season
        })
    return rows


def skill_with_ci(rows, n_boot, rng):
    """MAE and skill (1 - MAE / benchmark MAE) per method; interval from resampling whole test years."""
    years = np.array([r["season_year"] for r in rows])
    uy = np.unique(years)
    n = np.array([(years == y).sum() for y in uy], float)
    abs_err = {m: np.abs(np.array([r[m] - r["truth"] for r in rows])) for m in METHODS}
    per_year = {m: np.array([a[years == y].sum() for y in uy]) for m, a in abs_err.items()}
    draws = rng.integers(0, uy.size, (n_boot, uy.size))
    mae_bench_boot = per_year[BENCH][draws].sum(axis=1) / n[draws].sum(axis=1)
    out = []
    for m in METHODS:
        mae = abs_err[m].mean()
        row = {"method": m, "n_forecasts": len(rows), "MAE": mae}
        if m != BENCH:
            boot = 1 - (per_year[m][draws].sum(axis=1) / n[draws].sum(axis=1)) / mae_bench_boot
            row.update(skill=1 - mae / abs_err[BENCH].mean(),
                       ci_low=np.percentile(boot, (100 - LEVEL) / 2), ci_high=np.percentile(boot, 100 - (100 - LEVEL) / 2))
        out.append(row)
    return out


def scalar_forecasts(t, groups, n_boot, rng):
    skill_rows, detail_rows, details = [], [], {}
    for name, (values, mask) in target_values(t).items():
        series, moves = season_year_series(values, mask, t, groups)
        rows = rolling_forecasts(series, moves=moves)
        details[name] = rows
        for r in skill_with_ci(rows, n_boot, rng):
            skill_rows.append(_round({"target": name, **r}))
        for r in rows:
            detail_rows.append(_round({"target": name, **r, "season": SEASONS[r["season"]]}))
    return skill_rows, detail_rows, details


# ----------------------------------------------------------------------------- matrices
def empirical_P(counts):
    """Counts / row totals; rows with no moves stay put (so exit stays absorbing and no mass is lost)."""
    totals = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    empty = totals[:, 0] == 0
    P[empty] = np.eye(counts.shape[0])[empty]
    return P


def shrink(counts, prior, alpha):
    """Dirichlet-multinomial posterior mean (n_ij + alpha q_ij) / (n_i + alpha); rows sum to 1."""
    return (counts + alpha * prior) / (counts.sum(axis=1, keepdims=True) + alpha)


def floor(P, eps=EPS):
    return (1 - eps) * P + eps / P.shape[0]


def season_counts(t, n_states, weights=None):
    """{(season-year, season): count matrix} (dense, n_states x n_states)."""
    pair = t["start"] * n_states + t["end"]
    w = None if weights is None else weights[t["rank"]]
    return {k: np.bincount(pair[g], weights=None if w is None else w[g], minlength=n_states ** 2)
            .reshape(n_states, n_states).astype(float) for k, g in group_rows(t).items()}


def matrix_skill(t, groups, alphas, n_boot, rng):
    """Out-of-sample log-likelihood gain per move over the annual matrix, by alpha and method."""
    S = t["n_cells"] + 1
    counts = season_counts(t, S)
    methods = ("Seasonal", "Recency-weighted seasonal", "Last year's season")
    gain = {(m, a): np.zeros(t["n_drifters"]) for m in methods for a in alphas}
    moves = np.zeros(t["n_drifters"])
    for (year, s), idx in sorted(groups.items()):
        if year < FIRST_TEST_YEAR or idx.size < MIN_TEST_MOVES:
            continue
        prior_keys = [k for k in counts if k[0] < year]
        G = empirical_P(sum(counts[k] for k in prior_keys))
        i, j, r = t["start"][idx], t["end"][idx], t["rank"][idx]
        base = np.log(floor(G)[i, j])
        moves += np.bincount(r, minlength=t["n_drifters"])
        years = sorted({k[0] for k in prior_keys if k[1] == s})
        weights = {"Seasonal": {y: 1.0 for y in years},
                   "Recency-weighted seasonal": {y: 0.5 ** ((year - y) / HALFLIFE) for y in years},
                   "Last year's season": {years[-1]: 1.0} if years else {}}
        for m, w in weights.items():
            n = sum((w[y] * counts[(y, s)] for y in w), np.zeros((S, S)))
            for a in alphas:
                ll = np.log(floor(shrink(n, G, a))[i, j])
                gain[(m, a)] += np.bincount(r, weights=ll - base, minlength=t["n_drifters"])
    rows = []
    for (m, a), g in gain.items():
        est, lo, hi = boot_ratio(g, moves, n_boot, rng)
        rows.append(_round({"method": m, "alpha": a, "gain_per_move": est, "ci_low": lo, "ci_high": hi,
                            "test_moves": int(moves.sum())}, 5))
    return rows


# ----------------------------------------------------------------------------- part D
def load_coastal_cells(path=COASTAL_CELLS_PATH, cell_size=CELL_SIZE):
    """(release cells = coastal cells with drifters, coastal zone = all coastal cells) at this cell size."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if float(r["cell_size_deg"]) == cell_size]
    zone = np.array([int(r["cell"]) for r in rows])
    release = np.array([int(r["cell"]) for r in rows if int(r["start_drifters"]) > 0])
    return release, zone


def propagate_cycle(season_P, first_season, release, steps=RELEASE_STEPS):
    """Distribution after each count in `steps` when step k uses the matrix of the season it falls in.

    Release is at the start of first_season, which runs for STEPS_PER_SEASON steps, then the next season, and so on.
    """
    x = np.zeros(season_P[0].shape[0])
    x[release] = 1.0 / release.size
    kept = {}
    for k in range(1, max(steps) + 1):
        x = x @ season_P[(first_season + (k - 1) // STEPS_PER_SEASON) % 4]
        if k in steps:
            kept[k] = x.copy()
    return kept


def centre_of_mass(dist, region):
    """Probability-weighted mean (lat, lon) of cell centres over material still inside R."""
    lat_e, lon_e = grid_edges(region, CELL_SIZE)
    lat_c, lon_c = 0.5 * (lat_e[:-1] + lat_e[1:]), 0.5 * (lon_e[:-1] + lon_e[1:])
    inside = dist[:-1].reshape(len(lat_c), len(lon_c))
    w = inside / inside.sum()
    return float((w.sum(axis=1) * lat_c).sum()), float((w.sum(axis=0) * lon_c).sum())


def release_quantities(dist, zone, region):
    lat, lon = centre_of_mass(dist, region)
    return {"pct_left_R": 100 * dist[-1], "pct_in_coastal_zone": 100 * dist[zone].sum(),
            "centre_lat": lat, "centre_lon": lon}


def release_run(season_counts_by_season, release, zone, region, alpha=ALPHA_RELEASE):
    """{label: {steps: dist}} for each release season and for the all-season matrix."""
    total = sum(season_counts_by_season)
    G = empirical_P(total)
    P_s = [shrink(c, G, alpha) for c in season_counts_by_season]
    out = {SEASONS[s]: propagate_cycle(P_s, s, release) for s in range(4)}
    out["All seasons"] = propagate_cycle([G] * 4, 0, release)
    return out


def release_table(t, region, n_boot, rng, alpha=ALPHA_RELEASE):
    release, zone = load_coastal_cells()
    S = t["n_cells"] + 1
    pair = t["start"] * S + t["end"]
    by_season = [np.flatnonzero(t["season"] == s) for s in range(4)]

    def counts_for(weights):
        w = weights[t["rank"]]
        return [np.bincount(pair[idx], weights=w[idx], minlength=S * S).reshape(S, S) for idx in by_season]

    def quantities(weights):
        runs = release_run(counts_for(weights), release, zone, region, alpha)
        q = {(lab, k): release_quantities(d, zone, region) for lab, dd in runs.items() for k, d in dd.items()}
        for lab in SEASONS:                                   # season minus all-season, paired within a replicate
            for k in RELEASE_STEPS:
                q[(f"{lab} minus all seasons", k)] = {n: q[(lab, k)][n] - q[("All seasons", k)][n] for n in q[(lab, k)]}
        return q

    point = quantities(np.ones(t["n_drifters"]))
    draws = [quantities(rng.multinomial(t["n_drifters"], np.full(t["n_drifters"], 1.0 / t["n_drifters"])).astype(float))
             for _ in range(n_boot)]
    rows, summary = [], {}
    for (lab, k), vals in point.items():
        for name, est in vals.items():
            reps = np.array([d[(lab, k)][name] for d in draws])
            lo, hi = np.percentile(reps, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
            rows.append(_round({"release_season": lab, "steps": k, "days": 15 * k, "quantity": name,
                                "estimate": est, "ci_low": lo, "ci_high": hi}))
            summary[(lab, k, name)] = (est, lo, hi)
    dists = release_run(counts_for(np.ones(t["n_drifters"])), release, zone, region, alpha)
    return rows, dists, release, summary


# ----------------------------------------------------------------------------- plots
def draw_land(ax, path=LAND_PATH):
    with open(path) as f:
        features = json.load(f)["features"]
    for feature in features:
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for rings in polygons:
            ax.add_patch(Polygon(rings[0], closed=True, fc="#e6e6e6", ec="#7f7f7f", lw=0.6, zorder=3))


def plot_signal(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, (name, title) in zip(axes, TARGETS.items()):
        r = [x for x in rows if x["quantity"] == name]
        y = np.array([x["estimate"] for x in r])
        ax.errorbar(SEASONS, y, yerr=[y - [x["ci_low"] for x in r], [x["ci_high"] for x in r] - y],
                    fmt="o", capsize=5, color=BLUE)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Seasonal signal by season, {LEVEL}% drifter-bootstrap intervals", fontsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def broken_at_gaps(x, y, longest=0.25):
    """x and y with a NaN wherever the next season is missing, so a line doesn't run across years with no forecast."""
    xs, ys = [x[0]], [y[0]]
    for a, b, v in zip(x[:-1], x[1:], y[1:]):
        if b - a > longest + 1e-9:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(b)
        ys.append(v)
    return xs, ys


def plot_scalar(details, skill_rows, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for col, (name, title) in enumerate(TARGETS.items()):
        rows = details[name]
        x = [r["season_year"] + r["season"] / 4 for r in rows]
        ax = axes[0, col]
        ax.plot(*broken_at_gaps(x, [r["truth"] for r in rows]), "k-o", ms=3, label="Actual")
        ax.plot(*broken_at_gaps(x, [r["Seasonal mean"] for r in rows]), "--", color=BLUE, label="Seasonal mean forecast")
        ax.plot(*broken_at_gaps(x, [r[BENCH] for r in rows]), ":", color=ORANGE, label="Annual mean (benchmark)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Season-year")
        if col == 0:
            ax.legend(fontsize=7)
        sk = [r for r in skill_rows if r["target"] == name and r["method"] != BENCH]
        y = np.array([r["skill"] for r in sk])
        axes[1, col].barh([r["method"] for r in sk], y,
                          xerr=[y - [r["ci_low"] for r in sk], [r["ci_high"] for r in sk] - y], capsize=3, color=BLUE)
        axes[1, col].axvline(0, color="k", lw=1)
        axes[1, col].set_xlabel("Skill vs annual mean")
    fig.text(0.01, -0.03, "Top: each season-year's value and the forecasts made from earlier years only; lines break where a season has too few drifters. "
             "Bottom: skill = 1 - MAE / MAE of the annual mean, so above 0 the method beats ignoring the season.\n"
             f"Bars: {LEVEL}% intervals from resampling whole test years.", fontsize=9)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_matrix_skill(rows, path):
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    for m, c in zip(("Seasonal", "Recency-weighted seasonal", "Last year's season"), (BLUE, ORANGE, "#6b6b6b")):
        r = sorted((x for x in rows if x["method"] == m), key=lambda x: x["alpha"])
        a = [x["alpha"] for x in r]
        ax.plot(a, [x["gain_per_move"] for x in r], "-o", color=c, label=m)
        ax.fill_between(a, [x["ci_low"] for x in r], [x["ci_high"] for x in r], color=c, alpha=0.2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (pseudo-moves of the annual matrix)")
    ax.set_ylabel("Log-likelihood gain per move vs annual matrix\n(>0: the season helps)")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_release(dists, release, zone, region, path):
    lat_e, lon_e = grid_edges(region, CELL_SIZE)
    shape = (len(lat_e) - 1, len(lon_e) - 1)
    labels = list(SEASONS) + ["All seasons"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharey=True, layout="constrained")
    for ax, lab in zip(axes, labels):
        grid = dists[lab][STEPS_3M_KEY][:-1].reshape(shape) / (CELL_SIZE / 2) ** 2   # per 2 x 2 deg, as in the Week 6 maps
        mesh = ax.pcolormesh(lon_e, lat_e, np.ma.masked_less_equal(grid, 1e-6),
                             norm=colors.LogNorm(vmin=1e-4, vmax=0.3), cmap="Blues")
        draw_land(ax)
        ax.set_xlim(region["lon_min"], region["lon_max"])
        ax.set_ylim(region["lat_min"], region["lat_max"])
        ax.set_aspect("equal")
        ax.set_title(f"Released at start of {lab}" if lab != "All seasons" else "All-season matrix", fontsize=10)
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    fig.colorbar(mesh, ax=axes, label="Probability per 2 x 2 degree area", shrink=0.8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


STEPS_3M_KEY = 6


# ----------------------------------------------------------------------------- main
def main(n_boot=N_BOOT):
    region = load_region_bounds()
    t = load_table(region=region)
    groups = group_rows(t)
    rng = np.random.default_rng(SEED)
    fig_path = lambda n: os.path.join(FIGURES_DIR, n)
    res_path = lambda n: os.path.join(RESULTS_DIR, n)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"{t['start'].size:,} usable moves, {t['n_drifters']:,} drifters, "
          f"season-years {t['syear'].min()}-{t['syear'].max()}")
    write_csv(res_path("seasonal_coverage.csv"), coverage_table(t, groups))

    signal = seasonal_signal(t, n_boot, rng)
    write_csv(res_path("seasonal_signal.csv"), signal)
    plot_signal(signal, fig_path("seasonal_signal.png"))
    print("\nA. Seasonal signal (estimate [95% CI])")
    for r in signal:
        print(f"  {r['quantity']:12s} {r['season']}  {r['estimate']:8.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print(f"  Coastal box: {t['north_of_band'].sum():,} of {(t['in_band'] & ~t['exited']).sum():,} moves end north of it, "
          f"from {np.unique(t['rank'][t['north_of_band']]).size} drifters; {(t['in_band'] & t['exited']).sum()} leave R")

    skill, detail, details = scalar_forecasts(t, groups, n_boot, rng)
    write_csv(res_path("seasonal_forecast_scalar.csv"), skill)
    write_csv(res_path("seasonal_forecast_scalar_detail.csv"), detail)
    plot_scalar(details, skill, fig_path("seasonal_forecast_scalar.png"))
    print(f"\nB. Rolling-origin forecasts from {FIRST_TEST_YEAR} (skill vs annual mean, >0 = season helps)")
    for r in skill:
        if r["method"] != BENCH:
            print(f"  {r['target']:12s} {r['method']:32s} n={r['n_forecasts']:3d} skill {r['skill']:+.3f} "
                  f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")

    mskill = matrix_skill(t, groups, ALPHAS, n_boot, rng)
    write_csv(res_path("seasonal_matrix_skill.csv"), mskill)
    plot_matrix_skill(mskill, fig_path("seasonal_matrix_skill.png"))
    print("\nC. Whole-matrix forecast, log-likelihood gain per move vs annual matrix")
    for r in mskill:
        print(f"  {r['method']:26s} alpha {r['alpha']:5d}  {r['gain_per_move']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

    rel, dists, release_cells, summary = release_table(t, region, n_boot, rng)
    _, zone = load_coastal_cells()
    write_csv(res_path("seasonal_release_tau15d.csv"), rel)
    plot_release(dists, release_cells, zone, region, fig_path("seasonal_release_3months.png"))
    print("\nD. Release from the Benguela coastal cells, after 3 months (estimate [95% CI])")
    for lab in list(SEASONS) + ["All seasons"]:
        e = {q: summary[(lab, 6, q)] for q in ("pct_left_R", "pct_in_coastal_zone", "centre_lat", "centre_lon")}
        print(f"  {lab:11s} left R {e['pct_left_R'][0]:.2f}% | coastal zone {e['pct_in_coastal_zone'][0]:.2f}% "
              f"[{e['pct_in_coastal_zone'][1]:.2f}, {e['pct_in_coastal_zone'][2]:.2f}] | "
              f"centre {e['centre_lat'][0]:.1f}, {e['centre_lon'][0]:.1f}")

    sensitivity = [{"alpha": ALPHA_RELEASE, **r} for r in rel]
    for a in ALPHA_RELEASE_CHECKS:
        sensitivity += [{"alpha": a, **r} for r in release_table(t, region, n_boot, rng, alpha=a)[0]]
    write_csv(res_path("seasonal_release_alpha_sensitivity.csv"), sensitivity)
    print("\nD. DJF minus all seasons in the coastal zone after 3 months, by alpha")
    for r in sensitivity:
        if r["release_season"] == "DJF minus all seasons" and r["steps"] == 6 and r["quantity"] == "pct_in_coastal_zone":
            print(f"  alpha {r['alpha']:5d}  {r['estimate']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")

    diffs = season_differences(t, groups, n_boot, rng)
    write_csv(res_path("seasonal_signal_differences.csv"), diffs)
    print("\nA. Differences between seasons, resampling whole season-years")
    for r in diffs:
        print(f"  {r['quantity']:15s} {r['season']} - {r['minus']}  {r['difference']:+8.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
              f"  higher in {r['years_higher']} of {r['years_compared']} years")
    print(f"\nSaved tables to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap replicates (default %(default)s)")
    main(ap.parse_args().n_boot)
