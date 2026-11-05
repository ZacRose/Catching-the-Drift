"""
seasonal_matrices.py

Related task: RQ4 and the app's release-season switch. seasonal_forecast.py asks whether drift is seasonal, at 4 deg.
The app needs the same idea on its own 2 deg grid, where a season's own counts are far too thin to stand alone, and it
needs an answer to what happens when a release runs from one season into the next.

A season's matrix. The all-year matrix P (the product, or the undrogued one) is the prior, and the season's own counts
pull each row away from it, by the same formula as the product's 2 deg rows:
    P^s_ij = (n^s_ij + beta * P_ij) / (n^s_i + beta)
So a row with few moves in that season stays close to its all-year row, however thin, and a row with no moves equals it.

Choosing beta. Drifters in one year share that year's eddies, so a drifter split would reward a matrix for learning the
year rather than the season. A season has to recur, so beta is chosen by leaving out whole years: fit on the train and
valid drifters of every other year, score the same drifters' moves in the year left out (leave_year_out). The test
drifters are then scored the same way, so they are new drifters in a new year. Reported with the gain over the all-year
matrix, resampling test drifters.

Crossing seasons. Material doesn't wait for the season to end, so a chain uses, at each 15-day step, the matrix of the
season that step starts in. This is how the counts were made: a move belongs to the season of its start. The rule is
tested on test drifters at 30, 90, 180 and 360 days by the date each step really starts, against the all-year matrix and
against keeping the release season's matrix throughout (chain_scores). The matrices are fitted on the odd years and scored
on the even ones, and the other way round, so no fit sees the year a test move starts in (a long move can run into one).

In the app and in Product.forecast, a release is at the first day of its season and the matrix changes every
STEPS_PER_SEASON steps (90 days), as in seasonal_forecast.py Part D. A year is then 24 steps, 360 days: about 16 days
short by year 3, under the length of one step.

Status of the numbers. This is the typical year's pattern, not a forecast for a particular year. Shrinkage pulls the
seasons together, so "signal kept" reports how much of the raw gap in westward drift survives, and "other beta" how much
the release changes with beta. Keeping little is a result, not a fault of the method: the score counts every destination
of every row, and beyond the mean drift the detail of a season doesn't recur (a prior carrying a 4 deg seasonal shift
down to 2 deg did no better; project_plan.txt, decision 20). The seasons' grounding chance isn't separated (the app's
shore model scales each season's rows by the all-year chance): groundings are too few.

Intervals: drifter bootstrap, N_BOOT_SEASON replicates, refitting P and all four seasons with alpha and beta held fixed.

Outputs: data/processed/seasonal_counts_2deg_tau15d.npz (each set's season counts and beta; P^s is rebuilt from these
and P, so the four dense matrices aren't kept), outputs/results/seasonal_matrices_tau15d.csv and
outputs/figures/seasonal_matrices.png.

Run: python3 -m src.forecasting.seasonal_matrices   (about 3 minutes; after src.inference.drogue_comparison. The 180 and
360-day moves need the region and records caches, and are skipped without them)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.forecasting.seasonal_forecast import KM_PER_DEG, SEASONS, STEPS_PER_SEASON, season_index
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.trajectory_map import _degrees, draw_land
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS
from src.inference.bootstrap import LEVEL, SEED, drifter_totals, percentile_interval, resample_counts, resampled_means, shrinkage_fit
from src.inference.calibration import lag_table
from src.inference.drogue_comparison import UNDROGUED_STEM, undrogued_only
from src.inference.estimators import shrinkage_P
from src.inference.evaluate import pick
from src.inference.split import load_split, set_mask
from src.inference.transition_data import load_transitions, parent_states, subset, to_states
from src.linalg.connectivity import COAST_NAME, absorbing, box_cells, start_matrix

COUNTS_PATH = os.path.join(PROCESSED_DIR, f"seasonal_counts_2deg_tau{TAU_DAYS}d.npz")
BETAS = np.logspace(0.5, 5, 19)  # about 3 to 100,000 pseudo-moves of the all-year row
N_BOOT_SEASON = 200
CHAIN_DAYS = (30, 90, 180, 360)
OTHER_BETAS = (1 / 3, 3)  # the release repeated at these multiples of the chosen beta
RELEASE_STEPS = (1, 6, 12, 24)
SETS = ("all", "undrogued")
SET_NAMES = {"all": "All drifters", "undrogued": "Undrogued only"}
SEASON_NAMES = ("Summer", "Autumn", "Winter", "Spring")  # DJF, MAM, JJA, SON (southern hemisphere)
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")   # lags.py's three, then the next slot of the same validated palette
DJF, JJA = 0, 2


def season_and_year(start_time):
    """Each move's season index (into SEASONS) and calendar year, from its start."""
    months = np.asarray(start_time).astype("datetime64[M]")
    return season_index(months.astype(int) % 12 + 1), months.astype("datetime64[Y]").astype(int) + 1970


def season_counts(start, end, season, n_states):
    """Count matrices per season: (4, n_states, n_states)."""
    key = season.astype(np.int64) * n_states ** 2 + start.astype(np.int64) * n_states + end
    return np.bincount(key, minlength=4 * n_states ** 2).reshape(4, n_states, n_states).astype(float)


def season_matrices(P, counts, beta, n_water=None):
    """The four seasons' matrices: (4, n, n). Row i of season s is the all-year row pulled towards that season's moves.

    Inputs:
        P: the all-year matrix. counts: season_counts().
        n_water: the states that drift (the cells and "left the box"), if P has "ran aground" states after them. Then
        the chance of running aground stays the all-year one, and only the water part changes with the season.
    """
    n = P.shape[0] if n_water is None else n_water
    afloat = 1.0 - P[:n, n:].sum(axis=1, keepdims=True)
    water = np.divide(P[:n, :n], afloat, out=P[:n, :n].copy(), where=afloat > 0)
    out = np.repeat(P[None], 4, axis=0)
    for s in range(4):
        out[s, :n, :n] = shrinkage_P(counts[s, :n, :n], water, beta) * afloat
    return out


def chain_season(first_season, step, per_season=STEPS_PER_SEASON):
    """The season of a chain's step (0 = the first), for a release at the first day of first_season."""
    return (first_season + step // per_season) % 4


def season_replicates(start, end, season, drifter_id, n_states, fit, n_boot=N_BOOT_SEASON, seed=SEED):
    """Yield (P, counts) per replicate of the drifters: P = fit(all moves), counts as season_counts(). Same draws as bootstrap.py."""
    ids, rank = np.unique(drifter_id, return_inverse=True)
    size = 4 * n_states ** 2
    key = rank.astype(np.int64) * size + season.astype(np.int64) * n_states ** 2 + start.astype(np.int64) * n_states + end
    triples, n = np.unique(key, return_counts=True)
    drifter, pair = triples // size, triples % size
    for times_drawn in resample_counts(ids.size, n_boot, seed):
        counts = np.bincount(pair, weights=times_drawn[drifter] * n, minlength=size).reshape(4, n_states, n_states)
        yield fit(counts.sum(axis=0)), counts


def leave_year_out(start, end, season, year, fit_rows, score_rows, fit, betas, n_states):
    """log p of every scored move under matrices fitted without the year it starts in.

    Inputs:
        fit_rows, score_rows: bool per move. Only fit_rows' moves outside the scored move's year are fitted.
        fit: count matrix -> all-year P.
    Returns:
        (all-year log p, one per scored move; season log p, one row per beta).
    """
    idx = np.flatnonzero(score_rows)
    annual, seasonal = np.empty(idx.size), np.empty((len(betas), idx.size))
    for y in np.unique(year[idx]):
        train = fit_rows & (year != y)
        counts = season_counts(start[train], end[train], season[train], n_states)
        P = fit(counts.sum(axis=0))
        totals = counts.sum(axis=2)
        here = year[idx] == y
        a, b, s = start[idx[here]], end[idx[here]], season[idx[here]]
        annual[here] = np.log(P[a, b])
        for k, beta in enumerate(betas):
            seasonal[k, here] = np.log((counts[s, a, b] + beta * P[a, b]) / (totals[s, a] + beta))
    return annual, seasonal


def step_seasons(start_time, n_steps, tau=TAU_DAYS):
    """The season each step of a chain starts in, by date: (moves, n_steps)."""
    offset = (np.arange(n_steps) * tau).astype("timedelta64[D]")
    when = np.asarray(start_time).astype("datetime64[D]")[:, None] + offset
    return season_index(when.astype("datetime64[M]").astype(int) % 12 + 1)


def chain_scores(P, seasons, start, end, start_time, n_steps):
    """log p of each move under three chains of n_steps: the all-year matrix, the release season's throughout, and switching.

    Inputs:
        P: all-year matrix; seasons: (4, n, n), season_matrices(). start, end, start_time: one per move.
    Returns:
        dict of per-move arrays: "all year", "stuck", "switching".
    """
    with np.errstate(divide="ignore"):
        out = {"all year": np.log(np.linalg.matrix_power(P, n_steps)[start, end])}
    steps = step_seasons(start_time, n_steps)
    stuck = np.stack([np.linalg.matrix_power(seasons[s], n_steps) for s in range(4)])
    out["stuck"] = np.log(stuck[steps[:, 0], start, end])
    out["switching"] = np.empty(start.size)
    sequences, which = np.unique(steps, axis=0, return_inverse=True)
    which = which.ravel()
    for k, sequence in enumerate(sequences):  # a chain per distinct order of seasons, only from the cells it needs
        rows = np.flatnonzero(which == k)
        cells, at = np.unique(start[rows], return_inverse=True)
        X = np.zeros((cells.size, P.shape[0]))
        X[np.arange(cells.size), cells] = 1.0
        for s in sequence:
            X = X @ seasons[s]
        out["switching"][rows] = np.log(X[at, end[rows]])
    return out


def walk(S, matrices, coast, lat_c, lon_c, first_season, steps=RELEASE_STEPS):
    """A release through the seasons, or through one matrix if first_season is None.

    Inputs:
        matrices: (4, n, n) with first_season, or one (n, n) with None.
    Returns:
        {(quantity, step): one value per source}: pct_left, pct_coastal, pct_reached_coast, centre_lat, centre_lon.
    """
    n_cells = lat_c.size
    if first_season is None:
        matrices, first_season = np.repeat(matrices[None], 4, axis=0), 0
    Qs = [absorbing(matrices[s], coast) for s in range(4)]
    X, H, out = S.copy(), S.copy(), {}
    for n in range(max(steps)):
        s = chain_season(first_season, n)
        X, H = X @ matrices[s], H @ Qs[s]
        if n + 1 in steps:
            inside = X[:, :n_cells].sum(axis=1)
            out[("pct_left", n + 1)] = 100 * X[:, n_cells]
            out[("pct_coastal", n + 1)] = 100 * X[:, coast].sum(axis=1)
            out[("pct_reached_coast", n + 1)] = 100 * H[:, coast].sum(axis=1)
            out[("centre_lat", n + 1)] = X[:, :n_cells] @ lat_c / inside
            out[("centre_lon", n + 1)] = X[:, :n_cells] @ lon_c / inside
    return out


def westward_by_season(matrix_of_season, weights, zone, lat_c, lon_c):
    """Mean westward km per step from a set of cells, per season, moves that stay in the box, on cell centres.

    Inputs:
        matrix_of_season: (4, n, n) rows to average (P^s, or raw counts as probabilities).
        weights: (4, n) the moves each cell's row counts for.
    """
    n_cells = lat_c.size
    east = (lon_c[None, :] - lon_c[zone, None]) * KM_PER_DEG * np.cos(np.radians(0.5 * (lat_c[None, :] + lat_c[zone, None])))
    out = np.empty(4)
    for s in range(4):
        rows = matrix_of_season[s][zone][:, :n_cells]
        keep = rows.sum(axis=1)
        out[s] = -(weights[s, zone] * (rows * east).sum(axis=1)).sum() / (weights[s, zone] * keep).sum()
    return out


def track_centres(S, P, seasons, lat_c, lon_c, n_steps=24):
    """The centre per step for one source: {season index or None: (n_steps + 1, 2) lat, lon}."""
    n_cells = lat_c.size
    out = {}
    for first in (None, 0, 1, 2, 3):
        x, path = S.copy(), []
        for n in range(n_steps):
            x = x @ (P if first is None else seasons[chain_season(first, n)])
            path.append((x[:n_cells] @ lat_c / x[:n_cells].sum(), x[:n_cells] @ lon_c / x[:n_cells].sum()))
        out[first] = np.array(path)
    return out


def plot(region, paths, release_lat, release_lon, chain_rows, out_path):
    """(a) the centre of coast material through a year, by release season; (b) what switching gains at each horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    draw_land(ax)
    ax.plot(paths[None][:, 1], paths[None][:, 0], color="#6b7280", lw=2, ls=(0, (4, 3)), label="All year")
    for s, name in enumerate(SEASON_NAMES):
        p = paths[s]
        ax.plot(np.r_[release_lon, p[:, 1]], np.r_[release_lat, p[:, 0]], color=COLOURS[s], lw=2, label=f"Released at the start of {name.lower()}")
        ax.plot(p[STEPS_PER_SEASON - 1:: STEPS_PER_SEASON, 1], p[STEPS_PER_SEASON - 1:: STEPS_PER_SEASON, 0], ls="", marker="o", ms=6, mfc=COLOURS[s], mec="white", mew=1.2)
    ax.plot(release_lon, release_lat, marker="s", ms=7, color="#1b2733", ls="")
    ax.set_xlim(-20, 18)
    ax.set_ylim(-30, -16)
    ax.set_yticks(range(-30, -15, 5))
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(_degrees("lon"))
    ax.yaxis.set_major_formatter(_degrees("lat"))
    ax.set_title("(a) Centre of material from the Benguela coast, all drifters, over 360 days", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.25)

    ax = axes[1]
    days = sorted({r["days"] for r in chain_rows})
    x = np.arange(len(days))
    style = {("all", "switching minus all year"): (COLOURS[0], "o", "Switching vs all year: all drifters"),
             ("undrogued", "switching minus all year"): (COLOURS[1], "o", "Switching vs all year: undrogued"),
             ("all", "stuck minus all year"): (COLOURS[0], "x", "Keeping the release season: all drifters"),
             ("undrogued", "stuck minus all year"): (COLOURS[1], "x", "Keeping the release season: undrogued")}
    for k, ((key, what), (colour, marker, label)) in enumerate(style.items()):
        r = [next(v for v in chain_rows if v["set"] == key and v["compare"] == what and v["days"] == d) for d in days]
        y = np.array([v["gain_per_move"] for v in r])
        err = [y - np.array([v["ci_low"] for v in r]), np.array([v["ci_high"] for v in r]) - y]
        ax.errorbar(x + (k - 1.5) * 0.09, y, yerr=err, fmt=marker, ms=7, capsize=3, color=colour, label=label, ls="")
    ax.axhline(0, color="#1b2733", lw=1)
    ax.set_xticks(x, [f"{d} days" for d in days])
    ax.set_ylabel("Log-likelihood per move, gain over all year")
    ax.set_title("(b) Held-out drifters: does the rule for crossing seasons help?", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.01, -0.09, f"(a) 2° seasonal matrices stepped {TAU_DAYS} days at a time, changing every {STEPS_PER_SEASON * TAU_DAYS} days (dots). Square: the middle of the release cells. "
             "(b) Above zero: closer to where test drifters really went than the all-year matrix.\nFitted on odd years and scored on even ones, and the other way round; steps change season by the date. "
             f"Bars: 95% intervals from resampling test drifters. A typical year's pattern, not a forecast. Source: src/forecasting/seasonal_matrices.py.", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    base = load_transitions(region, verbose=False)
    assignment = load_split(base["drifter_id"])
    parent = parent_states(region, fine, coarse)
    cache_exists = os.path.exists(os.path.join(INTERIM_DIR, f"{region['name']}.npz"))
    land = land_fraction(region, fine, land_paths())
    coast = coastal_cells(region, fine, land)
    lat_edges, lon_edges = grid_edges(region, fine)
    lat_c, lon_c = (a.ravel() for a in np.meshgrid(0.5 * (lat_edges[:-1] + lat_edges[1:]), 0.5 * (lon_edges[:-1] + lon_edges[1:]), indexing="ij"))
    zone = box_cells(region, fine, region["coastal_zone"])
    orange = box_cells(region, fine, next(b for b in region["named_releases"] if b["name"] == "Orange Basin"))
    stems = {"all": f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d", "undrogued": UNDROGUED_STEM}
    lags = {d: lag_table(d, region, cache_exists) for d in CHAIN_DAYS}

    rows, chain_rows, saved, paths = [], [], {}, None
    for key in SETS:
        print(f"--- {SET_NAMES[key]}", flush=True)
        table = base if key == "all" else undrogued_only(base)
        saved_P = np.load(os.path.join(PROCESSED_DIR, f"{stems[key]}.npz"))
        P, alpha = saved_P["P"], float(saved_P["alpha"])
        fit = shrinkage_fit(parent, alpha)
        start, end, n_cells = to_states(table, region, fine)
        n_states = n_cells + 1
        season, year = season_and_year(table["start_time"])
        ids = table["drifter_id"]
        in_fit, in_test = set_mask(ids, assignment, ("train", "valid")), set_mask(ids, assignment, "test")

        # beta: leave out whole years
        annual, seasonal = leave_year_out(start, end, season, year, in_fit, in_fit, fit, BETAS, n_states)
        curve = seasonal.mean(axis=1)
        beta = pick(BETAS, curve, "beta")
        print(f"beta {beta:.0f}: all-year {annual.mean():.5f}, seasonal {curve.max():.5f} per move", flush=True)
        rows += [{"check": "choosing beta", "set": key, "beta": round(float(b), 2), "score_all_year": round(float(annual.mean()), 5),
                  "score_seasonal": round(float(c), 5), "gain_per_move": round(float(c - annual.mean()), 5), "chosen": bool(b == beta)} for b, c in zip(BETAS, curve)]

        # The test drifters: new drifters in a new year
        annual, seasonal = leave_year_out(start, end, season, year, in_fit, in_test, fit, (beta,), n_states)
        test_ids = ids[in_test]
        draws = resample_counts(np.unique(test_ids).size, 1000, SEED)
        gain = seasonal[0] - annual
        everything = np.ones(gain.size, dtype=bool)
        low, high = percentile_interval(resampled_means(*drifter_totals(gain, test_ids, everything), draws), LEVEL)
        rows.append({"check": "held-out score", "set": key, "days": TAU_DAYS, "beta": round(beta, 2), "moves": int(gain.size), "drifters": int(np.unique(test_ids).size),
                     "score_all_year": round(float(annual.mean()), 5), "score_seasonal": round(float(seasonal.mean()), 5),
                     "gain_per_move": round(float(gain.mean()), 5), "ci_low": round(float(low), 5), "ci_high": round(float(high), 5)})
        print(rows[-1], flush=True)

        # Crossing seasons: fit on odd years, score on even ones, and the other way round
        for days, moves in lags.items():
            if moves is None:
                print(f"{days} days: skipped (no saved table and no region cache)")
                continue
            moves = subset(moves, np.isin(moves["drifter_id"], list(assignment)))
            if key == "undrogued":
                moves = subset(moves, ~moves["drogued"].astype(bool))
            moves = subset(moves, set_mask(moves["drifter_id"], assignment, "test"))
            s_te, e_te, _ = to_states(moves, region, fine)
            _, y_te = season_and_year(moves["start_time"])
            scores = {k: np.empty(s_te.size) for k in ("all year", "stuck", "switching")}
            for odd in (0, 1):
                train = in_fit & (year % 2 == odd)
                counts = season_counts(start[train], end[train], season[train], n_states)
                P_half = fit(counts.sum(axis=0))
                here = y_te % 2 != odd
                got = chain_scores(P_half, season_matrices(P_half, counts, beta), s_te[here], e_te[here], moves["start_time"][here], days // TAU_DAYS)
                for k in scores:
                    scores[k][here] = got[k]
            draws = resample_counts(np.unique(moves["drifter_id"]).size, 1000, SEED)
            everything = np.ones(s_te.size, dtype=bool)
            for a, b in (("switching", "all year"), ("stuck", "all year"), ("switching", "stuck")):
                diff = scores[a] - scores[b]
                low, high = percentile_interval(resampled_means(*drifter_totals(diff, moves["drifter_id"], everything), draws), LEVEL)
                chain_rows.append({"check": "crossing seasons", "set": key, "days": days, "compare": f"{a} minus {b}", "moves": int(s_te.size), "drifters": int(np.unique(moves["drifter_id"]).size),
                                   "score_a": round(float(scores[a].mean()), 5), "score_b": round(float(scores[b].mean()), 5), "gain_per_move": round(float(diff.mean()), 5),
                                   "ci_low": round(float(low), 5), "ci_high": round(float(high), 5)})
                print(chain_rows[-1], flush=True)

        # The matrices on all the drifters, and what they say
        counts = season_counts(start, end, season, n_states)
        seasons = season_matrices(P, counts, beta)
        saved |= {f"{key}_beta": beta, f"{key}_season": np.nonzero(counts)[0].astype(np.int8), f"{key}_start": np.nonzero(counts)[1].astype(np.int16),
                  f"{key}_end": np.nonzero(counts)[2].astype(np.int16), f"{key}_n": counts[np.nonzero(counts)].astype(np.int32)}

        # Does the pattern survive the shrinkage?
        totals = counts[:, :, :n_cells].sum(axis=2)
        raw = np.divide(counts[:, :, :n_cells], totals[:, :, None], out=np.zeros_like(counts[:, :, :n_cells]), where=totals[:, :, None] > 0)
        west_raw = westward_by_season(raw, totals, zone, lat_c, lon_c)
        west_fit = westward_by_season(seasons, totals, zone, lat_c, lon_c)
        rows.append({"check": "signal kept", "set": key, "quantity": "westward km per 15 days from the Benguela zone, on cell centres",
                     **{f"raw_{SEASONS[s]}": round(float(west_raw[s]), 1) for s in range(4)}, **{f"fitted_{SEASONS[s]}": round(float(west_fit[s]), 1) for s in range(4)},
                     "raw_djf_minus_jja": round(float(west_raw[DJF] - west_raw[JJA]), 1), "fitted_djf_minus_jja": round(float(west_fit[DJF] - west_fit[JJA]), 1),
                     "pct_of_gap_kept": round(float(100 * (west_fit[DJF] - west_fit[JJA]) / (west_raw[DJF] - west_raw[JJA])), 0)})
        print(rows[-1], flush=True)

        # Releases, with intervals. Each season against the all-year matrix on the same replicate.
        release = coast[per_cell_counts(start, ids, n_cells)[1][coast] > 0]
        names = [COAST_NAME, "Orange Basin"]
        S = start_matrix([release, orange], n_states)
        est = {None: walk(S, P, coast, lat_c, lon_c, None)}
        est |= {s: walk(S, seasons, coast, lat_c, lon_c, s) for s in range(4)}
        reps = {first: {k: np.empty((N_BOOT_SEASON, S.shape[0])) for k in est[None]} for first in est}
        for b, (P_b, counts_b) in enumerate(season_replicates(start, end, season, ids, n_states, fit)):
            seasons_b = season_matrices(P_b, counts_b, beta)
            for first in est:
                got = walk(S, P_b if first is None else seasons_b, coast, lat_c, lon_c, first)
                for k, v in got.items():
                    reps[first][k][b] = v
        for first in est:
            for (q, step), value in est[first].items():
                for i, name in enumerate(names):
                    if q == "pct_reached_coast" and name == COAST_NAME:
                        continue
                    low, high = percentile_interval(reps[first][(q, step)][:, i], LEVEL)
                    row = {"check": "release", "set": key, "source": name, "released": "all year" if first is None else SEASONS[first], "quantity": q, "days": step * TAU_DAYS,
                           "estimate": round(float(value[i]), 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4)}
                    if first is not None:
                        d_low, d_high = percentile_interval(reps[first][(q, step)][:, i] - reps[None][(q, step)][:, i], LEVEL)
                        row |= {"minus_all_year": round(float(value[i] - est[None][(q, step)][i]), 4), "diff_low": round(float(d_low), 4), "diff_high": round(float(d_high), 4)}
                    rows.append(row)
        for factor in OTHER_BETAS:
            other = season_matrices(P, counts, beta * factor)
            for first in range(4):
                got = walk(S, other, coast, lat_c, lon_c, first)
                for q in ("pct_coastal", "centre_lon"):
                    for step in (6, 24):
                        rows.append({"check": "other beta", "set": key, "source": names[0], "released": SEASONS[first], "quantity": q, "days": step * TAU_DAYS, "beta": round(beta * factor, 2),
                                     "estimate": round(float(got[(q, step)][0]), 4), "minus_all_year": round(float(got[(q, step)][0] - est[None][(q, step)][0]), 4)})
        if key == "all":
            S0 = S[0]
            paths = (track_centres(S0, P, seasons, lat_c, lon_c), float(lat_c[release].mean()), float(lon_c[release].mean()))
        print("Releases done", flush=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez_compressed(COUNTS_PATH, **saved)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"seasonal_matrices_tau{TAU_DAYS}d.csv")
    everything = rows + chain_rows
    fields = list(dict.fromkeys(k for r in everything for k in r))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(everything)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_out = os.path.join(FIGURES_DIR, "seasonal_matrices.png")
    plot(region, paths[0], paths[1], paths[2], chain_rows, fig_out)
    print(f"Saved {COUNTS_PATH}, {out} and {fig_out}")
