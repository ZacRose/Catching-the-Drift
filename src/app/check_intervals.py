"""
check_intervals.py

Related task: Week 9 app - confirm that the app's 95% intervals (the drifter bootstrap) are what they
claim to be. export_app_data.py computes them the fast way: replicate counts through
bootstrap.drifter_pairs, and every start cell at once by stepping P backwards on a few vectors. This
script recomputes a sample of them the slow, obvious way and compares:
1. Counts: each replicate's count matrix weights every row of the transition table by how many times
   its drifter was drawn (one bincount over the rows), not through drifter_pairs' per-drifter triples.
2. Propagation: forward from each start, start @ P^n one step at a time (as Product.forecast).
3. Intervals: np.quantile over the replicates, not bootstrap.percentile_interval.
The replicates are the same draws (resample_counts with bootstrap.py's seed, drifters in sorted order),
so the two routes should agree to float32 rounding.

The same is done for the undrogued matrix, and for the seasons (release on the first day of each season, the matrix
changing every 90 days, forward from each start with season_matrices on the same draws). The seasons' intervals are stored
at some steps only, and the page reads the steps between off a straight line on a log scale, so the steps in BETWEEN are
compared with brute force too, to measure what that costs.

It also reports, per start and step:
- the bagged estimate (the mean over replicates) beside the full-data estimate the app shows. A large gap
  would mean the estimator is biased under resampling, and the intervals off-centre;
- the interval width beside the start cell's drifters, which should shrink as drifters grow (but see
  bootstrap.py: cells with under about 10 drifters can look falsely certain);
- whether the full-data estimate lies inside its interval.

Output: outputs/results/app_interval_check.csv
Run: python3 -m src.app.check_intervals   (about 40 minutes; after src.app.export_app_data)
"""
import csv
import json
import os
import numpy as np

from src.app.export_app_data import APP_DATA_PATH, QUANTITIES, SEASON_STEPS, SETS, build_models, cell_centres, from_log16, \
    great_circle_km, saved_season_counts, start_matrix, zone_cells
from src.common.config import PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.forecasting.seasonal_forecast import SEASONS
from src.forecasting.seasonal_matrices import N_BOOT_SEASON, chain_season, season_matrices, season_replicates
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.inference.bootstrap import LEVEL, N_BOOT, SEED, resample_counts
from src.inference.drogue_comparison import UNDROGUED_STEM, undrogued_only
from src.inference.transition_data import TAU_DAYS, TRANSITIONS_PATH, load_transitions
from src.linalg.connectivity import absorbing

STARTS = [-1, 103, 260, 353, 547, 368, 317, 396]  # the coast release, then cells from 137 down to 7 drifters
STEPS = [1, 6, 24, 72]
SEASON_CHECK_STEPS = [1, 6, 7, 13, 24, 27, 33, 72]  # 7, 13, 27 and 33 lie between the stored steps


def forward_quantities(X, H, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon):
    """QUANTITIES straight from distributions over states (one row per start). H is the same run with coastal cells
    absorbing, for having reached them."""
    water = X[:, :n_cells]
    inside = water.sum(axis=1)
    lat, lon = water @ lat_c / inside, water @ lon_c / inside
    return np.stack([100 * X[:, n_cells], 100 * X[:, benguela].sum(axis=1), 100 * X[:, coast].sum(axis=1), lat, lon,
                     great_circle_km(start_lat, start_lon, lat, lon), 100 * X[:, n_cells + 1:].sum(axis=1), 100 * H[:, coast].sum(axis=1)])


def brute_force(model, S, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon, steps=STEPS,
                n_boot=N_BOOT, seed=SEED):
    """Every replicate's QUANTITIES at the given steps: array (n_boot, len(QUANTITIES), len(steps), rows)."""
    k = model["n_states"]
    _, rank = np.unique(model["drifter_id"], return_inverse=True)
    pair = model["start"].astype(np.int64) * k + model["end"]
    out = np.empty((n_boot, len(QUANTITIES), len(steps), S.shape[0]))
    for b, times_drawn in enumerate(resample_counts(rank.max() + 1, n_boot, seed)):
        counts = np.bincount(pair, weights=times_drawn[rank], minlength=k * k).reshape(k, k)
        P_b = model["fit"](counts)
        Q_b = absorbing(P_b, coast)
        X, H = S.copy(), S.copy()
        for n in range(1, max(steps) + 1):
            X, H = X @ P_b, H @ Q_b
            if n in steps:
                out[b, :, steps.index(n)] = forward_quantities(X, H, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon)
    return out


def brute_force_season(model, first, beta, S, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon, steps=SEASON_CHECK_STEPS,
                       n_boot=N_BOOT_SEASON, seed=SEED):
    """As brute_force, for a release on the first day of one season, forward through the seasons."""
    out = np.empty((n_boot, len(QUANTITIES), len(steps), S.shape[0]))
    for b, (P_b, counts_b) in enumerate(season_replicates(model["start"], model["end"], model["season"], model["drifter_id"], model["n_states"], model["fit"], n_boot, seed)):
        seasons = season_matrices(P_b, counts_b, beta, model["n_water"])
        Qs = [absorbing(seasons[s], coast) for s in range(4)]
        X, H = S.copy(), S.copy()
        for n in range(1, max(steps) + 1):
            s = chain_season(first, n - 1)
            X, H = X @ seasons[s], H @ Qs[s]
            if n in steps:
                out[b, :, steps.index(n)] = forward_quantities(X, H, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon)
    return out


def read_off(values, stored, step):
    """The page's interval at a step: the stored value, or between two stored steps a straight line on a log scale (model.js).

    Inputs:
        values: one per stored step; stored: the stored steps.
    """
    if step in stored:
        return float(values[stored.index(step)])
    k = int(np.searchsorted(stored, step)) - 1
    f = (step - stored[k]) / (stored[k + 1] - stored[k])
    a, b = float(values[k]), float(values[k + 1])
    return 10 ** (np.log10(a) * (1 - f) + np.log10(b) * f) if a > 0 and b > 0 else a * (1 - f) + b * f


def app_intervals(path=APP_DATA_PATH):
    """The intervals the app ships: {set: {model: (low, high, rows, quantities)}} all year, each (quantities, steps, rows), and
    the same under "seasonal" with a first axis for the release season."""
    with open(path) as f:
        text = f.read()
    data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    out = {}
    for key, block in data["sets"].items():
        out[key] = {"all year": {}, "seasonal": {}}
        for name, iv in block["intervals"].items():
            shape = (len(iv["quantities"]), len(iv["steps"]), len(iv["rows"]))
            out[key]["all year"][name] = (from_log16(iv["low"]).reshape(shape), from_log16(iv["high"]).reshape(shape), iv["rows"], iv["quantities"])
        for name, iv in block["seasonal"]["intervals"].items():
            shape = (len(iv["quantities"]), len(iv["steps"]), len(iv["rows"]))
            low = np.stack([from_log16(v).reshape(shape) for v in iv["low"]])
            high = np.stack([from_log16(v).reshape(shape) for v in iv["high"]])
            out[key]["seasonal"][name] = (low, high, iv["rows"], iv["quantities"])
    return out


if __name__ == "__main__":
    region = load_region_bounds()
    fine = region["cell_size_deg"]
    base = load_transitions(region, verbose=False)
    with np.load(TRANSITIONS_PATH) as z:
        stored_all = {k: z[k] for k in z.files}
    undrogued_rows = ~stored_all["drogued"].astype(bool)
    stems = {"all": f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d", "undrogued": UNDROGUED_STEM}
    land = land_fraction(region, fine, land_paths())
    lat_c, lon_c = cell_centres(region, fine)
    coast = coastal_cells(region, fine, land)
    benguela = zone_cells(region, fine, region["coastal_zone"])
    shipped = app_intervals()
    q_lo, q_hi = (1 - LEVEL) / 2, (1 + LEVEL) / 2

    rows, summary, shores = [], [], None
    for key in SETS:
        table = base if key == "all" else undrogued_only(base)
        stored = stored_all if key == "all" else {k: (v[undrogued_rows] if v.shape == undrogued_rows.shape else v) for k, v in stored_all.items()}
        product = np.load(os.path.join(PROCESSED_DIR, f"{stems[key]}.npz"))
        models, n_cells, start_drifters, _ = build_models(region, table, stored, float(product["alpha"]), land, shores)
        shores = models["shore"]["shores"]
        release = coast[start_drifters[coast] > 0]
        starts = [c for c in STARTS if c < 0 or start_drifters[c] > 0]
        start_lat = np.array([lat_c[release].mean() if c < 0 else lat_c[c] for c in starts])
        start_lon = np.array([lon_c[release].mean() if c < 0 else lon_c[c] for c in starts])
        args = (n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon)
        _, beta = saved_season_counts(key, n_cells + 1)
        for name, m in models.items():
            S = start_matrix(starts, release, m["n_states"])
            Q = absorbing(m["P"], coast)
            X, H, est = S.copy(), S.copy(), {}
            for n in range(1, max(STEPS) + 1):
                X, H = X @ m["P"], H @ Q
                if n in STEPS:
                    est[n] = forward_quantities(X, H, *args)
            reps = brute_force(m, S, *args)
            lo, hi = np.quantile(reps, [q_lo, q_hi], axis=0)
            bagged = reps.mean(axis=0)
            app_lo, app_hi, app_rows, app_quantities = shipped[key]["all year"][name]
            worst, inside, total = 0.0, 0, 0
            for r, cell in enumerate(starts):
                a = app_rows.index(cell)
                drifters = "coast" if cell < 0 else int(start_drifters[cell])
                for s, n in enumerate(STEPS):
                    for quantity in app_quantities:
                        q = QUANTITIES.index(quantity)
                        e, l, h = est[n][q, r], lo[q, s, r], hi[q, s, r]
                        k = app_quantities.index(quantity)
                        al, ah = float(app_lo[k, n - 1, a]), float(app_hi[k, n - 1, a])
                        worst = max(worst, abs(al - l) / max(abs(l), 1e-3), abs(ah - h) / max(abs(h), 1e-3))
                        inside += l - 1e-9 <= e <= h + 1e-9
                        total += 1
                        rows.append({"set": key, "model": name, "released": "all year", "start_cell": cell, "start_drifters": drifters, "days": n * TAU_DAYS,
                                     "quantity": quantity, "estimate": round(float(e), 5), "bagged_mean": round(float(bagged[q, s, r]), 5),
                                     "ci_low": round(float(l), 5), "ci_high": round(float(h), 5), "app_ci_low": round(al, 5), "app_ci_high": round(ah, 5)})
            summary.append((key, name, "all year", worst, None, inside, total))
            print(summary[-1], flush=True)

            # The seasons. Three starts are enough to test the route, and the steps between the stored ones.
            few = [0, *range(1, len(starts), 3)]
            S_few, args_few = S[few], (n_cells, lat_c, lon_c, benguela, coast, start_lat[few], start_lon[few])
            app_lo, app_hi, app_rows, app_quantities = shipped[key]["seasonal"][name]
            worst, between = 0.0, 0.0
            for first in range(4):
                reps = brute_force_season(m, first, beta, S_few, *args_few)
                lo, hi = np.quantile(reps, [q_lo, q_hi], axis=0)
                for r, i in enumerate(few):
                    a = app_rows.index(starts[i])
                    for s, n in enumerate(SEASON_CHECK_STEPS):
                        for quantity in app_quantities:
                            q, k = QUANTITIES.index(quantity), app_quantities.index(quantity)
                            al = read_off(app_lo[first, k, :, a], SEASON_STEPS, n)
                            ah = read_off(app_hi[first, k, :, a], SEASON_STEPS, n)
                            gap = max(abs(al - lo[q, s, r]) / max(abs(lo[q, s, r]), 1e-3), abs(ah - hi[q, s, r]) / max(abs(hi[q, s, r]), 1e-3))
                            if n in SEASON_STEPS:
                                worst = max(worst, gap)
                            else:
                                between = max(between, gap)
                            rows.append({"set": key, "model": name, "released": SEASONS[first], "start_cell": starts[i], "start_drifters": "coast" if starts[i] < 0 else int(start_drifters[starts[i]]),
                                         "days": n * TAU_DAYS, "quantity": quantity, "ci_low": round(float(lo[q, s, r]), 5), "ci_high": round(float(hi[q, s, r]), 5),
                                         "app_ci_low": round(al, 5), "app_ci_high": round(ah, 5), "stored_step": n in SEASON_STEPS})
            summary.append((key, name, "seasons", worst, between, None, None))
            print(summary[-1], flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "app_interval_check.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for key, name, released, worst, between, inside, total in summary:
        text = f"{key}, {name}, {released}: largest relative gap between the app's intervals and the brute-force ones {worst:.1e}"
        if between is not None:
            text += f"; on steps between the stored ones {between:.1e}"
        if total:
            text += f"; estimate inside its interval in {inside} of {total}"
        print(text)
    print(f"Saved {out_path}")
