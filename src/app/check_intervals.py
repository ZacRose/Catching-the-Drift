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

It also reports, per start and step:
- the bagged estimate (the mean over replicates) beside the full-data estimate the app shows. A large gap
  would mean the estimator is biased under resampling, and the intervals off-centre;
- the interval width beside the start cell's drifters, which should shrink as drifters grow (but see
  bootstrap.py: cells with under about 10 drifters can look falsely certain);
- whether the full-data estimate lies inside its interval.

Output: outputs/results/app_interval_check.csv
Run: python3 -m src.app.check_intervals   (about 2 minutes; after src.app.export_app_data)
"""
import base64
import csv
import json
import os
import numpy as np

from src.app.export_app_data import APP_DATA_PATH, QUANTITIES, build_models, cell_centres, great_circle_km, \
    start_matrix, zone_cells
from src.common.config import PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.inference.bootstrap import LEVEL, N_BOOT, SEED, resample_counts
from src.inference.transition_data import TAU_DAYS, TRANSITIONS_PATH, load_transitions

STARTS = [-1, 103, 260, 353, 547, 368, 317, 396]  # the coast release, then cells from 137 down to 7 drifters
STEPS = [1, 6, 24, 72]


def forward_quantities(X, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon):
    """QUANTITIES straight from distributions over states (one row per start)."""
    water = X[:, :n_cells]
    inside = water.sum(axis=1)
    lat, lon = water @ lat_c / inside, water @ lon_c / inside
    return np.stack([100 * X[:, n_cells], 100 * X[:, benguela].sum(axis=1), 100 * X[:, coast].sum(axis=1), lat, lon,
                     great_circle_km(start_lat, start_lon, lat, lon), 100 * X[:, n_cells + 1:].sum(axis=1)])


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
        X = S.copy()
        for n in range(1, max(steps) + 1):
            X = X @ P_b
            if n in steps:
                out[b, :, steps.index(n)] = forward_quantities(X, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon)
    return out


def app_intervals(path=APP_DATA_PATH):
    """The intervals the app ships, per model: (low, high), each (QUANTITIES, steps, rows), and the row ids."""
    with open(path) as f:
        text = f.read()
    data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    out = {}
    for name, iv in data["intervals"].items():
        shape = (len(iv["quantities"]), iv["steps"], len(iv["rows"]))
        dec = [np.frombuffer(base64.b64decode(iv[k]), dtype="<f4").reshape(shape) for k in ("low", "high")]
        out[name] = (dec[0], dec[1], iv["rows"])
    return out


if __name__ == "__main__":
    region = load_region_bounds()
    fine = region["cell_size_deg"]
    table = load_transitions(region, verbose=False)
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    land = land_fraction(region, fine, land_paths())
    models, n_cells, start_drifters, _ = build_models(region, table, np.load(TRANSITIONS_PATH), float(product["alpha"]), land)
    lat_c, lon_c = cell_centres(region, fine)
    coast = coastal_cells(region, fine, land)
    release = coast[start_drifters[coast] > 0]
    benguela = zone_cells(region, fine, region["coastal_zone"])
    start_lat = np.array([lat_c[release].mean() if c < 0 else lat_c[c] for c in STARTS])
    start_lon = np.array([lon_c[release].mean() if c < 0 else lon_c[c] for c in STARTS])
    shipped = app_intervals()
    q_lo, q_hi = (1 - LEVEL) / 2, (1 + LEVEL) / 2

    rows, summary = [], []
    for name, m in models.items():
        S = start_matrix(STARTS, release, m["n_states"])
        X, est = S.copy(), {}
        for n in range(1, max(STEPS) + 1):
            X = X @ m["P"]
            if n in STEPS:
                est[n] = forward_quantities(X, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon)
        reps = brute_force(m, S, n_cells, lat_c, lon_c, benguela, coast, start_lat, start_lon)
        lo, hi = np.quantile(reps, [q_lo, q_hi], axis=0)
        bagged = reps.mean(axis=0)
        app_lo, app_hi, app_rows = shipped[name]
        worst, inside, total = 0.0, 0, 0
        for r, cell in enumerate(STARTS):
            a = app_rows.index(cell)
            drifters = "coast" if cell < 0 else int(start_drifters[cell])
            for s, n in enumerate(STEPS):
                for q, quantity in enumerate(QUANTITIES):
                    if name == "water" and quantity == "pct_aground":
                        continue
                    e, l, h = est[n][q, r], lo[q, s, r], hi[q, s, r]
                    al, ah = float(app_lo[q, n - 1, a]), float(app_hi[q, n - 1, a])
                    worst = max(worst, abs(al - l) / max(abs(l), 1e-3), abs(ah - h) / max(abs(h), 1e-3))
                    inside += l - 1e-9 <= e <= h + 1e-9
                    total += 1
                    rows.append({"model": name, "start_cell": cell, "start_drifters": drifters, "days": n * TAU_DAYS,
                                 "quantity": quantity, "estimate": round(float(e), 5), "bagged_mean": round(float(bagged[q, s, r]), 5),
                                 "ci_low": round(float(l), 5), "ci_high": round(float(h), 5),
                                 "app_ci_low": round(al, 5), "app_ci_high": round(ah, 5)})
        summary.append((name, worst, inside, total))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "app_interval_check.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for name, worst, inside, total in summary:
        print(f"{name}: largest relative gap between the app's intervals and the brute-force ones {worst:.1e}; "
              f"estimate inside its interval in {inside} of {total}")
    print(f"Saved {out_path}")
