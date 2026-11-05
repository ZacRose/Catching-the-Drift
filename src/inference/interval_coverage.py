"""
interval_coverage.py

Related task: Week 9 forensic review. Every number the project reports has a 95% interval from the drifter
bootstrap, but nothing has tested whether those intervals contain the truth 95% of the time, because the truth
is unknown. So this makes a world where it is known. The product P is taken as the true flow, a synthetic set of
drifters is drawn from it, and the whole estimate is redone as on the real data: shrinkage with the prior rebuilt
and alpha fixed, then the drifter bootstrap. The share of synthetic worlds whose interval contains the truth is
the coverage, and it should be near 95%. The gap between estimate and truth, averaged over worlds, is the bias.

The synthetic data keep the real design: every real move's drifter and start cell are kept, so each cell has
exactly the moves and drifters it really has. Only the destination is redrawn from the true P. Consecutive daily
moves overlap, so moves of one drifter from one cell share a destination over BLOCK days, standing in for that
dependence (the case the drifter bootstrap exists for). A first version instead let synthetic drifters walk from
their first position; they rarely reached the coast, whose cells then fell back on the prior a second time, so it
said more about the simulation than the intervals (see the AI log).

Coverage is reported for the coastal release (the project's headline numbers), and per start cell for the numbers
the app shows, grouped by the app's data flags (100+, 10-99, under 10 drifters). bootstrap.py warns that cells with
under about 10 drifters can look falsely certain; this measures how much. For the coast, the same worlds are also fitted
with next to no prior, which shows how much of any bias comes from the shrinkage. For single cells, mean_bias is the
average size of the bias, ignoring its sign.

What it can't show: the world here is exactly Markov and doesn't change from year to year, so this tests the
estimator and the bootstrap only. markov_check.py, calibration.py and year_intervals.py test those assumptions.

Output: outputs/results/interval_coverage_tau15d.csv
Run: python3 -m src.inference.interval_coverage   (about 6 minutes; --n-sim 20 for a quick look)
"""
import argparse
import csv
import os
import numpy as np

from src.common.config import PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS
from src.inference.adequacy import N_MIN
from src.inference.bootstrap import LEVEL, SEED, percentile_interval, replicate_matrices, shrinkage_fit
from src.inference.estimators import count_matrix
from src.inference.transition_data import load_transitions, parent_states, to_states

N_SIM = 100
N_BOOT_SIM = 200
BLOCK = TAU_DAYS  # days of one drifter's moves from one cell that share a destination
NO_SHRINKAGE = 1e-6  # alpha for the same fit with next to no prior, to show how much of a bias is the shrinkage's
STEPS = (1, 6, 24)
THIN = 10  # the app's "very thin" flag
FLAGS = {f"{N_MIN}+ drifters": (N_MIN, np.inf), f"{THIN}-{N_MIN - 1} drifters": (THIN, N_MIN), f"under {THIN} drifters": (1, THIN)}


def block_ids(start, drifter_id, start_time, block=BLOCK):
    """One id per group of moves that share a destination: the same drifter and start cell, within `block` days."""
    days = start_time.astype("datetime64[D]").astype(np.int64)
    _, rank = np.unique(drifter_id, return_inverse=True)
    key = np.stack([rank, start, days // block], axis=1)
    return np.unique(key, axis=0, return_inverse=True)[1].ravel()


def simulate(P, start, blocks, rng):
    """A destination for every move: drawn from the true row of its start cell, once per block.

    Inputs:
        P: the true matrix; start: the real start state of each move; blocks: block_ids().
    Returns:
        end state per move.
    """
    n_blocks = blocks.max() + 1
    first = np.full(n_blocks, -1)
    first[blocks[::-1]] = np.arange(blocks.size)[::-1]  # any move of the block gives its start cell
    cumulative = np.cumsum(P[start[first]], axis=1)
    cumulative[:, -1] = 1.0
    drawn = (cumulative < rng.random(n_blocks)[:, None]).sum(axis=1)
    return drawn[blocks]


def functionals(n_cells, lat_c, lon_c, zone):
    """Vectors over states: left the box, in the coastal zone, in the box, and lat and lon weighted by that."""
    F = np.zeros((n_cells + 1, 5))
    F[n_cells, 0] = 1.0
    F[zone, 1] = 1.0
    F[:n_cells, 2] = 1.0
    F[:n_cells, 3], F[:n_cells, 4] = lat_c, lon_c
    return F


def numbers(P, F0, starts, steps=STEPS):
    """{(quantity, step): one value per start row}, stepping P backwards on F0 (as the app's exporter)."""
    out, F = {}, F0
    for n in range(1, max(steps) + 1):
        F = P @ F
        if n in steps:
            G = starts @ F
            with np.errstate(invalid="ignore", divide="ignore"):
                out[("pct_left_box", n)] = 100 * G[:, 0]
                out[("pct_in_coastal_zone", n)] = 100 * G[:, 1]
                out[("centre_lat", n)] = G[:, 3] / G[:, 2]
                out[("centre_lon", n)] = G[:, 4] / G[:, 2]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n-sim", type=int, default=N_SIM, help="synthetic worlds (default %(default)s)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_SIM, help="bootstrap replicates per world (default %(default)s)")
    args = ap.parse_args()

    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region, verbose=False)
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    P_true, alpha = product["P"], float(product["alpha"])
    start, _, n_cells = to_states(table, region, fine)
    _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)
    fit = shrinkage_fit(parent_states(region, fine, coarse), alpha)
    fit_raw = shrinkage_fit(parent_states(region, fine, coarse), NO_SHRINKAGE)
    zone = coastal_cells(region, fine, land_fraction(region, fine, land_paths()))
    release = zone[drifters[zone] > 0]
    lat_edges, lon_edges = grid_edges(region, fine)
    lat_c, lon_c = (a.ravel() for a in np.meshgrid(0.5 * (lat_edges[:-1] + lat_edges[1:]), 0.5 * (lon_edges[:-1] + lon_edges[1:]), indexing="ij"))
    F0 = functionals(n_cells, lat_c, lon_c, zone)
    sampled = np.flatnonzero(drifters > 0)
    S = np.zeros((sampled.size + 1, n_cells + 1))
    S[np.arange(sampled.size), sampled] = 1.0
    S[-1, release] = 1.0 / release.size  # the last row is the whole-coast release
    truth = numbers(P_true, F0, S)
    blocks = block_ids(start, table["drifter_id"], table["start_time"])
    print(f"{start.size:,} moves in {blocks.max() + 1:,} blocks", flush=True)
    rng = np.random.default_rng(SEED)

    keys = list(truth)
    hits = {k: np.zeros(S.shape[0]) for k in keys}
    bias = {k: np.zeros(S.shape[0]) for k in keys}
    width = {k: np.zeros(S.shape[0]) for k in keys}
    valid = {k: np.zeros(S.shape[0]) for k in keys}
    bias_raw = {k: 0.0 for k in keys}  # whole-coast release only
    for sim in range(args.n_sim):
        e_sim = simulate(P_true, start, blocks, rng)
        counts_sim = count_matrix(start, e_sim, n_cells + 1)
        est = numbers(fit(counts_sim), F0, S)
        raw = numbers(fit_raw(counts_sim), F0, S)
        for k in keys:
            bias_raw[k] += float(raw[k][-1] - truth[k][-1])
        reps = {k: np.empty((args.n_boot, S.shape[0])) for k in keys}
        for b, P_b in enumerate(replicate_matrices(start, e_sim, table["drifter_id"], n_cells + 1, fit, args.n_boot, SEED + sim + 1)):
            out = numbers(P_b, F0, S)
            for k in keys:
                reps[k][b] = out[k]
        for k in keys:
            low, high = percentile_interval(reps[k], LEVEL)
            ok = np.isfinite(low) & np.isfinite(high)
            hits[k][ok] += ((low <= truth[k]) & (truth[k] <= high))[ok]
            bias[k][ok] += (est[k] - truth[k])[ok]
            width[k][ok] += (high - low)[ok]
            valid[k][ok] += 1
        print(f"world {sim + 1} of {args.n_sim}: coast, 1-year % left: truth {truth[('pct_left_box', 24)][-1]:.3f}, "
              f"estimate {est[('pct_left_box', 24)][-1]:.3f}; running coverage {100 * hits[('pct_left_box', 24)][-1] / (sim + 1):.0f}%", flush=True)

    rows = []
    for (q, step) in keys:
        days = step * TAU_DAYS
        c = -1
        rows.append({"starts": "whole Benguela coast", "start_rows": 1, "quantity": q, "days": days,
                     "pct_covered": round(100 * float(hits[(q, step)][c] / valid[(q, step)][c]), 1),
                     "mean_bias": round(float(bias[(q, step)][c] / valid[(q, step)][c]), 4), "truth": round(float(truth[(q, step)][c]), 4),
                     "mean_interval_width": round(float(width[(q, step)][c] / valid[(q, step)][c]), 4),
                     "mean_bias_without_shrinkage": round(bias_raw[(q, step)] / args.n_sim, 4)})
        for name, (lo, hi) in FLAGS.items():
            g = (drifters[sampled] >= lo) & (drifters[sampled] < hi)
            v = valid[(q, step)][:-1][g]
            rows.append({"starts": f"single cells, {name}", "start_rows": int(g.sum()), "quantity": q, "days": days,
                         "pct_covered": round(100 * float(hits[(q, step)][:-1][g].sum() / v.sum()), 1),
                         "mean_bias": round(float(np.mean(np.abs(bias[(q, step)][:-1][g] / np.maximum(v, 1)))), 4), "truth": "",
                         "mean_interval_width": round(float((width[(q, step)][:-1][g] / np.maximum(v, 1)).mean()), 4)})
    for r in rows:
        print(r)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"interval_coverage_tau{TAU_DAYS}d.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out} ({args.n_sim} worlds, {args.n_boot} replicates each)")
