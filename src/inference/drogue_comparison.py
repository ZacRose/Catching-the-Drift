"""
drogue_comparison.py

Related task: decision 6a, RQ3 and the app's drifter switch. All drifters, or only those that have lost their
drogue? Undrogued drifters feel the wind directly, so they are arguably the nearer stand-in for floating material
such as oil or debris (decision 6a, with its caution: a handful of studies, and no weathering or evaporation here).

robustness.py already reruns the headline numbers on undrogued drifters, one setting among many. That answers "does
the choice move the numbers?" but leaves three gaps, filled here:
1. The matrix itself wasn't kept, so nobody could use it. It is saved here with its bootstrap, in the product's format
   (load_product(drifters="undrogued")), with alpha chosen on that set's validation drifters as evaluate.py does.
2. Is a gap real? Undrogued moves are a subset of all moves, so the two answers aren't independent, and comparing their
   separate intervals misleads. Here both are refitted on the SAME resample of drifters, which gives an interval on
   the gap. Reported from the Benguela coast and the Orange Basin, at 15, 90 and 360 days.
3. Does it predict better? Held-out test drifters, scored under both matrices: mean log-likelihood per move at
   15 days and chained to 90 and 360 days. And are its chances right for undrogued drifters? calibration.py's checks,
   on undrogued test drifters only, under both matrices.

Status is read at the start of a move (issue e), which is clean for this set: no drifter's flag ever goes back from
undrogued to drogued (data_forensics.csv), so a move that starts undrogued stays undrogued. A move that starts drogued
may not (6% lose the drogue on the way), which is one more reason there is no drogued-only matrix here. The other is
data: 0.3% of 2 deg cells are adequate (decision 6a).

Cells where no undrogued drifter started fall back on their 4 deg area's undrogued moves. The 4 deg comparison named
in decision 6a is in robustness_tau15d.csv, as is the long-run pattern for each set.

Intervals: the drifter bootstrap, 1000 replicates. An answer's own interval resamples that set's drifters, so it is
the same as the app's; a gap's interval resamples all drifters and keeps each drawn drifter's undrogued moves.

Outputs: data/processed/shrinkage_P_2deg_tau15d_undrogued.npz and _undrogued_bootstrap.npz,
outputs/results/drogue_comparison_tau15d.csv and outputs/figures/drogue_comparison.png.

Run: python3 -m src.inference.drogue_comparison   (about 6 minutes; needs drifter_split.csv. The 360-day moves and the
having-left check need the region and records caches, and are skipped without them)
"""
import csv
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import cell_index, grid_edges, per_cell_counts
from src.gridding.trajectory_map import _degrees, draw_land
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS, build_transitions, save_grid_result
from src.ingest.drifter_records import load_records
from src.ingest.load import load_region
from src.inference.adequacy import N_MIN
from src.inference.bootstrap import (
    LEVEL, N_BOOT, SEED, drifter_pairs, drifter_totals, matrix_intervals, percentile_interval, replicate_count_matrix,
    replicate_matrices, resample_counts, resampled_means, shrinkage_fit,
)
from src.inference.calibration import coverage, lag_table, left_box, reliability
from src.inference.estimators import count_matrix, log_lik, prior_rows, shrinkage_P
from src.inference.evaluate import states_for, tune_alpha
from src.inference.split import load_split, set_mask
from src.inference.transition_data import load_transitions, parent_states, subset, to_states
from src.linalg.connectivity import COAST_NAME, absorbing, box_cells, start_matrix, subregion_map

HORIZONS = {1: 15, 6: 90, 24: 360}  # steps -> days
N_STEPS = max(HORIZONS)
UNDROGUED_STEM = f"shrinkage_P_2deg_tau{TAU_DAYS}d_undrogued"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # as lags.py
SERIES = {"all": ("All drifters", BLUE), "undrogued": ("Undrogued only", ORANGE)}


def undrogued_only(table):
    """The moves that start without a drogue (and so stay without one: data_forensics.csv)."""
    return subset(table, ~table["drogued"])


def fit_on_split(table, region, fine, coarse, assignment):
    """alpha from train -> valid and P on train + valid, as evaluate.py. Returns (alpha, P)."""
    s_tr, e_tr, n_cells, _ = states_for(table, region, fine, assignment, "train")
    s_va, e_va, _, _ = states_for(table, region, fine, assignment, "valid")
    parent = parent_states(region, fine, coarse)
    alpha, _ = tune_alpha(count_matrix(s_tr, e_tr, n_cells + 1), parent, s_va, e_va)
    s_fit, e_fit, _, _ = states_for(table, region, fine, assignment, ("train", "valid"))
    counts = count_matrix(s_fit, e_fit, n_cells + 1)
    return alpha, shrinkage_P(counts, prior_rows(counts, parent), alpha)


def paired_replicates(start, end, drifter_id, in_subset, n_states, fit_all, fit_subset, n_boot=N_BOOT, seed=SEED):
    """Yield (all-moves P, subset P) refitted on one resample of the drifters, so the gap has a fair interval.

    Inputs:
        start, end, drifter_id: every move. in_subset: bool per move; a drawn drifter brings its subset moves to the
        second matrix (a drifter with none adds nothing to it). The first matrix is bootstrap.replicate_matrices's.
    """
    pairs_all = drifter_pairs(start, end, drifter_id, n_states)
    pairs_sub = drifter_pairs(start[in_subset], end[in_subset], drifter_id[in_subset], n_states)
    where = np.searchsorted(pairs_all["drifter_ids"], pairs_sub["drifter_ids"])
    for times_drawn in resample_counts(pairs_all["drifter_ids"].size, n_boot, seed):
        yield fit_all(replicate_count_matrix(pairs_all, times_drawn)), fit_subset(replicate_count_matrix(pairs_sub, times_drawn[where]))


def track(S, P, coast, lat_c, lon_c, n_steps=N_STEPS):
    """What each source's material does, per step: {quantity: (n_steps, n_sources)}.

    pct_left: having left the box; pct_coastal: in the Benguela coastal cells that day; pct_reached_coast: entered them
    at any time up to then (connectivity.reached); centre_lat, centre_lon: over material still in the box.
    """
    n_cells = lat_c.size
    Q = absorbing(P, coast)
    X, H = S.copy(), S.copy()
    out = {k: np.empty((n_steps, S.shape[0])) for k in ("pct_left", "pct_coastal", "pct_reached_coast", "centre_lat", "centre_lon")}
    for n in range(n_steps):
        X, H = X @ P, H @ Q
        inside = X[:, :n_cells].sum(axis=1)
        out["pct_left"][n] = 100 * X[:, n_cells]
        out["pct_coastal"][n] = 100 * X[:, coast].sum(axis=1)
        out["pct_reached_coast"][n] = 100 * H[:, coast].sum(axis=1)
        out["centre_lat"][n] = X[:, :n_cells] @ lat_c / inside
        out["centre_lon"][n] = X[:, :n_cells] @ lon_c / inside
    return out


def subregion_shares(S, P, label, n_labels, n_steps=N_STEPS):
    """% of each source's material in each subregion, per step: (n_steps, n_sources, n_labels)."""
    to_label = np.zeros((P.shape[0], n_labels))
    to_label[np.arange(label.size), label] = 1.0
    X, out = S.copy(), np.empty((n_steps, S.shape[0], n_labels))
    for n in range(n_steps):
        X = X @ P
        out[n] = 100 * X @ to_label
    return out


def all_numbers(S, P, coast, lat_c, lon_c, label, n_labels):
    """track() and subregion_shares() in one dict, the numbers the bootstrap repeats."""
    out = track(S, P, coast, lat_c, lon_c)
    shares = subregion_shares(S, P, label, n_labels)
    for k in range(n_labels):
        out[f"pct_in_subregion_{k}"] = shares[:, :, k]
    return out


def stack_replicates(replicates):
    """A list of {quantity: array} into {quantity: array with replicates first}."""
    return {k: np.stack([r[k] for r in replicates]) for k in replicates[0]}


def heldout_rows(models, table, assignment, region, fine, steps, days, n_boot=N_BOOT, seed=SEED):
    """Mean log-likelihood per test move under each matrix, by the drogue status of the test move.

    Inputs:
        models: {"all": P^steps, "undrogued": P^steps}.
    Returns:
        rows: one for each kind of test move, with the gap (undrogued matrix minus all-drifter matrix) and its
        interval, resampling test drifters with the models held fixed (as score_intervals.py).
    """
    test = subset(table, set_mask(table["drifter_id"], assignment, "test"))
    start, end, _ = to_states(test, region, fine)
    ids = test["drifter_id"]
    lp = {k: log_lik(P, start, end) for k, P in models.items()}
    draws = resample_counts(np.unique(ids).size, n_boot, seed)
    rows = []
    for name, g in (("undrogued test moves", ~test["drogued"]), ("drogued test moves", test["drogued"].astype(bool))):
        if not g.any():
            continue
        low, high = percentile_interval(resampled_means(*drifter_totals(lp["undrogued"] - lp["all"], ids, g), draws), LEVEL)
        rows.append({"check": "held-out score", "days": days, "cells": name, "moves": int(g.sum()), "drifters": int(np.unique(ids[g]).size),
                     "score_all_drifters": round(float(lp["all"][g].mean()), 5), "score_undrogued_only": round(float(lp["undrogued"][g].mean()), 5),
                     "difference": round(float((lp["undrogued"] - lp["all"])[g].mean()), 5), "diff_low": round(float(low), 5), "diff_high": round(float(high), 5)})
    return rows


def plot(region, fine, dist, tracks, reps, release, drifters, out_path):
    """Top: where coast material is after a year, under each matrix. Bottom: its centre and having-left through time."""
    lat_edges, lon_edges = grid_edges(region, fine)
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    days = (np.arange(N_STEPS) + 1) * TAU_DAYS
    fig = plt.figure(figsize=(12.5, 7.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.82])
    norm = colors.LogNorm(vmin=1e-4, vmax=0.1)
    map_axes = [fig.add_subplot(grid[0, c]) for c in range(2)]
    for ax, key in zip(map_axes, SERIES):
        name, colour = SERIES[key]
        n = drifters[key]
        shown = np.ma.masked_where((n == 0).reshape(shape), np.maximum(dist[key][:-1].reshape(shape), norm.vmin))
        mesh = ax.pcolormesh(lon_edges, lat_edges, shown, cmap="Blues", norm=norm)
        draw_land(ax)
        for c in release[key]:
            r, k = divmod(int(c), shape[1])
            ax.add_patch(plt.Rectangle((lon_edges[k], lat_edges[r]), fine, fine, fill=False, ec="#d62728", lw=1.0, zorder=5))
        lat, lon = tracks[key]["centre_lat"][:, 0], tracks[key]["centre_lon"][:, 0]
        ax.plot(lon, lat, color="#1b2733", lw=1.2, ls=":", zorder=6)
        ax.plot(lon[-1], lat[-1], marker="D", ms=7, mfc="white", mec="#1b2733", zorder=7)
        ax.set_xlim(region["lon_min"], region["lon_max"])
        ax.set_ylim(region["lat_min"], region["lat_max"])
        ax.set_aspect("equal")
        ax.set_facecolor("#d9d9d9")
        ax.xaxis.set_major_formatter(_degrees("lon"))
        ax.yaxis.set_major_formatter(_degrees("lat"))
        ax.set_title(f"{name}: after {days[-1]} days from the Benguela coast", fontsize=10.5, loc="left")
    fig.colorbar(mesh, ax=map_axes, shrink=0.85, extend="both", label="probability per 2° cell (log scale)")

    for c, (quantity, ylabel, title, log) in enumerate((("centre_lon", "longitude (°E)", "Centre of the material still in the box", False),
                                                        ("pct_left", "% (log scale)", "Chance of having left the box", True))):
        ax = fig.add_subplot(grid[1, c])
        for key, (name, colour) in SERIES.items():
            y = tracks[key][quantity][:, 0]
            low, high = percentile_interval(reps[key][quantity][:, :, 0], LEVEL)
            ax.fill_between(days, low, high, color=colour, alpha=0.18, lw=0)
            ax.plot(days, y, color=colour, lw=2, label=name)
            ax.annotate(name, (days[-1], y[-1]), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=9, color="#1b2733")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("days after release")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlim(0, 490)  # room for the labels at the ends of the lines
        ax.legend(fontsize=9, frameon=False, loc="upper left" if log else "lower left")
        ax.grid(alpha=0.25)
    fig.text(0.01, -0.05, f"Each map is a 2° shrinkage matrix, {TAU_DAYS}-day steps; diamond and dots: the centre and its path; red: the release cells. Shading: 95% intervals from "
             f"{N_BOOT} drifter bootstrap replicates, each set's own.\nStatus is at the start of each move, and a drifter that has lost its drogue never regains it. "
             "Grey: land, and ocean cells with no drifters in that set. Source: src/inference/drogue_comparison.py.", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region, verbose=False)
    unt = undrogued_only(table)
    assignment = load_split(table["drifter_id"])
    start, end, n_cells = to_states(table, region, fine)
    n_states = n_cells + 1
    parent = parent_states(region, fine, coarse)
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    alpha_all = float(product["alpha"])

    # 1. Fit: alpha as evaluate.py, then on all of the set's drifters as shrinkage_fit.py
    alpha_all_check, P_fit_all = fit_on_split(table, region, fine, coarse, assignment)
    alpha_u, P_fit_u = fit_on_split(unt, region, fine, coarse, assignment)
    if abs(alpha_all_check - alpha_all) > 0.01:
        raise ValueError("The all-drifter alpha should reproduce the product's; check drifter_split.csv.")
    start_u, end_u, _ = to_states(unt, region, fine)
    fit_all, fit_u = shrinkage_fit(parent, alpha_all), shrinkage_fit(parent, alpha_u)
    counts_u = count_matrix(start_u, end_u, n_states)
    P_all, P_u = product["P"], fit_u(counts_u)
    _, drifters_all = per_cell_counts(start, table["drifter_id"], n_cells)
    _, drifters_u = per_cell_counts(start_u, unt["drifter_id"], n_cells)
    sampled = drifters_all > 0
    print(f"All drifters: alpha {alpha_all:.1f}, {np.unique(table['drifter_id']).size} drifters, {start.size:,} moves, "
          f"{100 * np.mean(drifters_all[sampled] >= N_MIN):.1f}% of cells adequate")
    print(f"Undrogued only: alpha {alpha_u:.1f}, {np.unique(unt['drifter_id']).size} drifters, {start_u.size:,} moves, "
          f"{100 * np.mean(drifters_u[sampled] >= N_MIN):.1f}% adequate, {int(np.sum(sampled & (drifters_u == 0)))} cells have only the prior")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_grid_result(os.path.join(PROCESSED_DIR, f"{UNDROGUED_STEM}.npz"), region, fine, TAU_DAYS, P=P_u, counts=counts_u, alpha=alpha_u, start_drifters=drifters_u)
    result = matrix_intervals(start_u, end_u, unt["drifter_id"], n_states, fit_u, P_u)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells with no drifters are NaN throughout
        tv_median = np.nanmedian(result["row_tv"], axis=0)
    save_grid_result(os.path.join(PROCESSED_DIR, f"{UNDROGUED_STEM}_bootstrap.npz"), region, fine, TAU_DAYS, se=result["se"].astype(np.float32),
                     ci_low=result["ci_low"].astype(np.float32), ci_high=result["ci_high"].astype(np.float32), row_tv_median=tv_median,
                     row_empty_share=result["row_empty_share"], start_drifters=drifters_u, n_boot=N_BOOT, seed=SEED, level=LEVEL, prior_weight=alpha_u)
    print("Saved the undrogued matrix and its bootstrap", flush=True)

    # 2. The answers and the gaps
    land = land_fraction(region, fine, land_paths())
    coast = coastal_cells(region, fine, land)
    names, label = subregion_map(region, fine, coast)
    lat_edges, lon_edges = grid_edges(region, fine)
    lat_c, lon_c = (a.ravel() for a in np.meshgrid(0.5 * (lat_edges[:-1] + lat_edges[1:]), 0.5 * (lon_edges[:-1] + lon_edges[1:]), indexing="ij"))
    orange = next(b for b in region["named_releases"] if b["name"] == "Orange Basin")
    orange_cells = box_cells(region, fine, orange)
    release_all, release_u = coast[drifters_all[coast] > 0], coast[drifters_u[coast] > 0]
    # Each set releases from its own sampled cells, as robustness.py and the app. "same cells" uses the cells both have.
    sources = [COAST_NAME, "Orange Basin", f"{COAST_NAME}, same cells"]
    S_all = start_matrix([release_all, orange_cells, release_u], n_states)
    S_u = start_matrix([release_u, orange_cells, release_u], n_states)
    est = {"all": all_numbers(S_all, P_all, coast, lat_c, lon_c, label, len(names)), "undrogued": all_numbers(S_u, P_u, coast, lat_c, lon_c, label, len(names))}

    own = {"all": [], "undrogued": []}
    for P_b in replicate_matrices(start, end, table["drifter_id"], n_states, fit_all):
        own["all"].append(all_numbers(S_all, P_b, coast, lat_c, lon_c, label, len(names)))
    for P_b in replicate_matrices(start_u, end_u, unt["drifter_id"], n_states, fit_u):
        own["undrogued"].append(all_numbers(S_u, P_b, coast, lat_c, lon_c, label, len(names)))
    own = {k: stack_replicates(v) for k, v in own.items()}
    print("Own intervals done", flush=True)

    gap = []
    for P_a, P_b in paired_replicates(start, end, table["drifter_id"], ~table["drogued"], n_states, fit_all, fit_u):
        a, b = all_numbers(S_all, P_a, coast, lat_c, lon_c, label, len(names)), all_numbers(S_u, P_b, coast, lat_c, lon_c, label, len(names))
        gap.append({k: b[k] - a[k] for k in a})
    gap = stack_replicates(gap)
    print("Paired intervals done", flush=True)

    rows = [{"check": "the data", "cells": name, "moves": int(s.size), "drifters": int(np.unique(ids).size), "alpha": round(a, 2),
             "pct_cells_adequate": round(float(100 * np.mean(d[sampled] >= N_MIN)), 1), "cells_only_prior": int(np.sum(sampled & (d == 0))),
             "release_cells": int(r.size)}
            for name, s, ids, a, d, r in (("all drifters", start, table["drifter_id"], alpha_all, drifters_all, release_all),
                                          ("undrogued only", start_u, unt["drifter_id"], alpha_u, drifters_u, release_u))]
    labels = {k: k for k in est["all"]} | {f"pct_in_subregion_{k}": f"pct_in {n}" for k, n in enumerate(names)}
    for step, days in HORIZONS.items():
        for q, name in labels.items():
            for i, source in enumerate(sources):
                if q == "pct_reached_coast" and source != "Orange Basin":
                    continue  # it starts there
                lo_a, hi_a = percentile_interval(own["all"][q][:, step - 1, i], LEVEL)
                lo_u, hi_u = percentile_interval(own["undrogued"][q][:, step - 1, i], LEVEL)
                lo_d, hi_d = percentile_interval(gap[q][:, step - 1, i], LEVEL)
                a, u = est["all"][q][step - 1, i], est["undrogued"][q][step - 1, i]
                rows.append({"check": "answers", "source": source, "quantity": name, "days": days, "all_drifters": round(float(a), 4),
                             "all_low": round(float(lo_a), 4), "all_high": round(float(hi_a), 4), "undrogued_only": round(float(u), 4),
                             "undrogued_low": round(float(lo_u), 4), "undrogued_high": round(float(hi_u), 4), "difference": round(float(u - a), 4),
                             "diff_low": round(float(lo_d), 4), "diff_high": round(float(hi_d), 4)})

    # 3. Which predicts undrogued drifters better, and are its chances right?
    cache_exists = os.path.exists(os.path.join(INTERIM_DIR, f"{region['name']}.npz"))
    checkpoints = None
    if cache_exists:
        data, records = load_region(region), load_records(region)
        built = build_transitions(data, records, TAU_DAYS)
        known = np.isin(built["drifter_id"], list(assignment))
        test_starts = known & set_mask(built["drifter_id"], assignment, "test") & ~built["drogued"].astype(bool)
        start_cell, _, _ = cell_index(built["start_lon"][test_starts].astype(float), built["start_lat"][test_starts].astype(float), region, fine)
        start_ids = built["drifter_id"][test_starts]
        checkpoints = np.stack([build_transitions(data, records, TAU_DAYS * k)["outcome"][test_starts] for k in range(1, N_STEPS + 1)])
    groups = {"all with drifters": sampled}
    power = {"all": np.eye(n_states), "undrogued": np.eye(n_states)}
    fitted = {"all": P_fit_all, "undrogued": P_fit_u}
    done = 0
    for steps, days in HORIZONS.items():
        for k in power:
            power[k] = power[k] @ np.linalg.matrix_power(fitted[k], steps - done)
        done = steps
        moves = lag_table(days, region, cache_exists)
        if moves is None:
            print(f"{days} days: skipped (no saved table and no region cache)")
            continue
        moves = subset(moves, np.isin(moves["drifter_id"], list(assignment)))
        moves["drogued"] = moves["drogued"].astype(bool)
        rows += heldout_rows(power, moves, assignment, region, fine, steps, days)
        test_u = subset(moves, set_mask(moves["drifter_id"], assignment, "test") & ~moves["drogued"])
        s_te, e_te, _ = to_states(test_u, region, fine)
        print(f"{days} days: {s_te.size:,} undrogued test moves from {np.unique(test_u['drifter_id']).size} drifters", flush=True)
        for key, (name, _) in SERIES.items():
            for r in coverage(power[key], n_cells, s_te, e_te, test_u["drifter_id"], groups):
                rows.append({"check": "likely area", "matrix": name, "days": days, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}})
            for r in reliability(power[key], s_te, e_te, test_u["drifter_id"]):
                rows.append({"check": "reliability", "matrix": name, "days": days, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}})
            if checkpoints is not None:
                for r in left_box(power[key], n_cells, start_cell, checkpoints[:steps], start_ids):
                    rows.append({"check": "having left the box", "matrix": name, "days": days, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"drogue_comparison_tau{TAU_DAYS}d.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    dist = {"all": S_all[0] @ np.linalg.matrix_power(P_all, N_STEPS), "undrogued": S_u[0] @ np.linalg.matrix_power(P_u, N_STEPS)}
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_out = os.path.join(FIGURES_DIR, "drogue_comparison.png")
    plot(region, fine, dist, est, own, {"all": release_all, "undrogued": release_u}, {"all": drifters_all, "undrogued": drifters_u}, fig_out)
    for r in rows:
        if r["check"] in ("the data", "held-out score") or (r["check"] == "answers" and r["days"] == 360 and r["source"] != f"{COAST_NAME}, same cells" and not r["quantity"].startswith("pct_in ")):
            print(r)
    print(f"Saved {out} and {fig_out}")
