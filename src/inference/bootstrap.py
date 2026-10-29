"""
bootstrap.py

Related task: Week 6 Computational - the drifter bootstrap (open issues c and i): uncertainty for
P and for every held-out score, by resampling whole drifters.

Why whole drifters: a drifter starts a 15-day move every day, and consecutive moves share 14 of
their 15 days, so the 776,777 counted moves are far from independent (about 15 per drifter per
2 deg cell). Formulas that treat each move as independent would give intervals that are much too
narrow. The independent unit is the drifter, so the drifter is what gets resampled.

One replicate: draw n drifters with replacement from the n available (some come up twice, some not
at all), keep every move of each drawn drifter (once per draw), and redo the calculation exactly
as on the real data. Repeating this N_BOOT = 1000 times gives a spread of answers. The 95% interval
is the 2.5th to 97.5th percentile of that spread, and the standard error is its standard deviation.
The seed is fixed, so every run gives the same replicates.

Two uses:
1. Uncertainty in P (objective O1; this script). P is refitted on each replicate of ALL drifters:
   the 2 deg shrinkage model (the product from shrinkage_fit.py) and the 4 deg naive P. For the
   shrinkage model the 4 deg prior row is rebuilt on every replicate, because it comes from the
   same data (empirical Bayes), but alpha stays at the product's value. How much the answer depends
   on alpha is the separate prior-sensitivity check (alpha_sensitivity.png).
2. Intervals on the held-out scores (issue i; score_intervals.py). There the TEST drifters are
   resampled and the fitted models are held fixed.
Anything else computed from P (the release maps, the long-run pattern) gets an interval the same
way: replicate_matrices() yields each refitted P, and the quantity is computed on each.

A replicate's counts are one weighted bincount: every drifter's moves are first reduced to
(drifter, start -> end, count) triples, about 123,000 at 2 deg, and a replicate weights each
triple by how many times its drifter was drawn. A 2 deg refit then takes under 10 ms.

Outputs:
- data/processed/shrinkage_P_2deg_tau15d_bootstrap.npz and naive_P_4deg_tau15d_bootstrap.npz:
  se, ci_low and ci_high for every entry of P (same layout as P), row_tv_median per cell, and the
  bootstrap settings, with the grid metadata from transitions.save_grid_result. Load P itself
  from the matching file without "_bootstrap".
- outputs/results/bootstrap_P_summary_tau15d.csv and outputs/figures/bootstrap_uncertainty_map.png.

Per-cell summary: row_tv_median, the median over replicates of the total variation distance
between the cell's refitted row and its row in P (half the sum of absolute differences): 0 means
the row never changes, 1 that it moves to entirely different destinations. It is the share of
the cell's material that would be sent somewhere else by a different sample of drifters.

Limitation: the bootstrap can only show variation that is in the data. A cell crossed by one or two
drifters has nearly the same row in every replicate that draws them, so its interval comes out
narrow even though the row rests on almost nothing (some such cells have row_tv_median near 0).
Read intervals for cells under about 10 drifters as unreliable, and use start_drifters beside them.

Run: python3 -m src.inference.bootstrap   (about 2 minutes; needs shrinkage_P_2deg_tau15d.npz from
src.inference.shrinkage_fit)
"""
import csv
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, save_grid_result
from src.inference.adequacy import N_MIN
from src.inference.estimators import count_matrix, naive_P, prior_rows, shrinkage_P
from src.inference.transition_data import TAU_DAYS, load_transitions, parent_states, to_states

N_BOOT = 1000
SEED = 3001
LEVEL = 0.95
MAX_BLOCK_BYTES = 2.5e8  # memory for one block of replicate rows (see matrix_intervals)


def shrinkage_fit(parent, alpha):
    """The product's estimator as a function of a count matrix: 2 deg shrinkage with the prior
    rebuilt from the same counts (shrinkage_fit.py) and alpha held fixed."""
    return lambda counts: shrinkage_P(counts, prior_rows(counts, parent), alpha)


def resample_counts(n_drifters, n_boot=N_BOOT, seed=SEED):
    """How many times each drifter is drawn in each replicate.

    Drawing n drifters with replacement, each equally likely, is one multinomial draw.
    Returns:
        (n_boot, n_drifters) integer array; every row sums to n_drifters.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_drifters, np.full(n_drifters, 1.0 / n_drifters), size=n_boot)


def drifter_pairs(start, end, drifter_id, n_states):
    """Every drifter's moves reduced to (drifter, start -> end pair, count) triples.

    Returns:
        dict with drifter_ids (sorted; replicate weights follow this order), and per triple the
        drifter's position in drifter_ids, the flat pair index start * n_states + end, and the
        number of such moves; plus n_states.
    """
    ids, rank = np.unique(drifter_id, return_inverse=True)
    n_pairs = n_states ** 2
    key = rank.astype(np.int64) * n_pairs + start.astype(np.int64) * n_states + end
    triples, n = np.unique(key, return_counts=True)
    return {"drifter_ids": ids, "drifter": triples // n_pairs, "pair": triples % n_pairs,
            "n": n, "n_states": n_states}


def replicate_count_matrix(pairs, times_drawn):
    """Count matrix of one replicate: each drifter's moves counted once per time it was drawn."""
    k = pairs["n_states"]
    weights = times_drawn[pairs["drifter"]] * pairs["n"]
    return np.bincount(pairs["pair"], weights=weights, minlength=k * k).reshape(k, k)


def replicate_matrices(start, end, drifter_id, n_states, fit, n_boot=N_BOOT, seed=SEED):
    """Yield fit(counts) for each bootstrap replicate of the drifters, in a fixed order.

    Inputs:
        start, end, drifter_id: one entry per transition (states as in transition_data.to_states).
        fit: function from a count matrix to a transition matrix, e.g. naive_P.
    """
    pairs = drifter_pairs(start, end, drifter_id, n_states)
    for times_drawn in resample_counts(pairs["drifter_ids"].size, n_boot, seed):
        yield fit(replicate_count_matrix(pairs, times_drawn))


def drifter_totals(values, drifter_id, mask=None):
    """Per-drifter sum of a per-transition quantity, and per-drifter number of transitions summed.

    Inputs:
        values, drifter_id: one entry per transition.
        mask: optional bool per transition; only these transitions are summed (e.g. one cell group).
    Returns:
        (totals, moves), each with one entry per drifter in sorted drifter_id order, the order the
        rows of resample_counts(np.unique(drifter_id).size) refer to.
    """
    ids, rank = np.unique(drifter_id, return_inverse=True)
    keep = np.ones(rank.size, dtype=bool) if mask is None else mask
    totals = np.bincount(rank[keep], weights=values[keep], minlength=ids.size)
    moves = np.bincount(rank[keep], minlength=ids.size).astype(float)
    return totals, moves


def resampled_means(totals, moves, draws):
    """Mean per transition in each replicate: the drawn drifters' totals over their transitions.

    draws: resample_counts() output, (n_boot, n_drifters). A drifter drawn twice counts twice.
    """
    with np.errstate(invalid="ignore", divide="ignore"):  # a replicate with no moves in a group gives NaN
        return (draws @ totals) / (draws @ moves)


def percentile_interval(samples, level=LEVEL, axis=0):
    """(low, high) percentile interval of bootstrap samples along an axis, ignoring NaN."""
    q = [(1 - level) / 2, (1 + level) / 2]
    if np.isnan(samples).any():
        low, high = np.nanquantile(samples, q, axis=axis)
    else:
        low, high = np.quantile(samples, q, axis=axis)
    return low, high


def matrix_intervals(start, end, drifter_id, n_states, fit, P_hat, n_boot=N_BOOT, seed=SEED,
                     level=LEVEL, max_block_bytes=MAX_BLOCK_BYTES):
    """Per-entry bootstrap standard error and percentile interval of a fitted P, and per-row spread.

    Keeping all replicates of a 2 deg P would need about 2.3 GB, so rows are done in blocks: the
    replicates are regenerated for each block (same seed, so the same replicates) and only that
    block's rows are kept. A replicate can leave a thin cell with no drifters at all; naive P then
    has no row there, and that replicate is left out for that row (NaN) rather than counted as zeros.

    Inputs:
        start, end, drifter_id, n_states, fit: as in replicate_matrices().
        P_hat: the matrix fitted on all the data (the centre that row_tv is measured from).
    Returns:
        dict: se, ci_low, ci_high (each n_states x n_states), row_tv (n_boot x n_states) and
        row_empty_share (share of replicates in which each row had no data).
    """
    rows_per_block = max(1, int(max_block_bytes // (n_boot * n_states * 4)))
    se = np.full((n_states, n_states), np.nan)
    low, high = np.full_like(se, np.nan), np.full_like(se, np.nan)
    row_tv = np.full((n_boot, n_states), np.nan)
    empty_share = np.zeros(n_states)
    for r0 in range(0, n_states, rows_per_block):
        r1 = min(r0 + rows_per_block, n_states)
        block = np.empty((n_boot, r1 - r0, n_states), dtype=np.float32)
        for b, P in enumerate(replicate_matrices(start, end, drifter_id, n_states, fit, n_boot, seed)):
            rows = P[r0:r1].astype(float)  # a copy, so the NaNs below don't touch P
            rows[rows.sum(axis=1) == 0] = np.nan
            row_tv[b, r0:r1] = 0.5 * np.abs(rows - P_hat[r0:r1]).sum(axis=1)
            block[b] = rows  # stored as float32 to halve the memory
        empty_share[r0:r1] = np.isnan(block[:, :, 0]).mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # rows with no data in any replicate stay NaN
            se[r0:r1] = np.nanstd(block, axis=0, ddof=1, dtype=np.float64)
            low[r0:r1], high[r0:r1] = percentile_interval(block, level)
    return {"se": se, "ci_low": low, "ci_high": high, "row_tv": row_tv, "row_empty_share": empty_share}


def summary_rows(name, P_hat, se, tv_median, counts, start_drifters, prior_weight):
    """Per-grid summary over cells with drifters, split by the N_MIN rule.

    prior_weight: pseudo-moves added to every row (alpha for shrinkage, 0 for naive). The
    independent-move standard error of the stay probability p is sqrt(p (1 - p) / (n_i + prior_weight + 1))
    for the shrinkage posterior and sqrt(p (1 - p) / n_i) for naive P (n_i = the cell's moves);
    the ratio of the bootstrap SE to it shows how far treating moves as independent understates
    the uncertainty (issues c and j).
    """
    n_cells = start_drifters.size
    moves = counts[:n_cells].sum(axis=1)
    p_stay = np.diag(P_hat)[:n_cells]
    extra = prior_weight + 1 if prior_weight else 0
    stay_se = np.diag(se)[:n_cells]
    with np.errstate(divide="ignore", invalid="ignore"):  # cells with no moves, or p of 0 or 1
        independent_se = np.sqrt(p_stay * (1 - p_stay) / (moves + extra))
        ratio = stay_se / independent_se
    max_se = np.nanmax(np.where(np.isnan(se[:n_cells]), 0, se[:n_cells]), axis=1)
    tv = tv_median[:n_cells]

    rows = []
    sampled = start_drifters > 0
    for cells, group in (("all with drifters", sampled), (f"under {N_MIN} drifters", sampled & (start_drifters < N_MIN)),
                         (f"{N_MIN}+ drifters", start_drifters >= N_MIN)):
        ok = group & (p_stay > 0) & (p_stay < 1) & np.isfinite(ratio)
        rows.append({
            "model": name, "cells": cells, "n_cells": int(group.sum()),
            "median_row_tv": round(float(np.median(tv[group])), 4),
            "median_max_entry_se": round(float(np.median(max_se[group])), 4),
            "pct_cells_max_entry_se_le_0.05": round(float(100 * np.mean(max_se[group] <= 0.05)), 1),
            "median_stay_se": round(float(np.median(stay_se[group])), 4),
            "median_stay_se_ratio_to_independent_moves": round(float(np.median(ratio[ok])), 2),
        })
    return rows


def plot(region, fine, coarse, maps, out_path):
    """(a) row_tv_median per 2 deg cell; (b) row_tv_median against drifters per cell, both grids."""
    lat_edges, lon_edges = grid_edges(region, fine)
    tv_fine, drifters_fine = maps[fine]
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.9, 1]})
    ax = axes[0]
    grid = np.ma.masked_where((drifters_fine == 0).reshape(shape), tv_fine.reshape(shape))
    mesh = ax.pcolormesh(lon_edges, lat_edges, grid, cmap="magma_r", vmin=0, vmax=0.3)
    ax.contour(0.5 * (lon_edges[:-1] + lon_edges[1:]), 0.5 * (lat_edges[:-1] + lat_edges[1:]),
               drifters_fine.reshape(shape) >= N_MIN, levels=[0.5], colors="#1f77b4", linewidths=1.2)
    ax.set_facecolor("#d9d9d9")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"(a) How much each {fine:g}° row of the shrinkage P changes between bootstrap replicates",
                 fontsize=10, loc="left")
    fig.colorbar(mesh, ax=ax, extend="max", shrink=0.85,
                 label="median share of the cell's material\nsent elsewhere (total variation)")

    ax = axes[1]
    for size, colour, label in ((coarse, "#ff7f0e", f"{coarse:g}° naive P"), (fine, "#1f77b4", f"{fine:g}° shrinkage P")):
        tv, drifters = maps[size]
        ok = drifters > 0
        ax.scatter(drifters[ok], tv[ok], s=9, alpha=0.6, color=colour, label=label, edgecolors="none")
    ax.axvline(N_MIN, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Distinct drifters starting a move in the cell (log scale)")
    ax.set_ylabel("Median total variation")
    ax.set_title("(b) Rows settle down as drifters accumulate", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    fig.text(0.01, -0.1,
             f"{N_BOOT} replicates, each drawing the {region['label']} drifters with replacement and refitting P "
             f"(tau = {TAU_DAYS} days). Total variation is half the summed absolute difference between a replicate's "
             "row and the fitted row.\nBlue line in (a) encloses cells with at least "
             f"{N_MIN} drifters; grey cells have no drifters. Dotted line in (b): {N_MIN} drifters. Cells with only 1-2 drifters "
             "can show little spread because resampling cannot add variation the data lack.", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region)
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    alpha = float(product["alpha"])
    parent = parent_states(region, fine, coarse)

    models = {
        fine: (f"shrinkage_P_{fine:g}deg", f"{fine:g} deg shrinkage (alpha {alpha:.1f})", alpha,
               shrinkage_fit(parent, alpha)),
        coarse: (f"naive_P_{coarse:g}deg", f"{coarse:g} deg naive", 0.0, naive_P),
    }
    summary, maps = [], {}
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for size, (stem, name, prior_weight, fit) in models.items():
        start, end, n_cells = to_states(table, region, size)
        counts = count_matrix(start, end, n_cells + 1)
        P_hat = fit(counts)
        if size == fine and not np.allclose(P_hat, product["P"]):
            raise ValueError("Refitted shrinkage P differs from the saved product; rerun src.inference.shrinkage_fit.")
        _, start_drifters = per_cell_counts(start, table["drifter_id"], n_cells)
        result = matrix_intervals(start, end, table["drifter_id"], n_cells + 1, fit, P_hat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # cells with no drifters are NaN throughout
            tv_median = np.nanmedian(result["row_tv"], axis=0)
        maps[size] = (tv_median[:n_cells], start_drifters)
        summary += summary_rows(name, P_hat, result["se"], tv_median, counts, start_drifters, prior_weight)

        path = os.path.join(PROCESSED_DIR, f"{stem}_tau{TAU_DAYS}d_bootstrap.npz")
        save_grid_result(path, region, size, TAU_DAYS,
                         se=result["se"].astype(np.float32), ci_low=result["ci_low"].astype(np.float32),
                         ci_high=result["ci_high"].astype(np.float32), row_tv_median=tv_median,
                         row_empty_share=result["row_empty_share"], start_drifters=start_drifters,
                         n_boot=N_BOOT, seed=SEED, level=LEVEL, prior_weight=prior_weight)
        print(f"{name}: saved {path}")

    for row in summary:
        print(row)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, f"bootstrap_P_summary_tau{TAU_DAYS}d.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "bootstrap_uncertainty_map.png")
    plot(region, fine, coarse, maps, fig_path)
    print(f"Saved {table_path} and {fig_path}")
