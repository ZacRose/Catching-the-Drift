"""
shrinkage_fit.py

Related task: Week 6 Quantitative - fit the 2 deg shrinkage model, and show how much borrowing
from the 4 deg cell adds to each cell, counted in drifters (open issue j).

1. alpha is chosen exactly as in evaluate.py: fitted on the training drifters, best score on the
   validation drifters (evaluate.tune_alpha). The test drifters played no part in choosing it.
2. With alpha fixed, the model is refitted on ALL drifters, as the product another analyst loads
   (objective O1). Held-out scores for it come from evaluate.py, which fits on train + valid only.
3. Effective sample size per cell, in drifters. The posterior row is a Dirichlet whose weight is
   n_i + alpha transitions, but those transitions are overlapping daily moves: a drifter makes
   about 15 of them per 2 deg cell (transition_adequacy.py), so counting moves overstates the
   information about 15 times (issue c). Stated in drifters instead:
       effective drifters = own drifters + alpha / (moves per drifter)
   The second term is what the 4 deg prior is worth. A cell counts as adequate under the same
   rule as the adequacy map (at least N_MIN = 100), which keeps the worst-case standard error
   of a transition probability at or below 0.05 (decision 3).

Outputs:
- data/processed/shrinkage_P_2deg_tau15d.npz: P, counts, alpha, start_drifters and
  effective_drifters, with the grid metadata from transitions.save_grid_result. Rows for cells
  with start_drifters == 0 are the 4 deg prior alone (land, or never sampled): mask them as needed.
- outputs/results/effective_sample_size_tau15d.csv and outputs/figures/effective_sample_size_map.png.

Run: python3 -m src.inference.shrinkage_fit   (needs outputs/results/drifter_split.csv)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, save_grid_result
from src.inference.adequacy import N_MIN
from src.inference.estimators import count_matrix, prior_rows, shrinkage_P
from src.inference.evaluate import states_for, tune_alpha
from src.inference.split import load_split
from src.inference.transition_adequacy import proxy_from_table
from src.inference.transition_data import TAU_DAYS, load_transitions, parent_states, to_states


def moves_per_drifter(counts, start_drifters):
    """Median, over cells with any drifters, of counted moves per distinct drifter starting there."""
    moves = counts[:-1].sum(axis=1)
    sampled = start_drifters > 0
    return float(np.median(moves[sampled] / start_drifters[sampled]))


def effective_drifters(start_drifters, alpha, per_drifter):
    """Posterior effective sample size per cell in drifters: own drifters plus the prior's worth."""
    return start_drifters + alpha / per_drifter


def plot(region, cell_size, ocean, naive, posterior, alpha, prior_worth, out_path):
    """Naive drifters per cell beside the posterior effective drifters, on one log colour scale."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    vmax = posterior[ocean].max()

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), layout="constrained")
    panels = (
        (naive, f"own drifters (naive estimate): {np.mean(naive[ocean] >= N_MIN):.1%} adequate"),
        (posterior, f"effective drifters with the {BASELINE_CELL_SIZE:g}° prior: "
                    f"{np.mean(posterior[ocean] >= N_MIN):.1%} adequate"),
    )
    for ax, (counts, title) in zip(axes, panels):
        grid = np.where(ocean, counts, 0).reshape(shape)
        mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_equal(grid, 0),
                             norm=colors.LogNorm(vmin=1, vmax=vmax), cmap="viridis")
        ax.contour(lon_c, lat_c, grid >= N_MIN, levels=[0.5], colors="black", linewidths=1.5)
        ax.set_aspect("equal")
        ax.set_title(f"{cell_size:g}° cells, {title}", fontsize=10)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    fig.colorbar(mesh, ax=axes, label="Distinct drifters per cell (log scale)", shrink=0.8)
    fig.suptitle(f"How much borrowing from the {BASELINE_CELL_SIZE:g}° cell adds, {region['label']}")
    fig.text(0.01, -0.08,
             f"Left: distinct drifters starting a {TAU_DAYS}-day move in each cell. Right: the same plus what the "
             f"{BASELINE_CELL_SIZE:g}° prior is worth, alpha = {alpha:.0f} pseudo-moves = {prior_worth:.2f} drifters "
             "(a drifter makes about 15 moves per cell).\n"
             f"Black line encloses cells with at least {N_MIN} (worst-case standard error of a transition "
             "probability <= 0.05). Blank cells have no drifters (mostly land).", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region)
    assignment = load_split(table["drifter_id"])
    parent = parent_states(region, fine, coarse)

    # 1. alpha from train -> valid, as in evaluate.py
    s_tr, e_tr, n_cells, _ = states_for(table, region, fine, assignment, "train")
    s_va, e_va, _, _ = states_for(table, region, fine, assignment, "valid")
    alpha, _ = tune_alpha(count_matrix(s_tr, e_tr, n_cells + 1), parent, s_va, e_va)

    # 2. Refit on all drifters
    start, end, _ = to_states(table, region, fine)
    counts = count_matrix(start, end, n_cells + 1)
    P = shrinkage_P(counts, prior_rows(counts, parent), alpha)
    _, start_drifters = per_cell_counts(start, table["drifter_id"], n_cells)

    # 3. Effective sample size in drifters, over the same ocean cells as the adequacy map
    per_drifter = moves_per_drifter(counts, start_drifters)
    prior_worth = alpha / per_drifter
    posterior = effective_drifters(start_drifters, alpha, per_drifter)
    ocean = proxy_from_table(region, fine)[1] > 0
    naive_ok, posterior_ok = start_drifters[ocean] >= N_MIN, posterior[ocean] >= N_MIN
    summary = {
        "cell_size_deg": fine,
        "tau_days": TAU_DAYS,
        "alpha": round(alpha, 1),
        "moves_per_drifter": per_drifter,
        "prior_worth_drifters": round(prior_worth, 2),
        "ocean_cells": int(ocean.sum()),
        "pct_adequate_naive": round(float(100 * naive_ok.mean()), 1),
        "pct_adequate_posterior": round(float(100 * posterior_ok.mean()), 1),
        "cells_made_adequate": int(np.sum(posterior_ok & ~naive_ok)),
        "ocean_cells_with_no_drifters": int(np.sum(start_drifters[ocean] == 0)),
    }
    print(summary)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    p_path = os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz")
    save_grid_result(p_path, region, fine, TAU_DAYS, P=P, counts=counts, alpha=alpha,
                     start_drifters=start_drifters, effective_drifters=posterior)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, f"effective_sample_size_tau{TAU_DAYS}d.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "effective_sample_size_map.png")
    plot(region, fine, ocean, start_drifters, posterior, alpha, prior_worth, fig_path)
    print(f"Saved {p_path}, {table_path} and {fig_path}")
