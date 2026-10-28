"""
long_run.py

Related task: Week 7 Quantitative - the long-run pattern (quasi-stationary distribution) and the
mixing time (MATH2501 Linear Algebra; proposal objective O2 and RQ1 "where material gathers").

Why "quasi"-stationary: "left R" is absorbing, so in the very long run everything has left the
box and the ordinary stationary distribution is just "100% left R", which says nothing. Instead,
take Q = P restricted to the cells inside R (drop the exit row and column). Q is sub-stochastic:
each row sums to 1 minus that cell's chance of leaving. Its leading left eigenvector,
    pi Q = lambda_1 pi,   pi >= 0, sum(pi) = 1,
is the quasi-stationary distribution (QSD): the shape that the material still inside R settles
into after a long time, whatever the starting point. It is where material gathers.
- lambda_1 (just below 1) is the share of that settled material still in R after each 15-day
  step, so the material's half-life in the box is ln 2 / (-ln lambda_1) steps, and its mean
  remaining time in R is 1 / (1 - lambda_1) steps.
- pi is a share per cell. Cells shrink towards the pole (a 2 deg cell at 41S is 24% smaller than
  one at 5S, decision 1), so it is divided by the cell's area inside the box and shown as a
  relative density (1 = the box average). Coastal cells are partly land, which this doesn't
  correct for, so their density is if anything understated.

Mixing time: how fast the system forgets where material started.
- Spectral: the gap to the leading eigenvalue sets the rate. Differences from the QSD shrink by
  about |lambda_2| / lambda_1 per step, so their e-folding time is -tau / ln(|lambda_2| / lambda_1).
- Direct: release material in one cell, step forward, and rescale what is still in R to sum to 1.
  The mixing time from that cell is the first step at which this is within total variation 0.25 of
  the QSD (the usual textbook cut-off). Reported as the median and the slowest over cells with at
  least N_MIN drifters.

Two matrices, so the answer can be checked across grids:
- 4 deg naive P (data/processed/naive_P_4deg_tau15d.npz), the baseline;
- 2 deg shrinkage P (data/processed/shrinkage_P_2deg_tau15d.npz), the product.
Cells with no drifters starting a move have no data of their own (land, or never sampled; for
shrinkage their row is the 4 deg prior alone), so they are dropped from Q: material that reaches
them is treated like material that left R. At 4 deg this also fixes the 39 moves into cells with
no row (open issue b): they are counted as lost, as before, but now explicitly.

Prior sensitivity (proposal: "how sensitive is it to alpha"): the 2 deg model is refitted on all
drifters at alpha / 4 and 4 x alpha, and the QSD, half-life and mixing time recomputed.

No uncertainty yet: intervals on all of this need the drifter bootstrap (issue c), which refits
P on resampled drifters; long_run_summary() takes any P, so the bootstrap can call it per refit.

Outputs: outputs/results/long_run_tau15d.csv, outputs/results/long_run_top_cells_tau15d.csv,
outputs/figures/quasi_stationary_map.png.

Run: python3 -m src.linalg.long_run   (seconds; needs the two .npz files above)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.grid import grid_edges
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS
from src.gridding.trajectory_map import draw_land
from src.inference.adequacy import N_MIN
from src.inference.estimators import prior_rows, shrinkage_P
from src.inference.transition_data import parent_states

EARTH_RADIUS_KM = 6371.0
TV_CUTOFF = 0.25
MAX_STEPS = 1000  # 1000 x 15 days, about 41 years
ALPHA_FACTORS = [0.25, 1.0, 4.0]


def quasi_stationary(Q):
    """Leading eigenvalue and left eigenvector of a sub-stochastic matrix, and |lambda_2|.

    Returns:
        (lambda_1, pi, abs_lambda_2), pi >= 0 summing to 1.
    """
    vals, vecs = np.linalg.eig(Q.T)
    order = np.argsort(-np.abs(vals))
    lead = order[0]
    if abs(vals[lead].imag) > 1e-9:
        raise ValueError("Leading eigenvalue is complex - Q isn't a sub-stochastic matrix.")
    pi = np.real(vecs[:, lead])
    pi = pi / pi.sum()
    if np.any(pi < -1e-10):
        raise ValueError("Leading eigenvector has mixed signs - Q may split into separate parts.")
    pi = np.clip(pi, 0, None)
    return float(vals[lead].real), pi / pi.sum(), float(np.abs(vals[order[1]]))


def cell_areas_km2(region, cell_size):
    """Area of each cell's part inside the box (flat array, cell ids as in grid.cell_index)."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat_lo = np.clip(lat_edges[:-1], region["lat_min"], region["lat_max"])
    lat_hi = np.clip(lat_edges[1:], region["lat_min"], region["lat_max"])
    lon_max = region["lon_max"] + 360 if region["lon_min"] > region["lon_max"] else region["lon_max"]
    lon_lo = np.maximum(lon_edges[:-1], region["lon_min"])
    lon_hi = np.minimum(lon_edges[1:], lon_max)
    band = np.sin(np.radians(lat_hi)) - np.sin(np.radians(lat_lo))  # per latitude row
    width = np.radians(np.maximum(lon_hi - lon_lo, 0))             # per longitude column
    return (EARTH_RADIUS_KM ** 2 * band[:, None] * width[None, :]).ravel()


def mixing_steps(Q, pi, max_steps=MAX_STEPS, cutoff=TV_CUTOFF):
    """For every start cell, the first step at which the surviving material's distribution is
    within total variation `cutoff` of the QSD (np.inf if not reached within max_steps)."""
    n = Q.shape[0]
    steps = np.full(n, np.inf)
    M = np.eye(n)
    for t in range(1, max_steps + 1):
        M = M @ Q
        M /= M.sum(axis=1, keepdims=True)  # rescale what is still in R
        tv = 0.5 * np.abs(M - pi[None, :]).sum(axis=1)
        newly = np.isinf(steps) & (tv <= cutoff)
        steps[newly] = t
        if not np.isinf(steps).any():
            break
    return steps


def long_run_summary(P, keep, adequate, areas, tau_days=TAU_DAYS):
    """QSD, residence time and mixing time for one transition matrix.

    Inputs:
        P: (n_cells + 1) x (n_cells + 1) transition matrix, last state "left R".
        keep: bool per cell, the cells kept in Q (those with drifters).
        adequate: bool per cell, the start cells the mixing time is summarised over.
        areas: cell areas (km^2) inside the box.
    Returns:
        (summary dict, density per cell (relative, NaN outside keep), pi per cell (0 outside keep)).
    """
    cells = np.flatnonzero(keep)
    Q = P[np.ix_(cells, cells)]
    lam1, pi_kept, lam2 = quasi_stationary(Q)
    steps = mixing_steps(Q, pi_kept)

    pi = np.zeros(keep.size)
    pi[cells] = pi_kept
    density = np.full(keep.size, np.nan)
    per_area = pi_kept / areas[cells]
    density[cells] = per_area / (pi_kept.sum() / areas[cells].sum())  # 1 = box average

    mix = steps[adequate[cells]]
    ratio = lam2 / lam1
    summary = {
        "cells_in_Q": int(cells.size),
        "lambda_1": lam1,
        "half_life_years": float(np.log(2) / -np.log(lam1) * tau_days / 365.25),
        "mean_time_in_R_years": float(1 / (1 - lam1) * tau_days / 365.25),
        "abs_lambda_2": lam2,
        "mixing_efold_years": float(-tau_days / np.log(ratio) / 365.25) if ratio < 1 else float("inf"),
        "mixing_median_years": float(np.median(mix) * tau_days / 365.25),
        "mixing_slowest_years": float(np.max(mix) * tau_days / 365.25),
        "pct_qsd_in_top_10pct_cells": round(float(100 * np.sort(pi_kept)[::-1][: max(1, cells.size // 10)].sum()), 1),
    }
    return summary, density, pi


def coarsen(pi_fine, parent):
    """Add up a fine-grid distribution over cells into its coarse parent cells."""
    return np.bincount(parent[:-1], weights=pi_fine, minlength=parent.max())


def top_cells(region, cell_size, density, n=5):
    """The n cells with the highest relative density, with their centres."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    n_lon = len(lon_edges) - 1
    order = np.argsort(-np.nan_to_num(density, nan=-1))[:n]
    return [{"grid": f"{cell_size:g} deg", "rank": r + 1,
             "lat_centre": float(lat_edges[i // n_lon] + cell_size / 2),
             "lon_centre": float(lon_edges[i % n_lon] + cell_size / 2),
             "relative_density": round(float(density[i]), 2)} for r, i in enumerate(order)]


def plot(region, panels, out_path):
    """Relative QSD density on each grid, one shared log colour scale."""
    vals = np.concatenate([d[np.isfinite(d) & (d > 0)] for _, d, _ in panels])
    norm = colors.LogNorm(vmin=max(vals.min(), 1e-2), vmax=vals.max())  # 0.01 and below share the darkest colour
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 4.6), layout="constrained")
    for ax, (cell_size, density, summary) in zip(axes, panels):
        lat_edges, lon_edges = grid_edges(region, cell_size)
        shape = (len(lat_edges) - 1, len(lon_edges) - 1)
        grid = np.ma.masked_invalid(np.where(density > 0, density, np.nan).reshape(shape))
        mesh = ax.pcolormesh(lon_edges, lat_edges, grid, norm=norm, cmap="viridis")
        draw_land(ax)
        ax.plot([region["lon_min"], region["lon_max"], region["lon_max"], region["lon_min"], region["lon_min"]],
                [region["lat_min"], region["lat_min"], region["lat_max"], region["lat_max"], region["lat_min"]],
                color="black", lw=1)
        ax.set_xlim(region["lon_min"] - 3, region["lon_max"] + 3)
        ax.set_ylim(region["lat_min"] - 3, region["lat_max"] + 3)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"{cell_size:g}° cells ({summary['estimator']}): half-life in R "
                     f"{summary['half_life_years']:.1f} years, mixing {summary['mixing_median_years']:.1f} years "
                     f"(median start)", fontsize=10)
    fig.colorbar(mesh, ax=axes, label="Relative density of material still in R (1 = box average, log scale)",
                 shrink=0.85, extend="min")
    fig.suptitle(f"Where material that stays in the box ends up: quasi-stationary distribution, {region['label']}")
    fig.text(0.01, -0.1,
             f"Leading left eigenvector of P restricted to cells inside R (tau = {TAU_DAYS} days), divided by each "
             "cell's area inside the box. Half-life: time for half of this settled material to leave R. Mixing: steps "
             f"until material released in one cell is within total variation {TV_CUTOFF} of this pattern.\n"
             "Blank cells have no drifters. 'Left R' is absorbing, so drifters that come back are not followed. "
             "No uncertainty yet (drifter bootstrap).", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    naive = np.load(os.path.join(PROCESSED_DIR, f"naive_P_{coarse:g}deg_tau{TAU_DAYS}d.npz"))
    shrunk = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    parent = parent_states(region, fine, coarse)

    rows, panels, densities, pis = [], [], {}, {}
    for cell_size, estimator, stored in ((coarse, "naive", naive), (fine, "shrinkage", shrunk)):
        drifters = stored["start_drifters"]
        summary, density, pi = long_run_summary(stored["P"], drifters > 0, drifters >= N_MIN,
                                                cell_areas_km2(region, cell_size))
        summary = {"grid": f"{cell_size:g} deg", "estimator": estimator,
                   "alpha": float(stored["alpha"]) if "alpha" in stored.files else "", **summary}
        rows.append(summary)
        panels.append((cell_size, density, summary))
        densities[cell_size], pis[cell_size] = density, pi

    # Same answer on both grids? Add the 2 deg QSD up into 4 deg cells and compare
    pi_fine_on_coarse = coarsen(pis[fine], parent)
    tv_grids = 0.5 * np.abs(pi_fine_on_coarse - pis[coarse]).sum()
    rows[1]["tv_to_4deg_qsd"] = float(tv_grids)

    # Prior sensitivity: refit the 2 deg model on all drifters at other alphas
    counts, alpha = shrunk["counts"], float(shrunk["alpha"])
    prior = prior_rows(counts, parent)
    drifters = shrunk["start_drifters"]
    for factor in ALPHA_FACTORS:
        if factor == 1.0:
            continue
        P = shrinkage_P(counts, prior, factor * alpha)
        summary, _, pi = long_run_summary(P, drifters > 0, drifters >= N_MIN, cell_areas_km2(region, fine))
        rows.append({"grid": f"{fine:g} deg", "estimator": f"shrinkage, alpha x {factor:g}",
                     "alpha": factor * alpha, **summary,
                     "tv_to_main_qsd": float(0.5 * np.abs(pi - pis[fine]).sum())})
    for row in rows:
        print(row)

    tops = top_cells(region, coarse, densities[coarse]) + top_cells(region, fine, densities[fine])
    for row in tops:
        print(row)

    write_csv(os.path.join(RESULTS_DIR, f"long_run_tau{TAU_DAYS}d.csv"), rows)
    write_csv(os.path.join(RESULTS_DIR, f"long_run_top_cells_tau{TAU_DAYS}d.csv"), tops)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "quasi_stationary_map.png")
    plot(region, panels, fig_path)
    print(f"Saved the two long_run CSVs in {RESULTS_DIR} and {fig_path}")
