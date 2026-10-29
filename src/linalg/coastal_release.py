"""
coastal_release.py

Related task: Week 6 Computational - redraw P, P^6 and P^24 from the Benguela coastal cells (Week 4
open issue a), with bootstrap intervals (objective O2: where material from the Benguela coast is
after 15 days, about 3 months and about a year, and the chance it has left the box or is in the
coastal zone).

Release: material spread evenly over the coastal cells that have drifters (src/gridding/coast.py;
9 cells at 2 deg, 5 at 4 deg). Coastal cells with no drifters can't be release points (no data for
their row), but they count as coastal zone when material arrives there. Two matrices:
- 4 deg naive P, the proposal's baseline (Figure 1). A 4 deg coastal cell reaches ~400 km offshore.
- 2 deg shrinkage P, the product (shrinkage_fit.py). Its rows for cells with no drifters are the
  4 deg prior alone; the share of material in such cells is reported, so it isn't read as data.
Each step multiplies by P (release_map.propagate). "Left R" is absorbing (decision 11), so its
entry is the chance the material has left the box by then. Mass sent into a 4 deg cell with no row
is lost (issue b) and reported; the shrinkage P has a row for every cell, so it loses none.

Intervals: N_BOOT replicates of all drifters (bootstrap.py); each refits P exactly as the original
(naive, or shrinkage with the prior rebuilt and alpha fixed) and is propagated from the same
release. Each number gets the 2.5th to 97.5th percentile over replicates.

The maps show probability per 2 x 2 degree area (4 deg values divided by 4), so the two grids share
one colour scale. Cell areas also shrink with latitude (decision 1); this is not corrected here.

Outputs: outputs/results/coastal_release_tau15d.csv and outputs/figures/coastal_release_tau15d.png.

Run: python3 -m src.linalg.coastal_release   (about a minute; needs shrinkage_P_2deg_tau15d.npz)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_paths, land_fraction
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.trajectory_map import draw_land, _degrees
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS
from src.inference.bootstrap import LEVEL, N_BOOT, percentile_interval, replicate_matrices, shrinkage_fit
from src.inference.estimators import count_matrix, naive_P
from src.inference.transition_data import load_transitions, parent_states, to_states
from src.linalg.release_map import STEPS, centre_of_mass, propagate

WHEN = {1: f"{TAU_DAYS} days", 6: "3 months", 24: "1 year"}
QUANTITIES = ["pct_left_R", "pct_in_coastal_zone", "pct_in_cells_without_drifters", "pct_lost_to_rowless_cells",
              "centre_lat", "centre_lon"]


def release_summary(dist, zone, no_data, region, cell_size):
    """The QUANTITIES for one distribution over states (last entry = left R)."""
    lat, lon = centre_of_mass(dist, region, cell_size)
    lost = 1 - dist.sum()
    lost = 0.0 if abs(lost) < 1e-12 else lost  # rows of P sum to 1 only to rounding error
    return {"pct_left_R": 100 * dist[-1], "pct_in_coastal_zone": 100 * dist[zone].sum(),
            "pct_in_cells_without_drifters": 100 * dist[:-1][no_data].sum(),
            "pct_lost_to_rowless_cells": 100 * lost, "centre_lat": lat, "centre_lon": lon}


def run_model(start, end, drifter_id, n_states, fit, release, zone, no_data, region, cell_size, n_boot=N_BOOT):
    """Distributions from the full-data P, and every quantity's estimate and replicates.

    Returns:
        (dists {step: array}, estimates {step: {quantity: value}}, replicates {step: {quantity: array}}).
    """
    P = fit(count_matrix(start, end, n_states))
    dists = propagate(P, release)
    estimates = {s: release_summary(d, zone, no_data, region, cell_size) for s, d in dists.items()}
    replicates = {s: {q: np.empty(n_boot) for q in QUANTITIES} for s in STEPS}
    for b, P_b in enumerate(replicate_matrices(start, end, drifter_id, n_states, fit, n_boot)):
        for s, d in propagate(P_b, release).items():
            for q, v in release_summary(d, zone, no_data, region, cell_size).items():
                replicates[s][q][b] = v
    return dists, estimates, replicates


def _pct(v):
    """1 decimal place from 1% up, 2 significant figures below (so 0.0057% doesn't print as 0.0)."""
    return f"{v:.1f}" if v >= 1 else f"{v:.2g}"


def plot(region, panels, out_path):
    """Rows: models; columns: STEPS. Probability per 2 x 2 degree area, coastal cells outlined."""
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(len(panels), len(STEPS), figsize=(13, 2.1 * len(panels) + 0.4),
                             layout="constrained", squeeze=False)
    norm = colors.LogNorm(vmin=1e-4, vmax=0.1)
    for row, (title, cell_size, dists, zone, sampled, stats) in zip(axes, panels):
        lat_edges, lon_edges = grid_edges(region, cell_size)
        shape = (len(lat_edges) - 1, len(lon_edges) - 1)
        for ax, step in zip(row, STEPS):
            per_area = dists[step][:-1].reshape(shape) / (cell_size / 2) ** 2
            shown = np.ma.masked_where(~sampled.reshape(shape), np.maximum(per_area, norm.vmin))
            mesh = ax.pcolormesh(lon_edges, lat_edges, shown, cmap="Blues", norm=norm)
            draw_land(ax)
            n_lon = shape[1]
            for c in zone:
                r, k = divmod(int(c), n_lon)
                ax.add_patch(plt.Rectangle((lon_edges[k], lat_edges[r]), cell_size, cell_size, fill=False,
                                           ec="#d62728", lw=0.9, zorder=5))
            ax.set_xlim(region["lon_min"], region["lon_max"])
            ax.set_ylim(region["lat_min"], region["lat_max"])
            ax.set_aspect("equal")
            ax.set_facecolor("#d9d9d9")
            ax.xaxis.set_major_formatter(_degrees("lon"))
            ax.yaxis.set_major_formatter(_degrees("lat"))
            ax.set_title(f"{title}: after {WHEN[step]} ($P^{{{step}}}$)", fontsize=8.5, loc="left")
            left, coast = stats[step]["pct_left_R"], stats[step]["pct_in_coastal_zone"]
            ax.text(region["lon_min"] + 1, region["lat_min"] + 1,
                    f"left box: {_pct(left[0])}% ({_pct(left[1])}-{_pct(left[2])})\n"
                    f"in coastal zone: {_pct(coast[0])}% ({_pct(coast[1])}-{_pct(coast[2])})",
                    fontsize=7, va="bottom", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.colorbar(mesh, ax=axes, shrink=0.6, extend="both", label="probability per 2° x 2° area (log scale)")
    fig.text(0.01, -0.04,
             "Material released evenly over the Benguela coastal cells that have drifters (34S-18S, cells the coastline "
             "passes through; red outlines, including cells with no drifters).\nTop: naive 4° matrix (proposal baseline). "
             "Bottom: 2° shrinkage matrix (the product). Grey: land, and the few ocean cells with no drifters. Brackets: 95% interval from "
             f"{N_BOOT} bootstrap replicates of the drifters.", fontsize=8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region)
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    alpha = float(product["alpha"])
    paths = land_paths()

    models = [(f"{coarse:g}° naive", coarse, naive_P),
              (f"{fine:g}° shrinkage", fine, shrinkage_fit(parent_states(region, fine, coarse), alpha))]
    rows, panels = [], []
    for title, size, fit in models:
        start, end, n_cells = to_states(table, region, size)
        _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)
        zone = coastal_cells(region, size, land_fraction(region, size, paths))
        release = zone[drifters[zone] > 0]
        if size == fine and not np.allclose(fit(count_matrix(start, end, n_cells + 1)), product["P"]):
            raise ValueError("Refitted shrinkage P differs from the saved product; rerun src.inference.shrinkage_fit.")
        print(f"{title}: release spread over {release.size} coastal cells {release.tolist()}")
        dists, estimates, replicates = run_model(start, end, table["drifter_id"], n_cells + 1, fit, release,
                                                 zone, drifters == 0, region, size)
        stats = {}
        for step in STEPS:
            stats[step] = {}
            for q in QUANTITIES:
                low, high = percentile_interval(replicates[step][q], LEVEL)
                stats[step][q] = (estimates[step][q], float(low), float(high))
                rows.append({"model": title.replace("°", " deg"), "release_cells": release.size, "steps": step,
                             "days": step * TAU_DAYS, "quantity": q, "estimate": round(float(estimates[step][q]), 4),
                             "ci_low": round(float(low), 4), "ci_high": round(float(high), 4)})
            e = estimates[step]
            print(f"  after {step * TAU_DAYS} days: left R {e['pct_left_R']:.2f}%, in coastal zone "
                  f"{e['pct_in_coastal_zone']:.2f}%, in cells without drifters {e['pct_in_cells_without_drifters']:.3f}%, "
                  f"lost {e['pct_lost_to_rowless_cells']:.3f}%, centre {e['centre_lat']:.1f} lat {e['centre_lon']:.1f} lon")
        panels.append((title, size, dists, zone, drifters > 0, stats))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, f"coastal_release_tau{TAU_DAYS}d.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, f"coastal_release_tau{TAU_DAYS}d.png")
    plot(region, panels, fig_path)
    print(f"Saved {table_path} and {fig_path}")
