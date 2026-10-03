"""
adequacy.py

Related task: Week 1 Quantitative - is the observation count actually enough given how
it's distributed across the box, not just the raw total?

A transition probability p out of a cell estimated from n independent drifters has
standard error sqrt(p(1-p)/n), which is largest at p = 0.5. Requiring SE <= 0.05
in that worst case gives 0.25/n <= 0.0025, i.e. n >= 100 (a 95% interval of about +/-0.10).
The stricter target of a 95% interval of +/-0.05 (n >= 1.96^2 * 0.25 / 0.05^2 = 385) was
considered and rejected; see project_plan.txt, decision 3.

Distinct drifters stand in for independent samples: hourly readings from the same
drifter are strongly correlated, so raw observation counts overstate the information.
The per-transition count at a chosen lag replaces this proxy in Week 3.

Each cell is judged on its own drifters only, i.e. the adequacy of the naive
(unregularised) estimate. Borrowing strength from neighbouring cells is the job of the
Week 6 shrinkage model, which reports its own effective sample size per cell.

Runs the check at 1 to 5 degrees so the working grid (2) and coarse baseline (4) rest on evidence.
Run: python3 -m src.inference.adequacy
"""
import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, RESULTS_DIR, load_region_bounds
from src.ingest.load import load_region
from src.gridding.grid import grid_edges, cell_index, per_cell_counts

N_MIN = 100
CELL_SIZES = [1.0, 2.0, 3.0, 4.0, 5.0]


def drifter_grid(data, region, cell_size):
    """Distinct drifters per cell as a 2-D (lat, lon) array, plus observations per cell (flat).

    Inputs:
        data: region readings from load_region().
        region: dict from config/region.yaml.
        cell_size: cell width in degrees.
    Returns:
        (drifter_counts_2d, obs_counts_flat).
    """
    cell, n_lat_cells, n_lon_cells = cell_index(data["lon"], data["lat"], region, cell_size)
    obs_counts, drifter_counts = per_cell_counts(cell, data["drifter_id"], n_lat_cells * n_lon_cells)
    return drifter_counts.reshape(n_lat_cells, n_lon_cells), obs_counts


def adequacy_summary(drifter_counts, obs_counts, cell_size):
    """One row of the resolution comparison table.

    Cells with zero observations are treated as land. In this box they line up with the
    African and South American coasts, so few if any are unsampled ocean.
    """
    ocean = drifter_counts[obs_counts > 0]
    return {
        "cell_size_deg": cell_size,
        "ocean_cells": int(ocean.size),
        "pct_cells_adequate": round(float(100 * np.mean(ocean >= N_MIN)), 1),
        "median_drifters": int(np.median(ocean)),
        "p10_drifters": int(np.percentile(ocean, 10)),
    }


def plot_adequacy(grids, summaries, region, out_path):
    """One map per resolution, two per row, on a shared log colour scale so they compare directly."""
    vmax = max(g.max() for g in grids)
    n_rows = math.ceil(len(grids) / 2)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3.4 * n_rows), layout="constrained")
    axes = axes.ravel()[: len(grids)]
    for spare in fig.axes[len(grids):]:
        spare.axis("off")

    for ax, grid, row in zip(axes, grids, summaries):
        lat_edges, lon_edges = grid_edges(region, row["cell_size_deg"])
        mesh = ax.pcolormesh(
            lon_edges, lat_edges, np.ma.masked_equal(grid, 0),
            norm=colors.LogNorm(vmin=1, vmax=vmax), cmap="viridis",
        )
        # Contour on cell centres traces the boundary between adequate and thin cells
        lon_centres = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_centres = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        ax.contour(lon_centres, lat_centres, grid >= N_MIN, levels=[0.5], colors="black", linewidths=1.5)
        ax.set_aspect("equal")
        ax.set_title(
            f"{row['cell_size_deg']:g}° cells: {row['pct_cells_adequate']}% of ocean cells adequate "
            f"(median {row['median_drifters']} drifters)"
        )
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    fig.colorbar(mesh, ax=axes, label="Distinct drifters per cell (log scale)", shrink=0.6)
    fig.suptitle(f"Data adequacy by grid resolution, {region['label']}")
    fig.text(
        0.02, -0.04,
        f"Black line encloses cells with at least {N_MIN} distinct drifters (worst-case SE of a "
        "transition probability <= 0.05).\nBlank cells have no observations (land).",
        fontsize=9,
    )
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    data = load_region(region)

    grids, summaries = [], []
    for cell_size in CELL_SIZES:
        grid, obs_counts = drifter_grid(data, region, cell_size)
        grids.append(grid)
        summaries.append(adequacy_summary(grid.ravel(), obs_counts, cell_size))
        print(summaries[-1])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, "adequacy_by_resolution.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    fig_path = os.path.join(FIGURES_DIR, "adequacy_map.png")
    plot_adequacy(grids, summaries, region, fig_path)
    print(f"Saved {table_path} and {fig_path}")
