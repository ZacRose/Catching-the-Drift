"""
coverage.py

Related tasks:
- Week 1 Computational - count real observations and distinct drifters in region R.
- Week 2 Computational - run the pipeline end-to-end and produce a coverage map
  (where and when drifters were present in R).

Run: python3 -m src.gridding.coverage
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, load_region_bounds
from src.ingest.load import load_region
from src.gridding.grid import grid_edges, cell_index


def coverage_map(data, region, out_path):
    """Save a two-panel coverage figure for region R.

    Inputs:
        data: region readings from load_region().
        region: dict from config/region.yaml (the working grid is used for the map).
        out_path: where to save the PNG.
    Left: observations per cell (log colour scale, since a few cells hold far more than the rest).
    Right: how many distinct drifters were active in R each month.
    """
    cell, n_lat_cells, n_lon_cells = cell_index(data["lon"], data["lat"], region)
    obs_grid = np.bincount(cell, minlength=n_lat_cells * n_lon_cells).reshape(n_lat_cells, n_lon_cells)
    lat_edges, lon_edges = grid_edges(region)

    months = data["time"].astype("datetime64[M]")
    month_drifter_pairs = np.unique(np.column_stack([months.astype(np.int64), data["drifter_id"]]), axis=0)
    month_ids, drifters_per_month = np.unique(month_drifter_pairs[:, 0], return_counts=True)

    fig, (ax_where, ax_when) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1]})

    mesh = ax_where.pcolormesh(
        lon_edges, lat_edges, np.ma.masked_equal(obs_grid, 0),
        norm=colors.LogNorm(), cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax_where, label="Hourly observations per cell (log scale)")
    ax_where.set_aspect("equal")
    ax_where.set_title(f"Where: observations per {region['cell_size_deg']:g}° cell")
    ax_where.set_xlabel("Longitude (°E)")
    ax_where.set_ylabel("Latitude (°N)")

    ax_when.plot(month_ids.astype("datetime64[M]"), drifters_per_month, lw=1)
    ax_when.set_title("When: drifters active in the region each month")
    ax_when.set_ylabel("Distinct drifters")
    ax_when.set_xlabel("Year")

    fig.suptitle(f"GDP hourly coverage, {region['label']}")
    fig.text(0.01, 0.01, "Blank cells have no observations (land, or never sampled).", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    data = load_region(region)

    # Week 1 deliverable: observation/drifter counts for the README
    print(f"Region: {region['name']}")
    print(f"Observations in R: {data['lon'].size:,}")
    print(f"Distinct drifters in R: {np.unique(data['drifter_id']).size:,}")
    first, last = data["time"].min(), data["time"].max()
    print(f"Time span: {first.astype('datetime64[D]')} to {last.astype('datetime64[D]')}")

    # Week 2 deliverable: where and when drifters were present
    out_path = os.path.join(FIGURES_DIR, "coverage_map.png")
    coverage_map(data, region, out_path)
    print(f"Saved coverage map to {out_path}")
