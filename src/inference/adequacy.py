import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.ingest.load import load_gdp
from src.gridding.coverage import load_region_bounds, region_mask_and_owners

# Define output path structure matching your coverage script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(REPO_ROOT, "outputs", "figures")

def get_grid_edges(region):
    """Generates lat/lon grid edges safely handling the international date line."""
    crossed_dateline = region["lon_min"] > region["lon_max"]
    lon_min = region["lon_min"]
    lon_max = region["lon_max"] + 360 if crossed_dateline else region["lon_max"]

    step = region["cell_size_deg"]
    lat_edges = np.arange(region["lat_min"], region["lat_max"] + step, step)
    lon_edges = np.arange(lon_min, lon_max + step, step)
    return lat_edges, lon_edges

def cell_index(lon, lat, region):
    # Normalise lon (same way as src.gridding.coverage.in_box)
    lon_norm = ((lon + 180) % 360) - 180

    crossed_dateline = region["lon_min"] > region["lon_max"]
    if crossed_dateline:
        lon_norm = np.where(lon_norm < 0, lon_norm + 360, lon_norm)

    # Use the edge helper function
    lat_edges, lon_edges = get_grid_edges(region)

    # Digitize, shift to 0-based, handle points exactly on lat_max/lon_max
    lat_idx = np.digitize(lat, lat_edges) - 1
    lon_idx = np.digitize(lon_norm, lon_edges) - 1
    lat_idx = np.clip(lat_idx, 0, len(lat_edges) - 2)
    lon_idx = np.clip(lon_idx, 0, len(lon_edges) - 2)

    # Combine (lat_i, lon_i) into one flat cell id
    n_lon_cells = len(lon_edges) - 1
    n_lat_cells = len(lat_edges) - 1
    cell_id = lat_idx * n_lon_cells + lon_idx

    return cell_id, n_lat_cells, n_lon_cells

def per_cell_counts(cell, owner, n_cells):
    df = pd.DataFrame({
        "cell": cell,
        "owner": owner
    })

    # Observations per cell
    obs_counts = df.groupby("cell").size()
    obs_counts = obs_counts.reindex(range(n_cells), fill_value=0)

    # Distinct drifters per cell
    drifter_counts = df.groupby("cell")["owner"].nunique()
    drifter_counts = drifter_counts.reindex(range(n_cells), fill_value=0)

    return obs_counts.values, drifter_counts.values

if __name__ == "__main__":
    region = load_region_bounds()
    ds = load_gdp()
    idx_true, owner = region_mask_and_owners(ds, region)

    # Pull lon/lat for idx_true only
    lon = ds.lon[idx_true]
    lat = ds.lat[idx_true]

    # Call the two functions, print the checks
    cell, n_lat_cells, n_lon_cells = cell_index(lon, lat, region)
    n_cells = n_lat_cells * n_lon_cells
    obs_counts, drifter_counts = per_cell_counts(cell, owner, n_cells)

    print(f"Number of cells in grid: {n_cells}")
    print(f"Sum of observations per cell: {obs_counts.sum()}")
    print(f"Largest distinct drifter count: {drifter_counts.max()}")
    print(f"Sum of distinct drifters per cell: {drifter_counts.sum()}")
    print(f"Number of cells with zero observations: {np.sum(obs_counts == 0)}")

    # Standard error for a transition probability:
    # Solved sqrt(0.5*(1-0.5)/n) <= 0.05 -> 0.25/n <= 0.0025 -> n >= 100
    N = 100

    # Assume cells with no observations are likely land
    ocean_drifter_counts = drifter_counts[obs_counts > 0]

    # Calculate the percentage of ocean cells meeting our safety threshold
    pass_rate = np.mean(ocean_drifter_counts >= N) * 100

    # Calculate distribution metrics across the ocean cells
    median_drifters = np.median(ocean_drifter_counts)
    p10_drifters = np.percentile(ocean_drifter_counts, 10)

    print(f"Percentage of ocean cells with >= {N} drifters: {pass_rate:.1f}%")
    print(f"Median drifters per ocean cell: {median_drifters:.0f}")
    print(f"10th percentile of drifters per ocean cell: {p10_drifters:.0f}")

    # Reshape 1D arrays back into the 2D spatial grid map
    drifter_grid = drifter_counts.reshape(n_lat_cells, n_lon_cells)

    # Mask out the 0 cells because log scales cannot process them (or land)
    masked_grid = np.ma.masked_equal(drifter_grid, 0)

    # Pull the edges into the main function scope so the plotting block can use them
    lat_edges, lon_edges = get_grid_edges(region)

    # Plot drifter concentration map with logarithmic scale and contour line for threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        lon_edges, 
        lat_edges, 
        masked_grid, 
        norm=colors.LogNorm(vmin=1, vmax=masked_grid.max()), 
        cmap="viridis",
        shading="auto"
    )

    # Add a boundary line showing where cells pass/fail the safety threshold
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    contour = ax.contour(
        lon_centers, 
        lat_centers, 
        drifter_grid >= N, 
        levels=[0.5], 
        colors="red", 
        linewidths=2
    )

    # Visual decorations and labeling
    cbar = fig.colorbar(mesh, ax=ax, label="Distinct Drifters (Log Scale)")
    ax.set_title(f"Data Adequacy Map (Red Line outlines cells with >= {N} Drifters)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Create a manual legend marker for our red contour lines
    custom_line = [plt.Line2D([0], [0], color="red", lw=2)]
    ax.legend(custom_line, [f"Adequate Data Threshold (N >= {N})"], loc="upper right")

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "adequacy_map.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Data adequacy map successfully generated and saved to {out_path}")
