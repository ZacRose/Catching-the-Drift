"""
sparsity_eda.py

Related task: Week 2 Quantitative - distribution of observations across cells, and an
early look at where sparsity is going to bite. (Where the thin cells are is mapped by
adequacy.py; this script covers how the data are distributed across them.)

Runs on the working grid in config/region.yaml.
Run: python3 -m src.inference.sparsity_eda
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, load_region_bounds
from src.ingest.load import load_region
from src.gridding.grid import cell_index, per_cell_counts
from src.inference.adequacy import N_MIN


def distribution_summary(counts, label):
    """Print spread and concentration for one per-cell count array.

    "Busiest 10%" = share of the total held by the top 10% of cells. A high share means
    the data sit in a small number of well-sampled cells rather than spread across the box.
    """
    top_n = max(1, int(np.ceil(0.1 * counts.size)))
    concentration = np.sort(counts)[::-1][:top_n].sum() / counts.sum()
    print(
        f"{label}: mean={counts.mean():,.0f}, median={np.median(counts):,.0f}, "
        f"min={counts.min():,}, max={counts.max():,}, busiest 10% of cells hold {concentration:.0%} of the total"
    )


def plot_count_distributions(obs_counts, drifter_counts, cell_size, out_path):
    """Histograms of observations and distinct drifters per ocean cell.

    Observations use a log x-axis because they are heavily right-skewed. The drifter
    histogram marks the adequacy threshold so the share of thin cells is visible.
    """
    fig, (ax_obs, ax_drift) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_obs.hist(obs_counts, bins=np.logspace(0, np.log10(obs_counts.max()), 40))
    ax_obs.set_xscale("log")
    ax_obs.set_title("Hourly observations per ocean cell")
    ax_obs.set_xlabel("Observations (log scale)")
    ax_obs.set_ylabel("Number of cells")

    ax_drift.hist(drifter_counts, bins=40)
    ax_drift.axvline(N_MIN, color="black", ls="--", label=f"Adequacy threshold ({N_MIN})")
    ax_drift.set_title("Distinct drifters per ocean cell")
    ax_drift.set_xlabel("Drifters")
    ax_drift.legend()

    fig.suptitle(f"How the data are spread across {cell_size:g}° cells")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    data = load_region(region)

    cell, n_lat_cells, n_lon_cells = cell_index(data["lon"], data["lat"], region)
    obs_counts, drifter_counts = per_cell_counts(cell, data["drifter_id"], n_lat_cells * n_lon_cells)
    ocean = obs_counts > 0
    obs_counts, drifter_counts = obs_counts[ocean], drifter_counts[ocean]

    print(f"Ocean cells: {ocean.sum()} of {ocean.size} ({region['cell_size_deg']:g}° grid)")
    distribution_summary(obs_counts, "Observations per ocean cell")
    distribution_summary(drifter_counts, "Drifters per ocean cell")

    # Hourly readings from one drifter are not independent, so judging adequacy by raw
    # observations would badly overstate how many cells can be trusted
    print(
        f"Cells with >= {N_MIN} observations: {np.mean(obs_counts >= N_MIN):.0%}; "
        f"with >= {N_MIN} distinct drifters: {np.mean(drifter_counts >= N_MIN):.0%}"
    )
    # Readings are hourly, so observations / drifters = hours each drifter spends in the cell
    hours_per_drifter = obs_counts / drifter_counts
    print(f"Hours a drifter spends in a cell: median over cells {np.median(hours_per_drifter):,.0f} "
          f"(~{np.median(hours_per_drifter) / 24:.0f} days)")

    out_path = os.path.join(FIGURES_DIR, "sparsity_distributions.png")
    plot_count_distributions(obs_counts, drifter_counts, region["cell_size_deg"], out_path)
    print(f"Saved {out_path}")
