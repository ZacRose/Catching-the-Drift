"""
coverage.py

Related tasks: 
- Week 1 Computational- count real observations and distinct drifters in region R.
- Week 2 Computational- run the pipeline end-to-end on a subsample of R's data and produce
a coverage map (where and when drifters were present), without crashing.
Region bounds from config/region.yaml (not hard-coded).

Quantitative EDA and README data-adequacy both use results from this script.
First-pass MVP
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.ingest.load import load_gdp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGION_CONFIG = os.path.join(REPO_ROOT, "config", "region.yaml")
FIGURES_DIR = os.path.join(REPO_ROOT, "outputs", "figures")

def load_region_bounds(yaml_path=REGION_CONFIG):
    """Load region R's box (lat/lon limits, grid cell size) from config/region.yaml."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def in_box(lon, lat, region):
    """True/False for every reading: is it inside region R's box?
    lon_norm maps longitude onto the -180..180 range so a drifter's
    reported longitude is always clearly comparable to the box. 
    The if/else logic handles a region that crosses the INTL dateline
    """
    lon_norm = ((lon + 180) % 360) - 180
    lon_min, lon_max = region["lon_min"], region["lon_max"]
    if lon_min <= lon_max:
        lon_ok = (lon_norm >= lon_min) & (lon_norm <= lon_max)
    else:
        lon_ok = (lon_norm >= lon_min) | (lon_norm <= lon_max)
    return lon_ok & (lat >= region["lat_min"]) & (lat <= region["lat_max"])


def region_mask_and_owners(ds, region):
    """Which readings fall inside region R and which drifters do they belong to?
    Every drifter's readings are stored one after another in one long list, and
    "rowsize" indicates how many readings each drifter contributed. 
    cum_rowsize with searchsorted: turns that into a mapping from each in-region
    reading to its drifter's index.
    """
    lon = ds["lon"].values
    lat = ds["lat"].values
    rowsize = ds["rowsize"].values

    mask = in_box(lon, lat, region)
    idx_true = np.flatnonzero(mask)
    cum_rowsize = np.cumsum(rowsize)
    owner = np.searchsorted(cum_rowsize, idx_true, side="right")
    return idx_true, owner


def subsample_by_drifter(idx_true, owner, n_drifters_sample=200, seed=0):
    """Randomly keep 200 whole drifters' in-region readings.
    Subsample by drifter (not reading) so each kept drifter's trajectory
    stays intact and continuous.
    """
    rng = np.random.default_rng(seed)
    unique_drifters = np.unique(owner)
    sample_drifters = rng.choice(
        unique_drifters, size=min(n_drifters_sample, unique_drifters.size), replace=False
    )
    keep = np.isin(owner, sample_drifters)
    return idx_true[keep], sample_drifters.size


def coverage_map(ds, region, sample_idx, out_path):
    """Build the coverage map for the subsample.
    Spatial map: a 2D grid of how many readings landed in each spatial cell of region R.
    Time series column graph: how many readings occurred per calendar month across time span. 
    """
    lon = ds["lon"].values[sample_idx]
    lat = ds["lat"].values[sample_idx]
    time = ds["time"].values[sample_idx]

    step = region["cell_size_deg"]
    lat_edges = np.arange(region["lat_min"], region["lat_max"] + step, step)
    lon_edges = np.arange(region["lon_min"], region["lon_max"] + step, step)

    months = time.astype("datetime64[M]")
    unique_months, month_counts = np.unique(months, return_counts=True)

    fig, (ax_where, ax_when) = plt.subplots(1, 2, figsize=(13,5))

    # counts.T prevents map being sideways
    counts, _, _ = np.histogram2d(lon, lat, bins=[lon_edges, lat_edges])
    mesh = ax_where.pcolormesh(lon_edges, lat_edges, counts.T, shading="auto")
    fig.colorbar(mesh, ax=ax_where, label="Observations per cell")
    ax_where.set_title("Spatial coverage in subsample")
    ax_where.set_xlabel("Longitude")
    ax_where.set_ylabel("Latitude")

    ax_when.bar(unique_months.astype(str), month_counts, width=1.0)
    ax_when.set_title("Monthly coverage in subsample")
    ax_when.set_ylabel("Observations")
    tick_step = max(len(unique_months) // 12, 1)
    ax_when.set_xticks(range(0, len(unique_months), tick_step))
    ax_when.set_xticklabels(unique_months.astype(str)[::tick_step], rotation=90)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    # Run "python3 -m src.gridding.coverage" to count observations and drifters,
    # and create coverage map
    region = load_region_bounds()
    print(f"Region: {region['name']}")

    print("Loading GDP dataset...")
    ds = load_gdp()  # Week 1: load raw dataset

    # Week 1 deliverable: observation/drifter counts for region for README
    idx_true, owner = region_mask_and_owners(ds, region)
    n_obs = int(idx_true.size)
    n_drifters = int(np.unique(owner).size)
    print(f"Observations in R: {n_obs:,}")
    print(f"Distinct drifters in R: {n_drifters:,}")

    # Week 2 deliverable: coverage map and monthly observation counts
    sample_idx, n_sample_drifters = subsample_by_drifter(idx_true, owner)
    out_path = os.path.join(FIGURES_DIR, "coverage_map.png")
    coverage_map(ds, region, sample_idx, out_path)
    print(f"Coverage map built from {n_sample_drifters} drifters ({len(sample_idx):,} observations)")
    print(f"Saved to {out_path}")