"""
data_quality.py

Related task: Week 2 Quantitative (EDA) - the data-quality checks that decide how the
transition matrix gets built, rather than description for its own sake:
1. Missing values, including how far hourly positions were interpolated between
   satellite fixes ("gap" = time between the fixes either side of each reading).
2. Drogue status - a drifter that loses its drogue (sea anchor) is pushed by wind as
   well as current. How much of the data is undrogued, and what would dropping it cost?
3. Speed outliers - are the fastest readings errors, or real fast currents?
4. Speed vs cell size - how long a typical drifter takes to cross a cell, which sets
   the smallest useful lag for the Week 3 transition counts.

Run: python3 -m src.inference.data_quality
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, load_region_bounds
from src.ingest.load import load_region
from src.gridding.grid import cell_index, per_cell_counts
from src.inference.adequacy import CELL_SIZES, N_MIN

KM_PER_DEG = 111.2  # one degree of latitude; longitude degrees are shorter away from the equator
OUTLIER_QUANTILE = 0.999


def missing_summary(data):
    """Print the percentage of missing (NaN/NaT) values for each float or time variable."""
    for name, values in data.items():
        if values.ndim == 0 or values.dtype == bool or np.issubdtype(values.dtype, np.integer):
            continue  # stored box bounds, flags and IDs can't be missing
        missing = np.isnat(values) if np.issubdtype(values.dtype, np.datetime64) else np.isnan(values)
        print(f"  {name}: {missing.mean():.3%} missing")


def drogue_adequacy(data, region, cell_size):
    """Share of ocean cells with at least N_MIN distinct drifters, using all readings,
    drogued readings only, and undrogued readings only. Returns the three shares (0 to 1)."""
    cell, n_lat_cells, n_lon_cells = cell_index(data["lon"], data["lat"], region, cell_size)
    n_cells = n_lat_cells * n_lon_cells
    drogued = data["drogue_status"]
    obs_all, drifters_all = per_cell_counts(cell, data["drifter_id"], n_cells)
    _, drifters_drogued = per_cell_counts(cell[drogued], data["drifter_id"][drogued], n_cells)
    _, drifters_undrogued = per_cell_counts(cell[~drogued], data["drifter_id"][~drogued], n_cells)
    ocean = obs_all > 0
    return (np.mean(drifters_all[ocean] >= N_MIN), np.mean(drifters_drogued[ocean] >= N_MIN),
            np.mean(drifters_undrogued[ocean] >= N_MIN))


def plot_quality(data, speed, region, out_path):
    """Speed distribution by drogue status, and where the fastest readings occur."""
    drogued = data["drogue_status"]
    threshold = np.nanquantile(speed, OUTLIER_QUANTILE)
    fast = speed > threshold
    fig, (ax_speed, ax_where) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})

    bins = np.linspace(0, np.nanmax(speed), 80)
    ax_speed.hist(speed[drogued], bins=bins, density=True, histtype="step", lw=1.5, label="Drogued")
    ax_speed.hist(speed[~drogued], bins=bins, density=True, histtype="step", lw=1.5, ls="--", label="Drogue lost")
    ax_speed.set_yscale("log")
    ax_speed.set_title("Current speed by drogue status")
    ax_speed.set_xlabel("Speed (m/s)")
    ax_speed.set_ylabel("Density (log scale)")
    ax_speed.legend()

    ax_where.scatter(data["lon"][fast], data["lat"][fast], s=1, color="black")
    ax_where.set_xlim(region["lon_min"], region["lon_max"])
    ax_where.set_ylim(region["lat_min"], region["lat_max"])
    ax_where.set_aspect("equal")
    ax_where.set_title(f"Where the fastest {1 - OUTLIER_QUANTILE:.1%} of readings occur")
    ax_where.set_xlabel("Longitude (°E)")
    ax_where.set_ylabel("Latitude (°N)")

    fig.suptitle(f"Drifter speeds and outliers, {region['label']}")
    fig.tight_layout()
    fig.text(
        0.01, -0.06,
        f"Left: speed by drogue (sea anchor) status. {np.mean(~drogued):.0%} of readings come from drifters that "
        f"have lost the drogue; dropping them would leave too few drifters per cell, so they are kept.\nRight: the fastest {1 - OUTLIER_QUANTILE:.1%} of readings (above {threshold:.2f} m/s) sit in "
        "known fast currents (Brazil and Brazil-Malvinas off South America, Agulhas in the southeast corner),\n"
        "so they are real flow, not errors, and are kept.",
        fontsize=9,
    )
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    data = load_region(region)
    speed = np.hypot(data["ve"], data["vn"])
    drogued = data["drogue_status"]
    gap_hours = data["gap"] / 3600

    print("Missing values:")
    missing_summary(data)
    print(f"  Interpolation gap: median {np.median(gap_hours) * 60:.0f} min, "
          f"99th percentile {np.quantile(gap_hours, 0.99):.1f} h, max {gap_hours.max():.0f} h")

    print("Drogue status:")
    print(f"  {np.mean(~drogued):.0%} of readings come from drifters that have lost their drogue")
    for cell_size in CELL_SIZES:
        pass_all, pass_drogued, pass_undrogued = drogue_adequacy(data, region, cell_size)
        print(f"  {cell_size:g}°: {pass_all:.1%} of ocean cells adequate using all drifters, "
              f"{pass_drogued:.1%} drogued readings only, {pass_undrogued:.1%} undrogued readings only")

    print("Speed outliers:")
    fast = speed > np.nanquantile(speed, OUTLIER_QUANTILE)
    print(f"  Fastest {1 - OUTLIER_QUANTILE:.1%}: speed > {np.nanquantile(speed, OUTLIER_QUANTILE):.2f} m/s "
          f"(max {np.nanmax(speed):.2f} m/s), from {np.unique(data['drifter_id'][fast]).size} different drifters")
    print(f"  Undrogued: {np.mean(~drogued[fast]):.0%} of the fastest readings vs {np.mean(~drogued):.0%} overall")

    # North-south crossing; east-west is a little quicker since longitude degrees shrink with latitude
    print("Days for a median-speed drogued drifter to cross one cell north to south:")
    km_per_day = np.nanmedian(speed[drogued]) * 86.4
    for cell_size in CELL_SIZES:
        print(f"  {cell_size:g}°: {cell_size * KM_PER_DEG / km_per_day:.0f} days")

    out_path = os.path.join(FIGURES_DIR, "data_quality.png")
    plot_quality(data, speed, region, out_path)
    print(f"Saved {out_path}")
