"""
transition_adequacy.py

Related task: Week 3 Quantitative - redo the adequacy check on the transition table.
This replaces the Week 1 proxy (distinct drifters with any reading in a cell) in the proposal.

The quantity is now the one the estimate actually uses: distinct drifters that START at least
one tau = 15 day transition in a cell. A cell's row is estimated only from those drifters, and
readings near the end of a record (or of a drifter that stopped within 15 days) no longer count.
Drifters are counted, not transitions: a drifter starts a transition every day it is in the cell,
and those daily starts share most of their path, so counting transitions would overstate the
information exactly as raw hourly readings did in Week 2. Same N = 100 rule as Week 1
(worst-case SE of a transition probability <= 0.05, project_plan.txt section 7 decision 3), and
the same ocean cells (cells with any reading), so the two percentages compare directly.

The Week 1 proxy needs the hourly region cache (data/interim/, gitignored). If it isn't on
this machine, the proxy is rebuilt from the transition table instead: every row is a daily
reading of a drifter in R, including the rows dropped for a gap or an ended record, so
"distinct drifters with any daily reading in the cell" is the Week 1 proxy sampled daily
instead of hourly. It can only differ for a drifter that crossed a cell in under a day, and
it avoids streaming the full cloud dataset (which needs more memory than most laptops have).
"Ocean cells" then means cells with a daily start rather than an hourly reading, so the
committed CSV records which source was used (week1_proxy_source); the committed version uses
the hourly cache and matches the official Week 1 figures.

Run: python3 -m src.inference.transition_adequacy
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, INTERIM_DIR, RESULTS_DIR, load_region_bounds
from src.ingest.load import load_region
from src.gridding.grid import cell_index, grid_edges, per_cell_counts
from src.inference.adequacy import N_MIN, drifter_grid
from src.gridding.transitions import BASELINE_CELL_SIZE
from src.inference.transition_data import TAU_DAYS, TRANSITIONS_PATH, load_transitions, to_states


def starting_drifters(table, region, cell_size):
    """Distinct drifters starting a transition, and transitions started, per cell (flat arrays)."""
    start, _, n_cells = to_states(table, region, cell_size)
    n_transitions, n_drifters = per_cell_counts(start, table["drifter_id"], n_cells)
    return n_drifters, n_transitions


def proxy_from_table(region, cell_size, path=TRANSITIONS_PATH):
    """Week 1 proxy from the transition table's daily starts (all outcomes, nothing dropped).

    Returns:
        (distinct drifters per cell, readings per cell), flat arrays.
    """
    stored = np.load(path)
    cell, n_lat, n_lon = cell_index(stored["start_lon"], stored["start_lat"], region, cell_size)
    readings, drifters = per_cell_counts(cell, stored["drifter_id"], n_lat * n_lon)
    return drifters, readings


def week1_proxy(region, cell_size, readings):
    """(drifters per cell, ocean mask) for the Week 1 proxy, from the cache if available."""
    if readings is not None:
        proxy_2d, obs_counts = drifter_grid(readings, region, cell_size)
        return proxy_2d.ravel(), obs_counts > 0
    drifters, n_readings = proxy_from_table(region, cell_size)
    return drifters, n_readings > 0


def summary_row(cell_size, ocean, proxy, n_drifters, n_transitions):
    """One CSV row comparing the Week 1 proxy with drifters starting a transition, over ocean cells."""
    return {
        "cell_size_deg": cell_size,
        "tau_days": TAU_DAYS,
        "ocean_cells": int(ocean.sum()),
        "pct_adequate_week1_proxy": round(float(100 * np.mean(proxy[ocean] >= N_MIN)), 1),
        "pct_adequate_transitions": round(float(100 * np.mean(n_drifters[ocean] >= N_MIN)), 1),
        "median_drifters_week1_proxy": int(np.median(proxy[ocean])),
        "median_drifters_transitions": int(np.median(n_drifters[ocean])),
        "p10_drifters_transitions": int(np.percentile(n_drifters[ocean], 10)),
        "median_transitions_per_drifter": round(float(np.median(
            n_transitions[ocean & (n_drifters > 0)] / n_drifters[ocean & (n_drifters > 0)])), 1),
        "ocean_cells_with_no_transitions": int(np.sum(ocean & (n_drifters == 0))),
    }


def plot(panels, region, out_path):
    """Rows = resolution, columns = Week 1 proxy vs transition starts, on one log colour scale."""
    vmax = max(max(p["proxy"].max(), p["drifters"].max()) for p in panels)
    fig, axes = plt.subplots(len(panels), 2, figsize=(15, 3.6 * len(panels)), layout="constrained", squeeze=False)
    for (ax_proxy, ax_new), p in zip(axes, panels):
        lat_edges, lon_edges = grid_edges(region, p["cell_size"])
        shape = (len(lat_edges) - 1, len(lon_edges) - 1)
        lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
        lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
        for ax, counts, pct, title in (
            (ax_proxy, p["proxy"], p["row"]["pct_adequate_week1_proxy"], "Week 1 proxy: drifters with any reading"),
            (ax_new, p["drifters"], p["row"]["pct_adequate_transitions"], f"Drifters starting a {TAU_DAYS}-day transition"),
        ):
            grid = counts.reshape(shape)
            mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_equal(grid, 0),
                                 norm=colors.LogNorm(vmin=1, vmax=vmax), cmap="viridis")
            ax.contour(lon_c, lat_c, grid >= N_MIN, levels=[0.5], colors="black", linewidths=1.5)
            ax.set_aspect("equal")
            ax.set_title(f"{p['cell_size']:g}°, {title}: {pct}% adequate", fontsize=10)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
    fig.colorbar(mesh, ax=axes, label="Distinct drifters per cell (log scale)", shrink=0.6)
    fig.suptitle(f"Which cells have enough drifters to estimate where material goes: {region['label']}")
    fig.text(0.02, -0.04, f"Left: drifters with any reading in the cell (Week 1 check). Right: drifters that "
             f"start a {TAU_DAYS}-day move in the cell, which is what the transition matrix is built from.\n"
             f"Black line encloses adequate cells (at least {N_MIN} distinct drifters). "
             "Blank cells have no drifters (mostly land).", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    cache = os.path.join(INTERIM_DIR, f"{region['name']}.npz")
    if os.path.exists(cache):
        readings = load_region(region)
        proxy_source = "hourly region cache"
    else:
        readings = None
        proxy_source = "transition table (daily readings; no region cache on this machine)"
    print(f"Week 1 proxy from: {proxy_source}")
    table = load_transitions(region)

    rows, panels = [], []
    for cell_size in (region["cell_size_deg"], BASELINE_CELL_SIZE):
        proxy, ocean = week1_proxy(region, cell_size, readings)
        n_drifters, n_transitions = starting_drifters(table, region, cell_size)
        row = summary_row(cell_size, ocean, proxy, n_drifters, n_transitions)
        row["week1_proxy_source"] = "cache" if readings is not None else "table"
        rows.append(row)
        panels.append({"cell_size": cell_size, "proxy": proxy, "drifters": n_drifters, "row": row})
        print(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, f"transition_adequacy_tau{TAU_DAYS}d.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "transition_adequacy_map.png")
    plot(panels, region, fig_path)
    print(f"Saved {table_path} and {fig_path}")
