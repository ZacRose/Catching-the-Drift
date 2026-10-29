"""
release_map.py

Related task: Week 4 Computational - preliminary results from the 4 deg baseline: where material
released at one point is after tau, about 3 months and about a year (P, P^6, P^24).
Makes Figure 1 of the proposal.

Material starts as probability 1 in the 4 deg cell holding region["release_point"], and each
step multiplies by the naive P (row vector times matrix). "Left R" is absorbing (decision 11),
so its entry is the chance the material has left the box by that time. The release point was
provisional; it is kept so Figure 1 reproduces, and the Week 6 maps release from the Benguela
coastal cells instead (src/linalg/coastal_release.py).

Inputs are committed files only (data/processed/), so this runs without the region cache:
    naive_P_4deg_tau15d.npz           the 4 deg matrix (panels b, c)
    transition_counts_2deg_tau15d.npz drifters starting a move in each 2 deg cell (panel a)

Run: python3 -m src.linalg.release_map
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter

from src.common.config import FIGURES_DIR, PROCESSED_DIR, load_region_bounds
from src.gridding.grid import cell_index, grid_edges
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS
from src.inference.adequacy import N_MIN

STEPS = (1, 6, 24)  # tau, about 3 months and about a year at tau = 15 days


def propagate(P, start_cell, steps=STEPS):
    """Distribution of released material after each number of steps in `steps`.

    Inputs:
        P: (n_cells + 1) x (n_cells + 1) transition matrix; the last state is "left R".
        start_cell: flat id of the release cell, or a list of cells to spread the release evenly over.
        steps: step counts to keep.
    Returns:
        dict {step: array of length n_cells + 1}. Entry n_cells is the chance of having left R.
        Mass sent to a cell with no row of its own is lost, so the total can fall just below 1.
    """
    cells = np.unique(np.atleast_1d(start_cell))
    x = np.zeros(P.shape[0])
    x[cells] = 1.0 / cells.size
    kept = {}
    for step in range(1, max(steps) + 1):
        x = x @ P
        if step in steps:
            kept[step] = x.copy()
    return kept


def centre_of_mass(dist, region, cell_size):
    """Probability-weighted mean (lat, lon) of the cell centres, over material still inside R."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    inside = dist[:-1].reshape(len(lat_c), len(lon_c))
    weights = inside / inside.sum()
    return float((weights.sum(axis=1) * lat_c).sum()), float((weights.sum(axis=0) * lon_c).sum())


def _degrees(axis):
    """Format a map axis as 20°S / 40°W rather than -20 / -40."""
    if axis == "lat":
        return FuncFormatter(lambda v, _: f"{abs(v):g}°{'S' if v < 0 else 'N' if v > 0 else ''}")
    return FuncFormatter(lambda v, _: f"{abs(v):g}°{'W' if v < 0 else 'E' if v > 0 else ''}")


def plot(region, drifters_fine, fine, dists, P_coarse, coarse, release_ll, out_path):
    """(a) drifters per working-grid cell; (b, c) where released material is after STEPS[1] and STEPS[2]."""
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.75), constrained_layout=True)

    def frame(ax, title):
        ax.set_facecolor("#d9d9d9")  # grey = no drifters (mostly land)
        ax.set_xlim(region["lon_min"], region["lon_max"])
        ax.set_ylim(region["lat_min"], region["lat_max"])
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(_degrees("lon"))
        ax.yaxis.set_major_formatter(_degrees("lat"))
        ax.tick_params(length=2, pad=1)
        ax.set_title(title, fontsize=7.5, loc="left", pad=3)
        ax.add_patch(plt.Rectangle(release_ll, coarse, coarse, fill=False, ec="#d62728", lw=1.2, zorder=5))

    lat_f, lon_f = grid_edges(region, fine)
    grid = drifters_fine.reshape(len(lat_f) - 1, len(lon_f) - 1)
    ax = axes[0]
    frame(ax, f"(a) Drifters per {fine:g}° cell ({TAU_DAYS}-day moves)")
    mesh = ax.pcolormesh(lon_f, lat_f, np.ma.masked_equal(grid, 0), cmap="viridis",
                         norm=colors.LogNorm(vmin=1, vmax=grid.max()))
    ax.contour(0.5 * (lon_f[:-1] + lon_f[1:]), 0.5 * (lat_f[:-1] + lat_f[1:]), grid,
               levels=[N_MIN], colors="black", linewidths=0.6)
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02, aspect=30)
    cb.set_label(f"distinct drifters (black line: {N_MIN})", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=1.5)

    lat_c, lon_c = grid_edges(region, coarse)
    shape = (len(lat_c) - 1, len(lon_c) - 1)
    has_row = (P_coarse[:-1].sum(axis=1) > 0).reshape(shape)
    norm = colors.LogNorm(vmin=1e-3, vmax=0.2)
    for ax, step, when in ((axes[1], STEPS[1], "3 months"), (axes[2], STEPS[2], "1 year")):
        frame(ax, f"({'bc'[step == STEPS[2]]}) After {when} ($P^{{{step}}}$), {coarse:g}° cells")
        ax.set_yticklabels([])
        inside = dists[step][:-1].reshape(shape)
        # floor at 1e-4 so ocean cells the material never reaches show as white, not grey like land
        shown = np.ma.masked_where(~has_row, np.maximum(inside, 1e-4))
        mesh = ax.pcolormesh(lon_c, lat_c, shown, cmap="Blues", norm=norm)
        ax.text(region["lon_min"] + 1.5, region["lat_min"] + 1.5, f"left box: {dists[step][-1]:.1%}",
                fontsize=6.5, bbox=dict(fc="white", ec="none", pad=1))
    cb = fig.colorbar(mesh, ax=list(axes[1:]), orientation="horizontal", shrink=0.5, pad=0.02,
                      aspect=30, extend="both")
    cb.set_label("probability of being in each cell (log scale)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=1.5)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    P = np.load(os.path.join(PROCESSED_DIR, f"naive_P_{coarse:g}deg_tau{TAU_DAYS}d.npz"))["P"]
    drifters_fine = np.load(os.path.join(
        PROCESSED_DIR, f"transition_counts_{fine:g}deg_tau{TAU_DAYS}d.npz"))["start_drifters"]

    lat, lon = region["release_point"]
    start, _, _ = cell_index(np.array([lon]), np.array([lat]), region, coarse)
    start = int(start[0])
    lat_edges, lon_edges = grid_edges(region, coarse)
    n_lon = len(lon_edges) - 1
    release_ll = (lon_edges[start % n_lon], lat_edges[start // n_lon])  # lower-left corner
    print(f"Release cell {start}: {release_ll[1]:g} to {release_ll[1] + coarse:g} lat, "
          f"{release_ll[0]:g} to {release_ll[0] + coarse:g} lon")

    dists = propagate(P, start)
    for step, dist in dists.items():
        c_lat, c_lon = centre_of_mass(dist, region, coarse)
        print(f"After {step * TAU_DAYS} days (P^{step}): {dist[start]:.1%} still in the release cell, "
              f"{dist[-1]:.1%} left R, {1 - dist.sum():.2%} lost to row-less cells, "
              f"centre of what remains at {c_lat:.1f} lat, {c_lon:.1f} lon")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f"release_map_tau{TAU_DAYS}d.png")
    plot(region, drifters_fine, fine, dists, P, coarse, release_ll, out_path)
    print(f"Saved {out_path}")
