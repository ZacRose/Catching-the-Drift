"""
drogue_compare.py

One map comparing where Benguela coastal material is after 360 days under the drogued-only and the
undrogued-only 2 deg shrinkage matrices: the mean position (diamond) and its path over the year
(dotted line) for each, with the coastal release cells (those with drifters in either subset) outlined. No probability shading, so the two
can share one map.

Each matrix is fitted the way robustness.py fits its drogue settings: drifters split by drogue status
at the START of each move (Week 4 issue e), alpha chosen on the train -> valid drifters of that subset
(evaluate.tune_alpha), then refitted on all of that subset's moves, with the 4 deg prior. The release
is spread evenly over the Benguela coastal cells that have drifters in that subset (as coastal_release.py).

Caveat: drogued-only data are thin (robustness.py: under 10% of 4 deg cells adequate, decision 6a),
so the drogued path leans more on the 4 deg prior and is indicative.

Outputs: outputs/figures/drogue_compare_360d.png and outputs/results/drogue_compare_360d.csv.

Run: python3 -m src.linalg.drogue_compare   (a minute or two; needs transitions_tau15d.npz and drifter_split.csv)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.common.config import FIGURES_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.trajectory_map import _degrees, draw_land
from src.inference.estimators import count_matrix
from src.inference.evaluate import states_for, tune_alpha
from src.inference.robustness import apply_filter, coarse_of, make_fit
from src.inference.split import load_split
from src.inference.transition_data import TAU_DAYS, load_transitions, parent_states, to_states
from src.linalg.release_map import centre_of_mass

DAYS = 360
CELL = 2.0
SUBSETS = [  # (filter, label, colour, marker face)
    ("drogued", "Drogued only", "#1f4e9c", "#1f4e9c"),
    ("undrogued", "Undrogued only", "#d95f02", "#d95f02"),
]


def fit_subset(table, filt, region, assignment):
    """2 deg shrinkage P for one drogue subset, alpha tuned on that subset's train -> valid drifters."""
    sub = apply_filter(table, filt)
    s_tr, e_tr, n_cells, _ = states_for(sub, region, CELL, assignment, "train")
    s_va, e_va, _, _ = states_for(sub, region, CELL, assignment, "valid")
    parent = parent_states(region, CELL, coarse_of(CELL))
    alpha, _ = tune_alpha(count_matrix(s_tr, e_tr, n_cells + 1), parent, s_va, e_va)
    start, end, n_cells = to_states(sub, region, CELL)
    P = make_fit("shrinkage", region, CELL, alpha)(count_matrix(start, end, n_cells + 1))
    _, drifters = per_cell_counts(start, sub["drifter_id"], n_cells)
    return P, alpha, drifters, sub


def centre_path(P, release, region, steps):
    """Mean (lat, lon) of material still in the box at release and after each step."""
    x = np.zeros(P.shape[0])
    x[release] = 1.0 / release.size
    path = [centre_of_mass(x, region, CELL)]
    for _ in range(steps):
        x = x @ P
        path.append(centre_of_mass(x, region, CELL))
    return np.array(path), x


def plot(region, zone, results, out_path):
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    draw_land(ax)
    lat_edges, lon_edges = grid_edges(region, CELL)
    n_lon = len(lon_edges) - 1
    for c in zone:
        r, k = divmod(int(c), n_lon)
        ax.add_patch(plt.Rectangle((lon_edges[k], lat_edges[r]), CELL, CELL, fill=False, ec="#d62728",
                                   lw=1.2, zorder=5))
    handles = []
    for res in results:
        path, colour, label = res["path"], res["colour"], res["label"]
        ax.plot(path[:, 1], path[:, 0], ls=(0, (1, 2)), lw=1.6, color=colour, zorder=6)
        ax.plot(path[-1, 1], path[-1, 0], marker="D", ms=10, mfc=colour, mec="white", mew=1.4, zorder=7)
        ax.annotate(f"{label}\n{abs(path[-1, 0]):.0f}°S, {abs(path[-1, 1]):.0f}°W", (path[-1, 1], path[-1, 0]),
                    xytext=(0, 14), textcoords="offset points", ha="center", va="bottom", fontsize=9.5,
                    color=colour, fontweight="bold", zorder=8)
        handles.append(Line2D([], [], color=colour, ls=(0, (1, 2)), lw=1.6, marker="D", ms=8, mfc=colour,
                              mec="white", label=f"{label} (mean position, path over {DAYS} days)"))
    handles.append(Line2D([], [], color="#d62728", marker="s", ls="none", mfc="none", ms=9,
                          label="Benguela coastal release cells"))
    ax.legend(handles=handles, loc="lower center", fontsize=9, framealpha=0.9)
    ax.set_xlim(region["lon_min"], region["lon_max"])
    ax.set_ylim(region["lat_min"], region["lat_max"])
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(_degrees("lon"))
    ax.yaxis.set_major_formatter(_degrees("lat"))
    ax.set_title(f"Drogued vs undrogued drifters: after {DAYS} days from the Benguela coast", loc="left")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    table = load_transitions(region, verbose=False)
    assignment = load_split(table["drifter_id"])
    zone = coastal_cells(region, CELL, land_fraction(region, CELL, land_paths()))
    results, rows, released = [], [], set()
    for filt, label, colour, _ in SUBSETS:
        P, alpha, drifters, sub = fit_subset(table, filt, region, assignment)
        release = zone[drifters[zone] > 0]
        released.update(release.tolist())
        path, x = centre_path(P, release, region, DAYS // TAU_DAYS)
        results.append({"label": label, "colour": colour, "path": path})
        rows.append({"drifters": filt, "alpha": round(alpha, 2), "n_moves": int(sub["drifter_id"].size),
                     "n_drifters": int(np.unique(sub["drifter_id"]).size), "release_cells": int(release.size),
                     "centre_lat_360d": round(float(path[-1, 0]), 3), "centre_lon_360d": round(float(path[-1, 1]), 3),
                     "pct_left_360d": round(float(100 * x[-1]), 3)})
        print(rows[-1])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"drogue_compare_{DAYS}d.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_out = os.path.join(FIGURES_DIR, f"drogue_compare_{DAYS}d.png")
    plot(region, sorted(released), results, fig_out)
    print(f"Saved {out} and {fig_out}")
