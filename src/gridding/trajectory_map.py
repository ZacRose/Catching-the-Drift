"""
trajectory_map.py

Related task: Week 5 poster - the study-region figure. Drifter tracks inside region R with the box
drawn on top, so one picture shows the data, the region and the Benguela flow (replaces the plain
box map from the poster draft).

Tracks use one reading per drifter-day (00:00 UTC, the same readings the transitions start from).
The region cache only holds readings inside R, so tracks stop at the box edge. Arrows are the mean
drifter velocity per 4 deg cell, drawn only where at least N_MIN distinct drifters were seen, so
the arrows follow the same adequacy rule as the rest of the project. All drifters are used,
drogued or not (decision 6).

Coastlines: Natural Earth 1:110m land (public domain), data/external/ne_110m_land.geojson.

Run: python3 -m src.gridding.trajectory_map
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter

from src.common.config import FIGURES_DIR, REPO_ROOT, load_region_bounds
from src.ingest.load import load_region
from src.gridding.grid import cell_index, grid_edges, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE
from src.inference.adequacy import N_MIN

LAND_PATH = os.path.join(REPO_ROOT, "data", "external", "ne_110m_land.geojson")
MAP_EXTENT = (-65, 25, -47, 1)  # lon_min, lon_max, lat_min, lat_max: a margin of coast around R


def split_tracks(lon, lat, drifter_id, time, max_step=np.timedelta64(1, "D")):
    """Cut readings into continuous track pieces for drawing.

    Inputs:
        lon, lat, drifter_id, time: readings sorted by drifter, then time.
        max_step: longest time step still drawn as a line.
    Returns:
        list of (n, 2) arrays of (lon, lat), one per piece with at least 2 readings.
        A new piece starts at each new drifter and after any longer step (a gap in the record,
        or the drifter leaving R and coming back), so no line jumps across water it never crossed.
    """
    new_piece = (drifter_id[1:] != drifter_id[:-1]) | (np.diff(time) > max_step)
    cuts = np.flatnonzero(new_piece) + 1
    points = np.column_stack([lon, lat])
    return [piece for piece in np.split(points, cuts) if len(piece) >= 2]


def mean_flow(lon, lat, drifter_id, ve, vn, region, cell_size):
    """Mean eastward and northward velocity (m/s) per cell, NaN where fewer than N_MIN drifters.

    Returns (lon_centres, lat_centres, u, v) with u, v shaped (n_lat_cells, n_lon_cells).
    """
    cell, n_lat, n_lon = cell_index(lon, lat, region, cell_size)
    n_cells = n_lat * n_lon
    counts, drifters = per_cell_counts(cell, drifter_id, n_cells)
    u = np.bincount(cell, weights=ve, minlength=n_cells) / np.maximum(counts, 1)
    v = np.bincount(cell, weights=vn, minlength=n_cells) / np.maximum(counts, 1)
    u[drifters < N_MIN] = np.nan
    v[drifters < N_MIN] = np.nan
    lat_edges, lon_edges = grid_edges(region, cell_size)
    return (0.5 * (lon_edges[:-1] + lon_edges[1:]), 0.5 * (lat_edges[:-1] + lat_edges[1:]),
            u.reshape(n_lat, n_lon), v.reshape(n_lat, n_lon))


def _degrees(axis):
    """Format a map axis as 20°S / 40°W rather than -20 / -40."""
    if axis == "lat":
        return FuncFormatter(lambda v, _: f"{abs(v):g}°{'S' if v < 0 else 'N' if v > 0 else ''}")
    return FuncFormatter(lambda v, _: f"{abs(v):g}°{'W' if v < 0 else 'E' if v > 0 else ''}")


def draw_land(ax, path=LAND_PATH):
    """Fill land from a Natural Earth GeoJSON (outer rings only; 1:110m land has no holes we need)."""
    with open(path) as f:
        features = json.load(f)["features"]
    for feature in features:
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for rings in polygons:
            ax.add_patch(Polygon(rings[0], closed=True, fc="#e6e6e6", ec="#7f7f7f", lw=0.6, zorder=3))


def plot(region, tracks, flow, n_drifters, out_path):
    """Tracks (light blue), mean flow (black arrows), region R (orange box) on one map."""
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    lon_min, lon_max, lat_min, lat_max = MAP_EXTENT
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(_degrees("lon"))
    ax.yaxis.set_major_formatter(_degrees("lat"))

    # very faint lines, so busy areas build up darker and the density reads as coverage
    ax.add_collection(LineCollection(tracks, colors="#4a90c2", linewidths=0.3, alpha=0.12, zorder=1))
    draw_land(ax)

    lon_c, lat_c, u, v = flow
    lon_g, lat_g = np.meshgrid(lon_c, lat_c)
    # R is 38 deg tall, so the top row of 4 deg cells overhangs it; its arrows would sit on the box edge
    overhang = lat_g >= region["lat_max"]
    u, v = np.where(overhang, np.nan, u), np.where(overhang, np.nan, v)
    arrows = ax.quiver(lon_g, lat_g, u, v, color="black", scale=4, width=0.003,
                       pivot="middle", zorder=4)
    ax.quiverkey(arrows, 0.72, 0.05, 0.2, "0.2 m/s mean flow", labelpos="E", coordinates="axes",
                 fontproperties={"size": 9})

    ax.add_patch(plt.Rectangle((region["lon_min"], region["lat_min"]),
                               region["lon_max"] - region["lon_min"], region["lat_max"] - region["lat_min"],
                               fill=False, ec="#d55e00", lw=2, zorder=5))
    ax.text(region["lon_min"] + 1, region["lat_max"] - 1, f"Region R ({n_drifters:,} drifters)",
            color="#d55e00", fontsize=10, fontweight="bold", va="top", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    # label sits in the arrow-free Angola Basin and points at the northwestward coastal flow
    ax.annotate("Benguela\nCurrent", xy=(11, -27), xytext=(2, -13), fontsize=10, fontweight="bold",
                ha="center", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    data = load_region(region)

    daily = data["time"] == data["time"].astype("datetime64[D]")  # 00:00 UTC readings only
    order = np.lexsort((data["time"][daily], data["drifter_id"][daily]))
    lon, lat = data["lon"][daily][order], data["lat"][daily][order]
    drifter_id, time = data["drifter_id"][daily][order], data["time"][daily][order]
    ve, vn = data["ve"][daily][order], data["vn"][daily][order]

    tracks = split_tracks(lon, lat, drifter_id, time)
    flow = mean_flow(lon, lat, drifter_id, ve, vn, region, BASELINE_CELL_SIZE)
    n_drifters = np.unique(drifter_id).size
    print(f"{lon.size:,} daily readings from {n_drifters:,} drifters, drawn as {len(tracks):,} track pieces")
    print(f"Arrows in {np.isfinite(flow[2]).sum()} of {flow[2].size} {BASELINE_CELL_SIZE:g} deg cells "
          f"(those with at least {N_MIN} drifters)")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "trajectory_map.png")
    plot(region, tracks, flow, n_drifters, out_path)
    print(f"Saved {out_path}")
