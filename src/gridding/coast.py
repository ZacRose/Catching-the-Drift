"""
coast.py

Related task: Week 6 Computational - define the Benguela coastal cells (Week 4 open issue a), so
the release maps can start from them and report the chance of being in the coastal zone, and
report how often records end, and drifters run aground, there (open issue d).

A cell is coastal if it lies inside the coastal_zone band in config/region.yaml and the coastline
passes through it: part land, part sea on the Natural Earth 1:110m outline
(data/external/ne_110m_land.geojson). The land share is measured on a LATTICE x LATTICE grid of
points covering the cell, edges included. Only the band and the outline decide which cells count;
the drifters don't, so a coastal cell can have no drifters (two between 18S and 22S have none:
drifters seldom come that close to the shore there). Such cells can't be release points (their row
of P is the 4 deg prior alone), but they still count as coastal-zone destinations.

On the 2 deg grid this gives 13 cells from Cape Town (34S) to northern Namibia (18S), 9 of them
with drifters. A 2 deg cell is about 200 km across, so "in the coastal zone" means within about
one cell of the shore, not on it.

Issue d: rows whose record ended aren't counted in P (decision 10). If drifters that run aground
(typedeath 1) cluster at the coast, coastal rows of P overstate how much material stays at sea.
outcome_shares() reports, for starts in the coastal cells and for the rest of R, the share of each
outcome and the share whose record ended within tau and whose drifter ran aground.

Outputs: outputs/results/coastal_cells.csv (the cells at 2 and 4 deg) and
outputs/results/coastal_outcomes_tau15d.csv (issue d).

Run: python3 -m src.gridding.coast   (needs data/processed/transitions_tau15d.npz)
"""
import csv
import json
import os
import numpy as np
from matplotlib.path import Path

from src.common.config import RESULTS_DIR, load_region_bounds
from src.gridding.grid import cell_index, grid_edges, per_cell_counts
from src.gridding.trajectory_map import LAND_PATH
from src.gridding.transitions import BASELINE_CELL_SIZE, ENDED, LEFT_R, OUTCOME_NAMES, TAU_DAYS
from src.inference.transition_data import TRANSITIONS_PATH

LATTICE = 21  # points per cell side (0.1 deg apart on the 2 deg grid)
RAN_AGROUND = 1  # GDP typedeath code


def land_paths(path=LAND_PATH):
    """Outer ring of every land polygon in a Natural Earth GeoJSON, as matplotlib Paths."""
    with open(path) as f:
        features = json.load(f)["features"]
    paths = []
    for feature in features:
        geom = feature["geometry"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        paths.extend(Path(np.asarray(rings[0])) for rings in polygons)
    return paths


def land_fraction(region, cell_size, paths=None, lattice=LATTICE):
    """Share of each cell's lattice points that fall on land (flat array, cells as in cell_index)."""
    paths = land_paths() if paths is None else paths
    lat_edges, lon_edges = grid_edges(region, cell_size)
    step = np.linspace(0, cell_size, lattice)
    lat0, lon0 = np.meshgrid(lat_edges[:-1], lon_edges[:-1], indexing="ij")  # row-major, as cell_index
    lon = (lon0.ravel()[:, None, None] + step[None, None, :]).repeat(lattice, axis=1)
    lat = (lat0.ravel()[:, None, None] + step[None, :, None]).repeat(lattice, axis=2)
    points = np.column_stack([lon.ravel(), lat.ravel()])
    on_land = np.zeros(len(points), dtype=bool)
    for p in paths:
        on_land |= p.contains_points(points)
    return on_land.reshape(lat0.size, -1).mean(axis=1)


def coastal_cells(region, cell_size, land=None):
    """Flat ids (sorted) of the cells inside region["coastal_zone"] that are part land, part sea.

    Inputs:
        region: dict from config/region.yaml, with a coastal_zone entry.
        cell_size: cell width in degrees.
        land: land_fraction() for this grid, if already computed.
    """
    zone = region["coastal_zone"]
    land = land_fraction(region, cell_size) if land is None else land
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat0, lon0 = (a.ravel() for a in np.meshgrid(lat_edges[:-1], lon_edges[:-1], indexing="ij"))
    tol = 1e-9
    inside = ((lat0 >= zone["lat_min"] - tol) & (lat0 + cell_size <= zone["lat_max"] + tol)
              & (lon0 >= zone["lon_min"] - tol) & (lon0 + cell_size <= zone["lon_max"] + tol))
    return np.flatnonzero(inside & (land > 0) & (land < 1))


def outcome_shares(table, region, cell_size, cells):
    """Share of each outcome, and of "record ended and the drifter ran aground", for starts in
    `cells` and for every other start in R (issue d).

    Inputs:
        table: the full transition table (all outcomes, as saved by transitions.py).
    Returns:
        list of two dicts, "coastal cells" then "rest of R".
    """
    start, _, _ = cell_index(table["start_lon"], table["start_lat"], region, cell_size)
    coastal = np.isin(start, cells)
    aground = (table["outcome"] == ENDED) & (table["typedeath"] == RAN_AGROUND)
    rows = []
    for name, m in (("coastal cells", coastal), ("rest of R", ~coastal)):
        row = {"cell_size_deg": cell_size, "starts_in": name, "starts": int(m.sum()),
               "drifters": int(np.unique(table["drifter_id"][m]).size)}
        for code, label in OUTCOME_NAMES.items():
            row[f"pct_{label.replace(' ', '_')}"] = round(float(100 * np.mean(table["outcome"][m] == code)), 2)
        row["pct_ended_and_ran_aground"] = round(float(100 * np.mean(aground[m])), 2)
        row["drifters_ended_and_ran_aground"] = int(np.unique(table["drifter_id"][m & aground]).size)
        rows.append(row)
    return rows


def _write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    region = load_region_bounds()
    table = dict(np.load(TRANSITIONS_PATH))
    table.pop("tau_days")
    paths = land_paths()

    cell_rows, outcome_rows = [], []
    for size in (region["cell_size_deg"], BASELINE_CELL_SIZE):
        land = land_fraction(region, size, paths)
        cells = coastal_cells(region, size, land)
        # drifters starting a counted (known-destination) move, as in the adequacy map
        known = table["outcome"] <= LEFT_R
        start, n_lat, n_lon = cell_index(table["start_lon"][known], table["start_lat"][known], region, size)
        _, drifters = per_cell_counts(start, table["drifter_id"][known], n_lat * n_lon)
        lat_edges, lon_edges = grid_edges(region, size)
        print(f"{size:g} deg: {cells.size} coastal cells, {np.sum(drifters[cells] > 0)} with drifters")
        for c in cells:
            r, k = divmod(int(c), n_lon)
            cell_rows.append({"cell_size_deg": size, "cell": int(c),
                              "lat_min": lat_edges[r], "lat_max": lat_edges[r + 1],
                              "lon_min": lon_edges[k], "lon_max": lon_edges[k + 1],
                              "land_fraction": round(float(land[c]), 3), "start_drifters": int(drifters[c])})
            print(f"  cell {c}: {lat_edges[r]:g} to {lat_edges[r + 1]:g} lat, {lon_edges[k]:g} to "
                  f"{lon_edges[k + 1]:g} lon, {land[c]:.1%} land, {drifters[c]} drifters")
        outcome_rows += outcome_shares(table, region, size, cells)

    for row in outcome_rows:
        print(row)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cells_path = os.path.join(RESULTS_DIR, "coastal_cells.csv")
    outcomes_path = os.path.join(RESULTS_DIR, f"coastal_outcomes_tau{TAU_DAYS}d.csv")
    _write_csv(cell_rows, cells_path)
    _write_csv(outcome_rows, outcomes_path)
    print(f"Saved {cells_path} and {outcomes_path}")
