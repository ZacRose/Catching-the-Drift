"""
product.py

Related task: Week 7 output - "a draft product: P plus its uncertainty, in a form another analyst
could pick up" (objective O1). Nothing new is estimated here: this gathers the files the pipeline
already saves into one object, with the two operations an analyst needs (which cell is this
position in; where is material from here after so many days).

Files read (all in data/processed/, made by the commands in the docstrings of each script):
- shrinkage_P_2deg_tau15d.npz          P, counts, alpha, start_drifters  (src.inference.shrinkage_fit)
- shrinkage_P_2deg_tau15d_bootstrap.npz se, ci_low, ci_high per entry    (src.inference.bootstrap)
- long_run_shrinkage_P_2deg_tau15d.npz  long-run pattern and intervals   (src.linalg.long_run; optional)

Example:
    from src.linalg.product import load_product
    prod = load_product()
    dist = prod.forecast(lat=-28.0, lon=14.0, days=90)   # 6 steps of 15 days
    prod.summary(dist)                                   # % left the box, centre, top cells

What it is and isn't (for the report and for anyone using it):
- States are 2 deg cells of the box in config/region.yaml, plus "left the box" (the last state,
  absorbing). Probabilities are for floating material moving with the near-surface flow, drogued and
  undrogued drifters pooled (decision 6), 1994-2022, all seasons together.
- Forecasts move in whole 15-day steps; a non-multiple of 15 days is refused rather than rounded.
  markov_check.py shows chaining 15-day steps predicts held-out 30- and 90-day moves at least as
  well as matrices fitted to those lags directly, but over-states exits (about 1 point at 30 days,
  3 points at 90 days, per cell).
- Rows for cells with no drifters (prior_only) are the 4 deg prior alone, and intervals for cells
  with under about 10 drifters are unreliable (bootstrap.py); check start_drifters.
"""
import os
from dataclasses import dataclass, field
import numpy as np

from src.common.config import PROCESSED_DIR
from src.gridding.grid import grid_edges
from src.gridding.transitions import TAU_DAYS
from src.ingest.load import in_box

BOX_KEYS = ("lat_min", "lat_max", "lon_min", "lon_max")


@dataclass
class Product:
    P: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    start_drifters: np.ndarray
    alpha: float
    region: dict
    cell_size: float
    tau_days: int
    lat_centre: np.ndarray  # per cell
    lon_centre: np.ndarray
    long_run: dict = field(default_factory=dict)

    @property
    def n_cells(self):
        return self.P.shape[0] - 1

    @property
    def prior_only(self):
        """True for cells with no drifters: their row is the 4 deg prior, not data."""
        return self.start_drifters == 0

    def cell_of(self, lat, lon):
        """Flat cell id of a position inside the box (ValueError outside it)."""
        lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
        if not np.all(in_box(lon, lat, self.region)):
            raise ValueError("Position is outside the box; the product has no row for it.")
        lat_edges, lon_edges = grid_edges(self.region, self.cell_size)
        row = np.clip(np.digitize(lat, lat_edges) - 1, 0, len(lat_edges) - 2)
        col = np.clip(np.digitize(lon, lon_edges) - 1, 0, len(lon_edges) - 2)
        ids = row * (len(lon_edges) - 1) + col
        return int(ids[0]) if ids.size == 1 else ids

    def forecast(self, lat=None, lon=None, days=None, start=None):
        """Distribution over states after `days` (a multiple of tau), from a position or a start distribution."""
        if days is None or days % self.tau_days:
            raise ValueError(f"days must be a multiple of {self.tau_days}.")
        if start is None:
            start = np.zeros(self.P.shape[0])
            start[self.cell_of(lat, lon)] = 1.0
        x = np.asarray(start, dtype=float)
        for _ in range(days // self.tau_days):
            x = x @ self.P
        return x

    def summary(self, dist, top=5):
        """% left the box, centre of the material still inside, and the cells holding most of it."""
        inside = dist[:-1]
        w = inside / inside.sum()
        order = np.argsort(-inside)[:top]
        return {"pct_left_box": 100 * float(dist[-1]),
                "centre_lat": float((w * self.lat_centre).sum()), "centre_lon": float((w * self.lon_centre).sum()),
                "top_cells": [(int(c), float(self.lat_centre[c]), float(self.lon_centre[c]), 100 * float(inside[c]),
                               int(self.start_drifters[c])) for c in order]}


def load_product(cell_size=2.0, tau_days=TAU_DAYS, processed_dir=PROCESSED_DIR):
    """The 2 deg shrinkage P with its bootstrap uncertainty (and the long-run pattern if saved)."""
    stem = os.path.join(processed_dir, f"shrinkage_P_{cell_size:g}deg_tau{tau_days}d")
    for path in (stem + ".npz", stem + "_bootstrap.npz"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: run src.inference.shrinkage_fit, then src.inference.bootstrap.")
    p, b = np.load(stem + ".npz"), np.load(stem + "_bootstrap.npz")
    region = {k: float(p[k]) for k in BOX_KEYS}
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat_c, lon_c = np.meshgrid(0.5 * (lat_edges[:-1] + lat_edges[1:]), 0.5 * (lon_edges[:-1] + lon_edges[1:]),
                               indexing="ij")
    long_run = {}
    lr_path = os.path.join(processed_dir, f"long_run_shrinkage_P_{cell_size:g}deg_tau{tau_days}d.npz")
    if os.path.exists(lr_path):
        long_run = {k: v for k, v in np.load(lr_path).items() if k not in BOX_KEYS}
    return Product(P=p["P"], se=b["se"], ci_low=b["ci_low"], ci_high=b["ci_high"],
                   start_drifters=p["start_drifters"], alpha=float(p["alpha"]), region=region,
                   cell_size=float(p["cell_size_deg"]), tau_days=int(p["tau_days"]),
                   lat_centre=lat_c.ravel(), lon_centre=lon_c.ravel(), long_run=long_run)


if __name__ == "__main__":
    prod = load_product()
    print(f"{prod.cell_size:g} deg, tau = {prod.tau_days} days, {prod.n_cells} cells "
          f"({int((~prod.prior_only).sum())} with drifters), alpha = {prod.alpha:.1f}")
    for days in (15, 90, 360):
        s = prod.summary(prod.forecast(lat=-28.0, lon=14.0, days=days))
        print(f"from 28S 14E after {days} days: left box {s['pct_left_box']:.2f}%, "
              f"centre {s['centre_lat']:.1f}, {s['centre_lon']:.1f}")
