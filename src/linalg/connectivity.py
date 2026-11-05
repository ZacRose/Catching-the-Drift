"""
connectivity.py

Related task: proposal RQ1 and objective O2, closed in the Week 9 forensic review. RQ1 asks which subregions
of the domain supply which others, and where, and at what rate, floating material leaves. O2 names release
points such as the Orange Basin. Earlier scripts answer "where from the Benguela coast" and "where does it
gather" (coastal_release.py, long_run.py), but not these.

Everything here uses the product (the 2 deg shrinkage P, 15-day steps) and adds no new estimate. Material is
spread evenly over a source's cells that have drifters, as in coastal_release.py.

1. Subregions (config/region.yaml, subregions): boxes covering R once each, plus the Benguela coastal cells
   (coast.py). A cell belongs to the box holding its centre. The boxes are a choice, made so their edges are
   physical features or match boxes used elsewhere in the project, so they're listed for the team to confirm.
2. In the destination at that moment: P^n from each source, added up over each destination, after 15 days,
   about 3 months and about a year.
3. Reached by then: the chance material has entered the destination at any time up to then (planners ask
   "could it get there, and how soon", which a snapshot understates, as material passes through). The
   destination's rows are made absorbing and P is stepped forward: first-passage probabilities. "How soon" is
   the first day this reaches 1% and 10%, out to about 3 years. Drifters that run aground stop reporting
   (decision 10), so reaching a coast is a lower bound.
4. Where it leaves. The chance of having left through each edge of the box by each time. A step's exit from
   a cell is split between edges by where that cell's drifters really crossed (data_forensics.py), or the
   nearest edge for a cell with no observed exit. The long-run rate is the share lost per step from the
   quasi-stationary distribution (long_run.py), split the same way.

Intervals: drifter bootstrap, as bootstrap.py: 1000 replicates for "in the destination" and exits, N_BOOT_REACH
for "reached" (eight absorbing matrices per replicate). They show sampling of drifters only, not year-to-year
change: see year_intervals.py.

Outputs: outputs/results/connectivity_tau15d.csv, outputs/figures/connectivity.png and
outputs/figures/orange_basin_release.png.

Run: python3 -m src.linalg.connectivity   (about 2 minutes; needs shrinkage_P_2deg_tau15d.npz, and
exit_edges_2deg_tau15d.npz from src.inference.data_forensics)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.trajectory_map import _degrees, draw_land
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS
from src.inference.bootstrap import LEVEL, N_BOOT, percentile_interval, replicate_matrices, shrinkage_fit
from src.inference.data_forensics import EDGES, EXIT_EDGES_PATH, nearest_edge
from src.inference.estimators import count_matrix
from src.inference.transition_data import load_transitions, parent_states, to_states
from src.linalg.coastal_release import _pct
from src.linalg.long_run import qsd_power, substochastic

COAST_NAME = "Benguela coast"
STEPS = (1, 6, 24)  # about 15 days, 3 months and a year
ARRIVAL_STEPS = 72  # about 3 years, as far as the app goes
ARRIVAL_LEVELS = (0.01, 0.10)
N_BOOT_REACH = 200


def box_cells(region, cell_size, box):
    """Flat ids of the cells whose centre lies inside a lat/lon box."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat_c, lon_c = np.meshgrid(0.5 * (lat_edges[:-1] + lat_edges[1:]), 0.5 * (lon_edges[:-1] + lon_edges[1:]), indexing="ij")
    return np.flatnonzero((lat_c.ravel() >= box["lat_min"]) & (lat_c.ravel() <= box["lat_max"])
                          & (lon_c.ravel() >= box["lon_min"]) & (lon_c.ravel() <= box["lon_max"]))


def subregion_map(region, cell_size, coast):
    """Every cell's subregion, as (names, label per cell). Raises an error unless the boxes cover R once each.

    Inputs:
        coast: the coastal cells, which become their own subregion whichever box holds them.
    """
    if "subregions" not in region:
        raise ValueError("This region has no subregions: list them in config/region.yaml (see the Benguela box there).")
    n_cells = (len(grid_edges(region, cell_size)[0]) - 1) * (len(grid_edges(region, cell_size)[1]) - 1)
    names = [COAST_NAME] + [b["name"] for b in region["subregions"]]
    label = np.full(n_cells, -1)
    for k, box in enumerate(region["subregions"], start=1):
        cells = box_cells(region, cell_size, box)
        if np.any(label[cells] >= 0):
            raise ValueError(f"Subregion {box['name']} overlaps another one in config/region.yaml.")
        label[cells] = k
    if np.any(label < 0):
        raise ValueError(f"{np.sum(label < 0)} cells are in no subregion; check config/region.yaml.")
    label[coast] = 0
    return names, label


def sources(names, label, drifters, region, cell_size):
    """(name, release cells): every subregion and every named release, over the cells that have drifters."""
    out = [(name, np.flatnonzero((label == k) & (drifters > 0))) for k, name in enumerate(names)]
    for box in region.get("named_releases", []):
        cells = box_cells(region, cell_size, box)
        out.append((box["name"], cells[drifters[cells] > 0]))
    return [(name, cells) for name, cells in out if cells.size]


def start_matrix(release_sets, n_states):
    """One row per source: its material spread evenly over its cells."""
    S = np.zeros((len(release_sets), n_states))
    for k, cells in enumerate(release_sets):
        S[k, cells] = 1.0 / cells.size
    return S


def in_destination(S, P, label, n_labels, steps=STEPS):
    """{step: (n_sources, n_labels + 1) array}: share in each subregion at that step; last column = left the box."""
    X, out, n_cells = S.copy(), {}, label.size
    to_label = np.zeros((P.shape[0], n_labels + 1))
    to_label[np.arange(n_cells), label] = 1.0
    to_label[n_cells, n_labels] = 1.0
    for step in range(1, max(steps) + 1):
        X = X @ P
        if step in steps:
            out[step] = X @ to_label
    return out


def absorbing(P, cells):
    """P with these cells' rows replaced by "stays here", so material that arrives is kept (first passage)."""
    Q = P.copy()
    Q[cells] = 0.0
    Q[cells, cells] = 1.0
    return Q


def reached(S, P, cells, steps=STEPS):
    """{step: share per source that has entered these cells by that step (at any time up to it)}."""
    Q = absorbing(P, cells)
    X, out = S.copy(), {}
    for step in range(1, max(steps) + 1):
        X = X @ Q
        if step in steps:
            out[step] = X[:, cells].sum(axis=1)
    return out


def arrival_days(S, P, cells, levels=ARRIVAL_LEVELS, max_steps=ARRIVAL_STEPS, tau=TAU_DAYS):
    """First day the chance of having reached these cells passes each level, per source (NaN if never)."""
    Q = absorbing(P, cells)
    X = S.copy()
    days = np.full((S.shape[0], len(levels)), np.nan)
    for step in range(1, max_steps + 1):
        X = X @ Q
        share = X[:, cells].sum(axis=1)
        for k, level in enumerate(levels):
            days[:, k] = np.where(np.isnan(days[:, k]) & (share >= level), step * tau, days[:, k])
    return days


def edge_shares(counts, nearest):
    """How each cell's exits are split between the box edges: where its drifters crossed, or else the nearest edge.

    Inputs:
        counts: (n_cells, n_edges) exits per cell through each edge (data_forensics.py).
        nearest: (n_cells) the nearest edge of each cell, for cells with no observed exit.
    """
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals > 0)
    empty = totals[:, 0] == 0
    shares[np.flatnonzero(empty), nearest[empty]] = 1.0
    return shares


def nearest_edges(region, cell_size):
    """Index into EDGES of the box edge nearest each cell's centre."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat_c, lon_c = (a.ravel() for a in np.meshgrid(0.5 * (lat_edges[:-1] + lat_edges[1:]), 0.5 * (lon_edges[:-1] + lon_edges[1:]), indexing="ij"))
    return nearest_edge(lat_c, lon_c, region)[0]


def left_by_edge(S, P, shares, steps=STEPS):
    """{step: (n_sources, n_edges) chance of having left through each edge by that step}."""
    n_cells = shares.shape[0]
    X, total, out = S.copy(), np.zeros((S.shape[0], shares.shape[1])), {}
    leave = P[:n_cells, n_cells][:, None] * shares
    for step in range(1, max(steps) + 1):
        total = total + X[:, :n_cells] @ leave
        X = X @ P
        if step in steps:
            out[step] = total.copy()
    return out


def long_run_exit(P, keep, shares, q_start=None):
    """Share lost per step from the quasi-stationary distribution, in total and through each edge (% per step)."""
    lam, q = qsd_power(substochastic(P, keep), q_start)
    n_cells = shares.shape[0]
    by_edge = q @ (P[keep, n_cells][:, None] * shares[keep])
    return 100 * by_edge, 100 * (1 - lam), q


def _ci(reps, axis=0):
    return percentile_interval(np.asarray(reps), LEVEL, axis=axis)


def plot_matrices(names_src, names_dst, tables, out_path):
    """One heat table per time: share of each source (row) in each destination (column) at that moment."""
    fig, axes = plt.subplots(1, len(tables), figsize=(16.5, 5.4), layout="constrained", sharey=True)
    norm = colors.LogNorm(vmin=0.1, vmax=100)
    for ax, (when, grid) in zip(axes, tables):
        shown = np.ma.masked_less(grid, 0.05)
        mesh = ax.imshow(shown, cmap="Blues", norm=norm, aspect="auto")
        ax.set_xticks(range(len(names_dst)), names_dst, rotation=45, ha="right", fontsize=8.5)
        ax.set_title(f"Where it is after {when}", fontsize=11, loc="left")
        for (r, c), v in np.ndenumerate(grid):
            if v >= 0.05:
                ax.text(c, r, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v >= 25 else "#1b2733")
        ax.set_xticks(np.arange(-0.5, len(names_dst)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names_src)), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="both", length=0)
    axes[0].set_yticks(range(len(names_src)), names_src, fontsize=9)
    axes[0].set_ylabel("Released evenly over (source)")
    fig.colorbar(mesh, ax=axes, shrink=0.75, label="% of the material (log scale)")
    fig.text(0.01, -0.05, f"From the 2° shrinkage matrix, {TAU_DAYS}-day steps. Each row adds up to 100% across destinations and "
             "'Left the box'. Blank: under 0.05%. Intervals are in outputs/results/connectivity_tau15d.csv.", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_release(region, cell_size, name, dists, release, drifters, stats, out_path):
    """Where material from one named release is after STEPS, as the coastal release maps."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    fig, axes = plt.subplots(1, len(STEPS), figsize=(13, 2.9), layout="constrained")
    norm = colors.LogNorm(vmin=1e-4, vmax=0.3)
    for ax, step in zip(axes, STEPS):
        shown = np.ma.masked_where((drifters == 0).reshape(shape), np.maximum(dists[step][:-1].reshape(shape), norm.vmin))
        mesh = ax.pcolormesh(lon_edges, lat_edges, shown, cmap="Blues", norm=norm)
        draw_land(ax)
        for c in release:
            r, k = divmod(int(c), shape[1])
            ax.add_patch(plt.Rectangle((lon_edges[k], lat_edges[r]), cell_size, cell_size, fill=False, ec="#d62728", lw=1.2, zorder=5))
        ax.set_xlim(region["lon_min"], region["lon_max"])
        ax.set_ylim(region["lat_min"], region["lat_max"])
        ax.set_aspect("equal")
        ax.set_facecolor("#d9d9d9")
        ax.xaxis.set_major_formatter(_degrees("lon"))
        ax.yaxis.set_major_formatter(_degrees("lat"))
        ax.set_title(f"{name}: after {step * TAU_DAYS} days ($P^{{{step}}}$)", fontsize=10, loc="left")
        ax.text(region["lon_min"] + 1, region["lat_min"] + 1, stats[step], fontsize=8, va="bottom", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.colorbar(mesh, ax=axes, shrink=0.8, extend="both", label="probability per 2° cell (log scale)")
    fig.text(0.01, -0.07, f"Red: the release cell ({int(drifters[release].sum())} drifters). 2° shrinkage matrix; brackets are 95% intervals from "
             f"{N_BOOT} drifter bootstrap replicates. Grey: land, and ocean cells with no drifters.", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region, verbose=False)
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    alpha = float(product["alpha"])
    start, end, n_cells = to_states(table, region, fine)
    n_states = n_cells + 1
    fit = shrinkage_fit(parent_states(region, fine, coarse), alpha)
    P = fit(count_matrix(start, end, n_states))
    if not np.allclose(P, product["P"]):
        raise ValueError("Refitted shrinkage P differs from the saved product; rerun src.inference.shrinkage_fit.")
    _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)
    keep = np.flatnonzero(drifters > 0)
    coast = coastal_cells(region, fine, land_fraction(region, fine, land_paths()))
    names, label = subregion_map(region, fine, coast)
    src = sources(names, label, drifters, region, fine)
    src_names = [s[0] for s in src]
    S = start_matrix([s[1] for s in src], n_states)
    dst_names = names + ["Left the box"]
    dst_cells = {k: np.flatnonzero(label == k) for k in range(len(names))}
    shares = edge_shares(np.load(EXIT_EDGES_PATH)["counts"], nearest_edges(region, fine))
    for name, cells in src:
        print(f"{name}: {cells.size} cells with drifters, median {int(np.median(drifters[cells]))} drifters per cell")

    # Full-data estimates
    est_in = in_destination(S, P, label, len(names))
    est_reach = {k: reached(S, P, cells) for k, cells in dst_cells.items()}
    est_edge = left_by_edge(S, P, shares)
    est_lr_edge, est_lr_total, q_hat = long_run_exit(P, keep, shares)
    arrivals = {k: arrival_days(S, P, cells) for k, cells in dst_cells.items()}

    # Replicates
    reps_in = {s: [] for s in STEPS}
    reps_edge = {s: [] for s in STEPS}
    reps_lr = []
    for P_b in replicate_matrices(start, end, table["drifter_id"], n_states, fit, N_BOOT):
        d, e = in_destination(S, P_b, label, len(names)), left_by_edge(S, P_b, shares)
        for s in STEPS:
            reps_in[s].append(d[s])
            reps_edge[s].append(e[s])
        by_edge, total, _ = long_run_exit(P_b, keep, shares, q_hat)
        reps_lr.append(np.append(by_edge, total))
    reps_reach = {k: {s: [] for s in STEPS} for k in dst_cells}
    for P_b in replicate_matrices(start, end, table["drifter_id"], n_states, fit, N_BOOT_REACH):
        for k, cells in dst_cells.items():
            r = reached(S, P_b, cells)
            for s in STEPS:
                reps_reach[k][s].append(r[s])
    print("bootstrap done")

    rows = []
    def add(source, quantity, destination, step, value, low, high, extra=""):
        rows.append({"source": source, "quantity": quantity, "destination": destination, "days": step * TAU_DAYS if step else "",
                     "estimate": round(float(value), 4), "ci_low": round(float(low), 4) if low is not None else "",
                     "ci_high": round(float(high), 4) if high is not None else "", "note": extra})

    for s in STEPS:
        lo, hi = _ci(reps_in[s])
        for i, sname in enumerate(src_names):
            for j, dname in enumerate(dst_names):
                add(sname, "pct_in_destination", dname, s, 100 * est_in[s][i, j], 100 * lo[i, j], 100 * hi[i, j])
        lo_e, hi_e = _ci(reps_edge[s])
        for i, sname in enumerate(src_names):
            for j, ename in enumerate(EDGES):
                add(sname, "pct_left_through_edge", ename, s, 100 * est_edge[s][i, j], 100 * lo_e[i, j], 100 * hi_e[i, j])
        for k, dname in enumerate(names):
            lo_r, hi_r = _ci(reps_reach[k][s])
            for i, (sname, cells) in enumerate(src):
                if np.all(label[cells] == k):  # it starts there, so "reaching" it says nothing
                    continue
                add(sname, "pct_reached_by_then", dname, s, 100 * est_reach[k][s][i], 100 * lo_r[i], 100 * hi_r[i],
                    "lower bound where the destination is coastal" if dname == COAST_NAME else "")
    for k, dname in enumerate(names):
        for i, (sname, cells) in enumerate(src):
            if np.all(label[cells] == k):
                continue
            for lev, level in enumerate(ARRIVAL_LEVELS):
                v = arrivals[k][i, lev]
                if np.isfinite(v):
                    rows.append({"source": sname, "quantity": f"first_day_{100 * level:.0f}pct_reached", "destination": dname,
                                 "days": "", "estimate": v, "ci_low": "", "ci_high": "", "note": "point estimate"})
    lr_lo, lr_hi = _ci(reps_lr)
    for j, ename in enumerate(EDGES):
        add("long run (QSD)", "pct_lost_per_step_through_edge", ename, 0, est_lr_edge[j], lr_lo[j], lr_hi[j])
    add("long run (QSD)", "pct_lost_per_step_total_including_cells_without_drifters", "", 0, est_lr_total, lr_lo[-1], lr_hi[-1])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"connectivity_tau{TAU_DAYS}d.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    tables = [(f"{s * TAU_DAYS} days", 100 * est_in[s]) for s in STEPS]
    plot_matrices(src_names, dst_names, tables, os.path.join(FIGURES_DIR, "connectivity.png"))

    for box in region.get("named_releases", []):
        i = src_names.index(box["name"])
        release = src[i][1]
        x = np.zeros(n_states)
        x[release] = 1.0 / release.size
        dists, stats = {}, {}
        lo_all = {s: _ci(reps_in[s]) for s in STEPS}
        for step in range(1, max(STEPS) + 1):
            x = x @ P
            if step in STEPS:
                dists[step] = x.copy()
                j = dst_names.index("Left the box")
                j2 = dst_names.index(COAST_NAME)
                lo, hi = lo_all[step]
                stats[step] = (f"left box: {_pct(100 * est_in[step][i, j])}% ({_pct(100 * lo[i, j])}-{_pct(100 * hi[i, j])})\n"
                               f"in Benguela coastal cells: {_pct(100 * est_in[step][i, j2])}% ({_pct(100 * lo[i, j2])}-{_pct(100 * hi[i, j2])})")
        plot_release(region, fine, box["name"], dists, release, drifters, stats,
                     os.path.join(FIGURES_DIR, f"{box['name'].lower().replace(' ', '_')}_release.png"))

    for sname in (COAST_NAME, "Orange Basin"):
        if sname in src_names:
            i = src_names.index(sname)
            print(f"\n{sname}:")
            for s in STEPS:
                top = sorted(zip(dst_names, 100 * est_in[s][i]), key=lambda x: -x[1])[:4]
                print(f"  after {s * TAU_DAYS} days: " + ", ".join(f"{n} {v:.2f}%" for n, v in top))
    print("\nLong run: lost per 15-day step " + f"{est_lr_total:.3f}%, through " + ", ".join(f"{e} {v:.3f}%" for e, v in zip(EDGES, est_lr_edge)))
    print(f"Saved {out} and the figures")
