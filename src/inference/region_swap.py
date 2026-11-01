"""
region_swap.py

Related task: Week 8 Quantitative - "run the code unchanged on a different region ('nothing depends on
R')", objective O4, for the quantitative half of the pipeline.

A truly new region needs a new ~9 minute extract from the cloud store (Computational, decision 4).
But a different box INSIDE our region can be made exactly from the tables we have, because a
transition's outcome only depends on where the drifter is at start + tau:
- starts outside the new box are dropped;
- "stayed in R" moves that end outside the new box become "left the box" (the drifter was still
  reporting, just not inside it: the same rule as decision 8);
- "left R", "record ended" and "gap" keep their codes (outside R is outside the new box too).
So the derived table is what src/gridding/transitions.py would build for the new box, without
rebuilding anything.

The test box: the Brazil-Malvinas Confluence, 42S-20S, 60W-30W, with its own coastal zone (the
Brazil / Uruguay shelf, 34S-22S west of 40W). Its edges don't line up with the 4 deg cells of the main
box, and its height (22 deg) isn't a whole number of 4 deg cells, so it also checks the grid overhang.
Every quant step then runs on it through the same functions, given the new box instead of
config/region.yaml: the adequacy count, the drifter split's held-out evaluation with every model
(evaluate.evaluate_grid), alpha and the 2 deg shrinkage fit, the drifter bootstrap of P, the coastal
release, the Markov check's held-out score at 30 days, and the long-run pattern.

Passing means: everything runs with no code change, the outputs pass the same internal checks as the
main box (rows sum to 1, no zero probabilities in the shrinkage P, QSD from eigenvectors = QSD from
power iteration), and the numbers make physical sense (more material leaves a box that is mostly
open boundary). The results are about this test box, not findings for the report.

Outputs: outputs/results/region_swap_tau15d.csv and outputs/figures/region_swap.png.

Run: python3 -m src.inference.region_swap   (about a minute)
"""
import copy
import csv
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from src.common.config import FIGURES_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.trajectory_map import _degrees, draw_land
from src.gridding.transitions import BASELINE_CELL_SIZE, IN_R, LEFT_R, table_path
from src.ingest.load import in_box
from src.inference.adequacy import N_MIN
from src.inference.bootstrap import matrix_intervals
from src.inference.estimators import count_matrix, prior_rows, shrinkage_P
from src.inference.evaluate import evaluate_grid, states_for, tune_alpha
from src.inference.markov_check import heldout_rows, tune_alphas
from src.inference.split import load_split
from src.inference.transition_data import TAU_DAYS, load_transitions, parent_states, to_states
from src.linalg.coastal_release import release_summary
from src.linalg.long_run import cell_areas_km2, density_ratio, qsd_power, substochastic, summarise
from src.linalg.release_map import propagate

TEST_BOX = {
    "name": "brazil_malvinas", "label": "Brazil-Malvinas Confluence (test box)",
    "lat_min": -42, "lat_max": -20, "lon_min": -60, "lon_max": -30, "cell_size_deg": 2.0,
    "coastal_zone": {"label": "Brazil / Uruguay shelf", "lat_min": -34, "lat_max": -22, "lon_min": -60, "lon_max": -40},
}
N_BOOT_SWAP = 100


def sub_box_table(stored, box):
    """A saved transition table (dict of arrays, every row) re-labelled for a smaller box inside R."""
    if not (box["lat_min"] >= -90 and box["lat_min"] < box["lat_max"] and box["lon_min"] < box["lon_max"]):
        raise ValueError("bad box")
    keep = in_box(stored["start_lon"], stored["start_lat"], box)
    out = {k: (v[keep].copy() if np.ndim(v) else v) for k, v in stored.items()}
    stayed = out["outcome"] == IN_R
    inside = np.zeros(stayed.size, dtype=bool)
    inside[stayed] = in_box(out["end_lon"][stayed], out["end_lat"][stayed], box)
    left = stayed & ~inside
    out["outcome"][left] = LEFT_R
    out["end_lon"][left], out["end_lat"][left] = np.nan, np.nan  # as transitions.py stores "left R" rows
    return out


def save_sub_table(tau_days, box, folder):
    """Write the re-labelled table for one lag and return its path."""
    stored = dict(np.load(table_path(tau_days)))
    path = os.path.join(folder, f"transitions_tau{tau_days}d.npz")
    np.savez(path, **sub_box_table(stored, box))
    return path


if __name__ == "__main__":
    main = load_region_bounds()
    box = copy.deepcopy(TEST_BOX)
    if not (box["lat_min"] >= main["lat_min"] and box["lat_max"] <= main["lat_max"]
            and box["lon_min"] >= main["lon_min"] and box["lon_max"] <= main["lon_max"]):
        raise ValueError("The test box must lie inside the main region, or its table can't be derived.")
    fine, coarse = box["cell_size_deg"], BASELINE_CELL_SIZE
    rows = []

    def record(step, quantity, value, check=""):
        rows.append({"step": step, "quantity": quantity, "value": value, "check": check})
        print(f"{step}: {quantity} = {value} {check}")

    with tempfile.TemporaryDirectory() as tmp:
        paths = {t: save_sub_table(t, box, tmp) for t in (TAU_DAYS, 2 * TAU_DAYS)}
        table = load_transitions(box, paths[TAU_DAYS], verbose=False)
        # Same starts in the same order at both lags (the filter is on start position), so match by row
        known = {t: np.isin(np.load(p)["outcome"], [IN_R, LEFT_R]) for t, p in paths.items()}
        both = known[TAU_DAYS] & known[2 * TAU_DAYS]
        matched = {t: load_transitions(box, p, verbose=False, rows=both) for t, p in paths.items()}
    assignment = load_split(table["drifter_id"])
    record("table", "usable moves", int(table["outcome"].size))
    record("table", "drifters", int(np.unique(table["drifter_id"]).size))
    record("table", "% of moves leaving the box", round(float(100 * table["exited"].mean()), 2))

    # Adequacy (Week 1 rule, on transitions)
    start, end, n_cells = to_states(table, box, fine)
    _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)
    sampled = drifters > 0
    record("adequacy", f"% of {fine:g} deg cells with drifters that have {N_MIN}+", round(float(100 * np.mean(drifters[sampled] >= N_MIN)), 1))

    # Held-out evaluation, every model (Week 3/6), on both grids
    for size, c in ((coarse, None), (fine, coarse)):
        results, _ = evaluate_grid(table, box, assignment, size, c)
        for r in results:
            if r["cells"] == "all":
                record("evaluate", f"{r['grid']} {r['model']} ({r['setting']}): mean held-out log-lik",
                       round(r["mean_loglik"], 4) if np.isfinite(r["mean_loglik"]) else "-inf",
                       f"{r['pct_zero_prob']}% of test moves given 0")

    # Shrinkage product for the test box (Week 6)
    parent = parent_states(box, fine, coarse)
    s_tr, e_tr, _, _ = states_for(table, box, fine, assignment, "train")
    s_va, e_va, _, _ = states_for(table, box, fine, assignment, "valid")
    alpha, _ = tune_alpha(count_matrix(s_tr, e_tr, n_cells + 1), parent, s_va, e_va)
    counts = count_matrix(start, end, n_cells + 1)
    fit = lambda C: shrinkage_P(C, prior_rows(C, parent), alpha)  # noqa: E731
    P = fit(counts)
    ok = np.allclose(P.sum(axis=1), 1) and P[:-1].min() > 0 and P[-1, -1] == 1  # exit row is absorbing by design
    record("shrinkage", "alpha (chosen on validation)", round(alpha, 2))
    record("shrinkage", "rows sum to 1, no zeros in cell rows, exit absorbing", bool(ok), "PASS" if ok else "FAIL")

    # Bootstrap of P (Week 6)
    result = matrix_intervals(start, end, table["drifter_id"], n_cells + 1, fit, P, n_boot=N_BOOT_SWAP)
    tv = np.nanmedian(result["row_tv"], axis=0)[:n_cells]
    record("bootstrap", f"median row total variation, cells with {N_MIN}+ / under {N_MIN} drifters",
           f"{np.median(tv[drifters >= N_MIN]):.3f} / {np.median(tv[sampled & (drifters < N_MIN)]):.3f}",
           f"{N_BOOT_SWAP} replicates")

    # Coastal release (Week 6)
    zone = coastal_cells(box, fine, land_fraction(box, fine, land_paths()))
    release = zone[drifters[zone] > 0]
    record("coastal release", "coastal cells / with drifters", f"{zone.size} / {release.size}")
    dists = propagate(P, release)
    for step, d in dists.items():
        r = release_summary(d, zone, drifters == 0, box, fine)
        record("coastal release", f"after {step * TAU_DAYS} days: % left box, % in coastal zone, centre",
               f"{r['pct_left_R']:.2f}, {r['pct_in_coastal_zone']:.2f}, ({r['centre_lat']:.1f}, {r['centre_lon']:.1f})")

    # Markov check held-out score at 30 days (Week 7), on starts known at both lags
    alphas = tune_alphas(matched, box, assignment, fine)
    m = heldout_rows(matched, box, assignment, fine, "shrinkage", alphas, 2 * TAU_DAYS, 2)[0]
    record("markov check", "30 d held-out, direct minus chained per move",
           f"{m['direct_minus_chained']:.3f} ({m['ci_low']:.3f} to {m['ci_high']:.3f})")

    # Long run (Week 7)
    keep = np.flatnonzero(sampled)
    area = cell_areas_km2(box, fine)
    est, q = summarise(P, keep, area, release, box, fine)
    lam_p, q_p = qsd_power(substochastic(P, keep))
    agree = abs(lam_p - est["lambda_1"]) < 1e-9 and np.abs(q_p - q).sum() < 1e-6
    record("long run", "QSD by eigenvectors = by power iteration", bool(agree), "PASS" if agree else "FAIL")
    for k in ("years_in_box_expected", "years_tmix_from_coast", "qsd_centre_lat", "qsd_centre_lon"):
        record("long run", k, round(est[k], 3))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"region_swap_tau{TAU_DAYS}d.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Figure: the test box inside the main box, its drifters per cell and its long-run pattern
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    lat_edges, lon_edges = grid_edges(box, fine)
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    dens = np.full(n_cells, np.nan)
    dens[keep] = density_ratio(q, area[keep])
    panels = ((np.ma.masked_equal(drifters, 0), colors.LogNorm(1, max(drifters.max(), 2)), "viridis",
               "drifters starting a 15-day move"),
              (np.ma.masked_invalid(dens), colors.LogNorm(0.1, 10), "viridis", "long-run density, times the box average"))
    for ax, (values, norm, cmap, label) in zip(axes, panels):
        mesh = ax.pcolormesh(lon_edges, lat_edges, values.reshape(shape), norm=norm, cmap=cmap)
        draw_land(ax)
        for c in zone:
            r_, k_ = divmod(int(c), shape[1])
            ax.add_patch(plt.Rectangle((lon_edges[k_], lat_edges[r_]), fine, fine, fill=False, ec="#d62728", lw=0.9))
        ax.set_xlim(box["lon_min"], box["lon_max"])
        ax.set_ylim(box["lat_min"], box["lat_max"])
        ax.set_aspect("equal")
        ax.set_facecolor("#d9d9d9")
        ax.xaxis.set_major_formatter(_degrees("lon"))
        ax.yaxis.set_major_formatter(_degrees("lat"))
        fig.colorbar(mesh, ax=ax, shrink=0.8, label=label)
    axes[0].set_title(f"(a) {box['label']}: {fine:g}° cells", fontsize=10, loc="left")
    axes[1].set_title("(b) Long-run pattern, same code as the main box", fontsize=10, loc="left")
    fig.text(0.01, -0.06, "Region-swap check (Week 8): the quantitative pipeline run unchanged on a different box, built from "
             "the same transition tables (moves ending outside the box count as leaving it).\nRed outlines: the test box's "
             "coastal zone (release cells). These are test results, not findings about the region.", fontsize=8.5)
    fig_out = os.path.join(FIGURES_DIR, "region_swap.png")
    fig.savefig(fig_out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out} and {fig_out}")
