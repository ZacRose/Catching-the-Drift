"""
robustness.py

Related task: Week 8 Quantitative - robustness: "rerun across grid sizes (1/2/4 deg) and lags; drogue
sensitivity (all drifters vs undrogued only at 4 deg, decision 6a, plus drogued vs undrogued where both
have data); years sensitivity (2010+ only, a filter on start_time)". The report needs results across
several settings, not one.

Each setting changes ONE thing from the base (the product: 2 deg shrinkage to 4 deg, tau = 15 days, all
drifters, all years), or, for the 4 deg settings, from the 4 deg naive baseline, and refits exactly as
that was fitted:
- alpha chosen on the train -> valid drifters of that setting (evaluate.tune_alpha), the model refitted
  on all of that setting's drifters; the same drifter split (drifter_split.csv) everywhere;
- naive P where a setting is the 4 deg baseline.
Grids: 1 deg (shrunk towards 2 deg, the same nesting idea one level down), 2 deg, 4 deg naive.
Lags: 5, 10, 15, 30, 60 and 90 days at 2 deg (Week 7 Computational tables).
Drogue: undrogued only (2 deg and 4 deg), drogued only (4 deg; under 10% of cells adequate, decision 6a,
so read as indicative), each judged on the drifter's drogue status at the START of the move (Week 4
issue e: no end-of-move status). Years: starts from 2010 on (2 and 4 deg).

Headline numbers per setting, as in the Week 6 and 7 work:
- adequacy: % of cells with drifters (any setting) that have 100+ drifters in this setting;
- held-out: mean test log-likelihood per move of the setting's model, and the share of test moves naive
  P gives probability 0 (comparable only within one grid, lag and drifter subset; see evaluate.py);
- coastal release (coastal_release.py) after 30, 90 and 360 days: % in the coastal zone, % left the box,
  centre. A horizon that isn't a whole number of steps (e.g. 30 days at tau = 60) is left blank;
- long run (long_run.py): expected years in the box, QSD centre, years to forget the coastal start, and
  how far the QSD is from the base's (total variation after adding both up to 4 deg cells, so every
  grid can be compared on the same cells).
Intervals: N_BOOT_ROBUST drifter bootstrap replicates per setting (for naive P, over the replicates
without a trap, as in long_run.py; pct_replicates_trapped says how many were set aside) (fewer than Week 6's 1000, to keep
the run to minutes; enough for a 95% interval to judge whether a setting moves a number by more than
its own noise). The 1 deg setting gets N_BOOT_FINE replicates only, as each takes about 9 seconds.

How to read it: a setting matters if its number falls outside the base's interval (and vice versa).
Differences between lags are partly the Markov question (markov_check.py) and partly that long-lag
matrices rest on longer-lived drifters (Week 7 Computational).

Limitations:
- the coastal cells at 1 deg come from the 1:110m coastline (the 1:10m file isn't in the repo yet;
  see Week 6 Computational), so the 1 deg coastal numbers are indicative;
- the 1 deg model rests almost entirely on its 2 deg prior (2.6% of 1 deg cells adequate at best).

Outputs: outputs/results/robustness_tau15d.csv and outputs/figures/robustness.png.

Run: python3 -m src.inference.robustness   (about 20 minutes; needs every transitions_tau*d.npz,
drifter_split.csv and shrinkage_P_2deg_tau15d.npz)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, table_path
from src.inference.adequacy import N_MIN
from src.inference.bootstrap import LEVEL, SEED, percentile_interval, replicate_matrices
from src.inference.estimators import count_matrix, log_lik, naive_P, prior_rows, shrinkage_P
from src.inference.evaluate import states_for, tune_alpha
from src.inference.split import load_split
from src.inference.transition_data import TAU_DAYS, load_transitions, parent_states, subset, to_states
from src.linalg.coastal_release import release_summary
from src.linalg.long_run import TRAP_TOL, cell_areas_km2, qsd_power, steps_to_mix, substochastic

N_BOOT_ROBUST = 200
N_BOOT_FINE = 50  # the 1 deg grid (about 3,000 states) takes ~9 s per replicate
HORIZONS = (30, 90, 360)  # days: one month, about 3 months, about a year
YEAR = np.datetime64("2010-01-01")
NUMBERS = [  # (column, label) of the numbers that get intervals and go in the figure
    ("pct_coastal_30d", "% in coastal zone after 30 days"),
    ("pct_left_360d", "% left the box after 360 days"),
    ("centre_lon_360d", "centre after 360 days (lon, deg)"),
    ("years_in_box", "expected years in the box (long run)"),
    ("qsd_centre_lon", "long-run centre (lon, deg)"),
    ("years_tmix_coast", "years to forget the coastal start"),
]


def settings():
    """(group, name, tau, cell_size, model, filter) for every setting; the first is the base."""
    base = ("base", "2 deg shrinkage, 15 d, all", TAU_DAYS, 2.0, "shrinkage", "all")
    out = [base,
           ("grid", "1 deg shrinkage (to 2 deg)", TAU_DAYS, 1.0, "shrinkage", "all"),
           ("grid", "4 deg naive", TAU_DAYS, 4.0, "naive", "all")]
    out += [("lag", f"2 deg shrinkage, {t} d", t, 2.0, "shrinkage", "all") for t in (5, 10, 30, 60, 90)]
    out += [("drogue", "2 deg shrinkage, undrogued only", TAU_DAYS, 2.0, "shrinkage", "undrogued"),
            ("drogue", "4 deg naive, undrogued only", TAU_DAYS, 4.0, "naive", "undrogued"),
            ("drogue", "4 deg naive, drogued only", TAU_DAYS, 4.0, "naive", "drogued"),
            ("years", "2 deg shrinkage, 2010+", TAU_DAYS, 2.0, "shrinkage", "2010+"),
            ("years", "4 deg naive, 2010+", TAU_DAYS, 4.0, "naive", "2010+")]
    return out


def apply_filter(table, name):
    """The rows of a transition table in one drifter subset (drogue status at the start, or years)."""
    if name == "all":
        return table
    if name == "undrogued":
        return subset(table, ~table["drogued"])
    if name == "drogued":
        return subset(table, table["drogued"])
    if name == "2010+":
        return subset(table, table["start_time"] >= YEAR)
    raise ValueError(f"unknown filter {name!r}")


def coarse_of(cell_size):
    """The prior grid for shrinkage at this cell size: one level up (1 -> 2, 2 -> 4)."""
    return 2.0 if cell_size == 1.0 else BASELINE_CELL_SIZE


def make_fit(model, region, cell_size, alpha=None):
    """Count matrix -> P for one setting."""
    if model == "naive":
        return naive_P
    parent = parent_states(region, cell_size, coarse_of(cell_size))
    return lambda counts: shrinkage_P(counts, prior_rows(counts, parent), alpha)


def to_coarse(q, keep, region, cell_size, coarse=BASELINE_CELL_SIZE):
    """A distribution over kept cells, added up into coarse cells (all coarse cells, in order)."""
    if cell_size == coarse:
        out = np.zeros((parent_states(region, coarse, coarse).size - 1))
        np.add.at(out, keep, q)
        return out
    parent = parent_states(region, cell_size, coarse)
    out = np.zeros(parent.max())
    np.add.at(out, parent[keep], q)
    return out


def headline(P, tau, keep, release, zone, no_data, area, region, cell_size, q_start=None):
    """Coastal release and long-run numbers for one P (cheap enough for the bootstrap)."""
    out = {}
    x = np.zeros(P.shape[0])
    x[release] = 1.0 / release.size
    step, dists = 0, {}
    for h in HORIZONS:
        if h % tau:
            continue
        while step < h // tau:
            x = x @ P
            step += 1
        dists[h] = x.copy()
    for h in HORIZONS:
        if h in dists:
            r = release_summary(dists[h], zone, no_data, region, cell_size)
            out[f"pct_coastal_{h}d"], out[f"pct_left_{h}d"] = r["pct_in_coastal_zone"], r["pct_left_R"]
            out[f"centre_lat_{h}d"], out[f"centre_lon_{h}d"] = r["centre_lat"], r["centre_lon"]
        else:
            for k in ("pct_coastal", "pct_left", "centre_lat", "centre_lon"):
                out[f"{k}_{h}d"] = float("nan")
    Q = substochastic(P, keep)
    lam1, q = qsd_power(Q, q_start)
    out["lambda_1"] = lam1
    out["years_in_box"] = tau / 365.25 / (1 - lam1)
    lat_edges, lon_edges = grid_edges(region, cell_size)
    row, col = np.divmod(keep, len(lon_edges) - 1)
    out["qsd_centre_lat"] = float((q * 0.5 * (lat_edges[:-1] + lat_edges[1:])[row]).sum())
    out["qsd_centre_lon"] = float((q * 0.5 * (lon_edges[:-1] + lon_edges[1:])[col]).sum())
    x0 = np.isin(keep, release).astype(float)
    out["years_tmix_coast"] = steps_to_mix(Q, q, x0 / x0.sum()) * tau / 365.25 if x0.any() else float("nan")
    return out, q


def run_setting(setting, tables, region, assignment, ocean, paths, n_boot=N_BOOT_ROBUST):
    """Fit one setting and return (row of numbers, bootstrap replicates, coarse QSD)."""
    group, name, tau, size, model, filt = setting
    table = apply_filter(tables[tau], filt)
    start, end, n_cells = to_states(table, region, size)
    n_states = n_cells + 1
    _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)

    # alpha from train -> valid; held-out score from train + valid -> test (as evaluate.py)
    s_fit, e_fit, _, _ = states_for(table, region, size, assignment, ("train", "valid"))
    s_te, e_te, _, _ = states_for(table, region, size, assignment, "test")
    alpha = None
    if model == "shrinkage":
        s_tr, e_tr, _, _ = states_for(table, region, size, assignment, "train")
        s_va, e_va, _, _ = states_for(table, region, size, assignment, "valid")
        alpha, _ = tune_alpha(count_matrix(s_tr, e_tr, n_states), parent_states(region, size, coarse_of(size)), s_va, e_va)
    fit = make_fit(model, region, size, alpha)
    C_fit = count_matrix(s_fit, e_fit, n_states)
    test_ll = log_lik(fit(C_fit), s_te, e_te)
    naive_ll = log_lik(naive_P(C_fit), s_te, e_te)

    P = fit(count_matrix(start, end, n_states))
    keep = np.flatnonzero(drifters > 0)
    zone = coastal_cells(region, size, land_fraction(region, size, paths))
    release = zone[drifters[zone] > 0]
    area = cell_areas_km2(region, size)
    est, q = headline(P, tau, keep, release, zone, drifters == 0, area, region, size)
    row = {"group": group, "setting": name, "tau_days": tau, "cell_size_deg": size, "model": model, "drifters": filt,
           "n_drifters": int(np.unique(table["drifter_id"]).size), "n_moves": int(start.size),
           "pct_cells_adequate": round(float(100 * np.mean(drifters[ocean[size]] >= N_MIN)), 1),
           "alpha": round(alpha, 2) if alpha is not None else "",
           "test_loglik": float(test_ll.mean()) if np.isfinite(test_ll).all() else float("-inf"),
           "naive_pct_test_zero_prob": round(float(100 * np.mean(~np.isfinite(naive_ll))), 2),
           "release_cells": int(release.size), **est}

    reps = {k: np.empty(n_boot) for k, _ in NUMBERS + [("lambda_1", "")]}
    for b, P_b in enumerate(replicate_matrices(start, end, table["drifter_id"], n_states, fit, n_boot, SEED)):
        out, _ = headline(P_b, tau, keep, release, zone, drifters == 0, area, region, size, q_start=q)
        for k in reps:
            reps[k][b] = out[k]
    return row, reps, to_coarse(q, keep, region, size)


def plot(rows, out_path):
    """One panel per headline number: every setting's estimate and 95% interval, the base shaded."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained", sharey=True)
    names = [r["setting"] for r in rows]
    y = np.arange(len(rows))[::-1]
    colours = {"base": "#000000", "grid": "#1f77b4", "lag": "#ff7f0e", "drogue": "#2ca02c", "years": "#9467bd"}
    for ax, (col, label) in zip(axes.ravel(), NUMBERS):
        base = rows[0]
        if np.isfinite(base[f"{col}_ci_low"]):
            ax.axvspan(base[f"{col}_ci_low"], base[f"{col}_ci_high"], color="grey", alpha=0.2, lw=0)
        for yi, r in zip(y, rows):
            v, lo, hi = r[col], r[f"{col}_ci_low"], r[f"{col}_ci_high"]
            if not np.isfinite(v):
                ax.text(0.02, yi, "n/a", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color="grey")
                continue
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]] if np.isfinite(lo) else None, fmt="o", ms=4,
                        color=colours[r["group"]], capsize=2, lw=1)
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(axis="x", lw=0.3)
    axes[0, 0].set_yticks(y, names, fontsize=8)
    axes[1, 0].set_yticks(y, names, fontsize=8)
    fig.suptitle("Week 8 robustness: does the answer change with the grid, the time step, drogue status or years? "
                 "(grey band: the product's 95% interval)", fontsize=11)
    fig.text(0.01, -0.03,
             f"Each setting changes one thing from the product (2° shrinkage, {TAU_DAYS}-day step, all drifters, all years; 4° settings: from the 4° naive baseline) "
             f"and is refitted the same way; bars are 95% intervals from {N_BOOT_ROBUST} drifter bootstrap replicates ({N_BOOT_FINE} at 1°).\n"
             "Release: evenly over the Benguela coastal cells with drifters. 'n/a': the horizon isn't a whole number of "
             "steps at that lag (e.g. 30 days at 60 or 90). Drogue status is at the start of each move.", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    paths = land_paths()
    lags = sorted({s[2] for s in settings()})
    tables = {t: load_transitions(region, table_path(t), verbose=False) for t in lags}
    assignment = load_split(tables[TAU_DAYS]["drifter_id"])
    ocean = {}
    for size in sorted({s[3] for s in settings()}):  # cells any drifter starts a 15-day move in
        s, _, n = to_states(tables[TAU_DAYS], region, size)
        ocean[size] = per_cell_counts(s, tables[TAU_DAYS]["drifter_id"], n)[1] > 0

    rows, base_q = [], None
    for setting in settings():
        n_boot = N_BOOT_FINE if setting[3] < 2 else N_BOOT_ROBUST
        row, reps, q4 = run_setting(setting, tables, region, assignment, ocean, paths, n_boot)
        row["n_boot"] = n_boot
        if setting[0] == "base":
            base_q = q4
            product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{setting[3]:g}deg_tau{TAU_DAYS}d.npz"))
            if abs(row["alpha"] - float(product["alpha"])) > 0.01:
                raise ValueError("The base setting should reproduce the product's alpha; check drifter_split.csv.")
        row["qsd_tv_vs_base_at_4deg"] = float(0.5 * np.abs(q4 - base_q).sum())
        # Naive P: a replicate that draws only a thin cell's drifters that never left it makes the cell a
        # trap (lambda_1 = 1; long_run.py). Report how often, and give intervals over the other replicates.
        trapped = reps["lambda_1"] >= 1 - TRAP_TOL
        row["pct_replicates_trapped"] = round(float(100 * trapped.mean()), 1)
        for k, _ in NUMBERS:
            ok = np.isfinite(reps[k]) & ~trapped
            low, high = percentile_interval(reps[k][ok], LEVEL) if ok.sum() >= 0.9 * (~trapped).sum() else (np.nan, np.nan)
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = float(low), float(high)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
               if not k.endswith(("_ci_low", "_ci_high"))})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"robustness_tau{TAU_DAYS}d.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_out = os.path.join(FIGURES_DIR, "robustness.png")
    plot(rows, fig_out)
    print(f"Saved {out} and {fig_out}")
