"""
markov_check.py

Related task: Week 7 Quantitative - test the Markov assumption by comparing P(tau)^2 with
P(2 tau) (the Chapman-Kolmogorov check, proposal section 4).

Why it matters: every P^n map (3 months = P^6, 1 year = P^24) assumes that where material goes
next depends only on the cell it is in now, not on how it got there. If that holds, two 15-day
steps should forecast 30 days as well as a matrix fitted directly to 30-day moves.

1. The 30-day moves. Computational's Week 7 task is to rebuild the transition table at other lags
   (that needs the per-drifter record cache). Until then, 30-day moves are built here by following
   each drifter through two consecutive 15-day rows of the committed table. Every row starts at
   00:00 UTC, so a drifter inside R at day 15 always has its own row starting then, and that row's
   destination is the drifter's position at day 30. This is exact for drifters still in R at day 15
   (checked in the code: the first row's end position equals the second row's start position).
   Drifters that left R by day 15 need a convention:
     - "absorbing" (used for the Markov check): they count as "left R" at day 30, which is exactly
       what P^2 assumes ("left R" is absorbing in every P, decision 11b). This way the check tests
       the Markov property alone, not the exit convention.
     - "observed": they count as in R if they have a row starting at day 30 (i.e. came back), and
       as "left R" otherwise. This is what a rebuilt 30-day table would show, except that a drifter
       that left R and then died before day 30 would be called "record ended" there and "left R"
       here (it can't be told apart without the record cache).
   Comparing the two conventions measures re-entry: how much the absorbing exit rule overstates
   the share of material that leaves the box (decision 11b).
   If data/processed/transitions_tau30d.npz exists (Computational's rebuild), it is used for the
   "observed" counts instead of the chained version.

2. Naive P on both grids (4 deg, the baseline, and 2 deg): for every cell with at least N_MIN
   drifters starting a 30-day move, compare row i of P(15)^2 with row i of P(30): total variation
   distance (half the sum of absolute differences: 0 = identical rows, 1 = no overlap), the chance
   of being in the same cell, and the chance of having left R. Noise reference, since there are no
   bootstrap intervals yet: drifters are split at random into two halves A and B, and
       Markov gap  = TV( P(15)^2 from A , P(30) from B )
       noise only  = TV( P(30) from A ,   P(30) from B )
   (and the same with A and B swapped). Both compare estimates from independent drifters, so if
   the Markov gap is no bigger than the noise-only distance, the data can't tell P(15)^2 from P(30).
   P(15)^2 is a little smoother than P(30) (it rests on twice as many moves), which if anything
   makes the Markov gap look smaller, so a gap above the noise is a real difference. The
   reference is still rough; the drifter bootstrap will replace it. Row entropy (how spread out a
   row is) shows the direction of any difference: two steps on coarse cells forget where in the
   cell a drifter was, so P(15)^2 tends to spread material too widely ("artificial dispersion",
   one of the proposal's stated limitations).

3. 2 deg, shrinkage (the product): held-out log-likelihood of the test drifters' 30-day moves,
   using the Week 3 split. "Direct" is shrinkage fitted to 30-day moves (alpha chosen on the
   validation drifters' 30-day moves); "Markov" is the 15-day shrinkage model (alpha chosen exactly
   as in evaluate.py) squared; the neighbour-only baseline fitted to 30-day moves gives the scale.
   Both models are fitted on train + valid, as in evaluate.py. Scores are split by cells under and
   over N_MIN drifters, as in Week 6.

Outputs: outputs/results/markov_check_tau15d.csv, outputs/results/markov_check_scores_tau15d.csv,
outputs/figures/markov_check.png.

Run: python3 -m src.inference.markov_check   (needs outputs/results/drifter_split.csv)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, ENDED, GAP, IN_R, LEFT_R
from src.inference.adequacy import N_MIN
from src.inference.estimators import count_matrix, naive_P, neighbour_P, prior_rows, shrinkage_P, log_lik, score
from src.inference.evaluate import seen_cells, states_for, tune_alpha, tune_neighbour
from src.inference.split import load_split
from src.inference.transition_data import (
    TAU_DAYS, load_raw_transitions, parent_states, subset, to_states, usable_transitions,
)

STEPS = 2  # compare P(tau)^STEPS with P(STEPS x tau)
HALF_SPLIT_SEED = 7
# 30-day moves leave their cell and go beyond its neighbours more often than 15-day ones, so the
# neighbour baseline's search runs to lower stay and higher leak values
NEIGHBOUR_STAYS_2TAU = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
NEIGHBOUR_LEAKS_2TAU = [0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
REBUILT_PATH = os.path.join(PROCESSED_DIR, f"transitions_tau{STEPS * TAU_DAYS}d.npz")


def _row_lookup(raw):
    """A function (row indices, days ahead) -> (found, index of that drifter's row starting then)."""
    hours = raw["start_time"].astype("datetime64[h]").astype(np.int64)
    if np.any(hours % 24):
        raise ValueError("Chaining needs every row to start at 00:00 UTC (transitions.py does this).")
    _, rank = np.unique(raw["drifter_id"], return_inverse=True)
    keys = (rank.astype(np.int64) << 32) | hours
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    def lookup(rows, days):
        target = keys[rows] + 24 * days
        pos = np.minimum(np.searchsorted(sorted_keys, target), sorted_keys.size - 1)
        return sorted_keys[pos] == target, order[pos]
    return lookup


def chain_transitions(raw, tau_days=TAU_DAYS, steps=STEPS, reentry="absorbing"):
    """Moves over steps x tau days, built by following each drifter through consecutive tau-day rows.

    Inputs:
        raw: a raw transition table (all outcomes), e.g. from load_raw_transitions().
        tau_days: the table's lag.
        steps: how many rows to chain.
        reentry: "absorbing" - a drifter that left R at an earlier step counts as "left R" at the end
                 (what P^steps assumes); "observed" - it counts as in R if it has a row starting at
                 the final time (it came back), otherwise as "left R".
    Returns:
        a raw table (same columns) with one row per original start; outcome and end position refer
        to start + steps x tau. Rows unknown at any step are ENDED or GAP as usual.
    """
    if reentry not in ("absorbing", "observed"):
        raise ValueError(f"reentry must be 'absorbing' or 'observed', not {reentry!r}")
    lookup = _row_lookup(raw)
    rows = np.arange(raw["outcome"].size)
    outcome = raw["outcome"].copy()
    end_lon, end_lat = raw["end_lon"].copy(), raw["end_lat"].copy()

    for k in range(1, steps):
        moving = outcome == IN_R  # still in R after k steps: follow its next row
        found, nxt = lookup(rows[moving], k * tau_days)
        if not found.all():
            raise ValueError(f"{np.sum(~found)} drifters were in R at day {k * tau_days} but have no row "
                             "starting then - the table isn't one row per drifter-day.")
        idx = np.flatnonzero(moving)
        outcome[idx] = raw["outcome"][nxt]
        end_lon[idx], end_lat[idx] = raw["end_lon"][nxt], raw["end_lat"][nxt]

    # Out of R (or in a gap) at an earlier step: look for a row at the final time
    if steps > 1:
        found, back = lookup(rows, steps * tau_days)
        came_back = found & ((outcome == GAP) | ((outcome == LEFT_R) & (reentry == "observed")))
        outcome[came_back] = IN_R
        end_lon[came_back], end_lat[came_back] = raw["start_lon"][back[came_back]], raw["start_lat"][back[came_back]]
    return {**raw, "outcome": outcome, "end_lon": end_lon, "end_lat": end_lat}


def reentry_summary(raw, tau_days=TAU_DAYS, steps=STEPS):
    """How many drifters that had left R after one step are back inside R at the final time."""
    lookup = _row_lookup(raw)
    left = np.flatnonzero(raw["outcome"] == LEFT_R)
    back, _ = lookup(left, steps * tau_days)
    return {"left_after_one_step": int(left.size), "back_in_R_at_end": int(back.sum()),
            "pct_back": round(float(100 * back.mean()), 1)}


def row_tv(A, B):
    """Total variation distance between matching rows of two transition matrices."""
    return 0.5 * np.abs(A - B).sum(axis=1)


def cell_drifters(table, region, cell_size):
    """Distinct drifters starting a usable move in each cell."""
    start, _, n_cells = to_states(table, region, cell_size)
    return per_cell_counts(start, table["drifter_id"], n_cells)[1]


def naive_from(table, region, cell_size):
    start, end, n_cells = to_states(table, region, cell_size)
    return naive_P(count_matrix(start, end, n_cells + 1))


def row_entropy(P):
    """Shannon entropy of each row (nats): higher = probability spread over more destinations."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log(P), 0.0).sum(axis=1)


def grid_comparison(t1, t2, region, cell_size, seed=HALF_SPLIT_SEED):
    """P(tau)^2 against P(2 tau), naive P, on cells with at least N_MIN drifters in the 2 tau table.

    Returns:
        (per-cell dict of arrays, summary dict).
    """
    P1, P2 = naive_from(t1, region, cell_size), naive_from(t2, region, cell_size)
    P1sq = P1 @ P1
    good = cell_drifters(t2, region, cell_size) >= N_MIN

    # Noise reference: independent halves of the drifters
    ids = np.unique(t1["drifter_id"])
    half_a = np.random.default_rng(seed).permutation(ids)[: ids.size // 2]
    a1, a2 = (np.isin(t["drifter_id"], half_a) for t in (t1, t2))
    P1a, P1b = naive_from(subset(t1, a1), region, cell_size), naive_from(subset(t1, ~a1), region, cell_size)
    P2a, P2b = naive_from(subset(t2, a2), region, cell_size), naive_from(subset(t2, ~a2), region, cell_size)
    # Halves can leave a cell with no row; only cells with a row in every estimate are compared
    has_row = np.all([M[:-1].sum(axis=1) > 0 for M in (P1a, P1b, P2a, P2b)], axis=0)
    both = good & has_row
    markov_half = 0.5 * (row_tv(P1a @ P1a, P2b) + row_tv(P1b @ P1b, P2a))[:-1]
    noise_half = row_tv(P2a, P2b)[:-1]

    cells = {
        "tv": row_tv(P1sq, P2)[:-1],
        "stay_markov": np.diag(P1sq)[:-1], "stay_direct": np.diag(P2)[:-1],
        "exit_markov": P1sq[:-1, -1], "exit_direct": P2[:-1, -1],
        "good": good, "P1sq": P1sq, "P2": P2,
    }
    summary = {
        "grid": f"{cell_size:g} deg", "cells_compared": int(good.sum()),
        "median_tv": float(np.median(cells["tv"][good])),
        "median_tv_markov_halves": float(np.median(markov_half[both])),
        "median_tv_noise_halves": float(np.median(noise_half[both])),
        "pct_cells_markov_gap_above_noise": round(float(100 * np.mean(markov_half[both] > noise_half[both])), 1),
        "median_stay_markov": float(np.median(cells["stay_markov"][good])),
        "median_stay_direct": float(np.median(cells["stay_direct"][good])),
        "median_exit_markov": float(np.median(cells["exit_markov"][good])),
        "median_exit_direct": float(np.median(cells["exit_direct"][good])),
        "median_entropy_markov": float(np.median(row_entropy(P1sq)[:-1][good])),
        "median_entropy_direct": float(np.median(row_entropy(P2)[:-1][good])),
    }
    # Box-wide exit share within 2 tau, weighting cells by where moves start
    start, _, _ = to_states(t2, region, cell_size)
    summary["pct_exit_markov_all_starts"] = round(float(100 * P1sq[start, -1].mean()), 2)
    summary["pct_exit_direct_all_starts"] = round(float(100 * P2[start, -1].mean()), 2)
    return cells, summary


def fine_scores(t1, t2, region, assignment, cell_size, coarse_size):
    """Held-out log-likelihood of the test drifters' 2 tau moves: direct shrinkage P(2 tau),
    the squared tau-day shrinkage model, and the neighbour-only baseline fitted to 2 tau moves."""
    parent = parent_states(region, cell_size, coarse_size)
    n_lat, n_lon = (len(e) - 1 for e in grid_edges(region, cell_size))

    def fitted(table):
        s_tr, e_tr, n_cells, _ = states_for(table, region, cell_size, assignment, "train")
        s_va, e_va, _, _ = states_for(table, region, cell_size, assignment, "valid")
        s_fit, e_fit, _, _ = states_for(table, region, cell_size, assignment, ("train", "valid"))
        C_tr = count_matrix(s_tr, e_tr, n_cells + 1)
        C_fit = count_matrix(s_fit, e_fit, n_cells + 1)
        alpha, _ = tune_alpha(C_tr, parent, s_va, e_va)
        return alpha, shrinkage_P(C_fit, prior_rows(C_fit, parent), alpha), C_tr, C_fit, s_va, e_va

    alpha1, S1, _, _, _, _ = fitted(t1)
    alpha2, S2, C2_tr, C2_fit, s_va, e_va = fitted(t2)
    stay, leak = tune_neighbour(n_lat, n_lon, seen_cells(C2_tr), s_va, e_va,
                                stays=NEIGHBOUR_STAYS_2TAU, leaks=NEIGHBOUR_LEAKS_2TAU)
    models = {
        "direct P(2 tau), shrinkage": (f"alpha={alpha2:.3g}", S2),
        "Markov P(tau)^2, shrinkage": (f"alpha={alpha1:.3g}", S1 @ S1),
        "neighbour-only, fitted to 2 tau": (f"stay={stay:g} leak={leak:g}",
                                            neighbour_P(n_lat, n_lon, seen_cells(C2_fit), stay, leak)),
    }

    s_te, e_te, n_cells, test_ids = states_for(t2, region, cell_size, assignment, "test")
    s_all, _, _ = to_states(t1, region, cell_size)  # the adequacy map's count, as in evaluate.py
    _, map_drifters = per_cell_counts(s_all, t1["drifter_id"], n_cells)
    thin = map_drifters[s_te] < N_MIN
    groups = {"all": np.ones(s_te.size, dtype=bool), f"under {N_MIN} drifters": thin, f"{N_MIN}+ drifters": ~thin}

    rows = []
    for name, (setting, P) in models.items():
        logp = log_lik(P, s_te, e_te)
        for cells, g in groups.items():
            rows.append({"grid": f"{cell_size:g} deg", "lag_days": STEPS * TAU_DAYS, "model": name,
                         "setting": setting, "cells": cells,
                         "test_drifters": int(np.unique(test_ids[g]).size),
                         "test_transitions": int(g.sum()), **score(logp[g])})
    return rows, S2, S1 @ S1


def plot(region, comparisons, reentry, out_path):
    """Entries of P(15)^2 against P(30) on the coarse grid, and the per-cell distance map on each grid."""
    lag1, lag2 = TAU_DAYS, STEPS * TAU_DAYS
    (coarse, (cells_c, sum_c)), (fine, (cells_f, sum_f)) = comparisons

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.55, 1.55]})
    ax = axes[0]
    good = cells_c["good"]
    a, b = cells_c["P2"][:-1][good].ravel(), cells_c["P1sq"][:-1][good].ravel()
    keep = (a > 0) | (b > 0)
    ax.scatter(a[keep], b[keep], s=4, alpha=0.3)
    ax.plot([0, 1], [0, 1], color="black", lw=1)
    ax.set_xlabel(f"P({lag2} d), fitted directly")
    ax.set_ylabel(f"P({lag1} d)$^2$, two {lag1}-day steps")
    ax.set_title(f"{coarse:g}°: every transition probability, {sum_c['cells_compared']} cells", fontsize=10)
    ax.set_aspect("equal")

    vmax = max(np.nanmax(np.where(c["good"], c["tv"], np.nan)) for c in (cells_c, cells_f))
    for ax, size, cells, summary in ((axes[1], coarse, cells_c, sum_c), (axes[2], fine, cells_f, sum_f)):
        lat_edges, lon_edges = grid_edges(region, size)
        shape = (len(lat_edges) - 1, len(lon_edges) - 1)
        grid = np.where(cells["good"], cells["tv"], np.nan).reshape(shape)
        mesh = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(grid), cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"{size:g}°: distance between the rows, median {summary['median_tv']:.2f}\n"
                     f"two halves of the drifters: Markov gap {summary['median_tv_markov_halves']:.2f}, "
                     f"noise only {summary['median_tv_noise_halves']:.2f}", fontsize=10)
    fig.colorbar(mesh, ax=axes[1:], label="Total variation distance (0 = identical rows)", shrink=0.85)

    fig.suptitle(f"Markov check (naive P): do two {lag1}-day steps match a direct {lag2}-day estimate?")
    fig.text(0.01, -0.13,
             f"Cells with at least {N_MIN} drifters starting a {lag2}-day move (blank: fewer). Drifters that left R by "
             f"day {lag1} count as having left at day {lag2}, as P$^2$ assumes. 'Markov gap' compares P({lag1} d)$^2$ "
             f"from one random half of the drifters with P({lag2} d) from the other half; 'noise only' compares\n"
             f"P({lag2} d) from each half. A gap no bigger than the noise means the data can't tell the two apart. "
             f"No bootstrap intervals yet. Re-entry: {reentry['pct_back']}% of drifters outside R at day {lag1} are back "
             f"inside at day {lag2}, which P$^2$ (exit absorbing) can't show.", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    raw = load_raw_transitions()
    t1 = usable_transitions(raw, region)
    print(f"\nChaining {STEPS} x {TAU_DAYS}-day rows into {STEPS * TAU_DAYS}-day moves")
    t2 = usable_transitions(chain_transitions(raw, reentry="absorbing"), region)
    if os.path.exists(REBUILT_PATH):
        print(f"Using Computational's rebuilt table for the observed exits: {REBUILT_PATH}")
        observed = usable_transitions(load_raw_transitions(REBUILT_PATH), region, verbose=False)
    else:
        observed = usable_transitions(chain_transitions(raw, reentry="observed"), region, verbose=False)
    assignment = load_split(t1["drifter_id"])

    # 1. Re-entry: how much the absorbing exit rule overstates exits over 2 tau
    reentry = reentry_summary(raw)
    reentry["pct_exit_absorbing"] = round(float(100 * t2["exited"].mean()), 2)
    reentry["pct_exit_observed"] = round(float(100 * observed["exited"].mean()), 2)
    print("Re-entry:", reentry)

    # 2. Both grids, naive P
    comparisons = [(size, grid_comparison(t1, t2, region, size)) for size in (coarse, fine)]
    for _, (_, summary) in comparisons:
        print("Grid comparison:", summary)

    # 3. Fine grid, shrinkage, held-out scores
    score_rows, _, _ = fine_scores(t1, t2, region, assignment, fine, coarse)
    for row in score_rows:
        print(row)

    write_csv(os.path.join(RESULTS_DIR, f"markov_check_tau{TAU_DAYS}d.csv"),
              [{**summary, **{f"reentry_{k}": v for k, v in reentry.items()}} for _, (_, summary) in comparisons])
    write_csv(os.path.join(RESULTS_DIR, f"markov_check_scores_tau{TAU_DAYS}d.csv"), score_rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "markov_check.png")
    plot(region, comparisons, reentry, fig_path)
    print(f"Saved the two markov_check CSVs in {RESULTS_DIR} and {fig_path}")
