"""
markov_check.py

Related task: Week 7 Quantitative - test the Markov assumption behind every P^n forecast
(proposal section 4): is two 15-day steps the same as one 30-day step, P(15 d)^2 = P(30 d), and
six steps the same as one 90-day step, P(15 d)^6 = P(90 d)? This is the Chapman-Kolmogorov
equation. If it holds, the 3-month and 1-year maps (P^6, P^24) can be trusted as far as the lag
goes; if it fails, they mix up a real flow with an artefact of chopping time into 15-day steps.

Two matrices at each grid, fitted the same way as the product and the baseline:
- 4 deg naive P (proposal Figure 1);
- 2 deg shrinkage to 4 deg (the product). Alpha is chosen on the validation drifters separately
  for each lag (evaluate.tune_alpha), so neither lag is handicapped by the other's alpha.
The direct matrices come from the Week 7 Computational tables (transitions_tau30d.npz,
transitions_tau90d.npz; src/gridding/lags.py).

Same starts at both lags ("matched", the main comparison). Row i is the same drifter-day start in
every table, so both matrices are fitted on the starts whose destination is known at BOTH lags.
Without this, P(90 d) would rest only on drifters that kept reporting for 90 days while P(15 d)
uses all of them (Week 7 Computational: 14.5% of 90-day starts have an ended record), and part of
any gap would come from which drifters each was built on. The unmatched version (every usable
start at each lag, as the product is fitted) is also reported, as point estimates, to show how
much that selection matters.

Three checks (plus the total variation as a size):
1. Held-out score (the main test). Fit both on the train + valid drifters and score the test
   drifters' real 30-day (or 90-day) moves: log of the probability P(15 d)^k or P(k x 15 d) gave
   to where each drifter actually was. If Markov holds, chaining short steps predicts long moves
   as well as a matrix fitted to them directly. Interval on the paired difference per move from
   resampling the test drifters (bootstrap.py, as score_intervals.py), models held fixed.
2. What differs, per cell (all drifters, matched): the chance of still being in the start cell,
   the chance of having left the box, the mean distance travelled (start cell centre to
   destination cell centre, over material still in R), and the total variation distance
   between the two rows (the share of the cell's material the two send to different places).
   Means over cells under and over N_MIN drifters, with 95% intervals on (direct - chained) from
   N_BOOT drifter bootstrap replicates; each replicate refits both lags on the same drawn drifters.
   The total variation gets no interval: resampling noise can only add to it, so its replicates sit
   above the full-data value (they did, on the first run) and a percentile interval would mislead.
3. Re-entry: a drifter that leaves the box and comes back counts as lost in P^k ("left R" is
   absorbing, decision 11b) but as in R in the direct table. The share of matched starts that are
   out of R at 15 days and back in R at 30 or 90 is reported, a lower bound on this effect.

How to read it: a total variation distance is never 0 between two estimates from finite data,
so it is a size, not a test. A split-half "noise floor" (direct from half the drifters against
direct from the other half) was tried and set aside: the chained matrix is smoother than a direct
one fitted from the same drifters, so it sits closer to an independent direct estimate than a
second direct estimate does, and the floor would make the Markov gap look negative (see the
2026-11-01 AI log entry). The held-out score accounts for both bias and noise, which is why it is
the main test.

Outputs: outputs/results/markov_check_tau15d.csv and outputs/figures/markov_check.png.

Run: python3 -m src.inference.markov_check   (a few minutes; needs transitions_tau30d.npz and
transitions_tau90d.npz from src.gridding.lags, and drifter_split.csv from src.inference.split)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.grid import grid_edges, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, IN_R, LEFT_R, table_path
from src.inference.adequacy import N_MIN
from src.inference.bootstrap import (
    LEVEL, N_BOOT, SEED, drifter_pairs, drifter_totals, percentile_interval, replicate_count_matrix,
    resample_counts, resampled_means,
)
from src.inference.estimators import count_matrix, log_lik, naive_P, prior_rows, shrinkage_P
from src.inference.evaluate import states_for, tune_alpha
from src.inference.split import load_split
from src.inference.transition_data import TAU_DAYS, load_transitions, parent_states, to_states
from src.linalg.release_map import _degrees

STEPS = (2, 6)  # P(15 d)^k against P(k x 15 d): 30 and 90 days
EARTH_RADIUS_KM = 6371.0


def known_rows(tau_days):
    """True for every saved row of a lag's table whose destination is known (in R or left R)."""
    path = table_path(tau_days)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run python3 -m src.gridding.lags (Week 7 Computational).")
    return np.isin(np.load(path)["outcome"], [IN_R, LEFT_R])


def same_starts(path_a, path_b):
    """Check that two lags' tables hold the same starts in the same order (Week 7 Computational)."""
    a, b = np.load(path_a), np.load(path_b)
    for col in ("drifter_id", "start_time"):
        if a[col].shape != b[col].shape or not np.array_equal(a[col], b[col]):
            raise ValueError(f"{os.path.basename(path_a)} and {os.path.basename(path_b)} differ in {col}: "
                             "rebuild both with src.gridding.lags.")


def reentry_share(tau_days, long_days):
    """Share of starts (known at both lags) out of R at tau_days and back in R at long_days."""
    short, long_ = np.load(table_path(tau_days))["outcome"], np.load(table_path(long_days))["outcome"]
    both = np.isin(short, [IN_R, LEFT_R]) & np.isin(long_, [IN_R, LEFT_R])
    return float(np.mean((short[both] == LEFT_R) & (long_[both] == IN_R)))


def cell_distances(region, cell_size):
    """Great-circle distance (km) between every pair of cell centres, (n_cells x n_cells)."""
    lat_edges, lon_edges = grid_edges(region, cell_size)
    lat_c = np.deg2rad(0.5 * (lat_edges[:-1] + lat_edges[1:]))
    lon_c = np.deg2rad(0.5 * (lon_edges[:-1] + lon_edges[1:]))
    lat, lon = (a.ravel() for a in np.meshgrid(lat_c, lon_c, indexing="ij"))  # row-major, as cell_index
    h = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def row_summaries(P, dist_km):
    """Per cell: chance of staying in the cell, of having left R, and mean km travelled within R."""
    n = dist_km.shape[0]
    inside = P[:n, :n]
    with np.errstate(invalid="ignore", divide="ignore"):  # rows with no data (naive P) give NaN
        km = (inside * dist_km).sum(axis=1) / inside.sum(axis=1)
    return {"stay": np.diag(P)[:n], "exit": P[:n, n], "km": km}


def total_variation(A, B):
    """Half the summed absolute difference between matching rows (cells only, exit row dropped)."""
    return 0.5 * np.abs(A[:-1] - B[:-1]).sum(axis=1)


def compare(P_short, P_long, k, dist_km):
    """Per-cell chained (P_short^k) and direct (P_long) summaries, and their total variation."""
    chained = np.linalg.matrix_power(P_short, k)
    return row_summaries(chained, dist_km), row_summaries(P_long, dist_km), total_variation(chained, P_long)


def group_means(values, groups):
    """Mean of a per-cell array over each group of cells (NaN cells ignored)."""
    return {g: float(np.nanmean(values[m])) if m.any() else float("nan") for g, m in groups.items()}


def make_fit(model, region, cell_size, alpha=None):
    """Count matrix -> P: naive, or shrinkage to the 4 deg cell with the given alpha."""
    if model == "naive":
        return naive_P
    parent = parent_states(region, cell_size, BASELINE_CELL_SIZE)
    return lambda counts: shrinkage_P(counts, prior_rows(counts, parent), alpha)


def tune_alphas(tables, region, assignment, cell_size):
    """Shrinkage alpha for each lag's table, chosen on train -> valid exactly as in evaluate.py."""
    parent = parent_states(region, cell_size, BASELINE_CELL_SIZE)
    alphas = {}
    for lag, table in tables.items():
        s_tr, e_tr, n_cells, _ = states_for(table, region, cell_size, assignment, "train")
        s_va, e_va, _, _ = states_for(table, region, cell_size, assignment, "valid")
        alphas[lag], _ = tune_alpha(count_matrix(s_tr, e_tr, n_cells + 1), parent, s_va, e_va)
    return alphas


def heldout_rows(tables, region, assignment, cell_size, model, alphas, long_days, k):
    """Check 1: score test drifters' long-lag moves under P(15 d)^k and P(long) fitted on train + valid."""
    s_fit, e_fit, n_cells, _ = states_for(tables[TAU_DAYS], region, cell_size, assignment, ("train", "valid"))
    S_fit, E_fit, _, _ = states_for(tables[long_days], region, cell_size, assignment, ("train", "valid"))
    s_te, e_te, _, ids = states_for(tables[long_days], region, cell_size, assignment, "test")
    n_states = n_cells + 1
    P_short = make_fit(model, region, cell_size, alphas.get(TAU_DAYS))(count_matrix(s_fit, e_fit, n_states))
    P_long = make_fit(model, region, cell_size, alphas.get(long_days))(count_matrix(S_fit, E_fit, n_states))

    chained = log_lik(np.linalg.matrix_power(P_short, k), s_te, e_te)
    direct = log_lik(P_long, s_te, e_te)
    finite = np.isfinite(chained) & np.isfinite(direct)
    with np.errstate(invalid="ignore"):  # -inf - -inf where both give 0; masked out below
        diff = np.where(finite, direct - chained, 0.0)
    draws = resample_counts(np.unique(ids).size, N_BOOT, SEED)
    totals, moves = drifter_totals(diff, ids, finite)
    low, high = percentile_interval(resampled_means(totals, moves, draws), LEVEL)
    alpha_note = (f"alpha {TAU_DAYS}d={alphas[TAU_DAYS]:.3g}, {long_days}d={alphas[long_days]:.3g}"
                  if alphas else "")
    return [{
        "check": "held-out score", "quantity": "mean log-likelihood per test move (where both finite)",
        "cells": "all", "n_cells": "", "chained": float(chained[finite].mean()), "direct": float(direct[finite].mean()),
        "direct_minus_chained": float(diff[finite].mean()), "ci_low": float(low), "ci_high": float(high),
        "note": (f"{int(finite.sum()):,} test moves, {np.unique(ids).size} test drifters; "
                 f"zero probability: chained {100 * np.mean(~np.isfinite(chained)):.2f}%, "
                 f"direct {100 * np.mean(~np.isfinite(direct)):.2f}%; {alpha_note}").rstrip("; "),
    }]


def matrix_rows(tables, region, cell_size, model, alphas, k, dist_km, groups, n_boot=N_BOOT):
    """Check 2: per-cell differences between P(15 d)^k and P(k x 15 d) on all drifters, with intervals."""
    states, pairs = {}, {}
    for lag, table in tables.items():
        start, end, n_cells = to_states(table, region, cell_size)
        states[lag] = count_matrix(start, end, n_cells + 1)
        pairs[lag] = drifter_pairs(start, end, table["drifter_id"], n_cells + 1)
    ids = pairs[TAU_DAYS]["drifter_ids"]
    for lag in tables:  # matched rows: the same drifters at every lag, so one draw weights all of them
        if not np.array_equal(pairs[lag]["drifter_ids"], ids):
            raise ValueError("Lags hold different drifters; the matched tables should share every start.")
    fits = {lag: make_fit(model, region, cell_size, alphas.get(lag)) for lag in tables}

    def summarise(counts):
        P = {lag: fits[lag](c) for lag, c in counts.items()}
        chained, direct, tv = compare(P[TAU_DAYS], P[k * TAU_DAYS], k, dist_km)
        out = {q: (group_means(chained[q], groups), group_means(direct[q], groups)) for q in chained}
        return out

    estimate = summarise(states)
    estimate["tv"] = (group_means(
        compare(fits[TAU_DAYS](states[TAU_DAYS]), fits[k * TAU_DAYS](states[k * TAU_DAYS]), k, dist_km)[2], groups), None)
    reps = []
    for draw in resample_counts(ids.size, n_boot, SEED):
        reps.append(summarise({lag: replicate_count_matrix(pairs[lag], draw) for lag in tables}))

    names = {"stay": "chance still in the start cell", "exit": "chance of having left the box",
             "km": "mean distance travelled within R (km)", "tv": "total variation between the two rows"}
    rows = []
    for q, label in names.items():
        for g, mask in groups.items():
            chained, direct = estimate[q][0][g], (estimate[q][1] or {}).get(g)
            if q == "tv":  # a size, not a test: resampling noise only adds to it, so no interval
                rows.append({"check": "per-cell", "quantity": label, "cells": g, "n_cells": int(mask.sum()),
                             "chained": chained, "direct": "", "direct_minus_chained": "", "ci_low": "", "ci_high": "",
                             "note": "no interval: total variation is biased upward by resampling noise"})
                continue
            samples = np.array([r[q][1][g] - r[q][0][g] for r in reps])
            low, high = percentile_interval(samples, LEVEL)
            rows.append({"check": "per-cell", "quantity": label, "cells": g, "n_cells": int(mask.sum()),
                         "chained": chained, "direct": direct, "direct_minus_chained": direct - chained,
                         "ci_low": float(low), "ci_high": float(high),
                         "note": f"interval on direct - chained, {n_boot} drifter bootstrap replicates"})
    return rows


def run_grid(region, assignment, cell_size, model, matched, unmatched, dist_km):
    """All checks for one grid and model; returns (csv rows, per-cell arrays for the figure)."""
    # Cell groups by the adequacy map's count (all drifters starting a 15-day move), as in evaluate.py
    start, _, n_cells = to_states(unmatched[TAU_DAYS], region, cell_size)
    _, drifters = per_cell_counts(start, unmatched[TAU_DAYS]["drifter_id"], n_cells)
    sampled = drifters > 0
    groups = {"all with drifters": sampled, f"under {N_MIN} drifters": sampled & (drifters < N_MIN),
              f"{N_MIN}+ drifters": drifters >= N_MIN}
    name = f"{cell_size:g} deg {'naive' if model == 'naive' else 'shrinkage'}"

    rows, figure = [], {}
    for k in STEPS:
        long_days = k * TAU_DAYS
        tables = matched[k]
        alphas = tune_alphas(tables, region, assignment, cell_size) if model == "shrinkage" else {}
        for r in heldout_rows(tables, region, assignment, cell_size, model, alphas, long_days, k):
            rows.append({"grid_model": name, "lag_days": long_days, "sample": "matched", **r})
        for r in matrix_rows(tables, region, cell_size, model, alphas, k, dist_km, groups):
            rows.append({"grid_model": name, "lag_days": long_days, "sample": "matched", **r})

        # Per-cell arrays for the figure, from the full-data matched fits
        P = {}
        for lag, table in tables.items():
            s, e, _ = to_states(table, region, cell_size)
            P[lag] = make_fit(model, region, cell_size, alphas.get(lag))(count_matrix(s, e, n_cells + 1))
        chained, direct, tv = compare(P[TAU_DAYS], P[long_days], k, dist_km)
        figure[k] = {"chained": chained, "direct": direct, "tv": tv, "drifters": drifters}

        # Unmatched: every usable start at each lag (as the product is fitted), point estimates only
        P_un = {}
        for lag in (TAU_DAYS, long_days):
            s, e, _ = to_states(unmatched[lag], region, cell_size)
            P_un[lag] = make_fit(model, region, cell_size, alphas.get(lag))(count_matrix(s, e, n_cells + 1))
        chained_u, direct_u, tv_u = compare(P_un[TAU_DAYS], P_un[long_days], k, dist_km)
        for q, label in (("stay", "chance still in the start cell"), ("exit", "chance of having left the box"),
                         ("km", "mean distance travelled within R (km)")):
            for g, mask in groups.items():
                c, d = group_means(chained_u[q], {g: mask})[g], group_means(direct_u[q], {g: mask})[g]
                rows.append({"grid_model": name, "lag_days": long_days, "sample": "all usable starts",
                             "check": "per-cell", "quantity": label, "cells": g, "n_cells": int(mask.sum()),
                             "chained": c, "direct": d, "direct_minus_chained": d - c, "ci_low": "", "ci_high": "",
                             "note": "no interval: shows the effect of fitting each lag on its own starts"})
        for g, mask in groups.items():
            rows.append({"grid_model": name, "lag_days": long_days, "sample": "all usable starts", "check": "per-cell",
                         "quantity": "total variation between the two rows", "cells": g, "n_cells": int(mask.sum()),
                         "chained": group_means(tv_u, {g: mask})[g], "direct": "", "direct_minus_chained": "",
                         "ci_low": "", "ci_high": "", "note": ""})
    return rows, figure


def plot(region, figures, out_path):
    """(a) total variation per 2 deg cell at 30 days; (b) stay chance and (c) km travelled, chained vs direct."""
    fine = region["cell_size_deg"]
    lat_edges, lon_edges = grid_edges(region, fine)
    shape = (len(lat_edges) - 1, len(lon_edges) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained", gridspec_kw={"width_ratios": [1.7, 1, 1]})

    f = figures[f"{fine:g} deg shrinkage"][STEPS[0]]
    ax = axes[0]
    grid = np.ma.masked_where((f["drifters"] == 0).reshape(shape), f["tv"].reshape(shape))
    mesh = ax.pcolormesh(lon_edges, lat_edges, grid, cmap="magma_r", vmin=0, vmax=0.4)
    ax.contour(0.5 * (lon_edges[:-1] + lon_edges[1:]), 0.5 * (lat_edges[:-1] + lat_edges[1:]),
               f["drifters"].reshape(shape) >= N_MIN, levels=[0.5], colors="#1f77b4", linewidths=1.2)
    ax.set_facecolor("#d9d9d9")
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(_degrees("lon"))
    ax.yaxis.set_major_formatter(_degrees("lat"))
    ax.set_title(f"(a) {fine:g}° shrinkage: share of each cell's material that $P({TAU_DAYS}d)^2$ and\n"
                 f"$P({STEPS[0] * TAU_DAYS}d)$ send to different places (total variation)", fontsize=9.5, loc="left")
    fig.colorbar(mesh, ax=ax, extend="max", shrink=0.85, label="total variation")

    styles = {(BASELINE_CELL_SIZE, STEPS[0]): ("#ff7f0e", "o"), (BASELINE_CELL_SIZE, STEPS[1]): ("#ff7f0e", "^"),
              (fine, STEPS[0]): ("#1f77b4", "o"), (fine, STEPS[1]): ("#1f77b4", "^")}
    for ax, q, label in ((axes[1], "stay", "chance still in the start cell"),
                         (axes[2], "km", "mean distance travelled within R (km)")):
        top, every = 0, []
        for name, per_k in figures.items():
            size = float(name.split()[0])
            for k, f in per_k.items():
                colour, marker = styles[(size, k)]
                ok = f["drifters"] >= N_MIN
                x, y = f["direct"][q][ok], f["chained"][q][ok]
                ax.scatter(x, y, s=8, alpha=0.5, color=colour, marker=marker, edgecolors="none",
                           label=f"{name}, {k * TAU_DAYS} days")
                top = max(top, np.nanmax(x), np.nanmax(y))
                every.append(np.concatenate([x, y]))
        if q == "km":  # a few fast-current cells (Agulhas, Brazil Current) would squash the rest
            every = np.concatenate(every)
            top = float(np.nanquantile(every, 0.99)) * 1.05
            off = int(np.sum(np.isfinite(every) & (every > top)))
            ax.text(0.98, 0.02, f"{off} of {np.isfinite(every).sum()} values off scale", transform=ax.transAxes,
                    ha="right", fontsize=7.5, color="grey")
        ax.plot([0, top], [0, top], color="grey", lw=1, ls=":")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel(f"direct matrix: {label}")
        ax.set_ylabel(f"chained $P({TAU_DAYS}d)^k$: {label}")
        ax.set_aspect("equal")
    axes[1].set_title(f"(b) Staying put, cells with {N_MIN}+ drifters", fontsize=9.5, loc="left")
    axes[2].set_title(f"(c) Distance travelled, cells with {N_MIN}+ drifters", fontsize=9.5, loc="left")
    axes[1].legend(fontsize=7.5, markerscale=2)
    fig.text(0.01, -0.1,
             f"Chapman-Kolmogorov check, {region['label']}: k steps of the {TAU_DAYS}-day matrix against a matrix fitted "
             f"directly to {STEPS[0] * TAU_DAYS}-day (circles) or {STEPS[1] * TAU_DAYS}-day (triangles) moves, both from the "
             "same drifter-day starts.\nPoints on the dotted line agree. Below it in (b): the chain empties cells faster "
             "than drifters really leave them. Above it in (c): the chain spreads material further than drifters really go.\n"
             f"Blue line in (a) encloses cells with at least {N_MIN} drifters; grey cells have no drifters.", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    lags = [TAU_DAYS] + [k * TAU_DAYS for k in STEPS]
    for lag in lags[1:]:
        same_starts(table_path(TAU_DAYS), table_path(lag))
    known = {lag: known_rows(lag) for lag in lags}

    unmatched = {lag: load_transitions(region, table_path(lag), verbose=False) for lag in lags}
    matched = {}
    for k in STEPS:
        both = known[TAU_DAYS] & known[k * TAU_DAYS]
        matched[k] = {lag: load_transitions(region, table_path(lag), verbose=False, rows=both)
                      for lag in (TAU_DAYS, k * TAU_DAYS)}
        print(f"{k * TAU_DAYS} days: {both.sum():,} starts known at both lags "
              f"({unmatched[TAU_DAYS]['outcome'].size:,} at {TAU_DAYS} days); "
              f"out of R at {TAU_DAYS} days and back in R at {k * TAU_DAYS}: "
              f"{100 * reentry_share(TAU_DAYS, k * TAU_DAYS):.3f}%")
    assignment = load_split(unmatched[TAU_DAYS]["drifter_id"])

    rows, figures = [], {}
    for size, model in ((coarse, "naive"), (fine, "shrinkage")):
        grid_rows, figures[f"{size:g} deg {model}"] = run_grid(region, assignment, size, model, matched, unmatched,
                                                               cell_distances(region, size))
        rows += grid_rows
        print(f"{size:g} deg {model}: done")
    for k in STEPS:
        rows.append({"grid_model": "all", "lag_days": k * TAU_DAYS, "sample": "matched", "check": "re-entry",
                     "quantity": f"% of starts out of R at {TAU_DAYS} days and back in R at {k * TAU_DAYS}",
                     "cells": "all", "n_cells": "", "chained": "", "direct": 100 * reentry_share(TAU_DAYS, k * TAU_DAYS),
                     "direct_minus_chained": "", "ci_low": "", "ci_high": "", "note": "lower bound on re-entry"})

    for r in rows:
        for key in ("chained", "direct", "direct_minus_chained", "ci_low", "ci_high"):
            if isinstance(r[key], float):
                r[key] = round(r[key], 5)
        if r["check"] != "per-cell" or r["cells"] == f"{N_MIN}+ drifters":
            print({k: r[k] for k in ("grid_model", "lag_days", "sample", "quantity", "cells", "chained", "direct",
                                     "direct_minus_chained", "ci_low", "ci_high")})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_out = os.path.join(RESULTS_DIR, f"markov_check_tau{TAU_DAYS}d.csv")
    with open(table_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_out = os.path.join(FIGURES_DIR, "markov_check.png")
    plot(region, figures, fig_out)
    print(f"Saved {table_out} and {fig_out}")
