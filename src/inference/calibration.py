"""
calibration.py

Related task: Week 9 forensic review. The primary research question asks how reliable each estimate is. The
held-out log-likelihood (evaluate.py) ranks models but doesn't say whether a stated probability means what it
says. These checks ask that directly, on the test drifters (never used to fit or tune), with P fitted on train
+ valid and chained to longer times exactly as a user would.

1. Reliability. Take every (start cell, destination cell) entry of P^k, weighted by the test moves from that
   cell, and bin by predicted probability. In each bin, compare the average predicted probability with the
   share of those entries that really happened. A well-calibrated P has the two equal.
2. Likely areas (the app's lines). For each start cell, the smallest set of cells holding at least 50% (and 90%)
   of P^k's material still in the box, most likely first: the rule behind the app's outlines (model.js,
   likelyAreas). The share of real test moves that end inside should match. The set is whole cells, so it holds
   a little more than the level; "predicted" is what it really holds.
3. Leaving the box. "Left R" is absorbing, so P^k gives the chance of HAVING LEFT at some point by then. Reading
   it against moves that are outside on day k (as markov_check.py's per-cell exit does) compares two different
   things, because about a fifth of the drifters that leave are back in the box on the day. And a drifter
   that stops reporting first is neither "stayed" nor "left". So the observed chance is a Kaplan-Meier
   estimate on the same 15-day checkpoints as the chain: a start is at risk until it is first seen outside R
   (the event) or first has no known position (censored). This assumes stopping reporting is unrelated to
   leaving, which grounding breaks at the coast, so read it as a check on the open ocean.
4. Horizons: 15 days (the fitted step), 90 days (P^6) and 360 days (P^24), the proposal's three horizons. Longer
   moves rest on longer-lived drifters. The 360-day table and the checkpoints aren't kept in the repo, so they
   are built here, which needs the region and records caches, and are skipped without them.

Intervals: resampling the test drifters (bootstrap.py), models held fixed.

Outputs: outputs/results/calibration_tau15d.csv and outputs/figures/calibration.png.
Run: python3 -m src.inference.calibration   (about 2 minutes; needs drifter_split.csv)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, INTERIM_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.grid import cell_index, per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, IN_R, LEFT_R, TAU_DAYS, build_transitions, table_path
from src.ingest.drifter_records import load_records
from src.ingest.load import load_region
from src.inference.adequacy import N_MIN
from src.inference.bootstrap import LEVEL, N_BOOT, SEED, percentile_interval, resample_counts
from src.inference.estimators import count_matrix, prior_rows, shrinkage_P
from src.inference.evaluate import states_for, tune_alpha
from src.inference.split import load_split, set_mask
from src.inference.transition_data import load_transitions, parent_states, subset, to_states

HORIZONS = {1: 15, 6: 90, 24: 360}  # steps -> days
BIN_EDGES = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0000001)
LEVELS = (0.5, 0.9)
EXIT_BINS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0000001)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # as lags.py


def reliability(P, start, end, drifter_id, bin_edges=BIN_EDGES, n_boot=N_BOOT, seed=SEED):
    """Predicted against observed frequency in each probability bin.

    Inputs:
        P: the chained matrix; start, end, drifter_id: one entry per test move at that horizon.
    Returns:
        list of dicts, over destination cells: bin edges, entries weighted by moves, mean predicted %, observed % and its interval.
    """
    ids, rank = np.unique(drifter_id, return_inverse=True)
    n = ids.size
    draws = resample_counts(n, n_boot, seed)
    which = np.digitize(P, bin_edges) - 1  # the bin of every entry of P
    which[:, -1] = -1  # destination cells only; leaving the box is checked by left_box()
    rows = []
    for b in range(len(bin_edges) - 1):
        in_bin = which == b
        per_start_trials = in_bin.sum(axis=1)
        per_start_pred = (P * in_bin).sum(axis=1)
        trials = np.bincount(rank, weights=per_start_trials[start], minlength=n)
        pred = np.bincount(rank, weights=per_start_pred[start], minlength=n)
        hits = np.bincount(rank, weights=in_bin[start, end].astype(float), minlength=n)
        if trials.sum() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            obs_reps = 100 * (draws @ hits) / (draws @ trials)
        low, high = percentile_interval(obs_reps, LEVEL)
        rows.append({"bin_low": bin_edges[b], "bin_high": min(bin_edges[b + 1], 1.0), "entries_weighted": int(trials.sum()),
                     "hits": int(hits.sum()), "predicted_pct": 100 * float(pred.sum() / trials.sum()),
                     "observed_pct": 100 * float(hits.sum() / trials.sum()), "ci_low": float(low), "ci_high": float(high)})
    return rows


def likely_sets(P, n_cells, level):
    """For each row of P: the smallest set of cells (most likely first) holding at least `level` of the material
    still in the box, as model.js likelyAreas.

    Returns:
        (mask, held): bool (n_cells x n_cells), and the share of the in-box material each set really holds.
    """
    inside = P[:n_cells, :n_cells]
    order = np.argsort(-inside, axis=1, kind="stable")
    sorted_p = np.take_along_axis(inside, order, axis=1)
    total = sorted_p.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cum = np.cumsum(sorted_p, axis=1) / total
    upto = np.minimum((cum < level).sum(axis=1), n_cells - 1)  # the first position reaching the level
    sorted_mask = np.arange(n_cells)[None, :] <= upto[:, None]
    mask = np.zeros_like(sorted_mask)
    np.put_along_axis(mask, order, sorted_mask, axis=1)
    return mask, np.take_along_axis(cum, upto[:, None], axis=1).ravel()


def coverage(P, n_cells, start, end, drifter_id, groups, levels=LEVELS, n_boot=N_BOOT, seed=SEED):
    """How often test moves ending in the box end inside each likely area, by group of start cells.

    Inputs:
        groups: name -> bool per cell.
    Returns:
        list of dicts: group, level, moves, what the sets predict (%), observed (%) and its interval.
    """
    stayed = end < n_cells
    s, e, ids = start[stayed], end[stayed], drifter_id[stayed]
    unique, rank = np.unique(ids, return_inverse=True)
    draws = resample_counts(unique.size, n_boot, seed)
    rows = []
    for level in levels:
        mask, held = likely_sets(P, n_cells, level)
        inside = mask[s, e]
        for name, cells in groups.items():
            g = cells[s]
            if not g.any():
                continue
            hits = np.bincount(rank[g], weights=inside[g].astype(float), minlength=unique.size)
            moves = np.bincount(rank[g], minlength=unique.size).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                low, high = percentile_interval(100 * (draws @ hits) / (draws @ moves), LEVEL)
            rows.append({"cells": name, "level_pct": 100 * level, "moves": int(g.sum()), "drifters": int(np.unique(ids[g]).size),
                         "predicted_pct": 100 * float(held[s[g]].mean()), "observed_pct": 100 * float(inside[g].mean()),
                         "ci_low": float(low), "ci_high": float(high)})
    return rows


def first_exit_counts(outcomes):
    """Per start and checkpoint, for a Kaplan-Meier estimate of having left the box.

    Inputs:
        outcomes: (n_checkpoints, n_starts) outcome codes at each 15-day checkpoint.
    Returns:
        (at_risk, left): bool arrays of the same shape. A start is at risk at a checkpoint if it hasn't yet been
        seen outside R and its position has been known at every checkpoint up to and including this one; left
        marks the checkpoint where it is first seen outside.
    """
    known = np.isin(outcomes, [IN_R, LEFT_R])
    all_known = np.logical_and.accumulate(known, axis=0)
    outside_before = np.concatenate([np.zeros((1, outcomes.shape[1]), dtype=bool),
                                     np.logical_or.accumulate(outcomes == LEFT_R, axis=0)[:-1]])
    at_risk = all_known & ~outside_before
    return at_risk, at_risk & (outcomes == LEFT_R)


def back_in_box(outcomes):
    """Of the starts seen outside R at some checkpoint and known at the last one, the share back in R at the last.

    This is why "outside on that day" understates "has left at some point".
    """
    left_before = (outcomes == LEFT_R).any(axis=0)
    known = np.isin(outcomes[-1], [IN_R, LEFT_R]) & left_before
    return {"moves": int(known.sum()), "observed_pct": 100 * float(np.mean(outcomes[-1][known] == IN_R))}


def left_box(P_steps, n_cells, start_cell, outcomes, drifter_id, bin_edges=EXIT_BINS, n_boot=N_BOOT, seed=SEED):
    """Predicted chance of having left against the Kaplan-Meier estimate, overall and by bin of predicted chance.

    Inputs:
        P_steps: P^k for k = the number of checkpoints.
        start_cell, drifter_id: per test start (all outcomes, as saved in the table).
        outcomes: (k, n_starts) outcome codes at checkpoints 1..k.
    Returns:
        list of dicts: group, starts, drifters, predicted %, observed % and its interval.
    """
    predicted = P_steps[start_cell, n_cells]
    at_risk, left = first_exit_counts(outcomes)
    unique, rank = np.unique(drifter_id, return_inverse=True)
    draws = resample_counts(unique.size, n_boot, seed)
    groups = [("all starts", np.ones(predicted.size, dtype=bool))]
    groups += [(f"predicted {100 * a:g}-{100 * min(b, 1):g}%", (predicted >= a) & (predicted < b)) for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    rows = []
    for name, g in groups:
        if not g.any():
            continue
        survive, reps = 1.0, np.ones(n_boot)
        for c in range(outcomes.shape[0]):
            n = np.bincount(rank[g], weights=at_risk[c][g].astype(float), minlength=unique.size)
            d = np.bincount(rank[g], weights=left[c][g].astype(float), minlength=unique.size)
            if n.sum() > 0:
                survive *= 1 - d.sum() / n.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                reps = reps * (1 - np.nan_to_num((draws @ d) / (draws @ n)))
        low, high = percentile_interval(100 * (1 - reps), LEVEL)
        rows.append({"cells": name, "moves": int(g.sum()), "drifters": int(np.unique(drifter_id[g]).size),
                     "predicted_pct": 100 * float(predicted[g].mean()), "observed_pct": 100 * (1 - survive),
                     "ci_low": float(low), "ci_high": float(high)})
    return rows


def plot(reliability_rows, exit_rows, coverage_rows, out_path):
    """(a) reliability of destination cells; (b) having left the box; (c) likely-area coverage, all start cells."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    style = {15: (BLUE, "o"), 90: (ORANGE, "s"), 360: (AQUA, "^")}

    def diagonal(ax, rows):
        ok = [v for v in rows if v["observed_pct"] > 0]
        low, top = min(v["predicted_pct"] for v in ok), max(max(v["predicted_pct"], v["ci_high"]) for v in ok)
        ax.plot([low, top], [low, top], color="0.6", ls=":", lw=1.5, label="predicted = observed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, frameon=False)

    def series(ax, rows):
        for days, (colour, marker) in style.items():
            r = [x for x in rows if x["days"] == days and x["observed_pct"] > 0]
            if not r:
                continue
            x, y = np.array([v["predicted_pct"] for v in r]), np.array([v["observed_pct"] for v in r])
            err = [y - np.array([v["ci_low"] for v in r]), np.array([v["ci_high"] for v in r]) - y]
            ax.errorbar(x, y, yerr=np.maximum(err, 0), color=colour, marker=marker, ms=7, lw=2, capsize=3, label=f"{days} days")

    ax = axes[0]
    rows = [r for r in reliability_rows if r["entries_weighted"] >= 5000]  # a handful of entries says little
    series(ax, rows)
    diagonal(ax, rows)
    ax.set_xlabel("Predicted chance of being in a cell (%)")
    ax.set_ylabel("How often test drifters were there (%)")
    ax.set_title("(a) Destination cells: do the probabilities hold?", fontsize=11, loc="left")

    ax = axes[1]
    rows = [r for r in exit_rows if r["cells"] != "all starts"]
    series(ax, rows)
    diagonal(ax, rows)
    ax.set_xlabel("Predicted chance of having left the box (%)")
    ax.set_ylabel("Observed, allowing for records that end (%)")
    ax.set_title("(b) Leaving the box", fontsize=11, loc="left")

    ax = axes[2]
    days_list = [d for d in style if any(c["days"] == d for c in coverage_rows)]
    for k, level in enumerate((50.0, 90.0)):
        r = [next(c for c in coverage_rows if c["days"] == d and c["level_pct"] == level and c["cells"] == "all with drifters") for d in days_list]
        x = np.arange(len(days_list)) + (k - 0.5) * 0.3
        y = np.array([v["observed_pct"] for v in r])
        err = [y - np.array([v["ci_low"] for v in r]), np.array([v["ci_high"] for v in r]) - y]
        ax.errorbar(x, y, yerr=err, fmt="o", ms=8, capsize=4, color=(BLUE, ORANGE)[k], label=f"{level:.0f}% area: observed")
        ax.scatter(x, [v["predicted_pct"] for v in r], marker="_", s=420, color="#1b2733", zorder=5,
                   label="what the area should hold" if k == 0 else None)
    ax.set_xticks(range(len(days_list)), [f"{d} days" for d in days_list])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test moves ending inside the area (%)")
    ax.set_title("(c) The app's 50% and 90% lines", fontsize=11, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.01, -0.09, f"Held-out test drifters, never used to fit or tune; the 2° shrinkage matrix fitted on train + valid drifters and chained in {TAU_DAYS}-day "
             "steps. Bars: 95% intervals from resampling test drifters.\nLonger horizons rest on longer-lived drifters. (b) is a Kaplan-Meier estimate on 15-day "
             "checkpoints, which counts a drifter that leaves and returns as having left. Source: src/inference/calibration.py.", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def lag_table(days, region, cache_exists):
    """The moves at a lag (as load_transitions), from the saved table or, for lags not saved, built from the caches."""
    path = table_path(days)
    if os.path.exists(path):
        return load_transitions(region, path, verbose=False)
    if not cache_exists:
        return None
    built = build_transitions(load_region(region), load_records(region), days)
    keep = np.isin(built["outcome"], [IN_R, LEFT_R])
    table = {k: built[k][keep] for k in ("drifter_id", "start_time", "start_lon", "start_lat", "end_lon", "end_lat", "drogued", "outcome")}
    table["exited"] = table["outcome"] == LEFT_R
    for k in ("start_lon", "start_lat", "end_lon", "end_lat"):
        table[k] = table[k].astype(float)
    return table


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    base = load_transitions(region, verbose=False)
    assignment = load_split(base["drifter_id"])
    parent = parent_states(region, fine, coarse)

    # As evaluate.py: alpha from train -> valid, fit on train + valid
    s_tr, e_tr, n_cells, _ = states_for(base, region, fine, assignment, "train")
    s_va, e_va, _, _ = states_for(base, region, fine, assignment, "valid")
    alpha, _ = tune_alpha(count_matrix(s_tr, e_tr, n_cells + 1), parent, s_va, e_va)
    s_fit, e_fit, _, _ = states_for(base, region, fine, assignment, ("train", "valid"))
    C_fit = count_matrix(s_fit, e_fit, n_cells + 1)
    P_fit = shrinkage_P(C_fit, prior_rows(C_fit, parent), alpha)
    s_all, _, _ = to_states(base, region, fine)
    _, drifters = per_cell_counts(s_all, base["drifter_id"], n_cells)
    groups = {"all with drifters": drifters > 0, f"under {N_MIN} drifters": (drifters > 0) & (drifters < N_MIN),
              f"{N_MIN}+ drifters": drifters >= N_MIN}
    cache_exists = os.path.exists(os.path.join(INTERIM_DIR, f"{region['name']}.npz"))

    # Checkpoints every 15 days out to the longest horizon, from the caches (row i is the same start at every lag)
    checkpoints, test_starts = None, None
    if cache_exists:
        data, records = load_region(region), load_records(region)
        built = build_transitions(data, records, TAU_DAYS)
        test_starts = set_mask(built["drifter_id"], assignment, "test") & np.isin(built["drifter_id"], list(assignment))
        start_cell, _, _ = cell_index(built["start_lon"][test_starts].astype(float), built["start_lat"][test_starts].astype(float), region, fine)
        start_ids = built["drifter_id"][test_starts]
        checkpoints = np.stack([build_transitions(data, records, TAU_DAYS * k)["outcome"][test_starts] for k in range(1, max(HORIZONS) + 1)])

    rel_rows, exit_rows, cov_rows, back_rows = [], [], [], []
    power = np.eye(P_fit.shape[0])
    done = 0
    for steps, days in HORIZONS.items():
        power = power @ np.linalg.matrix_power(P_fit, steps - done)
        done = steps
        table = lag_table(days, region, cache_exists)
        if table is None:
            print(f"{days} days: skipped (no saved table and no region cache)")
            continue
        missing = np.setdiff1d(table["drifter_id"], list(assignment))
        if missing.size:
            table = subset(table, ~np.isin(table["drifter_id"], missing))
        test = subset(table, set_mask(table["drifter_id"], assignment, "test"))
        start, end, _ = to_states(test, region, fine)
        ids = test["drifter_id"]
        print(f"{days} days: {ids.size:,} test moves from {np.unique(ids).size} drifters")
        for row in reliability(power, start, end, ids):
            rel_rows.append({"days": days, **row})
        for row in coverage(power, n_cells, start, end, ids, groups):
            cov_rows.append({"days": days, **row})
        if checkpoints is not None:
            for row in left_box(power, n_cells, start_cell, checkpoints[:steps], start_ids):
                exit_rows.append({"days": days, **row})
            if steps > 1:
                back_rows.append({"days": days, **back_in_box(checkpoints[:steps])})

    rows = [{"check": "reliability", **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}} for r in rel_rows]
    rows += [{"check": "likely area", **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}} for r in cov_rows]
    rows += [{"check": "having left the box", **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}} for r in exit_rows]
    rows += [{"check": "back in the box", **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}} for r in back_rows]
    for r in rows:
        print(r)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"calibration_tau{TAU_DAYS}d.csv")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_out = os.path.join(FIGURES_DIR, "calibration.png")
    plot(rel_rows, exit_rows, cov_rows, fig_out)
    print(f"Saved {out} and {fig_out}")
