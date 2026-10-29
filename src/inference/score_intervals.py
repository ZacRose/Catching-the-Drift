"""
score_intervals.py

Related task: Week 6 Computational - intervals on the held-out scores (open issue i), using the
drifter bootstrap, so a claim that one model beats another can be checked before it is made.

The models are fitted exactly as in evaluate.py (evaluate.fit_models: tuned on train -> valid,
refitted on train + valid) and then held fixed. Only the 378 TEST drifters are resampled: each
replicate draws 378 of them with replacement and recomputes every score from the drawn drifters'
moves. The interval says how much a score would change with a different set of test drifters. It
doesn't include the variation from refitting on different training drifters (bootstrap.py does
that for P itself).

Scores are means of log P(start -> end) over test moves (higher = better), for all test moves and
separately for moves starting in cells under and over N_MIN drifters, as in evaluate.py. Naive P
gives some test moves probability 0 (log = -inf), so for naive the interval is on the mean where
finite, plus one on the share of moves given probability 0.

Differences: for every pair of models on a grid, the later model's score minus the earlier one's
in the order persistence, neighbour-only, naive, shrinkage (positive = the later model is better).
Both scores come from the same replicate, so the interval is on the paired difference, which is
much tighter than comparing two separate intervals. Where naive is in the pair, only the moves
naive can score are used. pct_replicates_above_zero is the share of replicates in which the later
model came out ahead. A difference whose 95% interval excludes 0 is one we can report as real for
these test drifters.

Outputs: outputs/results/score_intervals_tau15d.csv and outputs/figures/score_intervals.png.

Run: python3 -m src.inference.score_intervals   (about a minute; needs drifter_split.csv)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.common.config import FIGURES_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.transitions import BASELINE_CELL_SIZE
from src.inference.bootstrap import (
    LEVEL, N_BOOT, SEED, drifter_totals, percentile_interval, resample_counts, resampled_means,
)
from src.inference.estimators import log_lik
from src.inference.evaluate import fit_models
from src.inference.split import load_split
from src.inference.transition_data import TAU_DAYS, load_transitions

ORDER = ["persistence", "neighbour", "naive", "shrinkage"]  # simplest first; differences are later minus earlier


def _rank(name):
    return next(i for i, key in enumerate(ORDER) if name.startswith(key))


def interval_row(grid, quantity, model, versus, cells, moves_used, values, mask, drifter_id, draws):
    """One output row: the estimate on the real test drifters and its bootstrap interval."""
    totals, moves = drifter_totals(values, drifter_id, mask)
    replicates = resampled_means(totals, moves, draws)
    low, high = percentile_interval(replicates, LEVEL)
    row = {"grid": grid, "quantity": quantity, "model": model, "versus": versus, "cells": cells,
           "moves": moves_used, "test_drifters": int(np.sum(moves > 0)), "test_transitions": int(mask.sum()),
           "estimate": float(values[mask].mean()), "ci_low": float(low), "ci_high": float(high),
           "pct_replicates_above_zero": ""}
    if quantity == "difference":
        row["pct_replicates_above_zero"] = round(float(100 * np.mean(replicates > 0)), 1)
    return row


def grid_intervals(table, region, assignment, cell_size, coarse_size=None, n_boot=N_BOOT, seed=SEED):
    """Score and difference intervals for every model on one grid (rows for the output table)."""
    models, test, _ = fit_models(table, region, assignment, cell_size, coarse_size)
    drifter_id = test["drifter_id"]
    draws = resample_counts(np.unique(drifter_id).size, n_boot, seed)
    logp = {name: log_lik(P, test["start"], test["end"]) for name, (_, P) in models.items()}
    names = sorted(models, key=_rank)
    grid = f"{cell_size:g} deg"

    rows = []
    for cells, group in test["groups"].items():
        for name in names:
            finite = np.isfinite(logp[name])
            if finite.all():
                rows.append(interval_row(grid, "mean log-likelihood", name, "", cells, "all",
                                         logp[name], group, drifter_id, draws))
            else:
                rows.append(interval_row(grid, "mean log-likelihood", name, "", cells, "where finite",
                                         np.where(finite, logp[name], 0.0), group & finite, drifter_id, draws))
                rows.append(interval_row(grid, "% of moves given probability 0", name, "", cells, "all",
                                         100.0 * ~finite, group, drifter_id, draws))
        for i, earlier in enumerate(names):
            for later in names[i + 1:]:
                both = np.isfinite(logp[earlier]) & np.isfinite(logp[later])
                used = "all" if both.all() else "where naive finite"
                diff = np.zeros(both.size)
                diff[both] = logp[later][both] - logp[earlier][both]
                rows.append(interval_row(grid, "difference", later, earlier, cells, used,
                                         diff, group & both, drifter_id, draws))
    return rows


def plot(rows, out_path):
    """One small panel per model comparison: the paired difference and its interval, per cell group."""
    diffs = [r for r in rows if r["quantity"] == "difference"]
    pairs = list(dict.fromkeys((r["grid"], r["model"], r["versus"]) for r in diffs))
    groups = list(dict.fromkeys(r["cells"] for r in diffs))
    colours = ["black", "#d62728", "#1f77b4"]
    n_col = 3
    n_row = int(np.ceil(len(pairs) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(12, 2.1 * n_row), layout="constrained", squeeze=False)
    for ax, (grid, later, earlier) in zip(axes.ravel(), pairs):
        for k, (cells, colour) in enumerate(zip(groups, colours)):
            r = next(r for r in diffs if (r["grid"], r["model"], r["versus"], r["cells"]) == (grid, later, earlier, cells))
            ax.errorbar(r["estimate"], -k, xerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]],
                        fmt="o", color=colour, ms=4, capsize=3, label=cells)
        ax.axvline(0, color="grey", lw=0.8, ls=":")
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, 0), max(hi, 0))
        ax.set_yticks([])
        ax.set_ylim(-len(groups) + 0.4, 0.6)
        moves = next(r["moves"] for r in diffs if (r["grid"], r["model"], r["versus"]) == (grid, later, earlier))
        note = " (moves naive can score)" if moves != "all" else ""
        ax.set_title(f"{grid}: {later} minus {earlier}{note}", fontsize=8.5, loc="left")
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.supxlabel("Difference in mean held-out log-likelihood per move (right of the dotted line: the first-named model is better)",
                  fontsize=9)
    fig.text(0.01, -0.07,
             f"Points: score on the 378 test drifters. Bars: 95% interval from {N_BOOT} bootstrap replicates of the test "
             f"drifters (models held fixed; tau = {TAU_DAYS} days).\nA bar that doesn't cross the dotted line is a "
             "difference that holds up for these test drifters. Each panel has its own scale.", fontsize=8.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region)
    assignment = load_split(table["drifter_id"])

    rows = grid_intervals(table, region, assignment, coarse) + grid_intervals(table, region, assignment, fine, coarse)
    for r in rows:
        if r["quantity"] == "difference":
            print(f"{r['grid']} {r['model']} - {r['versus']} [{r['cells']}, {r['moves']}]: {r['estimate']:.4f} "
                  f"({r['ci_low']:.4f} to {r['ci_high']:.4f}), later ahead in {r['pct_replicates_above_zero']}%")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, f"score_intervals_tau{TAU_DAYS}d.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "score_intervals.png")
    plot(rows, fig_path)
    print(f"Saved {table_path} and {fig_path}")
