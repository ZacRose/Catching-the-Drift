"""
year_intervals.py

Related task: Week 9 forensic review. Every interval so far comes from the drifter bootstrap, which answers
"how much would this change with a different sample of drifters?" Drifters in the same year share that year's
ocean, so it says little about the question a planner has next: "how much does this change from year to year,
and does the past predict later years?" Week 8 hinted at this: two 2010+ numbers fell just outside the product's
intervals. Two checks, on the product itself (2 deg shrinkage, alpha held at the product's value):

1. Whole-year intervals. Calendar years are resampled with replacement instead of drifters, using the same
   code (bootstrap.replicate_matrices: a group is a group), so a year's moves stay together. Each replicate
   refits P and recomputes the headline numbers. The width is compared with the drifter interval. There are
   only 29 years, of which the first nine are thin, so read these as a sensitivity check rather than a better
   interval: the ratio is the finding.

2. Does the past predict later years? The drifter split (split.py) holds out random drifters from every year,
   so training drifters share each test drifter's years. Here half of the drifters first seen from SPLIT_YEAR
   on are held out, and their moves are scored by a matrix fitted to moves that finished before SPLIT_YEAR. A
   control matrix is fitted to the same number of moves from drifters of all years (never the test drifters).
   Same test moves, same amount of data, so the gap is the cost of having no data from those years. Interval
   from resampling the test drifters, models held fixed, as score_intervals.py.

Outputs: outputs/results/year_intervals_tau15d.csv
Run: python3 -m src.inference.year_intervals   (about a minute; after src.linalg.coastal_release and
src.linalg.long_run, whose drifter intervals it compares with)
"""
import csv
import os
import numpy as np

from src.common.config import PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE, TAU_DAYS
from src.inference.bootstrap import (
    LEVEL, N_BOOT, SEED, drifter_totals, percentile_interval, replicate_matrices, resample_counts, resampled_means,
    shrinkage_fit,
)
from src.inference.estimators import count_matrix, log_lik
from src.inference.transition_data import load_transitions, parent_states, subset, to_states
from src.linalg.coastal_release import release_summary
from src.linalg.long_run import cell_areas_km2, centre, qsd_power, steps_to_mix, substochastic, years
from src.linalg.release_map import STEPS, propagate

SPLIT_YEAR = 2015
QUANTITIES = ("pct_left_R", "pct_in_coastal_zone", "centre_lat", "centre_lon")
LONG_RUN = ("years_in_box_expected", "qsd_centre_lat", "qsd_centre_lon", "years_tmix_from_coast")


def start_years(table):
    """Calendar year of each move's start."""
    return table["start_time"].astype("datetime64[Y]").astype(int) + 1970


def headline(P, release, zone, no_data, keep, region, cell_size, q_start=None):
    """The numbers compared: coastal release at STEPS and the long-run pattern."""
    out = {}
    for step, d in propagate(P, release).items():
        r = release_summary(d, zone, no_data, region, cell_size)
        for q in QUANTITIES:
            out[(q, step)] = r[q]
    Q = substochastic(P, keep)
    lam, q = qsd_power(Q, q_start)
    out[("years_in_box_expected", 0)] = float(years(1 / (1 - lam)))
    out[("qsd_centre_lat", 0)], out[("qsd_centre_lon", 0)] = centre(q, keep, region, cell_size)
    x0 = np.isin(keep, release).astype(float)
    out[("years_tmix_from_coast", 0)] = float(years(steps_to_mix(Q, q, x0 / x0.sum())))
    return out, q


def drifter_intervals(results_dir=RESULTS_DIR):
    """The saved drifter-bootstrap intervals for the same numbers: {(quantity, step): (low, high)}."""
    out = {}
    with open(os.path.join(results_dir, f"coastal_release_tau{TAU_DAYS}d.csv")) as f:
        for r in csv.DictReader(f):
            if r["model"].endswith("shrinkage") and r["quantity"] in QUANTITIES:
                out[(r["quantity"], int(r["steps"]))] = (float(r["ci_low"]), float(r["ci_high"]))
    with open(os.path.join(results_dir, f"long_run_tau{TAU_DAYS}d.csv")) as f:
        for r in csv.DictReader(f):
            if r["model"].endswith("shrinkage") and r["quantity"] in LONG_RUN and r["ci_low"]:
                out[(r["quantity"], 0)] = (float(r["ci_low"]), float(r["ci_high"]))
    return out


def holdout_sets(table, split_year=SPLIT_YEAR, seed=SEED, tau_days=TAU_DAYS):
    """Rows for the later-years test.

    Returns:
        (test, past, control), bool masks over the table's rows: test is half of the drifters first seen from
        split_year on; past is every other drifter's moves that finished before split_year; control is moves
        from a random draw of all non-test drifters, about as many as past. No test drifter's moves are in
        past or control.
    """
    ids, rank = np.unique(table["drifter_id"], return_inverse=True)
    first = np.full(ids.size, np.datetime64("2100-01-01"), dtype=table["start_time"].dtype)
    np.minimum.at(first, rank, table["start_time"])
    cut = np.datetime64(f"{split_year}-01-01").astype(table["start_time"].dtype)
    later = np.flatnonzero(first >= cut)
    rng = np.random.default_rng(seed)
    is_test = np.zeros(ids.size, dtype=bool)
    is_test[rng.permutation(later)[: later.size // 2]] = True
    test = is_test[rank]
    finished = table["start_time"] + np.timedelta64(tau_days, "D") < cut
    past = ~test & finished

    moves = np.bincount(rank[~test], minlength=ids.size)
    order = rng.permutation(np.flatnonzero(~is_test))
    take = order[: np.searchsorted(np.cumsum(moves[order]), past.sum()) + 1]
    chosen = np.zeros(ids.size, dtype=bool)
    chosen[take] = True
    return test, past, chosen[rank] & ~test


def later_years_score(table, region, fine, coarse, alpha, split_year=SPLIT_YEAR):
    """Score the later-years test moves under the past-only and the control matrix (see the docstring)."""
    test, past, control = holdout_sets(table, split_year)
    fit = shrinkage_fit(parent_states(region, fine, coarse), alpha)
    s_te, e_te, n_cells = to_states(subset(table, test), region, fine)
    ids = table["drifter_id"][test]
    n_states = n_cells + 1
    P = {}
    for name, mask in (("past", past), ("control", control)):
        s, e, _ = to_states(subset(table, mask), region, fine)
        P[name] = fit(count_matrix(s, e, n_states))
    lp_past, lp_control = log_lik(P["past"], s_te, e_te), log_lik(P["control"], s_te, e_te)
    draws = resample_counts(np.unique(ids).size, N_BOOT, SEED)
    all_moves = np.ones(ids.size, dtype=bool)
    low, high = percentile_interval(resampled_means(*drifter_totals(lp_past - lp_control, ids, all_moves), draws), LEVEL)
    return {"test_drifters": int(np.unique(ids).size), "test_moves": int(ids.size), "past_moves": int(past.sum()),
            "control_moves": int(control.sum()), "score_past_only": float(lp_past.mean()),
            "score_control": float(lp_control.mean()), "past_minus_control": float((lp_past - lp_control).mean()),
            "ci_low": float(low), "ci_high": float(high)}


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region, verbose=False)
    alpha = float(np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))["alpha"])
    start, end, n_cells = to_states(table, region, fine)
    _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)
    keep = np.flatnonzero(drifters > 0)
    zone = coastal_cells(region, fine, land_fraction(region, fine, land_paths()))
    release = zone[drifters[zone] > 0]
    fit = shrinkage_fit(parent_states(region, fine, coarse), alpha)
    P = fit(count_matrix(start, end, n_cells + 1))
    est, q_hat = headline(P, release, zone, drifters == 0, keep, region, fine)

    groups = start_years(table)
    print(f"Resampling {np.unique(groups).size} years, {N_BOOT} replicates")
    reps = {k: np.empty(N_BOOT) for k in est}
    for b, P_b in enumerate(replicate_matrices(start, end, groups, n_cells + 1, fit, N_BOOT)):
        out, _ = headline(P_b, release, zone, drifters == 0, keep, region, fine, q_hat)
        for k, v in out.items():
            reps[k][b] = v
    saved = drifter_intervals()

    rows = []
    for (q, step), value in est.items():
        low, high = percentile_interval(reps[(q, step)], LEVEL)
        d_low, d_high = saved.get((q, step), (np.nan, np.nan))
        ratio = (high - low) / (d_high - d_low) if d_high > d_low else np.nan
        rows.append({"check": "whole-year intervals", "quantity": q, "days": step * TAU_DAYS if step else "long run",
                     "estimate": round(float(value), 4), "drifter_ci_low": round(d_low, 4), "drifter_ci_high": round(d_high, 4),
                     "year_ci_low": round(float(low), 4), "year_ci_high": round(float(high), 4), "width_ratio": round(float(ratio), 2),
                     "note": ""})
        print(rows[-1])

    s = later_years_score(table, region, fine, coarse, alpha)
    note = (f"{s['test_moves']} moves from {s['test_drifters']} drifters first seen from {SPLIT_YEAR}; "
            f"fitted on {s['past_moves']} past moves vs {s['control_moves']} moves from all years")
    rows.append({"check": "later years", "quantity": "mean log-likelihood per move, past only minus control", "days": TAU_DAYS,
                 "estimate": round(s["past_minus_control"], 5), "drifter_ci_low": round(s["ci_low"], 5),
                 "drifter_ci_high": round(s["ci_high"], 5), "year_ci_low": "", "year_ci_high": "", "width_ratio": "", "note": note})
    for name in ("score_past_only", "score_control"):
        rows.append({"check": "later years", "quantity": name, "days": TAU_DAYS, "estimate": round(s[name], 5), "drifter_ci_low": "",
                     "drifter_ci_high": "", "year_ci_low": "", "year_ci_high": "", "width_ratio": "", "note": ""})
    print(s)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"year_intervals_tau{TAU_DAYS}d.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")
