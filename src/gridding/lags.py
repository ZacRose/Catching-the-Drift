"""
lags.py

Related task: Week 7 Computational - build the transition table at other time steps (proposal
section 5 timeline). Week 3 built it for tau = 15 days only.

Lags built, and what each one is for:
    30 days   the Markov check in the proposal (section 4): does applying the 15-day matrix twice
              match a matrix fitted directly to 30-day moves, P(15 d)^2 = P(30 d)? Quant's
              src/inference/markov_check.py (branch week7-quant) reads
              data/processed/transitions_tau30d.npz when it exists, instead of 30-day moves
              chained together from the 15-day table.
    90 days   the same check for the 3-month map we headline: P(15 d)^6 against P(90 d).
    60 days   two months (P(15 d)^4, P(30 d)^2), between 30 and 90 days, so a trend with the
              lag is visible rather than two points.
    5, 10     shorter steps for the Week 8 time-step robustness check. 5 days is still longer
              than the 2-3 days after which a drifter's motion stops depending on its recent
              past (proposal section 4), and close to the ~8 days a median drogued drifter takes
              to cross a 1 deg cell (data_quality.py), which the Week 8 1 deg rerun needs.
15 days is rebuilt too, but only to check it against the committed table; that file isn't rewritten.

Every table is built by the same function as the 15-day one (transitions.build_transitions with
a different tau_days): the same starts (each drifter's 00:00 UTC reading in R, every day), the
same outcome codes and the same columns, saved as transitions.table_path(tau_days). So row i is
the same drifter-day start in every table, and any script that reads the 15-day table reads the
others unchanged if given the other path.

Reading a long-lag table: the destination is where the drifter is at start + tau, whatever
happened in between. A drifter that left the box and came back by then is an ordinary in-R move
here, whereas P(15 d)^k counts it as lost, because "left R" is absorbing (decision 11b). So a
gap between P(15 d)^k and a directly fitted P(k x 15 d) mixes two things: memory in the motion
(the Markov question) and re-entry. markov_check.py separates the two.

Checks, run every time (outputs/results/lag_checks.csv; the script stops before saving if any fail):
1. Rebuild: the 15-day table built here is identical, array for array, to the committed
   data/processed/transitions_tau15d.npz, so the new lags come from the pipeline behind every
   result so far.
2. Composition: a drifter in R at start + a has its own row in the b-day table starting at
   start + a (one row per drifter-day), and that row's destination is the drifter's position at
   start + a + b. So for every such start, the (a + b)-day table must give exactly that row's
   outcome and end position. Checked for 5 + 5 = 10, 15 + 15 = 30, 30 + 30 = 60 and
   30 + 60 = 90 days, on every start where it applies.
3. Brute force: for a random sample of each outcome at each lag, the answer is worked out again
   from the raw readings and record stretches with plain masks, not the sorted-key search that
   build_transitions uses. This is the Week 3 check (project_plan.txt, Week 3 Computational
   step 6), now in code rather than only in the AI log (part of open issue f).

Summary (outputs/results/lag_summary.csv, outputs/figures/lag_summary.png), one row per lag:
- outcome shares. "Record ended" also counts starts whose start + tau falls after the last
  reading in the dataset (2022-10-31): that is the data stopping, not the drifter, and it grows
  with tau, so it is reported separately (pct_past_end_of_data, part of pct_record_ended);
- median distance moved, for moves that end in R;
- the share of counted moves that end in the cell they started in, at 1, 2 and 4 deg. Near 100%
  means P is close to "stay put" at that lag and grid, so P^n needs many steps to say anything
  and each step adds artificial dispersion (proposal section 2);
- the share of ocean cells with at least N_MIN distinct drifters starting a counted move (the
  adequacy rule, decision 3; ocean cells as in transition_adequacy.py).

Run: python3 -m src.gridding.lags
(needs the region and records caches in data/interim/; 2-3 minutes on the full data, 2026-10-29)
"""
import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, RESULTS_DIR, load_region_bounds
from src.ingest.load import load_region
from src.ingest.drifter_records import load_records
from src.gridding.transitions import (
    BASELINE_CELL_SIZE, ENDED, GAP, IN_R, LEFT_R, OUTCOME_NAMES, TAU_DAYS,
    _hours, _key, build_transitions, table_path, transition_counts,
)
from src.inference.adequacy import N_MIN
from src.inference.displacement_eda import displacement_km
from src.inference.transition_adequacy import week1_proxy

LAGS = [5, 10, 15, 30, 60, 90]  # days; 15 is the committed table, rebuilt only as a check
COMPOSITIONS = [(5, 5), (15, 15), (30, 30), (30, 60)]  # (a, b): table a then table b must equal table a + b
BRUTE_FORCE_PER_OUTCOME = 25
SEED = 3001


def _same(a, b):
    """Element-wise equality that counts two NaNs as equal (end positions are NaN unless in R)."""
    return (a == b) | (np.isnan(a) & np.isnan(b))


def rebuild_check(table, committed):
    """Number of columns that differ between a rebuilt table and the committed one (0 = identical).

    Inputs:
        table: from build_transitions().
        committed: dict loaded from the committed .npz (tau_days already taken out).
    """
    if sorted(table) != sorted(committed):
        return len(set(table) ^ set(committed))
    differ = 0
    for column, values in table.items():
        if values.dtype.kind == "f":
            differ += int(not np.all(_same(values, committed[column])))
        else:
            differ += int(not np.array_equal(values, committed[column]))
    return differ


def composition_check(first, second, combined, first_days):
    """Compare the (a + b)-day table with following each drifter through the a-day, then the b-day table.

    Inputs:
        first, second, combined: tables for lags a, b and a + b, built from the same readings
            (so row i is the same start in all three).
        first_days: a, the first table's lag in days.
    Returns:
        dict: rows_checked (starts still in R after a days, the only ones where the rule applies)
        and mismatches (rows where combined disagrees on outcome or end position, plus any start
        whose drifter has no row at start + a; both should be 0).
    """
    for table in (second, combined):
        if not (np.array_equal(first["drifter_id"], table["drifter_id"])
                and np.array_equal(first["start_time"], table["start_time"])):
            raise ValueError("Tables don't share the same starts; build them from the same readings.")
    _, rank = np.unique(first["drifter_id"], return_inverse=True)
    keys = _key(rank, _hours(first["start_time"]))
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    in_r = np.flatnonzero(first["outcome"] == IN_R)
    target = keys[in_r] + 24 * first_days
    pos = np.minimum(np.searchsorted(sorted_keys, target), sorted_keys.size - 1)
    found = sorted_keys[pos] == target
    nxt = order[pos]  # the drifter's row starting a days later (only meaningful where found)

    agree = ((combined["outcome"][in_r] == second["outcome"][nxt])
             & _same(combined["end_lon"][in_r], second["end_lon"][nxt])
             & _same(combined["end_lat"][in_r], second["end_lat"][nxt]))
    return {"rows_checked": int(in_r.size), "mismatches": int(np.sum(~found) + np.sum(found & ~agree))}


def brute_force_check(data, records, table, tau_days, per_outcome=BRUTE_FORCE_PER_OUTCOME, seed=SEED):
    """Re-derive a sample of rows straight from the readings and record stretches, one row at a time.

    Inputs:
        data, records: what the table was built from (load_region(), load_records()).
        table: from build_transitions(data, records, tau_days).
        per_outcome: rows sampled from each outcome (all of them if fewer).
    Returns:
        dict: rows_checked and mismatches (rows where the outcome or end position differs; should be 0).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for code in OUTCOME_NAMES:
        candidates = np.flatnonzero(table["outcome"] == code)
        rows.extend(rng.choice(candidates, size=min(per_outcome, candidates.size), replace=False))

    mismatches = 0
    for i in rows:
        drifter = table["drifter_id"][i]
        target = table["start_time"][i] + np.timedelta64(tau_days, "D")
        hit = np.flatnonzero((data["drifter_id"] == drifter) & (data["time"] == target))
        if hit.size:  # a reading inside R at exactly start + tau
            expected = IN_R
            end = (np.float32(data["lon"][hit[0]]), np.float32(data["lat"][hit[0]]))
        else:
            mine = records["seg_id"] == drifter
            reporting = np.any((records["seg_start"][mine] <= target) & (target <= records["seg_end"][mine]))
            if reporting:
                expected = LEFT_R
            elif target > records["seg_end"][mine].max():
                expected = ENDED
            else:
                expected = GAP
            end = (np.nan, np.nan)
        same_end = _same(np.array([table["end_lon"][i], table["end_lat"][i]]), np.array(end, dtype=np.float32))
        mismatches += int(table["outcome"][i] != expected or not same_end.all())
    return {"rows_checked": len(rows), "mismatches": mismatches}


def lag_summary(table, tau_days, region, ocean, dataset_end, cell_sizes):
    """One CSV row describing a lag's table (see the module docstring for each quantity).

    Inputs:
        table: from build_transitions().
        tau_days: its lag.
        region: dict from config/region.yaml.
        ocean: dict cell size -> boolean mask of ocean cells (cells with any hourly reading).
        dataset_end: time of the last reading of any drifter in the table.
        cell_sizes: grid sizes to report, in degrees.
    """
    outcome = table["outcome"]
    target = table["start_time"] + np.timedelta64(tau_days, "D")
    row = {"tau_days": tau_days, "starts": int(outcome.size),
           "drifters": int(np.unique(table["drifter_id"]).size)}
    for code, name in OUTCOME_NAMES.items():
        row[f"pct_{name.replace(' ', '_')}"] = round(float(100 * np.mean(outcome == code)), 2)
    row["pct_past_end_of_data"] = round(float(100 * np.mean((outcome == ENDED) & (target > dataset_end))), 2)
    row["counted_moves"] = int(np.sum(outcome <= LEFT_R))

    in_r = outcome == IN_R
    moves = {k: table[k][in_r] for k in ("start_lon", "start_lat", "end_lon", "end_lat")}
    _, _, dist = displacement_km({**moves, "exited": np.zeros(in_r.sum(), dtype=bool)})
    row["median_km"] = round(float(np.median(dist)), 1)

    for size in cell_sizes:
        counts, start_drifters = transition_counts(table, region, size)
        inside = counts[:-1, :-1]  # moves that end in a cell of R ("left R" excluded)
        row[f"pct_same_cell_{size:g}deg"] = round(float(100 * np.trace(inside) / inside.sum()), 1)
        row[f"pct_adequate_{size:g}deg"] = round(float(100 * np.mean(start_drifters[ocean[size]] >= N_MIN)), 1)
    return row


def plot(rows, cell_sizes, out_path):
    """Three panels against the lag: outcome shares, moves ending in their start cell, adequate cells."""
    # Colour-blind-checked categorical colours (blue, orange, aqua, yellow); markers differ too,
    # so no line depends on colour alone
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    markers = ["o", "s", "^", "D"]
    lags = [r["tau_days"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

    outcome_lines = [("pct_left_R", "left the box"), ("pct_record_ended", "record ended (all)"),
                     ("pct_past_end_of_data", "  of which: the dataset stops (2022-10-31)"),
                     ("pct_gap_in_record", "gap in record")]
    panels = [
        (axes[0], outcome_lines, "% of daily starts", "(a) Starts that don't end in a cell of the box"),
        (axes[1], [(f"pct_same_cell_{s:g}deg", f"{s:g}° cells") for s in cell_sizes],
         "% of moves within the box", "(b) Moves that end in the cell they started in"),
        (axes[2], [(f"pct_adequate_{s:g}deg", f"{s:g}° cells") for s in cell_sizes],
         "% of ocean cells", f"(c) Cells with at least {N_MIN} drifters (adequate)"),
    ]
    for ax, lines, ylabel, title in panels:
        for (key, label), colour, marker in zip(lines, colours, markers):
            ax.plot(lags, [r[key] for r in rows], color=colour, marker=marker, markersize=7, linewidth=2, label=label)
        ax.axvline(TAU_DAYS, color="0.6", linestyle=":", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xticks(lags, [str(t) for t in lags])
        ax.minorticks_off()
        ax.set_xlabel("Time step tau (days, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    axes[1].set_ylim(0, 100)
    axes[2].set_ylim(0, 100)
    fig.suptitle("How the transition table changes with the time step tau (dotted line: tau = 15 days, used so far)")
    fig.text(0.01, -0.2,
             "Each table has one start per drifter per day in the box; the destination is the same drifter's position "
             "tau days later.\n(a) Longer steps lose more starts: more drifters leave the box or stop reporting first. "
             "Part of 'record ended' is only the dataset stopping, not the drifter. Starts with no known destination\n"
             "aren't counted, so long-step matrices rest only on drifters that kept reporting that long.\n"
             "(b) Where nearly every move stays "
             "in its cell, the matrix is close to 'stay put' at that step and grid. (c) Distinct drifters starting a "
             "move with a known destination, per cell.\nSource: src/gridding/lags.py; numbers in "
             "outputs/results/lag_summary.csv.", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    t0 = time.time()
    region = load_region_bounds()
    data = load_region(region)
    records = load_records(region)
    # 1 deg too, because Week 8 reruns at 1, 2 and 4 deg and needs to know which lags suit which grid
    cell_sizes = (1.0, region["cell_size_deg"], BASELINE_CELL_SIZE)
    ocean = {size: week1_proxy(region, size, data)[1] for size in cell_sizes}
    dataset_end = records["seg_end"].max()

    tables = {}
    for tau in LAGS:
        tables[tau] = build_transitions(data, records, tau)
        print(f"tau = {tau} days: {tables[tau]['outcome'].size:,} starts built ({time.time() - t0:.0f} s)")

    checks = []
    committed = dict(np.load(table_path(TAU_DAYS)))
    committed.pop("tau_days")
    checks.append({"check": "rebuilt table matches the committed one (columns that differ)",
                   "lag_days": TAU_DAYS, "rows_checked": int(committed["outcome"].size),
                   "mismatches": rebuild_check(tables[TAU_DAYS], committed)})
    for a, b in COMPOSITIONS:
        result = composition_check(tables[a], tables[b], tables[a + b], a)
        checks.append({"check": f"{a} days then {b} days equals {a + b} days", "lag_days": a + b, **result})
    for tau in LAGS:
        result = brute_force_check(data, records, tables[tau], tau)
        checks.append({"check": f"brute-force lookup, up to {BRUTE_FORCE_PER_OUTCOME} rows per outcome",
                       "lag_days": tau, **result})
    for c in checks:
        print(c)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    checks_path = os.path.join(RESULTS_DIR, "lag_checks.csv")
    _write_csv(checks, checks_path)
    if any(c["mismatches"] for c in checks):
        raise SystemExit(f"A check failed (see {checks_path}); no tables saved.")

    for tau in LAGS:
        if tau == TAU_DAYS:
            continue  # identical to the committed table (check 1), so leave that file alone
        np.savez_compressed(table_path(tau), **tables[tau], tau_days=tau)
        print(f"Saved {table_path(tau)}")

    rows = [lag_summary(tables[tau], tau, region, ocean, dataset_end, cell_sizes) for tau in LAGS]
    for row in rows:
        print(row)
    summary_path = os.path.join(RESULTS_DIR, "lag_summary.csv")
    _write_csv(rows, summary_path)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    figure_path = os.path.join(FIGURES_DIR, "lag_summary.png")
    plot(rows, cell_sizes, figure_path)
    print(f"Saved {checks_path}, {summary_path} and {figure_path} ({time.time() - t0:.0f} s in all)")
