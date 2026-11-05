"""
run_all.py

Related task: objective O4, "documented, tested code that reproduces the analysis". Every script, in the order
one's output feeds the next, so the analysis can be redone from the data with one command. Each runs as it would
from the command line, in its own process. Everything is reproducible: the seeds are fixed, and rerunning gives
the same numbers (see the forensic review in project_plan.txt, section 8, for the check and the timings).

The first two steps read the cloud copy of the GDP once and cache it in data/interim/ (about 10 minutes, and about
3 GB of memory). The caches aren't in git, but the transition tables are, so --from-tables starts after them and
needs no internet or caches; the scripts that use hourly readings are then left out.

Run: python3 -m src.run_all                 (about 1 hour 50 minutes on a 10-core laptop once the caches exist: each step timed on
                                             2026-11-05 and added up, see project_plan.txt section 9)
     python3 -m src.run_all --from-tables   (no caches needed; about 1 hour 45 minutes)
     python3 -m src.run_all --list          (just print the order)
"""
import argparse
import subprocess
import sys
import time

from src.common.config import REPO_ROOT

NEEDS_CACHE = [
    "src.ingest.drifter_records",       # one reading per drifter record, cached
    "src.gridding.coverage",            # Week 2: where and when drifters were present
    "src.inference.adequacy",           # Week 1: how many cells have enough drifters
    "src.inference.sparsity_eda",       # Week 2: how the data are spread across cells
    "src.inference.data_quality",       # Week 2: missing values, drogue, speed
    "src.gridding.trajectory_map",      # Week 5: the study-region figure
    "src.gridding.transitions",         # Week 3: the transition table, 2 deg counts, 4 deg naive P
    "src.gridding.lags",                # Week 7: the table at other time steps, with its checks
    "src.inference.data_forensics",     # Week 9: positions against velocities, drogue flags, decision 8, exit edges
]
FROM_TABLES = [
    "src.inference.split",              # Week 3: train, validation and test drifters
    "src.inference.evaluate",           # Weeks 3, 6: held-out scores
    "src.inference.transition_adequacy",  # Week 3: adequacy on the moves themselves
    "src.inference.displacement_eda",   # Week 3: displacement by drogue, season and era
    "src.linalg.release_map",           # Week 4: proposal Figure 1
    "src.inference.shrinkage_fit",      # Week 6: the product
    "src.gridding.coast",               # Week 6: the Benguela coastal cells
    "src.inference.bootstrap",          # Week 6: uncertainty in P
    "src.inference.score_intervals",    # Week 6: intervals on the held-out scores
    "src.linalg.coastal_release",       # Week 6: where material from the coast goes
    "src.inference.markov_check",       # Week 7: is P(15 d)^k the same as P(k x 15 d)?
    "src.linalg.long_run",              # Week 7: the long-run pattern
    "src.inference.prior_sensitivity",  # Week 7: do the results depend on the prior?
    "src.linalg.product",               # Week 7: the product another analyst loads
    "src.inference.robustness",         # Week 8: grids, lags, drogue status, years
    "src.inference.region_swap",        # Week 8: the same code on another box
    "src.forecasting.seasonal_forecast",   # Weeks 6-8: RQ4
    "src.forecasting.seasonal_robustness",  # Week 9: does the seasonal signal survive each choice?
    "src.linalg.connectivity",          # Week 9: RQ1, which subregions supply which, and the Orange Basin
    "src.inference.calibration",        # Week 9: do the probabilities mean what they say?
    "src.inference.year_intervals",     # Week 9: year-to-year change, and predicting later years
    "src.inference.drogue_comparison",  # Week 9: all drifters against undrogued only, and the undrogued matrix
    "src.forecasting.seasonal_matrices",  # Week 9: 2 deg seasonal matrices, and crossing seasons
    "src.inference.interval_coverage",  # Week 9: do the intervals contain a known truth?
    "src.app.export_app_data",          # Week 9: the data the app loads
    "src.app.check_intervals",          # Week 9: the app's intervals, recomputed the slow way
]


def steps(from_tables=False):
    """The modules to run, in order."""
    return FROM_TABLES if from_tables else NEEDS_CACHE + FROM_TABLES


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--from-tables", action="store_true", help="start from the committed transition tables")
    ap.add_argument("--list", action="store_true", help="print the steps and stop")
    args = ap.parse_args()
    t0 = time.time()
    for module in steps(args.from_tables):
        if args.list:
            print(module)
            continue
        print(f"--- {module} ({(time.time() - t0) / 60:.1f} min so far)", flush=True)
        subprocess.run([sys.executable, "-m", module], cwd=REPO_ROOT, check=True)
    if not args.list:
        print(f"Done in {(time.time() - t0) / 60:.1f} minutes.")
