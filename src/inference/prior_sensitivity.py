"""
prior_sensitivity.py

Related task: Week 7 Quantitative - the prior-sensitivity check (proposal RQ2: how sensitive is the
shrinkage estimate to alpha?), carried through to the answers we report, not just the score.

alpha_sensitivity.png (evaluate.py) already shows the held-out score for every alpha. That says
which alpha predicts best, not whether the alpha choice changes what the report tells a reader.
So here every headline number is recomputed under other priors and set beside the product's:
- alpha = 0.1, 0.3, 1 (the product, 56.2), 3 and 10 times the chosen value, with the prior built as
  in the product (each 4 deg destination shared among its 2 deg cells by arrivals);
- the same alpha with the 4 deg destination shared evenly among its four 2 deg cells (the other
  reasonable reading of "borrow from the 4 deg cell", Week 4 issue k);
- no prior at all (2 deg naive P), for reference.
For each: the held-out test score (fit on train + valid, as evaluate.py), how far the matrix itself
moves (median total variation of a row from the product's), the Benguela coastal release after 15
days, 3 months and 1 year (coastal_release.py), and the long-run numbers (long_run.py).

How to read it: a prior change matters only if it moves a number by more than the number's own
bootstrap interval (coastal_release_tau15d.csv, long_run_tau15d.csv). The figure shades that interval
around the product's value. The test drifters were never used to choose alpha; scoring every alpha
on them here is a check, not a selection (as in evaluate.py).

Outputs: outputs/results/prior_sensitivity_tau15d.csv and outputs/figures/prior_sensitivity.png.

Run: python3 -m src.inference.prior_sensitivity   (about a minute; needs shrinkage_P_2deg_tau15d.npz,
and, for the shaded intervals only, coastal_release_tau15d.csv and long_run_tau15d.csv)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, PROCESSED_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.coast import coastal_cells, land_fraction, land_paths
from src.gridding.grid import per_cell_counts
from src.gridding.transitions import BASELINE_CELL_SIZE
from src.inference.adequacy import N_MIN
from src.inference.estimators import count_matrix, log_lik, naive_P, prior_rows, shrinkage_P
from src.inference.evaluate import states_for
from src.inference.split import load_split
from src.inference.transition_data import TAU_DAYS, load_transitions, parent_states, to_states
from src.linalg.coastal_release import release_summary
from src.linalg.long_run import cell_areas_km2, summarise
from src.linalg.release_map import STEPS, propagate

MULTIPLIERS = [0.1, 0.3, 1.0, 3.0, 10.0]
HEADLINES = [  # (column, label for the figure, file and model/quantity holding its bootstrap interval)
    ("test_loglik_all", "held-out score per move", None),
    ("release_pct_coastal_15d", f"% in coastal zone after {TAU_DAYS} days",
     ("coastal_release", f"{TAU_DAYS}", "pct_in_coastal_zone")),
    ("release_pct_left_1y", "% left the box after 1 year", ("coastal_release", f"{24 * TAU_DAYS}", "pct_left_R")),
    ("release_centre_lon_1y", "centre after 1 year (lon, deg)", ("coastal_release", f"{24 * TAU_DAYS}", "centre_lon")),
    ("years_in_box_expected", "expected years in the box (long run)", ("long_run", "", "years_in_box_expected")),
    ("years_tmix_from_coast", "years to forget the coastal start", ("long_run", "", "years_tmix_from_coast")),
]


def variants(alpha):
    """(name, split, alpha) for every prior checked; alpha None = naive P."""
    out = [(f"alpha x {m:g}", "arrivals", m * alpha) for m in MULTIPLIERS]
    out.append(("even split, alpha x 1", "even", alpha))
    out.append(("no prior (naive)", None, None))
    return out


def fit_variant(counts, parent, split, alpha):
    """P for one prior variant (naive P when alpha is None)."""
    if alpha is None:
        return naive_P(counts)
    return shrinkage_P(counts, prior_rows(counts, parent, split=split), alpha)


def heldout(C_fit, parent, split, alpha, s_te, e_te, thin):
    """Mean held-out log-likelihood over all test moves and by cell group; % given probability 0."""
    logp = log_lik(fit_variant(C_fit, parent, split, alpha), s_te, e_te)
    out = {"test_pct_zero_prob": float(100 * np.mean(~np.isfinite(logp)))}
    for name, g in (("all", np.ones_like(thin)), ("thin", thin), ("adequate", ~thin)):
        out[f"test_loglik_{name}"] = float(logp[g].mean()) if np.isfinite(logp[g]).all() else float("-inf")
    return out


def bootstrap_bands():
    """{(file, days, quantity): (low, high)} for the product, from the Week 6 and Week 7 CSVs (if present)."""
    bands = {}
    path = os.path.join(RESULTS_DIR, f"coastal_release_tau{TAU_DAYS}d.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["model"].endswith("shrinkage"):
                bands[("coastal_release", r["days"], r["quantity"])] = (float(r["ci_low"]), float(r["ci_high"]))
    path = os.path.join(RESULTS_DIR, f"long_run_tau{TAU_DAYS}d.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["model"].endswith("shrinkage") and r["ci_low"]:
                bands[("long_run", "", r["quantity"])] = (float(r["ci_low"]), float(r["ci_high"]))
    return bands


def plot(rows, alpha, bands, out_path):
    """One small panel per headline number against alpha, with the product's bootstrap interval shaded."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
    sweep = [r for r in rows if r["prior"].startswith("alpha")]
    even = next(r for r in rows if r["prior"].startswith("even"))
    naive = next(r for r in rows if r["prior"].startswith("no prior"))
    x = np.array([r["alpha"] for r in sweep])
    for ax, (col, label, band) in zip(axes.ravel(), HEADLINES):
        if band in bands:
            ax.axhspan(*bands[band], color="#1f77b4", alpha=0.15, lw=0, label="product's 95% bootstrap interval")
        ax.plot(x, [r[col] for r in sweep], marker="o", color="#1f77b4", label="4° prior, shared by arrivals")
        ax.scatter([alpha], [even[col]], marker="s", color="#ff7f0e", zorder=5, label="4° prior, shared evenly")
        if np.isfinite(naive[col]):
            ax.axhline(naive[col], color="grey", ls="--", lw=1, label="no prior (naive)")
        ax.axvline(alpha, color="grey", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("prior strength alpha")
        ax.set_title(label, fontsize=10, loc="left")
    axes[0, 0].text(0.02, 0.04, "naive: -inf (gives probability 0\nto some test moves)", transform=axes[0, 0].transAxes,
                    fontsize=7.5, color="grey")
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8.5)
    fig.suptitle(f"Do the 2° results depend on the prior? (dotted line: the chosen alpha = {alpha:.0f}, picked on "
                 "validation drifters)", fontsize=11)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region, verbose=False)
    assignment = load_split(table["drifter_id"])
    product = np.load(os.path.join(PROCESSED_DIR, f"shrinkage_P_{fine:g}deg_tau{TAU_DAYS}d.npz"))
    alpha = float(product["alpha"])
    parent = parent_states(region, fine, coarse)

    # Held-out: fit on train + valid, score on test (as evaluate.py)
    s_fit, e_fit, n_cells, _ = states_for(table, region, fine, assignment, ("train", "valid"))
    s_te, e_te, _, _ = states_for(table, region, fine, assignment, "test")
    C_fit = count_matrix(s_fit, e_fit, n_cells + 1)

    # All drifters: the product's data, cells, coastal release and areas
    start, end, _ = to_states(table, region, fine)
    counts = count_matrix(start, end, n_cells + 1)
    _, drifters = per_cell_counts(start, table["drifter_id"], n_cells)
    thin = drifters[s_te] < N_MIN
    keep = np.flatnonzero(drifters > 0)
    area = cell_areas_km2(region, fine)
    zone = coastal_cells(region, fine, land_fraction(region, fine, land_paths()))
    release = zone[drifters[zone] > 0]
    P_product = product["P"]
    _, q_product = summarise(P_product, keep, area, release, region, fine, full=False)

    rows = []
    for name, split, a in variants(alpha):
        P = fit_variant(counts, parent, split, a)
        if name == "alpha x 1" and not np.allclose(P, P_product):
            raise ValueError("alpha x 1 should reproduce the saved product; rerun src.inference.shrinkage_fit.")
        row = {"prior": name, "alpha": a if a is not None else 0.0,
               **heldout(C_fit, parent, split, a, s_te, e_te, thin)}
        tv = 0.5 * np.abs(P[keep] - P_product[keep]).sum(axis=1)
        row["median_row_tv_vs_product"] = float(np.median(tv))
        row["max_row_tv_vs_product"] = float(np.max(tv))
        dists = propagate(P, release)
        for step, when in ((1, f"{TAU_DAYS}d"), (6, "3mo"), (24, "1y")):
            r = release_summary(dists[step], zone, drifters == 0, region, fine)
            row[f"release_pct_left_{when}"] = r["pct_left_R"]
            row[f"release_pct_coastal_{when}"] = r["pct_in_coastal_zone"]
            row[f"release_centre_lat_{when}"] = r["centre_lat"]
            row[f"release_centre_lon_{when}"] = r["centre_lon"]
        long_run, q = summarise(P, keep, area, release, region, fine, full=False, q_start=q_product)
        for k in ("lambda_1", "years_in_box_expected", "years_mixing_efold", "years_tmix_from_coast",
                  "qsd_centre_lat", "qsd_centre_lon"):
            row[k] = long_run[k]
        row["qsd_tv_vs_product"] = float(0.5 * np.abs(q - q_product).sum())
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    assert STEPS == (1, 6, 24)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_out = os.path.join(RESULTS_DIR, f"prior_sensitivity_tau{TAU_DAYS}d.csv")
    with open(table_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_out = os.path.join(FIGURES_DIR, "prior_sensitivity.png")
    plot(rows, alpha, bootstrap_bands(), fig_out)
    print(f"Saved {table_out} and {fig_out}")
