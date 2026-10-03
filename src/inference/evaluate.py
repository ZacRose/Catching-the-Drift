"""
evaluate.py

Related task: Week 3 Quantitative - the evaluation design, run once as a preliminary
check (Week 7 reruns it on the final model, alongside the bootstrap).

Score = held-out log-likelihood: the average of log P(start -> end) over the test drifters'
transitions. Higher is better, and it rewards a model for putting probability where drifters
actually went. Three models:
- naive P (counts / row totals) - any unseen transition scores log 0 = -inf, so we also report
  the share of test transitions it gives zero probability; that share is the case for shrinkage;
- persistence ("stays in its cell"), with move probability eps chosen on the validation set;
- Dirichlet shrinkage of 2 deg rows toward their 4 deg parent, alpha chosen on validation.

Settings are tuned with models fitted on train and scored on valid; the chosen settings are
then refitted on train + valid and scored once on test. The alpha curve on the test set is
the prior-sensitivity check (how much the answer depends on alpha); alpha is never chosen from it.

Scores are only comparable within one grid: a grid with fewer cells has fewer places to put
probability, so the 4 deg scores are higher than the 2 deg ones automatically.

Run: python3 -m src.inference.evaluate   (needs outputs/results/drifter_split.csv; run split first)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from src.common.config import FIGURES_DIR, RESULTS_DIR, load_region_bounds
from src.gridding.transitions import BASELINE_CELL_SIZE
from src.inference.transition_data import TAU_DAYS, load_transitions, subset, to_states, parent_states
from src.inference.split import load_split, set_mask
from src.inference.estimators import (
    count_matrix, naive_P, persistence_P, prior_rows, shrinkage_P, log_lik, score,
)

ALPHAS = np.logspace(-1, 3.5, 19)  # 0.1 to ~3000 pseudo-transitions
EPSILONS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def states_for(table, region, cell_size, assignment, names):
    """(start, end, n_cells) for the transitions of drifters in the named sets."""
    rows = subset(table, set_mask(table["drifter_id"], assignment, names))
    return to_states(rows, region, cell_size)


def pick(grid, scores, name):
    """Value in grid with the best validation score; warn if it sits on the grid's edge."""
    scores = np.asarray(scores)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Validation log-likelihood is -inf for some {name}: a model gave probability 0.")
    i = int(np.argmax(scores))
    if i in (0, len(grid) - 1):
        print(f"  warning: best {name} = {grid[i]:g} is at the edge of the search grid - widen it")
    return float(grid[i])


def evaluate_grid(table, region, assignment, cell_size, coarse_size=None):
    """Score naive, persistence and (if coarse_size) shrinkage on one grid.

    Returns:
        (rows for the results table, alpha curve dict or None).
    """
    s_tr, e_tr, n_cells = states_for(table, region, cell_size, assignment, "train")
    s_va, e_va, _ = states_for(table, region, cell_size, assignment, "valid")
    s_fit, e_fit, _ = states_for(table, region, cell_size, assignment, ("train", "valid"))
    s_te, e_te, _ = states_for(table, region, cell_size, assignment, "test")
    n_states = n_cells + 1
    C_tr = count_matrix(s_tr, e_tr, n_states)
    C_fit = count_matrix(s_fit, e_fit, n_states)

    # Tune on train -> valid, then refit on train + valid and score once on test
    eps = pick(EPSILONS, [log_lik(persistence_P(n_states, e), s_va, e_va).mean() for e in EPSILONS], "eps")
    models = {"naive": ("", naive_P(C_fit)),
              "persistence": (f"eps={eps:g}", persistence_P(n_states, eps))}

    curve = None
    if coarse_size:
        # Tune alpha on validation, then score every alpha on test for the sensitivity curve
        parent = parent_states(region, cell_size, coarse_size)
        prior_tr, prior_fit = prior_rows(C_tr, parent), prior_rows(C_fit, parent)
        valid_ll = [log_lik(shrinkage_P(C_tr, prior_tr, a), s_va, e_va).mean() for a in ALPHAS]
        alpha = pick(ALPHAS, valid_ll, "alpha")
        test_ll = [log_lik(shrinkage_P(C_fit, prior_fit, a), s_te, e_te).mean() for a in ALPHAS]
        models[f"shrinkage to {coarse_size:g} deg"] = (f"alpha={alpha:.3g}", shrinkage_P(C_fit, prior_fit, alpha))
        curve = {"alpha": ALPHAS, "valid": np.array(valid_ll), "test": np.array(test_ll), "chosen": alpha}

    # Also score every model only where naive is finite: a comparison on naive's own terms
    naive_finite = np.isfinite(log_lik(models["naive"][1], s_te, e_te))
    test_ids = table["drifter_id"][set_mask(table["drifter_id"], assignment, "test")]
    results = []
    for name, (setting, P) in models.items():
        logp = log_lik(P, s_te, e_te)
        results.append({
            "grid": f"{cell_size:g} deg", "model": name, "setting": setting,
            "test_drifters": int(np.unique(test_ids).size),
            **score(logp),
            "mean_loglik_where_naive_finite": float(logp[naive_finite].mean()),
        })
    return results, curve


def plot_alpha_curve(curve, fine, coarse, out_path):
    """Held-out log-likelihood against prior strength alpha, for validation and test."""
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    ax.plot(curve["alpha"], curve["valid"], marker="o", label="Validation (fit on train)")
    ax.plot(curve["alpha"], curve["test"], marker="s", ls="--", label="Test (fit on train + valid)")
    ax.axvline(curve["chosen"], color="grey", lw=1, ls=":", label=f"Chosen alpha = {curve['chosen']:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("Prior strength alpha (pseudo-transitions per row)")
    ax.set_ylabel("Mean held-out log-likelihood")
    ax.set_title(f"How much the {fine:g}° estimate depends on alpha (rows pulled toward their "
                 f"{coarse:g}° cell, tau = {TAU_DAYS} days)", fontsize=10)
    ax.legend()
    fig.text(0.01, -0.09, "Higher is better: the average log-probability the model gave to where held-out "
             "drifters actually went.\nSmall alpha trusts each cell's own counts; large alpha copies "
             f"the {coarse:g}° row. Alpha was chosen on the validation drifters only.", fontsize=8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    region = load_region_bounds()
    fine, coarse = region["cell_size_deg"], BASELINE_CELL_SIZE
    table = load_transitions(region)
    assignment = load_split(table["drifter_id"])

    coarse_rows, _ = evaluate_grid(table, region, assignment, coarse)
    fine_rows, curve = evaluate_grid(table, region, assignment, fine, coarse)
    rows = coarse_rows + fine_rows
    for row in rows:
        print(row)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table_path = os.path.join(RESULTS_DIR, f"evaluation_tau{TAU_DAYS}d.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURES_DIR, "alpha_sensitivity.png")
    plot_alpha_curve(curve, fine, coarse, fig_path)
    print(f"Saved {table_path} and {fig_path}")
