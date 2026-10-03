"""
estimators.py

Related task: Week 3 Quantitative - "settle the estimator". Three transition matrices,
each (n_states x n_states) with the last state meaning "left R":

1. naive_P: counts / row totals (transitions.row_normalise, shared with Computational).
   Accurate where data is plentiful, but gives probability 0 to every transition it never
   saw, and has no row at all for cells with no training data.
2. persistence_P: the baseline "material stays where it is", smoothed so moves get a small
   probability eps (otherwise any held-out drifter that moves scores log 0 = -inf).
3. shrinkage_P: a weighted average of a cell's own counts and a "prior" guess for that cell's
   row (a Dirichlet-multinomial model). In plain terms: before looking at cell i's data we
   pretend we have already seen alpha transitions from it, spread according to the prior
   row m_i (these are "pseudo-transitions"). Adding them to the real counts n_ij gives
        p_ij = (n_ij + alpha * m_ij) / (n_i + alpha),
   which is the posterior mean (the model's best single estimate after seeing the data).
   With few drifters it leans on the prior row m_i; with many it follows the data. A row
   with n_i = alpha real transitions is half data, half prior.

   The prior row m_i is the 4 deg parent's row ("borrowing strength from neighbours",
   project_plan.txt section 7, decision 2). The parent row is a distribution over 4 deg
   destinations, so each 4 deg destination's mass is spread over its 2 deg children in
   proportion to how often training drifters arrived in each child, plus 0.5 so a child no
   training drifter reached still gets a little (land cells get that 0.5 too; it is wasted but
   tiny). The parent row itself gets one pseudo-transition spread evenly over all states, so no
   transition ever has probability exactly 0 - a held-out drifter can't score -inf.

The fitting here is for Week 3 design and a preliminary check; Week 6 fits it properly and
Week 7 adds the bootstrap.
"""
import numpy as np

from src.gridding.transitions import row_normalise as naive_P  # noqa: F401 (re-exported for the quant scripts)


def count_matrix(start, end, n_states):
    """n_ij = number of transitions from state i to state j."""
    counts = np.zeros((n_states, n_states))
    np.add.at(counts, (start, end), 1)
    return counts


def persistence_P(n_states, eps):
    """Stay with probability 1 - eps, otherwise move to any other state equally. Exit is absorbing."""
    P = np.full((n_states, n_states), eps / (n_states - 1))
    np.fill_diagonal(P, 1 - eps)
    P[-1] = 0.0
    P[-1, -1] = 1.0
    return P


def prior_rows(counts_fine, parent, smoothing=1.0):
    """The prior row m_i for every fine state, built from the coarse parent grid.

    Inputs:
        counts_fine: training count matrix on the fine grid.
        parent: parent_states() map, fine state -> coarse state (exit -> exit).
        smoothing: total pseudo-transitions spread evenly over each coarse row.
    Returns:
        (n_fine x n_fine) matrix whose rows sum to 1.
    """
    n_fine, n_coarse = parent.size, parent.max() + 1

    # Coarse counts by aggregating fine counts, so prior and data come from the same transitions
    agg = np.zeros((n_fine, n_coarse))
    agg[np.arange(n_fine), parent] = 1.0
    counts_coarse = agg.T @ counts_fine @ agg
    P_coarse = (counts_coarse + smoothing / n_coarse) / (counts_coarse.sum(axis=1, keepdims=True) + smoothing)

    # Within each coarse destination, share its mass among fine children by arrivals
    # (+0.5 for every cell, so a cell no training drifter reached still gets a little)
    weight = counts_fine.sum(axis=0) + 0.5
    weight[-1] = 1.0  # exit -> exit
    share = weight / np.bincount(parent, weights=weight)[parent]

    # m_ij = P_coarse[parent(i), parent(j)] * share_j
    prior = P_coarse[np.ix_(parent, parent)] * share[None, :]
    prior[-1] = 0.0
    prior[-1, -1] = 1.0  # exit is absorbing
    return prior


def shrinkage_P(counts, prior, alpha):
    """Posterior mean of a Dirichlet(alpha * prior_i) per row: (n_ij + alpha m_ij) / (n_i + alpha)."""
    return (counts + alpha * prior) / (counts.sum(axis=1, keepdims=True) + alpha)


def log_lik(P, start, end):
    """log p(start -> end) for every held-out transition (-inf where P gives 0)."""
    with np.errstate(divide="ignore"):
        return np.log(P[start, end])


def score(logp):
    """Summary of held-out log-likelihoods for one model."""
    finite = np.isfinite(logp)
    return {
        "mean_loglik": float(logp.mean()) if finite.all() else float("-inf"),
        "pct_zero_prob": round(float(100 * (~finite).mean()), 2),
        "mean_loglik_where_finite": float(logp[finite].mean()) if finite.any() else float("nan"),
    }
