# Methods and evaluation (Quantitative stream) - Week 3 draft for the proposal

Numbers in [square brackets] come from the real-data run of the scripts named beside them; fill them in before submission.

## Data adequacy on transitions

The Week 1 adequacy check counted distinct drifters with any reading in a cell. We now count the quantity the estimate actually uses: distinct drifters that start at least one 15-day transition in the cell (`src/inference/transition_adequacy.py`). We count drifters rather than transitions because a drifter starts a transition every day it spends in a cell, and those daily starts share most of their path, so they are far from independent. A cell is adequate if at least 100 distinct drifters start there, which caps the worst-case standard error of any transition probability out of that cell at 0.05. On this definition [X]% of ocean cells are adequate at 2° (median [Y] drifters) and [X]% at 4° (median [Y]), against 46% and 80% on the Week 1 proxy (`outputs/figures/transition_adequacy_map.png`). The thin cells are concentrated in [describe, e.g. the north-east, off Angola and northern Namibia], which is the gap the shrinkage estimator below is designed to fill.

## Estimator

For each start cell *i*, the transitions out of it are multinomial with probabilities *p_i* over the destination cells plus one "left the region" state. Counting and normalising (the naive estimate) is unbiased where data is plentiful, but it gives probability zero to any transition it never observed and has no estimate at all for cells without data. We therefore use a Dirichlet-multinomial model. Each 2° row gets a Dirichlet(α·*m_i*) prior, whose posterior mean is

  *p̂_ij* = (*n_ij* + α·*m_ij*) / (*n_i* + α).

A row with few drifters leans on its prior row *m_i*; a well-sampled row follows its own data. The prior row is the transition row of the 4° cell containing *i*, which is exactly the 2×2 block of neighbouring 2° cells, so a thin cell borrows strength from its immediate neighbours. Each 4° destination's probability is shared among its four 2° sub-cells in proportion to how often training drifters arrived there. α is the prior's weight, measured in transitions, and is chosen on held-out drifters rather than fixed by hand. Varying α across two orders of magnitude around the chosen value is our sensitivity check (`outputs/figures/alpha_sensitivity.png`). A drifter-level bootstrap (resampling whole drifters, Week 7) will give a model-free interval to compare against the posterior uncertainty.

## Evaluation design

We fixed the assessment before fitting any model. Drifters, not readings, are split at random: 20% are held out as a test set, and 20% of the remainder form a validation set used only to choose α and the persistence baseline's smoothing (`src/inference/split.py`; the split is committed so every later model is scored on the same drifters). Splitting by reading would leak information, because one drifter's transitions are strongly correlated. Each transition matrix is scored by its mean held-out log-likelihood, the average log-probability it gave to where test drifters actually went (`src/inference/evaluate.py`). We compare three models: the naive estimate, a persistence baseline ("material stays in its cell", with a small probability of moving anywhere else), and the shrinkage estimate. Because the naive estimate assigns zero probability to unseen moves, we also report the share of test transitions it scores as impossible ([X]% at 2°, [X]% at 4°); this is the measurable case for shrinkage. Preliminary 2° results: shrinkage [X], naive [X] (where finite), persistence [X].

## Exploratory association with the response

Figure `outputs/figures/displacement_association.png` shows 15-day displacement split by drogue status, season (southern-hemisphere seasons) and start before or from 2010 (`src/inference/displacement_eda.py`). [One or two sentences on what differs.] Transitions that leave the region have no end position and are excluded from the boxes; their share is shown per group. These splits are not independent (the share of undrogued drifters changes over time), so they are a first look rather than a controlled comparison. They inform two decisions in the plan: keeping undrogued drifters in the main estimate, and using all years with 2010+ as a sensitivity check.
