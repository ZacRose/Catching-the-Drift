"""
split.py

Related task: Week 3 Quantitative - "plan the assessment before fitting anything".

Splits by drifter, never by reading or transition: one drifter's transitions are strongly
correlated (consecutive daily starts share 14 of their 15 days), so splitting rows would
leak the test set into training. 20% of drifters are held out as the test set, and 20% of
the rest are a validation set used only to choose tuning settings (the prior strength,
the persistence smoothing). Choosing those on the test set would make the test score
optimistic. That leaves 64% / 16% / 20% of drifters for train / valid / test.

The split is saved to outputs/results/drifter_split.csv and committed, so every model in
Weeks 3-8 is scored on the same held-out drifters. Rerunning with the same seed and the
same drifters reproduces it exactly.

Run: python3 -m src.inference.split
"""
import csv
import os
import numpy as np

from src.common.config import RESULTS_DIR, load_region_bounds
from src.inference.transition_data import load_transitions

SEED = 3001
TEST_FRAC = 0.2
VALID_FRAC = 0.2  # of the non-test drifters
SPLIT_PATH = os.path.join(RESULTS_DIR, "drifter_split.csv")


def split_drifters(drifter_ids, test_frac=TEST_FRAC, valid_frac=VALID_FRAC, seed=SEED):
    """Assign each distinct drifter to "train", "valid" or "test".

    Returns:
        dict drifter_id -> set name.
    """
    ids = np.unique(drifter_ids)
    shuffled = np.random.default_rng(seed).permutation(ids)
    n_test = int(round(test_frac * ids.size))
    n_valid = int(round(valid_frac * (ids.size - n_test)))
    labels = ["test"] * n_test + ["valid"] * n_valid + ["train"] * (ids.size - n_test - n_valid)
    return {int(d): s for d, s in zip(shuffled, labels)}


def save_split(assignment, path=SPLIT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["drifter_id", "set"])
        writer.writerows(sorted(assignment.items()))


def load_split(path=SPLIT_PATH):
    """dict drifter_id -> "train"/"valid"/"test" from the committed CSV."""
    with open(path) as f:
        return {int(row["drifter_id"]): row["set"] for row in csv.DictReader(f)}


def set_mask(drifter_ids, assignment, names):
    """True for transitions whose drifter is in any of the named sets."""
    names = {names} if isinstance(names, str) else set(names)
    keep = np.array(sorted(d for d, s in assignment.items() if s in names))
    return np.isin(drifter_ids, keep)


if __name__ == "__main__":
    table = load_transitions(load_region_bounds())
    assignment = split_drifters(table["drifter_id"])
    save_split(assignment)
    for name in ("train", "valid", "test"):
        mask = set_mask(table["drifter_id"], assignment, name)
        n_drifters = sum(s == name for s in assignment.values())
        print(f"{name}: {n_drifters} drifters, {mask.sum()} transitions")
    print(f"Saved {SPLIT_PATH}")
