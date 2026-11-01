"""The insurance submission: 100 dataset grids, likely valid and spread out (rule 3.1).

Where this fits
---------------
If the design generator (Stage 3) has no designs by the end of 6 November, rule 3.1
(docs/decisions.md) submits 100 dataset grids instead. ``insurance()`` takes the unrated grids
approved for all four advisors (surrogates.py) and picks 100 by greedy max-min diversity: the
likeliest to pass all four (calibration.py) first, then, again and again, the candidate farthest
from its nearest pick. ``reference_selection()`` is rule 3.6's yardstick for the Stage 3 pilot.

Reads   data/cache/ratings/ (python surrogates.py first), the grids, the calibration inputs
Writes  submissions/insurance_<YYYYMMDD_HHMMSS>.npy (100 grids) and .json (stats); new each run
Run     python select_submission.py

Surprises
---------
* A grid's chance of passing all four is the product of its four calibrated chances. That
  assumes the advisors' errors are independent (rule 2.8 measures how far they are not).
* Grids with a development rating are never picked: the models trained on most of them, so
  their predictions are too hopeful. Test-set grids look unrated here, so they can be picked.
* Under 100 candidates: all are taken, then the distinct grids closest to approval (it says so).
"""

import json
import os
from datetime import datetime

import numpy as np

import calibration
import scoring
import surrogates
import validity
from validity import ADVISORS

N = scoring.N_GRIDS  # grids per submission
DRAWS = 500  # Monte Carlo draws for the expected score
REFERENCE_POOL = 300  # rule 3.6: a random 300 of the grids approved for all four
SUBMISSIONS_DIR = os.path.join(validity.PS2_ROOT, "submissions")
INDEPENDENCE = "the four advisors' errors are independent"


def shortfall(pred, cuts):
    """Per grid: the sum over advisors of how far pred falls below the cut-off (0: approved)."""
    return np.maximum(0, np.array([cuts[a] for a in ADVISORS]) - pred).sum(1)


def p_all_four(pred, spread, cals):
    """Per grid: the product of the four calibrated chances of passing (assumes independence)."""
    p = np.ones(len(pred))
    for a, advisor in enumerate(ADVISORS):
        p *= calibration.pass_probability(cals[advisor], pred[:, a], spread[:, a])
    return p


def greedy_maxmin(grids, priority, n):
    """Positions of n grids: the highest priority first; then, repeatedly, the grid whose
    nearest chosen grid is farthest away (in cells differing), ties to the higher priority."""
    flat = grids.reshape(len(grids), -1)
    chosen = [int(np.argmax(priority))]
    nearest = np.full(len(flat), np.inf)  # cells differing from the nearest chosen grid
    while len(chosen) < n:
        # Only the newest pick can bring a grid's nearest chosen grid closer.
        nearest = np.minimum(nearest, (flat != flat[chosen[-1]]).sum(1))
        far = np.flatnonzero(nearest == nearest.max())
        chosen.append(int(far[np.argmax(priority[far])]))
    return np.array(chosen)


def expected_score(grids, p, draws=DRAWS, seed=0):
    """Mean course score over draws in which each grid is valid with chance p (others unscored)."""
    valid = np.random.default_rng(seed).random((draws, len(p))) < p
    return float(np.mean([scoring.diversity(grids[v]) for v in valid]))


def insurance(pred, spread, cuts, cals, n=N, grids=None, allowed=None):
    """(indices, stats): n grids approved for all four, picked by greedy max-min diversity from
    the rows where `allowed` is True (default all). grids defaults to the dataset's 500,000."""
    grids = validity.load_dev()[0] if grids is None else grids
    rows = np.arange(len(pred)) if allowed is None else np.flatnonzero(allowed)
    p = p_all_four(pred, spread, cals)
    short = shortfall(pred, cuts)
    candidates = rows[short[rows] == 0]
    if len(candidates) >= n:
        idx = candidates[greedy_maxmin(grids[candidates], p[candidates], n)]
    else:  # every candidate, then the grids closest to approval; repeated grids once only
        flat = grids[rows].reshape(len(rows), -1)
        rows = rows[np.sort(np.unique(flat, axis=0, return_index=True)[1])]
        idx = rows[np.argsort(short[rows], kind="stable")][:n]
    stats = {
        "candidates": len(candidates),
        "filled_below_cutoffs": int((short[idx] > 0).sum()),
        "diversity_if_all_valid": scoring.diversity(grids[idx]),
        "mean_p_all_four": float(p[idx].mean()),
        "expected_valid": float(p[idx].sum()),
        "expected_score": expected_score(grids[idx], p[idx]),
    }
    return idx, stats


def reference_selection(pred, cuts, seed=0, grids=None):
    """Rule 3.6's reference: the diversity of 100 grids picked by greedy max-min from a random
    300 of the grids approved for all four. With no calibrated chances, its start and its
    ties go by each grid's smallest margin above the cut-offs."""
    grids = validity.load_dev()[0] if grids is None else grids
    approved = np.flatnonzero(shortfall(pred, cuts) == 0)
    rng = np.random.default_rng(seed)
    pool = rng.choice(approved, min(REFERENCE_POOL, len(approved)), replace=False)
    margin = (pred[pool] - np.array([cuts[a] for a in ADVISORS])).min(1)
    picked = pool[greedy_maxmin(grids[pool], margin, min(N, len(pool)))]
    return scoring.diversity(grids[picked])


def report(s, reference):
    if s["filled_below_cutoffs"]:
        print(f"WARNING: {s['filled_below_cutoffs']} picks miss the cut-offs.")
    print(f"Insurance set: {N} of the {s['candidates']:,} grids approved for all four")
    print(f"  diversity, if all {N} are valid    {s['diversity_if_all_valid']:.3f}")
    print(f"  mean chance of passing all four   {s['mean_p_all_four']:.3f}")
    print(f"  expected number valid             {s['expected_valid']:.1f}")
    print(f"  expected score ({DRAWS} draws)       {s['expected_score']:.3f}")
    print(f"  (the chance of passing all four assumes {INDEPENDENCE})")
    print(f"Rule 3.6 reference: max-min, a random {REFERENCE_POOL} of the approved")
    print(f"  diversity, if all {N} are valid    {reference:.3f}")


def main():
    grids, dev = validity.load_dev()  # dev only to see which grids are rated
    pred, spread, info = surrogates.load_ratings()  # stops if the incumbents changed
    cuts = info["cutoffs"]  # so "approved" means what surrogates.py counted
    unrated = np.isnan(dev).all(1)  # rule 3.1: rated grids trained the models
    cals = calibration.fit_all()
    idx, stats = insurance(pred, spread, cuts, cals, grids=grids, allowed=unrated)
    reference = reference_selection(pred, cuts, grids=grids)
    submission = grids[idx].astype(np.int64)
    scoring.check_submission(submission)
    commit, dirty = validity._git_state()  # the code that made the chances and picks
    record = stats | {"reference_diversity": reference, "assumes": INDEPENDENCE}
    record |= {"commit": commit, "uncommitted": dirty, "ratings": info}
    record["rows"] = idx.tolist()
    stem = os.path.join(SUBMISSIONS_DIR, f"insurance_{datetime.now():%Y%m%d_%H%M%S}")
    np.save(stem + ".npy", submission)
    with open(stem + ".json", "w") as f:
        json.dump(record, f, indent=1)
    report(stats, reference)
    print(f"Saved {stem}.npy and .json")


if __name__ == "__main__":
    main()
