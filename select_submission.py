"""Pick 100 grids to submit: dataset grids (the insurance set, rule 3.1) or generated designs.

Where this fits
---------------
If the design generator (Stage 3) has no designs by the end of 6 November, rule 3.1
(docs/decisions.md) submits 100 dataset grids instead. ``insurance()`` takes the unrated grids
approved for all four advisors (surrogates.py) and picks 100 by greedy max-min diversity: the
likeliest to pass all four (calibration.py) first, then, again and again, the candidate farthest
from its nearest pick. ``reference_selection()`` is rule 3.6's yardstick for the Stage 3 pilot.
``--designs`` makes the same pick from generate.py's pools instead, one submission per margin.

Reads   data/cache/ratings/ (python surrogates.py first), the grids, the calibration inputs,
        data/designs/pool_* (--designs only)
Writes  submissions/insurance_<YYYYMMDD_HHMMSS>.npy (100 grids) and .json (stats); new each run
        --designs: submissions/designs_m<margin>_<...>.npy and .json; results/designs_<date>.xlsx
Run     python select_submission.py
        python select_submission.py --designs 0.03 0.10

Surprises
---------
* A grid's chance of passing all four is the product of its four calibrated chances. That
  assumes the advisors' errors are independent (rule 2.8 measures how far they are not).
* Grids with a development rating are never picked: the models trained on most of them, so
  their predictions are too hopeful. Test-set grids look unrated here, so they can be picked.
* Under 100 candidates: all are taken, then the distinct grids closest to approval (it says so).
* Designs' chances are hopeful: calibration learned from dataset grids, and designs were pushed
  until the models liked them. Rules 3.3 and 3.5, or the scoreboard, say how hopeful.
"""

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np

import calibration
import generate
import scoring
import surrogates
import tables
import validity
from validity import ADVISORS

N = scoring.N_GRIDS  # grids per submission
DRAWS = 500  # Monte Carlo draws for the expected score
REFERENCE_POOL = 300  # rule 3.6: a random 300 of the grids approved for all four
SUBMISSIONS_DIR = os.path.join(validity.PS2_ROOT, "submissions")
DESIGNS_DIR = os.path.join(validity.PS2_ROOT, "data", "designs")  # generate.py's pools
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


def _stats(grids, p):
    """The picked grids' diversity if all are valid, and what their chances p expect."""
    return {
        "diversity_if_all_valid": scoring.diversity(grids),
        "mean_p_all_four": float(p.mean()),
        "expected_valid": float(p.sum()),
        "expected_score": expected_score(grids, p),
    }


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
    }
    return idx, stats | _stats(grids[idx], p[idx])


def load_pools(margin, cuts):
    """(grids, pred, spread) of every pool generate.py made at this margin, in seed order, each
    distinct design once: batches drew their starts separately, so some reached the same design."""
    parts = []
    for path in glob.glob(os.path.join(DESIGNS_DIR, "pool_*.json")):
        with open(path) as f:
            settings = json.load(f)["settings"]
        if settings["cutoffs"] != cuts:  # other models made it: its ratings don't apply
            raise SystemExit(f"{path} has other cut-offs: re-rate or regenerate it")
        if settings["margin"] == margin:
            parts.append((settings["seed"], np.load(path[: -len(".json")] + ".npz")))
    if not parts:
        raise SystemExit(f"no pools at margin {margin} in {DESIGNS_DIR}")
    parts.sort(key=lambda part: part[0])
    grids, pred, spread = (
        np.concatenate([z[k] for _, z in parts]) for k in ("grids", "pred", "spread")
    )
    first = np.sort(
        np.unique(grids.reshape(len(grids), -1), axis=0, return_index=True)[1]
    )
    return grids[first], pred[first], spread[first]


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


def _save(name, submission, record):
    """Check the submission, then save it and its record as submissions/<name>_<time>.npy/.json."""
    scoring.check_submission(submission)
    commit, dirty = validity._git_state()  # the code that made the chances and picks
    record |= {"assumes": INDEPENDENCE, "commit": commit, "uncommitted": dirty}
    stem = os.path.join(SUBMISSIONS_DIR, f"{name}_{datetime.now():%Y%m%d_%H%M%S}")
    np.save(stem + ".npy", submission)
    with open(stem + ".json", "w") as f:
        json.dump(record, f, indent=1)
    print(f"Saved {stem}.npy and .json")
    return stem


def designs(margins, cuts, cals, info):
    """One submission per margin: 100 of that margin's pooled designs, by greedy max-min."""
    rows = []
    for margin in margins:
        grids, pred, spread = load_pools(margin, cuts)
        p = p_all_four(pred, spread, cals)
        idx = greedy_maxmin(grids, p, N)
        stats = {"margin": margin, "candidates": len(grids)} | _stats(
            grids[idx], p[idx]
        )
        print(
            f"Designs at margin {margin}: {N} of {len(grids):,} distinct pooled designs"
        )
        print(
            f"  diversity, if all {N} are valid       {stats['diversity_if_all_valid']:.3f}"
        )
        print(
            f"  expected score, if the chances hold  {stats['expected_score']:.3f} (hopeful)"
        )
        stem = _save(
            f"designs_{generate.margin_label(margin)}",
            grids[idx].astype(np.int64),
            stats | {"ratings": info},
        )
        rows.append(stats | {"file": os.path.basename(stem) + ".npy"})
    print(
        f"Table: {tables.save(f'designs_{datetime.now():%Y-%m-%d_%H%M}', {'Designs': rows})}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--designs",
        type=generate.parse_margin,
        nargs="+",
        metavar="MARGIN",
        help="pick from generate.py's pools at each margin, not from dataset grids",
    )
    args = ap.parse_args(argv)
    pred, spread, info = surrogates.load_ratings()  # stops if the incumbents changed
    cuts = info["cutoffs"]  # so "approved" means what surrogates.py counted
    cals = calibration.fit_all()
    if args.designs:
        return designs(args.designs, cuts, cals, info)
    grids, dev = validity.load_dev()  # dev only to see which grids are rated
    unrated = np.isnan(dev).all(1)  # rule 3.1: rated grids trained the models
    idx, stats = insurance(pred, spread, cuts, cals, grids=grids, allowed=unrated)
    reference = reference_selection(pred, cuts, grids=grids)
    report(stats, reference)
    record = stats | {
        "reference_diversity": reference,
        "ratings": info,
        "rows": idx.tolist(),
    }
    _save("insurance", grids[idx].astype(np.int64), record)


if __name__ == "__main__":
    main()
