"""Spread 100 approved designs apart by simulated annealing (Stage 4's last step).

Where this fits
---------------
select_submission.py picks 100 varied designs from generate.py's pool; this script moves them
apart. One proposal changes one cell of one design. Its effect on the course score (every
design's mean distance to its 10 nearest, /100) is exact; it is kept by the Metropolis rule,
which also keeps some worse proposals early on (temperature T falls from T0 to T1), so the set
can leave a local peak. Greedy one-cell spreading stops at about 0.755 even with no approval
rule; annealing reaches 0.774 (7 Nov, docs/decisions.md).

Every design must stay approved: each advisor's surrogate prediction >= cut-off + margin.
* hard rule (--hard): a proposal that leaves its design unapproved is rejected.
* soft rule (default): the objective is score - lam * shortfall (summed over designs and
  advisors), with lam rising from LAM0 to LAM1, so designs may dip below a bar early and must
  climb back; any still below at the end are repaired by generate.design's hill-climb.

Reads / writes
--------------
reads   --start (a 100-grid .npy); the surrogate models and cut-offs (surrogates.py)
writes  submissions/anneal_m<margin>_<time>.npy and .json (settings, commit, result)

How to run
----------
    python anneal.py --start submissions/<file>.npy --margin 0.06 [--hard] [--steps 1500000]

Surprises
---------
* Like generate.py, it runs only from a clean, committed tree.
* Rating one grid recomputes its features once per model (21 times); a cache here makes each
  rating ~2x faster without touching the models' code. It is checked to change no prediction.
* Scoreboard tries set the margin: annealed designs at 0.03 lost ~5 of 100, at 0.06 none.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import features
import features_lab
import generate
import scoring
import select_submission as sel
import surrogates
import validity
from validity import ADVISORS

T0, T1 = 5e-5, 1e-6  # temperature: a score drop of T is kept with chance 1/e
LAM0, LAM1 = 3e-3, 3.0  # soft rule's weight on shortfall, start and end
STEPS = 1_500_000
REPORT = 100_000  # proposals between progress lines


def _cached(fn):
    """fn, remembering its last input's result: the models share features for one grid."""
    memo = {}

    def wrapper(grids):
        key = np.asarray(grids).tobytes()
        if key not in memo:
            memo.clear()
            memo[key] = fn(grids)
        return memo[key]

    return wrapper


def _cache_features():
    features.engineered = _cached(features.engineered)
    features.onehot = _cached(features.onehot)
    features_lab.lab = _cached(features_lab.lab)
    for name in list(validity.FEATURIZERS):
        validity.FEATURIZERS[name] = _cached(validity.FEATURIZERS[name])


def score(D):
    """The course score of a set with distance matrix D (scoring.diversity, all valid)."""
    nearest = np.partition(D, 10, axis=1)[
        :, :11
    ]  # the 11 smallest include the 0 to itself
    return nearest.sum() / 10 / len(D)


def shortfall(models, grids, bar):
    """Per grid: summed over advisors, how far the prediction falls below its bar."""
    return np.maximum(0, bar - surrogates.rate(models, grids)[0]).sum(1)


def anneal(grids, models, bar, steps=STEPS, hard=False, seed=0):
    """(grids, info): the set after `steps` proposals. Every design starts approved."""
    flat = grids.reshape(len(grids), -1).copy()
    short = shortfall(models, grids, bar)
    assert not short.any(), "every design must start approved"
    D = (flat[:, None] != flat[None]).mean(2)
    now, rng, tic = score(D), np.random.default_rng(seed), time.perf_counter()
    for k in range(steps):
        t = T0 * (T1 / T0) ** (k / steps)
        lam = np.inf if hard else LAM0 * (LAM1 / LAM0) ** (k / steps)
        i, cell = rng.integers(len(flat)), rng.integers(flat.shape[1])
        old = flat[i, cell]
        flat[i, cell] = (old + rng.integers(1, 5)) % 5
        row = (flat[i] != flat).mean(1)
        row[i] = 0
        D2 = D.copy()
        D2[i], D2[:, i] = row, row
        gain, need = score(D2) - now, t * np.log(rng.random())  # keep if change >= need
        slack = 0.0 if hard else lam * short[i]  # the most a better rating could add
        if gain + slack >= need:
            new = shortfall(models, flat[i].reshape(1, 7, 7), bar)[0]
            cost = (np.inf if new else 0.0) if hard else lam * (new - short[i])
            if gain - cost >= need:
                D, now, short[i] = D2, now + gain, new
                continue
        flat[i, cell] = old
        if (k + 1) % REPORT == 0:
            print(
                f"  {k + 1:>9,} proposals: score if all pass {now:.4f}, below a bar "
                f"{int((short > 0).sum())} ({time.perf_counter() - tic:.0f} s)",
                flush=True,
            )
    out = flat.reshape(grids.shape)
    repaired = np.flatnonzero(short > 0)
    for i in repaired:  # soft rule only: climb back over every bar
        cuts = dict(zip(ADVISORS, bar))
        out[i], info = generate.design(out[i], models, cuts, 0.0, rng)
        assert info["shortfall"] == 0, f"design {i} could not be repaired"
    return out, {"repaired": len(repaired), "seconds": time.perf_counter() - tic}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", required=True, help="a .npy of 100 approved designs")
    ap.add_argument("--margin", type=float, required=True, help="cushion over cut-offs")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument(
        "--hard", action="store_true", help="never leave a design unapproved"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _, uncommitted = validity._git_state()
    if uncommitted:  # _save records the commit, which must hold the code that ran
        raise SystemExit(f"Commit first. Not committed: {uncommitted[:5]}")
    cuts = surrogates.load_ratings()[2]["cutoffs"]
    bar = np.array([cuts[a] for a in ADVISORS]) + args.margin
    models = surrogates.fit_models()
    surrogates.check_refits(models)
    start = np.load(args.start).astype(np.int64)
    before = surrogates.rate(models, start)[0]
    _cache_features()
    assert np.array_equal(surrogates.rate(models, start)[0], before), (
        "cache changed a rating"
    )
    print(
        f"Annealing {args.start} at cut-off + {args.margin}: {scoring.diversity(start):.4f}"
    )
    grids, info = anneal(start, models, bar, args.steps, args.hard, args.seed)
    assert (surrogates.rate(models, grids)[0] >= bar).all()
    result = {"score_if_all_pass": scoring.diversity(grids)} | info
    print(f"Done: score if all pass {result['score_if_all_pass']:.4f}")
    settings = vars(args) | {
        "T0": T0,
        "T1": T1,
        "lam0": LAM0,
        "lam1": LAM1,
        "cutoffs": cuts,
    }
    sel._save(f"anneal_m{args.margin:.2f}", grids, {"settings": settings} | result)


if __name__ == "__main__":
    main()
