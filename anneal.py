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

--neighbours (try F's method, 7 Nov): each update takes one design, rates all its neighbours
(the 196 one-cell changes plus 196 random swaps of two cells, which keep the district counts)
on 10 worker processes, and moves it to one that stays approved, or keeps it, with chance
proportional to exp(score gain / T): a heat-bath choice. Designs below a bar at the start are
first repaired (greedy steps that raise their lowest margin). --models upgraded rates with
upgraded.py's models (neighbours only).

Reads / writes
--------------
reads   --start (a 100-grid .npy); the models and cut-offs (surrogates.py or upgraded.py)
writes  submissions/anneal_<models>_m<margin>_<time>.npy and .json (settings, commit, result);
        --neighbours also saves data/designs/anneal_partial_seed<seed>.npy at each progress line

How to run
----------
    python anneal.py --start submissions/<file>.npy --margin 0.06 [--hard] [--steps 1500000]
    python anneal.py --start <file> --margin 0.06 --models upgraded --neighbours [--steps 100000]

Surprises
---------
* Like generate.py, it runs only from a clean, committed tree.
* Rating one grid recomputes its features once per model (21 times); a cache here makes each
  rating ~2x faster without touching the models' code. It is checked to change no prediction.
* Scoreboard tries set the margin: annealed designs at 0.03 lost ~5 of 100, at 0.06 none.
* --steps counts proposals (default 1,500,000), or with --neighbours updates (100,000: about
  2.3 h with the upgraded models on 10 cores).
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import features
import features_lab
import generate
import scoring
import select_submission as sel
import surrogates
import upgraded
import validity
from validity import ADVISORS

T0, T1 = 5e-5, 1e-6  # temperature: a score drop of T is kept with chance 1/e
LAM0, LAM1 = 3e-3, 3.0  # soft rule's weight on shortfall, start and end
STEPS = 1_500_000
REPORT = 100_000  # proposals between progress lines
UPDATES = 100_000  # --neighbours: designs moved (or kept), one at a time
WORKERS = 10  # --neighbours: designs whose neighbours are rated at once, one per process
SWAPS = 196  # two-cell swaps tried per update, beside the 196 one-cell changes
PARTIAL = os.path.join(validity.PS2_ROOT, "data", "designs", "anneal_partial_seed{}.npy")
_PAIRS = np.array([(a, b) for a in range(49) for b in range(a + 1, 49)])
_W = {}  # a worker's rating function and bars


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


def _init(which, bar, models):
    """Worker setup: how to rate grids with the `which` models, and the bars."""
    if which == "upgraded":
        torch.set_num_threads(1)  # the workers share the cores
        saved = upgraded.load()
        _W["rate"] = lambda grids: upgraded.rate(saved, grids)
    else:
        _cache_features()
        _W["rate"] = lambda grids: surrogates.rate(models, grids)[0]
    _W["bar"] = bar


def _slack(grids):
    """Per grid: its lowest margin over the four bars (>= 0: approved)."""
    return (_W["rate"](grids) - _W["bar"]).min(1)


def _approved(grids):
    return _slack(grids) >= 0


def neighbours(grid, rng, k=SWAPS):
    """The 196 one-cell changes of grid, plus k random swaps of two cells holding different
    districts."""
    flat = grid.reshape(49)
    pairs = _PAIRS[flat[_PAIRS[:, 0]] != flat[_PAIRS[:, 1]]]
    pairs = pairs[rng.choice(len(pairs), min(k, len(pairs)), replace=False)]
    swapped = np.repeat(flat[None], len(pairs), 0)
    rows = np.arange(len(pairs))
    swapped[rows, pairs[:, 0]] = flat[pairs[:, 1]]
    swapped[rows, pairs[:, 1]] = flat[pairs[:, 0]]
    changed = generate.one_cell_changes(grid).reshape(-1, 49)
    return np.concatenate([changed, swapped]).reshape(-1, 7, 7)


def _repair(grid, steps=300):
    """(grid, steps taken; -1 if it failed): greedy moves to the neighbour with the largest
    lowest margin, until approved."""
    rng = np.random.default_rng(int(grid.sum()))
    for step in range(steps):
        if _slack(grid[None])[0] >= 0:
            return grid, step
        near = neighbours(grid, rng)
        grid = near[_slack(near).argmax()]
    return grid, -1


def _score_with_row(D, i, row):
    D2 = D.copy()
    D2[i], D2[:, i] = row, row
    return score(D2)


def anneal_neighbours(grids, pool, updates=UPDATES, t0=T0, seed=0):
    """(grids, info): heat-bath annealing over approved neighbours. Every design starts
    approved; `pool` holds WORKERS processes set up by _init."""
    flat = grids.reshape(len(grids), -1).copy()
    D = (flat[:, None] != flat[None]).mean(2)
    now, rng, tic = score(D), np.random.default_rng(seed), time.perf_counter()
    done = moved = 0
    while done < updates:
        batch = rng.choice(len(flat), WORKERS, replace=False)
        near = [neighbours(flat[i].reshape(7, 7), rng) for i in batch]
        for i, cand, ok in zip(batch, near, list(pool.map(_approved, near))):
            t = t0 * (T1 / t0) ** (done / updates)
            done += 1
            good = cand.reshape(-1, 49)[ok]
            if not len(good):
                continue
            rows = (good[:, None] != flat[None]).mean(2)
            rows[:, i] = 0
            gains = np.array([_score_with_row(D, i, row) for row in rows]) - now
            logits = np.append(gains, 0.0) / t  # the last option: stay
            p = np.exp(logits - logits.max())
            pick = rng.choice(len(p), p=p / p.sum())
            if pick < len(good):
                flat[i], D[i], D[:, i] = good[pick], rows[pick], rows[pick]
                now += gains[pick]
                moved += 1
        if done % 10_000 < WORKERS:
            np.save(PARTIAL.format(seed), flat.reshape(grids.shape))
            print(
                f"  {done:>7,} updates: score if all pass {now:.4f}, moves {moved:,} "
                f"({time.perf_counter() - tic:.0f} s)",
                flush=True,
            )
    seconds = time.perf_counter() - tic
    return flat.reshape(grids.shape), {"moves": moved, "seconds": seconds}


def _run_neighbours(start, args, bar, models):
    """Repair any design below a bar, then anneal_neighbours; checks the result."""
    spawn = multiprocessing.get_context("spawn")  # a forked one can hang LightGBM
    setup = (args.models, bar, models)
    with ProcessPoolExecutor(WORKERS, spawn, initializer=_init, initargs=setup) as pool:
        split = np.array_split(start, WORKERS)
        bad = np.flatnonzero(~np.concatenate(list(pool.map(_approved, split))))
        for i, (fixed, step) in zip(bad, pool.map(_repair, list(start[bad]))):
            assert step >= 0, f"design {i} could not be repaired"
            start[i] = fixed
        distinct = len(np.unique(start.reshape(len(start), -1), axis=0))
        assert distinct == len(start), "a repair made a duplicate"
        print(f"  {len(bad)} designs repaired: {scoring.diversity(start):.4f}", flush=True)
        grids, info = anneal_neighbours(start, pool, args.steps, args.t0, args.seed)
        split = np.array_split(grids, WORKERS)
        assert np.concatenate(list(pool.map(_approved, split))).all()
    return grids, info | {"repaired": len(bad)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", required=True, help="a .npy of 100 designs")
    ap.add_argument("--margin", type=float, required=True, help="cushion over cut-offs")
    ap.add_argument("--steps", type=int, help=f"default {STEPS:,}, --neighbours {UPDATES:,}")
    ap.add_argument(
        "--hard", action="store_true", help="never leave a design unapproved"
    )
    ap.add_argument("--neighbours", action="store_true", help="try F's method")
    ap.add_argument("--models", choices=("final", "upgraded"), default="final")
    ap.add_argument("--t0", type=float, default=T0, help="--neighbours: start temperature")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.models == "upgraded" and not args.neighbours:
        ap.error("--models upgraded needs --neighbours")
    args.steps = args.steps or (UPDATES if args.neighbours else STEPS)
    _, uncommitted = validity._git_state()
    if uncommitted:  # _save records the commit, which must hold the code that ran
        raise SystemExit(f"Commit first. Not committed: {uncommitted[:5]}")
    if args.models == "upgraded":
        models, cuts = None, upgraded.load()["cutoffs"]
    else:
        models, cuts = surrogates.fit_models(), surrogates.load_ratings()[2]["cutoffs"]
        surrogates.check_refits(models)
    bar = np.array([cuts[a] for a in ADVISORS]) + args.margin
    start = np.load(args.start).astype(np.int64)
    print(
        f"Annealing {args.start} at cut-off + {args.margin}: {scoring.diversity(start):.4f}"
    )
    if args.neighbours:
        grids, info = _run_neighbours(start, args, bar, models)
    else:
        before = surrogates.rate(models, start)[0]
        _cache_features()
        assert np.array_equal(surrogates.rate(models, start)[0], before), (
            "cache changed a rating"
        )
        grids, info = anneal(start, models, bar, args.steps, args.hard, args.seed)
        assert (surrogates.rate(models, grids)[0] >= bar).all()
    result = {"score_if_all_pass": scoring.diversity(grids)} | info
    print(f"Done: score if all pass {result['score_if_all_pass']:.4f}")
    settings = vars(args) | {
        "T0": args.t0 if args.neighbours else T0,
        "T1": T1,
        "lam0": LAM0,
        "lam1": LAM1,
        "cutoffs": cuts,
    }
    name = f"anneal_{args.models}_m{args.margin:.2f}"
    sel._save(name, grids, {"settings": settings} | result)


if __name__ == "__main__":
    main()
