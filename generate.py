"""Design new city grids that the surrogate models predict all four advisors approve (v1).

Where this fits
---------------
Stage 3's generator (docs/plan.md, "Designing new grids"), a game of "hot and cold". From a
random dataset grid, rate all 196 one-cell changes (49 cells x 4 other districts) and keep the
one that most lowers the *total shortfall*: summed over the four advisors, how far each
prediction falls below that advisor's cut-off + margin. Stop at 0. When no change helps,
change 3 random cells (a "kick") and carry on, for at most MAX_STEPS moves.

v1's only gate drops near-copies: designs under MIN_CELLS_DIFFERENT cells from a pooled one.
Later: the familiar-ground gate (rule 3.4), the check-model veto (3.5), and rule 3.2's push
away from pooled designs and scoring by each ensemble's most pessimistic model (v1 uses the
mean of the three). Designs are only *predicted* to pass until rules 3.3 and 3.5 clear them.

Reads / writes
--------------
reads   the grids (validity.load_dev()); the models and cut-offs (surrogates.py)
writes  data/designs/pool_<seed>.npz   pooled designs: grids, pred and spread (n, 4), steps
                                       and kicks (moves), seconds, start (dataset row)
        data/designs/pool_<seed>.json  the settings, and a summary of the run

How to run
----------
    python generate.py --n 20 --margin 0.03 --workers 10 --seed 0

Surprises
---------
* Like search.py, it runs only from a clean, committed tree; and it never overwrites a pool.
* The models are fitted once, here, checked against the saved ones, and sent to each worker.
* The gate runs in design order after all designs finish: a seed always gives the same pool.
* Climbs stop the moment every bar is cleared, so designs end just above their bars: the
  margin is their only cushion. A start already over every bar is counted, not pooled.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import surrogates
import validity

DESIGNS_DIR = os.path.join(validity.PS2_ROOT, "data", "designs")
MARGIN = 0.03  # a starting value; the margin experiment (rule 3.3) will set it
MAX_STEPS = 200  # moves (kept changes plus kicks) before a design gives up
KICK_CELLS = 3
MIN_CELLS_DIFFERENT = 10  # the near-copy gate, out of 49 cells
WORKERS = os.cpu_count() or 1

N_CELLS, N_DISTRICTS = 49, 5
# One-cell change k adds SHIFTS[k] (1-4, mod 5) to cell CELLS[k], so it never repeats the grid.
CELLS = np.repeat(np.arange(N_CELLS), N_DISTRICTS - 1)
SHIFTS = np.tile(np.arange(1, N_DISTRICTS), N_CELLS)


def one_cell_changes(grid):
    """The 196 grids that differ from grid in exactly one cell."""
    flat = np.repeat(grid.reshape(1, N_CELLS), len(CELLS), axis=0)
    rows = np.arange(len(CELLS))
    flat[rows, CELLS] = (flat[rows, CELLS] + SHIFTS) % N_DISTRICTS
    return flat.reshape(-1, *grid.shape)


def kick(grid, rng, cells=KICK_CELLS):
    """A copy of grid with `cells` random cells, each changed to a different district."""
    flat = grid.flatten()
    where = rng.choice(N_CELLS, cells, replace=False)
    flat[where] = (flat[where] + rng.integers(1, N_DISTRICTS, cells)) % N_DISTRICTS
    return flat.reshape(grid.shape)


def design(start, models, cuts, margin, rng, max_steps=MAX_STEPS):
    """Hill-climb from start until every advisor's prediction reaches cut-off + margin.

    Returns (grid, info). info["shortfall"] is 0 if it got there within max_steps moves.
    """
    bar = np.array([cuts[a] for a in validity.ADVISORS]) + margin
    grid, kicks = np.asarray(start), 0
    for step in range(max_steps + 1):
        # Row 0 is the grid itself, so its rating comes free, even just after a kick.
        batch = np.concatenate([grid[None], one_cell_changes(grid)])
        pred, spread = surrogates.rate(models, batch)
        # Total shortfall: summed over advisors, how far each prediction is below its bar.
        short = np.maximum(0.0, bar - pred).sum(1)
        if short[0] == 0 or step == max_steps:
            break
        if short.min() < short[0]:  # a change beats the grid itself
            # Draw among equal bests: always taking the first would favour top-left cells.
            grid = batch[rng.choice(np.flatnonzero(short == short.min()))]
        else:
            grid, kicks = kick(grid, rng), kicks + 1
    info = {"pred": pred[0], "spread": spread[0], "shortfall": float(short[0])}
    return grid, {**info, "steps": step, "kicks": kicks}


_WORKER = {}  # design()'s other arguments, set once in each worker process


def _start_worker(models, cuts, margin, max_steps):
    _WORKER.update(models=models, cuts=cuts, margin=margin, max_steps=max_steps)


def _design_task(start, seed, index):
    """One design, timed. Its kicks use the index-th child of the run's seed."""
    tic = time.perf_counter()
    # A child seed, not [seed, index]: numpy treats [0, 0] like 0, the start-row draw's seed.
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))
    grid, info = design(start, rng=rng, **_WORKER)
    return grid, {**info, "seconds": time.perf_counter() - tic}


def _design_all(starts, setup, seed, workers):
    """_design_task from every start, results in start order. setup is _start_worker's args."""
    args = (starts, [seed] * len(starts), range(len(starts)))
    if workers == 1:  # in this process, where tests' monkeypatches apply
        _start_worker(*setup)
        return list(map(_design_task, *args))
    # "spawn", as in search.py: forked workers inherit thread pools that can hang LightGBM.
    context = multiprocessing.get_context("spawn")
    workers = min(workers, len(starts))
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=_start_worker, initargs=setup
    ) as pool:
        return list(pool.map(_design_task, *args))


def near_copy_gate(grids, min_cells=MIN_CELLS_DIFFERENT):
    """Indices kept, in order: each differs from every grid kept before it in at least
    min_cells cells. Dropped grids don't count against later ones."""
    flat = grids.reshape(len(grids), N_CELLS)  # -1 fails for 0 grids
    kept = []
    for i, g in enumerate(flat):
        if not kept or (flat[kept] != g).sum(1).min() >= min_cells:
            kept.append(i)
    return kept


def run(n, margin=MARGIN, workers=WORKERS, seed=0, max_steps=MAX_STEPS):
    """Design n grids from random dataset starts; save and return the pool that passes:
    a dict of the arrays saved in pool_<seed>.npz, plus "settings" and "summary"."""
    commit, uncommitted = validity._git_state()
    if uncommitted:  # the saved commit must hold the code that made the pool
        raise PermissionError(f"Commit first. Not committed: {uncommitted[:5]}")
    path = os.path.join(DESIGNS_DIR, f"pool_{seed}")
    if os.path.exists(path + ".json"):  # never overwrite a pool
        raise FileExistsError(f"{path}.json exists: use a new seed, or delete it")
    grids, _ = validity.load_dev()  # the grids only
    rows = np.random.default_rng(seed).choice(len(grids), n, replace=False)
    print(f"fitting the surrogate models, then designing {n} grids ...", flush=True)
    cuts = {a: float(c) for a, c in surrogates.cutoffs().items()}
    models = surrogates.fit_models()
    surrogates.check_refits(models)  # else the cut-offs aren't these models' cut-offs
    results = _design_all(grids[rows], (models, cuts, margin, max_steps), seed, workers)

    designs = np.stack([grid for grid, _ in results])
    per = {key: np.array([info[key] for _, info in results]) for key in results[0][1]}
    reached, moved = per["shortfall"] == 0, per["steps"] > 0
    passed = np.flatnonzero(reached & moved)  # unmoved starts: dataset grids
    kept = passed[near_copy_gate(designs[passed])]
    pool = {k: per[k][kept] for k in ("pred", "spread", "steps", "kicks", "seconds")}
    pool.update(grids=designs[kept], start=rows[kept])
    pool["settings"] = {"margin": margin, "seed": seed, "max_steps": max_steps}
    pool["settings"] |= {"workers": workers, "kick_cells": KICK_CELLS, "commit": commit}
    pool["settings"] |= {"min_cells_different": MIN_CELLS_DIFFERENT, "cutoffs": cuts}
    pool["summary"] = {
        "tried": n,
        "already_passing": int((reached & ~moved).sum()),
        "succeeded": len(passed),  # reached every bar after at least one move
        "near_copies": len(passed) - len(kept),
        "pooled": len(kept),
        "mean_steps": per["steps"].mean(),  # this and the next two: per design tried
        "mean_kicks": per["kicks"].mean(),
        "mean_seconds": per["seconds"].mean(),
    }
    _save(pool, path)
    return pool


def _save(pool, path):
    os.makedirs(DESIGNS_DIR, exist_ok=True)
    arrays = {k: v for k, v in pool.items() if isinstance(v, np.ndarray)}
    np.savez(path + ".npz", **arrays)
    with open(path + ".json", "w") as f:
        json.dump({k: pool[k] for k in ("settings", "summary")}, f, indent=2)


def parse_margin(text):
    """'0.1' -> 0.1, one margin for all four advisors; '0.2,0.1,0.1,0.1' -> one per advisor."""
    values = [float(v) for v in text.split(",")]
    if len(values) not in (1, len(validity.ADVISORS)):
        raise argparse.ArgumentTypeError("give one margin, or one per advisor")
    return values[0] if len(values) == 1 else values


def margin_label(margin):
    """For file names: m0.10, or mWellness0.20-Tax0.10-Transportation0.10-Business0.10."""
    if np.ndim(margin) == 0:
        return f"m{margin:.2f}"
    return "m" + "-".join(f"{a}{m:.2f}" for a, m in zip(validity.ADVISORS, margin))


def print_summary(pool):
    cfg, s = pool["settings"], pool["summary"]
    margins = np.broadcast_to(cfg["margin"], len(cfg["cutoffs"]))
    bars = ", ".join(
        f"{a} {c + m:.3f}" for (a, c), m in zip(cfg["cutoffs"].items(), margins)
    )
    print(f"""
Bars (cut-off + margin {cfg["margin"]}): {bars}
{s["tried"]} designs tried (seed {cfg["seed"]}), at most {cfg["max_steps"]} moves each:
  passed at the start    {s["already_passing"]} (dataset grids, not designs: not pooled)
  reached every bar      {s["succeeded"]}
  near-copies dropped    {s["near_copies"]} (under {MIN_CELLS_DIFFERENT} cells from a pooled one)
  pooled                 {s["pooled"]}, in data/designs/pool_{cfg["seed"]}.npz and .json
  moves per design       {s["mean_steps"]:.1f} on average, {s["mean_kicks"]:.1f} of them kicks
  seconds per design     {s["mean_seconds"]:.1f} on average
Predicted to pass, not yet trusted: rules 3.3 and 3.5 must clear these designs first.""")


def main():
    ap = argparse.ArgumentParser(description="Stage 3's generator v1 (docs/plan.md).")
    ap.add_argument("--n", type=int, default=20, help="designs to try (default 20)")
    ap.add_argument(
        "--margin",
        type=parse_margin,
        default=MARGIN,
        help=f"default {MARGIN}; "
        "or one per advisor, e.g. 0.2,0.1,0.1,0.1 (Wellness, Tax, Transportation, Business)",
    )
    ap.add_argument("--workers", type=int, default=WORKERS, help=f"default {WORKERS}")
    ap.add_argument("--seed", type=int, default=0, help="default 0; one pool per seed")
    args = ap.parse_args()
    print_summary(run(args.n, args.margin, args.workers, args.seed))


if __name__ == "__main__":
    main()
