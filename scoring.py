"""How a submission is scored: the format rules, and the diversity score.

Where this fits
---------------
Stage 4 chooses the final 100 grids by *expected* diversity: every candidate set is scored
hundreds of times over simulated "which grids failed" scenarios. That needs a scorer with no
heavy imports. ``diversity()`` is it, and tests/test_scoring.py proves it gives exactly the
same number as the course's own ``utils_public.diversity_score`` on random sets, so "fast"
never means "different". ``check_submission()`` is the last gate before an array is saved
as a submission.

The score, in words (starter notebook, "The Task"):
    distance(a, b) = fraction of the 49 cells where grids a and b differ  (0 = identical)
    each valid grid contributes the mean distance to its 10 nearest OTHER valid grids
    score = sum of contributions / 100      (always /100, however many are valid)

Reads / writes
--------------
Nothing. Pure functions.

How to run
----------
Imported. For example::

    import numpy as np, scoring
    grids = np.random.default_rng(0).integers(0, 5, (100, 7, 7))
    scoring.diversity(grids)              # ≈ 0.70 for random grids
    scoring.check_submission(grids)       # raises ValueError if anything is off

Surprises
---------
* ``diversity()`` scores whatever grids you hand it, as if all were valid. To score a scenario
  where some fail, drop the failures first; that's how Stage 4's Monte Carlo uses it.
* ``check_submission()`` is stricter than the course in one way: it rejects duplicate grids.
  They're legal, but a duplicate's nearest neighbour is its twin at distance 0, so it can only
  cost points.
"""

from __future__ import annotations

import numpy as np

N_GRIDS = 100  # grids per submission, and the score's fixed divisor
K_NEIGHBOURS = 10  # each grid is compared with its 10 nearest others
SIZE = 7
N_DISTRICTS = 5


def diversity(grids, k=K_NEIGHBOURS, divisor=N_GRIDS):
    """The course's diversity score for these grids, all treated as valid."""
    grids = np.asarray(grids)
    n = len(grids)
    if n < 2:
        return 0.0
    flat = grids.reshape(n, -1)
    # Fraction of the 49 cells that differ, for every pair: the normalized Hamming distance.
    dist = (flat[:, None, :] != flat[None, :, :]).mean(-1)
    np.fill_diagonal(dist, np.inf)  # a grid is not its own neighbour
    k = min(k, n - 1)
    nearest = np.sort(dist, axis=1)[:, :k]
    return float(np.sum(np.mean(nearest, axis=1)) / divisor)


def check_submission(arr):
    """Raise ValueError, listing every problem, unless arr is a valid CP2 submission.

    The rules mirror the starter notebook's asserts: shape (100, 7, 7), dtype int (int64),
    values 0-4. Plus: no duplicate grids (see the module notes).
    """
    a = np.asarray(arr)
    problems = []
    if a.shape != (N_GRIDS, SIZE, SIZE):
        problems.append(f"shape is {a.shape}, must be ({N_GRIDS}, {SIZE}, {SIZE})")
    if a.dtype != np.int64:
        problems.append(f"dtype is {a.dtype}, must be int (int64); use .astype(int)")
    if a.size and np.issubdtype(a.dtype, np.number):
        lo, hi = a.min(), a.max()
        if lo < 0 or hi > N_DISTRICTS - 1 or not np.all(a == np.round(a)):
            problems.append(
                f"values span {lo}..{hi}; every cell must be a whole number 0-4"
            )
    if a.ndim == 3 and len(a):
        n_unique = len(np.unique(a.reshape(len(a), -1), axis=0))
        if n_unique < len(a):
            problems.append(
                f"{len(a) - n_unique} duplicate grid(s); each would score 0"
            )
    if problems:
        raise ValueError("not a valid submission: " + "; ".join(problems))
