"""Turn city grids into the numbers our models read.

Where this fits
---------------
Every model in this project sees a grid through one of the two functions here:

* ``onehot(grids)``: 245 yes/no values, "is cell (r, c) district k?". It describes every
  cell, but says nothing about how cells relate to each other.
* ``engineered(grids)``: 75 hand-designed numbers about *relationships*. That means how much
  of each district there is, what sits next to what, how far apart things are, how clumped
  they are, and edge versus centre.

The 75 answer five questions (README, "Stage 1"):

    counts        5   How much of each district?
    adjacency    30   What sits next to what? (side-by-side and diagonal, every district pair)
    distance     20   On average, how far is a cell of X from the nearest Y?
    clusters     10   How many separate blobs of each district, and how big is the biggest?
    edge/centre  10   How many sit on the border, and how far from the middle on average?

Every engineered feature is unchanged by the 8 rotations and mirrors of a grid (checked in
tests/test_features.py). That's deliberate: the planning probes found the advisors appear not
to care about orientation. If Stage 2's position test finds otherwise, that advisor gets extra
position-aware features on top; these 75 stay as they are.

This module also owns the 8 symmetries themselves (``orientations``, ``augment``), because
"rotate a grid" and "what stays the same when you rotate it" belong together.

Reads / writes
--------------
Nothing. Pure functions: arrays in, arrays out.

How to run
----------
Imported, not run. To see one grid's 75 values::

    python -c "import features, numpy as np; \\
        g = np.random.default_rng(0).integers(0, 5, (1, 7, 7)); \\
        print(dict(zip(features.ENGINEERED_NAMES, features.engineered(g)[0])))"

Surprises
---------
* District codes: 0 Residential, 1 Industrial, 2 Commercial, 3 Park, 4 Office.
* Distance is in grid steps, counting only up/down/left/right moves ("Manhattan" distance).
  If a grid has no district Y at all, distance-to-Y is 13. That's one more than the farthest
  two cells can be on a 7x7 grid (12), so "absent" reads as "very far".
* Blobs only join side-by-side neighbours. Two cells touching at a corner are separate blobs.
* Bump FEATURE_VERSION whenever a definition changes. It's part of validity.py's cache key,
  so stale predictions can never be reused (risk R14).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DISTRICTS = ("Residential", "Industrial", "Commercial", "Park", "Office")
N_DISTRICTS = len(DISTRICTS)
SIZE = 7

# Part of validity.py's cache key. Change it whenever any feature definition changes.
FEATURE_VERSION = "engineered-75-v1"

ABSENT_DISTANCE = 13.0  # one more than the 12 steps between opposite corners

_COORDS = np.array([(r, c) for r in range(SIZE) for c in range(SIZE)])
# Steps between every pair of the 49 cells.
_STEPS = np.abs(_COORDS[:, None, :] - _COORDS[None, :, :]).sum(-1).astype(np.float32)
_ON_BORDER = np.zeros((SIZE, SIZE), bool)
_ON_BORDER[[0, -1], :] = True
_ON_BORDER[:, [0, -1]] = True
_ON_BORDER = _ON_BORDER.ravel()
_STEPS_FROM_CENTRE = np.abs(_COORDS - SIZE // 2).sum(1).astype(np.float32)
# Blob labelling runs on a whole stack of grids at once; this structure joins side-by-side
# cells within a grid but never links one grid in the stack to the next.
_JOIN_SIDE_BY_SIDE = np.zeros((3, 3, 3), bool)
_JOIN_SIDE_BY_SIDE[1] = ndimage.generate_binary_structure(2, 1)


# --------------------------------------------------------------------------------------
# the two views of a grid
# --------------------------------------------------------------------------------------


def onehot(grids):
    """(n, 7, 7) district codes -> (n, 245) float32 of 0s and 1s."""
    grids = np.asarray(grids)
    is_district = np.arange(N_DISTRICTS, dtype=np.int8) == grids[..., None]
    return is_district.reshape(len(grids), -1).astype(np.float32)


def engineered(grids):
    """(n, 7, 7) district codes -> (n, 75) float32, columns in ENGINEERED_NAMES order."""
    return np.stack([column for _, column in _build(np.asarray(grids))], axis=1).astype(
        np.float32
    )


def _pair_count(a, b, i, j):
    """How many neighbour pairs (one cell from a, its neighbour from b) are districts {i, j}."""
    if i == j:
        return ((a == i) & (b == i)).sum((1, 2))
    return (((a == i) & (b == j)) | ((a == j) & (b == i))).sum((1, 2))


def _blobs(grids, k):
    """Per grid: how many separate blobs of district k, and the size of the biggest."""
    n = len(grids)
    labels, _ = ndimage.label(grids == k, structure=_JOIN_SIDE_BY_SIDE)
    flat = labels.reshape(n, -1)
    grid_of_cell, cell = np.nonzero(flat)
    blob_of_cell = flat[grid_of_cell, cell]
    blob_sizes = np.bincount(labels.ravel())
    biggest = np.zeros(n)
    np.maximum.at(biggest, grid_of_cell, blob_sizes[blob_of_cell])
    # Each distinct (grid, blob) pair is one blob; count them per grid.
    width = int(labels.max()) + 1
    one_per_blob = np.unique(grid_of_cell.astype(np.int64) * width + blob_of_cell)
    count = np.bincount(one_per_blob // width, minlength=n).astype(float)
    return count, biggest


def _build(grids):
    """Every engineered feature as (name, column). The single source of the column order."""
    n = len(grids)
    flat = grids.reshape(n, -1)
    is_k = [flat == k for k in range(N_DISTRICTS)]
    out = []

    # 1. How much of each district?
    for k, name in enumerate(DISTRICTS):
        out.append((f"count {name}", is_k[k].sum(1)))

    # 2. What sits next to what? Side-by-side means sharing an edge; diagonal, a corner.
    side = ((grids[:, :, :-1], grids[:, :, 1:]), (grids[:, :-1, :], grids[:, 1:, :]))
    diag = (
        (grids[:, :-1, :-1], grids[:, 1:, 1:]),
        (grids[:, :-1, 1:], grids[:, 1:, :-1]),
    )
    for i in range(N_DISTRICTS):
        for j in range(i, N_DISTRICTS):
            pair = f"{DISTRICTS[i]}-{DISTRICTS[j]}"
            out.append(
                (f"side-by-side {pair}", sum(_pair_count(a, b, i, j) for a, b in side))
            )
            out.append(
                (f"diagonal {pair}", sum(_pair_count(a, b, i, j) for a, b in diag))
            )

    # 3. How far is a cell of X from the nearest Y, on average?
    for b in range(N_DISTRICTS):
        # For every cell: steps to the nearest b cell (ABSENT_DISTANCE if the grid has none).
        nearest_b = np.where(
            is_k[b][:, None, :], _STEPS[None], np.float32(ABSENT_DISTANCE)
        ).min(-1)
        for a in range(N_DISTRICTS):
            if a == b:
                continue
            n_a = is_k[a].sum(1)
            mean = (nearest_b * is_k[a]).sum(1) / np.maximum(n_a, 1)
            out.append(
                (
                    f"distance {DISTRICTS[a]}->{DISTRICTS[b]}",
                    np.where(n_a > 0, mean, ABSENT_DISTANCE),
                )
            )

    # 4. Clumped or scattered?   5. Edge or centre?
    for k, name in enumerate(DISTRICTS):
        count, biggest = _blobs(grids, k)
        n_k = is_k[k].sum(1)
        out.append((f"blobs {name}", count))
        out.append((f"biggest blob {name}", biggest))
        out.append((f"border {name}", (is_k[k] & _ON_BORDER).sum(1)))
        out.append(
            (
                f"centre distance {name}",
                np.where(
                    n_k > 0,
                    (is_k[k] * _STEPS_FROM_CENTRE).sum(1) / np.maximum(n_k, 1),
                    0.0,
                ),
            )
        )
    return out


ENGINEERED_NAMES = tuple(name for name, _ in _build(np.zeros((1, SIZE, SIZE), np.int8)))

# Which columns answer which of the five questions. Stage 2's ablation hides one group at a time.
FEATURE_GROUPS = {
    "counts": [i for i, n in enumerate(ENGINEERED_NAMES) if n.startswith("count ")],
    "adjacency": [
        i
        for i, n in enumerate(ENGINEERED_NAMES)
        if n.startswith(("side-by-side ", "diagonal "))
    ],
    "distance": [
        i for i, n in enumerate(ENGINEERED_NAMES) if n.startswith("distance ")
    ],
    "clusters": [
        i
        for i, n in enumerate(ENGINEERED_NAMES)
        if n.startswith(("blobs ", "biggest blob "))
    ],
    "edge/centre": [
        i
        for i, n in enumerate(ENGINEERED_NAMES)
        if n.startswith(("border ", "centre distance "))
    ],
}


# --------------------------------------------------------------------------------------
# the 8 symmetries
# --------------------------------------------------------------------------------------


def _turn(grids, quarter_turns):
    return np.rot90(grids, quarter_turns, axes=(-2, -1))


SYMMETRIES = {
    "as is": lambda g: g,
    "rotated 90°": lambda g: _turn(g, 1),
    "rotated 180°": lambda g: _turn(g, 2),
    "rotated 270°": lambda g: _turn(g, 3),
    "mirrored": lambda g: g[..., ::-1],
    "mirrored, rotated 90°": lambda g: _turn(g[..., ::-1], 1),
    "mirrored, rotated 180°": lambda g: _turn(g[..., ::-1], 2),
    "mirrored, rotated 270°": lambda g: _turn(g[..., ::-1], 3),
}


def orientations(grids):
    """(n, 7, 7) or (7, 7) -> the 8 orientations stacked on a new first axis."""
    grids = np.asarray(grids)
    return np.stack([np.ascontiguousarray(f(grids)) for f in SYMMETRIES.values()])


def augment(grids, labels):
    """All 8 orientations of every grid, each keeping its grid's label.

    Only ever call this on a TRAINING fold, after the split. Rotating before splitting puts
    rotated copies of validation grids into training (tests/test_protocol.py checks this).
    """
    turned = orientations(grids)
    return turned.reshape(-1, SIZE, SIZE), np.tile(np.asarray(labels), len(SYMMETRIES))
