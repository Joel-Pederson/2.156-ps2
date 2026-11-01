"""Slot 3's new feature groups: access, crowding and commuting (the owner's ideas, 1 Nov).

Where this fits
---------------
features.py's 75 features see each district on its own, or cell-to-nearest-cell. These ask
how *homes* relate to the places they use: parks, shops (Commercial), offices, and jobs
(Industrial, Commercial or Office). Slot 3's search chooses which groups each advisor uses.

    crowding        27  homes within reach of each park/shop/office cell: busiest, average,
                        share with none (over- and under-crowding); reach 1, 2, 3 steps
    coverage        15  share of homes with a park/shop/office within reach; worst-served home;
                        parks/shops/offices per home
    cluster access  15  steps from each group (cluster) of homes to the nearest park/shop/office
    commute         17  homes per job; each home's trip to its nearest job; the busiest job
                        cell (congestion); idle jobs; job crowding and coverage by reach

Inputs/outputs: ``lab(grids)`` maps (n, 7, 7) district codes to (n, 74) float32, columns in
LAB_NAMES order; LAB_GROUPS maps group name to column indices.

Surprises
---------
* Distance is in grid steps (Manhattan), as in features.py. "Within reach r" means at most r steps.
* A district that is absent gives 0 for crowding and coverage and 13 steps for distances
  (features.ABSENT_DISTANCE), so "absent" reads as "very far".
* A home with several equally near jobs splits its trip equally between them, so every feature
  is unchanged by rotations and mirrors (tests/test_features_lab.py).
"""

import numpy as np
from scipy import ndimage

from features import _JOIN_SIDE_BY_SIDE, _STEPS, ABSENT_DISTANCE, SIZE

HOME, INDUSTRIAL, SHOP, PARK, OFFICE = range(5)
PLACES = {"Park": [PARK], "Commercial": [SHOP], "Office": [OFFICE]}
JOBS = [INDUSTRIAL, SHOP, OFFICE]
REACH = (1, 2, 3)
_WITHIN = {r: (_STEPS <= r).astype(np.float32) for r in REACH}


def _nearest(mask):
    """(n, 49) steps from every cell to the nearest cell in mask (ABSENT_DISTANCE if none)."""
    return np.where(mask[:, None, :], _STEPS[None], np.float32(ABSENT_DISTANCE)).min(-1)


def _share(hits, mask):
    """Share of the mask's cells where hits is True (0 if the mask is empty)."""
    return (hits & mask).sum(1) / np.maximum(mask.sum(1), 1)


def _crowding(homes, place, name):
    """Homes within reach of each place cell: busiest, average, share with none."""
    out = []
    for r in REACH:
        near = homes.astype(np.float32) @ _WITHIN[r]  # homes within r of every cell
        out.append((f"{name} r{r} busiest", np.where(place, near, 0).max(1)))
        out.append(
            (
                f"{name} r{r} average",
                (near * place).sum(1) / np.maximum(place.sum(1), 1),
            )
        )
        out.append((f"{name} r{r} share empty", _share(near == 0, place)))
    return out


def _cluster_access(grids, homes, dist, name):
    """Steps from each home's cluster (its nearest cell) to the place, averaged over homes."""
    labels = ndimage.label(grids == HOME, structure=_JOIN_SIDE_BY_SIDE)[0].reshape(
        len(grids), -1
    )
    best = np.full(labels.max() + 1, np.float32(ABSENT_DISTANCE))
    np.minimum.at(best, labels[homes], dist[homes])
    per_home = np.where(homes, best[labels], 0)
    n_homes = np.maximum(homes.sum(1), 1)
    out = [
        (f"cluster access {name} average", per_home.sum(1) / n_homes),
        (f"cluster access {name} worst", per_home.max(1)),
    ]
    out += [
        (f"cluster access {name} r{r}", _share(per_home <= r, homes)) for r in REACH
    ]
    return out


def _build(grids):
    """Every lab feature as (name, group, column)."""
    flat = grids.reshape(len(grids), SIZE * SIZE)
    homes = flat == HOME
    out = []
    for name, codes in PLACES.items():
        place = np.isin(flat, codes)
        dist = _nearest(place)
        out += [
            (n, "crowding", c) for n, c in _crowding(homes, place, f"crowding {name}")
        ]
        out += [
            (f"coverage {name} r{r}", "coverage", _share(dist <= r, homes))
            for r in REACH
        ]
        out.append((f"worst home->{name}", "coverage", np.where(homes, dist, 0).max(1)))
        out.append(
            (f"{name} per home", "coverage", place.sum(1) / np.maximum(homes.sum(1), 1))
        )
        out += [
            (n, "cluster access", c)
            for n, c in _cluster_access(grids, homes, dist, name)
        ]

    jobs = np.isin(flat, JOBS)
    n_homes, n_jobs = homes.sum(1), jobs.sum(1)
    steps = np.where(jobs[:, None, :], _STEPS[None], np.inf)  # (n, home cell, job cell)
    trip = steps.min(-1, keepdims=True)
    split = (steps == trip) & np.isfinite(trip)  # each home's equally-nearest jobs
    load = (
        homes[:, :, None] * split / np.maximum(split.sum(-1, keepdims=True), 1)
    ).sum(1)
    trip = np.where(homes & (n_jobs > 0)[:, None], trip[..., 0], 0)
    trip = np.where(homes & (n_jobs == 0)[:, None], ABSENT_DISTANCE, trip)
    out += [
        ("commute homes per job", "commute", n_homes / np.maximum(n_jobs, 1)),
        ("commute average trip", "commute", trip.sum(1) / np.maximum(n_homes, 1)),
        ("commute longest trip", "commute", trip.max(1)),
        ("commute busiest job", "commute", load.max(1)),
        ("commute idle jobs", "commute", _share(load == 0, jobs)),
    ]
    out += [(n, "commute", c) for n, c in _crowding(homes, jobs, "commute crowding")]
    out += [
        (f"commute coverage r{r}", "commute", _share(_nearest(jobs) <= r, homes))
        for r in REACH
    ]
    return out


def lab(grids):
    """(n, 7, 7) district codes -> (n, 74) float32, columns in LAB_NAMES order."""
    grids = np.asarray(grids)
    if not len(grids):
        return np.zeros((0, len(LAB_NAMES)), np.float32)
    return np.stack([c for _, _, c in _build(grids)], axis=1).astype(np.float32)


_SPEC = _build(np.zeros((1, SIZE, SIZE), np.int8))
LAB_NAMES = tuple(n for n, _, _ in _SPEC)
LAB_GROUPS = {
    g: [i for i, (_, gi, _) in enumerate(_SPEC) if gi == g]
    for g in ("crowding", "coverage", "cluster access", "commute")
}
