"""Slot 3's candidate feature groups (1 Nov): the owner's access, crowding and commute ideas,
plus groups suggested by a screen of the model's misses.

Where this fits
---------------
features.py's 75 features see each district on its own, or cell-to-nearest-cell. Slot 3's
search (search.GroupSearchGBM) starts from those 75 and adds up to 3 of these groups per
advisor. "Home" is Residential; "shop" is Commercial; "jobs" are Industrial, Commercial or
Office. Reach r means within r steps; r = 1, 2, 3 are all computed and the GBM picks.

    park / shop / office access    10 each  share of homes within reach; worst-served home;
                                            places per home; steps from each home cluster
    park / shop / office crowding   9 each  homes within reach of each place cell: busiest,
                                            average, share with none (under-crowding)
    commute                        17       homes per job; trip to the nearest job; busiest
                                            job cell (congestion); idle jobs; job crowding
    count ratios                   20       every district count over every other (+1)
    nuisance                        4       homes near industry
    mix                             3       how varied the districts are, locally and overall
    rings                          20       each district's count in each ring from the centre
    halves                         20       each district's count in the top/bottom/left/right
                                            three rows or columns: position-aware
    home surroundings               5       share of a home's 8 neighbours of each district
    park quality                    2       parks of 3+ cells; largest park per home

Inputs/outputs: ``lab(grids)`` maps (n, 7, 7) district codes to (n, 148) float32, columns in
LAB_NAMES order; LAB_GROUPS maps group name to column indices. ``neighbourhood(grids)`` gives
the upgraded models' 600 neighbourhood counts (7 Nov; upgraded.py), not a slot 3 group.

Surprises
---------
* Distance is in grid steps (Manhattan), as in features.py; absent districts read as 13 steps.
* Every group but "halves" is unchanged by rotations and mirrors (tests/test_features_lab.py);
  a home with several equally near jobs splits its trip between them to keep it that way.
* Change LAB_VERSION whenever a definition changes: slot 3's saved results record it.
"""

import numpy as np
from scipy import ndimage

from features import (
    _COORDS,
    _JOIN_SIDE_BY_SIDE,
    _STEPS,
    ABSENT_DISTANCE,
    DISTRICTS,
    SIZE,
)

LAB_VERSION = "lab-148-v1"
HOME, INDUSTRIAL, SHOP, PARK, OFFICE = range(5)
PLACES = {
    "park": ("Park", PARK),
    "shop": ("Commercial", SHOP),
    "office": ("Office", OFFICE),
}
JOBS = [INDUSTRIAL, SHOP, OFFICE]
REACH = (1, 2, 3)
_WITHIN = {r: (_STEPS <= r).astype(np.float32) for r in REACH}
_CHEB = np.abs(_COORDS[:, None] - _COORDS[None]).max(
    -1
)  # king-move steps between cells
_NEIGHBOURS = (_CHEB == 1).astype(np.float32)
_BLOCK = (_CHEB <= 1).astype(np.float32)  # a cell's 3x3 block
_RING = np.abs(_COORDS - SIZE // 2).max(1)  # 0 at the centre, 3 on the border
_HALVES = {
    "top": _COORDS[:, 0] < 3,
    "bottom": _COORDS[:, 0] > 3,
    "left": _COORDS[:, 1] < 3,
    "right": _COORDS[:, 1] > 3,
}


def _nearest(mask):
    """(n, 49) steps from every cell to the nearest cell in mask (ABSENT_DISTANCE if none)."""
    return np.where(mask[:, None, :], _STEPS[None], np.float32(ABSENT_DISTANCE)).min(-1)


def _share(hits, mask):
    """Share of the mask's cells where hits is True (0 if the mask is empty)."""
    return (hits & mask).sum(1) / np.maximum(mask.sum(1), 1)


def _per(a, b):
    return a / np.maximum(b, 1)


def _crowding(homes, place, name):
    """Homes within reach of each place cell: busiest, average, share with none."""
    out = []
    for r in REACH:
        near = homes.astype(np.float32) @ _WITHIN[r]  # homes within r of every cell
        out.append((f"{name} r{r} busiest", np.where(place, near, 0).max(1)))
        out.append((f"{name} r{r} average", _per((near * place).sum(1), place.sum(1))))
        out.append((f"{name} r{r} share empty", _share(near == 0, place)))
    return out


def _access(grids, homes, place, name):
    """How well homes reach the place: by cell, by home cluster, and places per home."""
    dist = _nearest(place)
    out = [(f"coverage {name} r{r}", _share(dist <= r, homes)) for r in REACH]
    out.append((f"worst home->{name}", np.where(homes, dist, 0).max(1)))
    out.append((f"{name} per home", _per(place.sum(1), homes.sum(1))))
    labels = ndimage.label(grids == HOME, structure=_JOIN_SIDE_BY_SIDE)[0]
    labels = labels.reshape(len(grids), -1)
    best = np.full(labels.max() + 1, np.float32(ABSENT_DISTANCE))
    np.minimum.at(best, labels[homes], dist[homes])  # each cluster's nearest cell
    cluster = np.where(homes, best[labels], 0)
    out.append((f"cluster access {name} average", _per(cluster.sum(1), homes.sum(1))))
    out.append((f"cluster access {name} worst", cluster.max(1)))
    out += [(f"cluster access {name} r{r}", _share(cluster <= r, homes)) for r in REACH]
    return out


def _commute(homes, jobs):
    """Each home's trip to its nearest job, and how the trips pile up on job cells."""
    n_homes, n_jobs = homes.sum(1), jobs.sum(1)
    steps = np.where(jobs[:, None, :], _STEPS[None], np.inf)  # (n, home cell, job cell)
    trip = steps.min(-1, keepdims=True)
    split = (steps == trip) & np.isfinite(trip)  # each home's equally-nearest jobs
    load = (
        homes[:, :, None] * split / np.maximum(split.sum(-1, keepdims=True), 1)
    ).sum(1)
    trip = np.where(
        homes, np.where(n_jobs[:, None] > 0, trip[..., 0], ABSENT_DISTANCE), 0
    )
    out = [
        ("commute homes per job", _per(n_homes, n_jobs)),
        ("commute average trip", _per(trip.sum(1), n_homes)),
        ("commute longest trip", trip.max(1)),
        ("commute busiest job", load.max(1)),
        ("commute idle jobs", _share(load == 0, jobs)),
    ]
    out += _crowding(homes, jobs, "commute crowding")
    near = _nearest(jobs)
    return out + [(f"commute coverage r{r}", _share(near <= r, homes)) for r in REACH]


def _screened(grids, flat, is_k):
    """The groups suggested by the screen: {group: [(name, column)]}."""
    counts = [k.sum(1) for k in is_k]
    homes = is_k[HOME]
    industry_near = [is_k[INDUSTRIAL].astype(np.float32) @ _WITHIN[r] for r in (1, 2)]
    share = np.stack(counts, 1) / (SIZE * SIZE)
    unlike = (grids[:, :, :-1] != grids[:, :, 1:]).sum((1, 2)) + (
        grids[:, :-1] != grids[:, 1:]
    ).sum((1, 2))
    parks = ndimage.label(grids == PARK, structure=_JOIN_SIDE_BY_SIDE)[0].reshape(
        len(grids), -1
    )
    sizes = np.bincount(parks.ravel())
    owner = np.repeat(np.arange(len(grids)), SIZE * SIZE)[
        np.unique(parks.ravel(), True)[1]
    ]
    big, largest = np.zeros(len(grids)), np.zeros(len(grids))
    np.add.at(big, owner[1:], sizes[1:] >= 3)  # label 0 is "not a park"
    np.maximum.at(largest, owner[1:], sizes[1:])
    near_k = [k.astype(np.float32) @ _NEIGHBOURS for k in is_k]
    seen = (homes * _NEIGHBOURS.sum(1)).sum(1)  # neighbour cells around all homes
    return {
        "count ratios": [
            (f"ratio {DISTRICTS[i]}/{DISTRICTS[j]}", counts[i] / (counts[j] + 1))
            for i in range(5)
            for j in range(5)
            if i != j
        ],
        "nuisance": [
            ("homes with industry within 1", _share(industry_near[0] > 0, homes)),
            ("homes with industry within 2", _share(industry_near[1] > 0, homes)),
            (
                "industry within 1 per home",
                _per((industry_near[0] * homes).sum(1), counts[0]),
            ),
            ("industry within 1 of worst home", (industry_near[0] * homes).max(1)),
        ],
        "mix": [
            (
                "district entropy",
                -(share * np.log(np.where(share > 0, share, 1))).sum(1),
            ),
            ("districts per 3x3 block", sum((k @ _BLOCK) > 0 for k in is_k).mean(1)),
            ("unlike side-by-side pairs", unlike),
        ],
        "rings": [
            (f"ring {r} {DISTRICTS[k]}", (is_k[k] & (_RING == r)).sum(1))
            for k in range(5)
            for r in range(4)
        ],
        "halves": [
            (f"{h} {DISTRICTS[k]}", (is_k[k] & m).sum(1))
            for k in range(5)
            for h, m in _HALVES.items()
        ],
        "home surroundings": [
            (f"home neighbours {DISTRICTS[k]}", _per((near_k[k] * homes).sum(1), seen))
            for k in range(5)
        ],
        "park quality": [
            ("parks of 3+ cells", big),
            ("largest park per home", _per(largest, counts[0])),
        ],
    }


def _build(grids):
    """Every lab feature as (name, group, column)."""
    flat = grids.reshape(len(grids), SIZE * SIZE)
    is_k = [flat == k for k in range(5)]
    homes = is_k[HOME]
    out = []
    for group, (name, code) in PLACES.items():
        place = is_k[code]
        out += [
            (n, f"{group} access", c) for n, c in _access(grids, homes, place, name)
        ]
        out += [
            (n, f"{group} crowding", c)
            for n, c in _crowding(homes, place, f"crowding {name}")
        ]
    out += [(n, "commute", c) for n, c in _commute(homes, np.isin(flat, JOBS))]
    for group, columns in _screened(grids, flat, is_k).items():
        out += [(n, group, c) for n, c in columns]
    return out


def lab(grids):
    """(n, 7, 7) district codes -> (n, 148) float32, columns in LAB_NAMES order."""
    grids = np.asarray(grids)
    if not len(grids):
        return np.zeros((0, len(LAB_NAMES)), np.float32)
    return np.stack([c for _, _, c in _build(grids)], axis=1).astype(np.float32)


_SPEC = _build(np.zeros((1, SIZE, SIZE), np.int8))
LAB_NAMES = tuple(n for n, _, _ in _SPEC)
LAB_GROUPS = {
    g: [i for i, (_, gi, _) in enumerate(_SPEC) if gi == g] for _, g, _ in _SPEC
}


def _shifted(m, dr, dc):
    """m (n, 7, 7) moved dr rows down and dc columns right; cells moved in are 0."""
    out = np.zeros_like(m)
    out[:, max(dr, 0) : SIZE + min(dr, 0), max(dc, 0) : SIZE + min(dc, 0)] = m[
        :, max(-dr, 0) : SIZE + min(-dr, 0), max(-dc, 0) : SIZE + min(-dc, 0)
    ]
    return out


def neighbourhood(grids):
    """(n, 7, 7) -> (n, 600): the upgraded models' neighbourhood library (upgraded.py).
    For each reach d = 1, 2, 3 (in steps, or as a square block) and each ordered pair of
    districts (a, b): the a-b pairs within reach, and the a-cells with at least 1, 2 or 3
    b-cells within reach. 2 shapes x 3 reaches x 25 pairs x 4 = 600 counts."""
    masks = [(np.asarray(grids) == k).astype(np.int16) for k in range(5)]
    cols = []
    for square in (False, True):
        for d in REACH:
            offsets = [
                (r, c)
                for r in range(-d, d + 1)
                for c in range(-d, d + 1)
                if (r, c) != (0, 0) and (square or abs(r) + abs(c) <= d)
            ]
            for b in range(5):
                near = sum(_shifted(masks[b], r, c) for r, c in offsets)
                for a in range(5):
                    cols.append((masks[a] * near).sum((1, 2)))
                    cols += [(masks[a] * (near >= k)).sum((1, 2)) for k in (1, 2, 3)]
    return np.stack(cols, 1).astype(float)
