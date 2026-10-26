"""A slow, obviously-correct version of features.engineered, for tests only.

Deliberately written independently of features.py: plain loops, no numpy tricks, and
its own copy of the district names. If the fast vectorized code has a bug, this version
won't share it, so the comparison test catches it. One grid at a time; returns a dict
{feature name: value}, keyed by the same names features.ENGINEERED_NAMES uses.
"""

from collections import deque

DISTRICTS = ("Residential", "Industrial", "Commercial", "Park", "Office")
ABSENT = 13.0


def engineered_reference(grid):
    g = [[int(v) for v in row] for row in grid]
    cells = [(r, c) for r in range(7) for c in range(7)]
    at = lambda p: g[p[0]][p[1]]
    out = {}

    for k, name in enumerate(DISTRICTS):
        out[f"count {name}"] = sum(1 for p in cells if at(p) == k)

    side = [((r, c), (r, c + 1)) for r in range(7) for c in range(6)] + [
        ((r, c), (r + 1, c)) for r in range(6) for c in range(7)
    ]
    diag = [((r, c), (r + 1, c + 1)) for r in range(6) for c in range(6)] + [
        ((r, c + 1), (r + 1, c)) for r in range(6) for c in range(6)
    ]
    for i in range(5):
        for j in range(i, 5):
            want = {i, j}
            key = f"{DISTRICTS[i]}-{DISTRICTS[j]}"
            out[f"side-by-side {key}"] = sum(
                1 for a, b in side if {at(a), at(b)} == want
            )
            out[f"diagonal {key}"] = sum(1 for a, b in diag if {at(a), at(b)} == want)

    for a in range(5):
        for b in range(5):
            if a == b:
                continue
            a_cells = [p for p in cells if at(p) == a]
            b_cells = [p for p in cells if at(p) == b]
            if not a_cells or not b_cells:
                value = ABSENT
            else:
                value = sum(
                    min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in b_cells)
                    for p in a_cells
                ) / len(a_cells)
            out[f"distance {DISTRICTS[a]}->{DISTRICTS[b]}"] = value

    for k, name in enumerate(DISTRICTS):
        seen, sizes = set(), []
        for start in cells:
            if at(start) != k or start in seen:
                continue
            queue, size = deque([start]), 0  # flood-fill one blob
            seen.add(start)
            while queue:
                r, c = queue.popleft()
                size += 1
                for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nb[0] < 7
                        and 0 <= nb[1] < 7
                        and nb not in seen
                        and at(nb) == k
                    ):
                        seen.add(nb)
                        queue.append(nb)
            sizes.append(size)
        mine = [p for p in cells if at(p) == k]
        out[f"blobs {name}"] = len(sizes)
        out[f"biggest blob {name}"] = max(sizes, default=0)
        out[f"border {name}"] = sum(1 for r, c in mine if r in (0, 6) or c in (0, 6))
        out[f"centre distance {name}"] = (
            sum(abs(r - 3) + abs(c - 3) for r, c in mine) / len(mine) if mine else 0.0
        )
    return out
