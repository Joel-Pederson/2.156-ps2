"""Shared test setup: where the course data lives, and a zoo of grids to test on.

Tests that need the course data use the ``needs_data`` fixture. Without the data
(./setup_data.sh not run yet) they are SKIPPED locally, with a reason. In CI, where
CP2_REQUIRE_DATA=1, a missing download FAILS instead, so it can't hide behind a skip.
"""

import os

import numpy as np
import pytest

os.environ.setdefault(
    "MPLBACKEND", "Agg"
)  # utils_public imports matplotlib; never open windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURSE_ROOT = os.path.join(ROOT, "data", "cp2-repo")


@pytest.fixture
def needs_data():
    if os.path.isdir(os.path.join(COURSE_ROOT, "datasets")):
        return
    if os.environ.get("CP2_REQUIRE_DATA") == "1":
        pytest.fail("course data missing (CP2_REQUIRE_DATA=1): run ./setup_data.sh")
    pytest.skip("course data not downloaded: run ./setup_data.sh")


@pytest.fixture(scope="session")
def grid_zoo():
    """200 random grids, 50 clumpy ones like the dataset's, and hand-made edge cases."""
    rng = np.random.default_rng(0)
    random = rng.integers(0, 5, size=(200, 7, 7))
    # Clumpy: copy a neighbour's district most of the time, so blobs form.
    clumpy = random[:50].copy()
    for g in clumpy:
        for r in range(7):
            for c in range(1, 7):
                if rng.random() < 0.6:
                    g[r, c] = g[r, c - 1]
    edge = [
        np.full((7, 7), 3),  # all park: four districts absent
        np.zeros((7, 7), int),  # all residential
        (np.indices((7, 7)).sum(0) % 2) * 3,  # checkerboard of homes and parks
        np.where(np.arange(49).reshape(7, 7) < 25, 0, 4),  # two halves
        np.arange(49).reshape(7, 7) % 5,  # every district, striped
    ]
    return np.concatenate([random, clumpy, np.stack(edge)]).astype(np.int8)
