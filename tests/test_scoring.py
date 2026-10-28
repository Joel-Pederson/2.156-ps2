"""Plan tests 5-6: we score submissions exactly like the course does.

5. scoring.diversity() equals the course's utils_public.diversity_score.
6. The submission checker rejects every way a submission can be malformed.
"""

import sys

import numpy as np
import pytest
from conftest import COURSE_ROOT

import scoring


def test_our_diversity_equals_the_course_function(needs_data):
    sys.path.insert(0, COURSE_ROOT)
    from utils_public import diversity_score

    rng = np.random.default_rng(2)
    for n in (2, 3, 11, 50, 100, 140):
        grids = rng.integers(0, 5, size=(n, 7, 7))
        if n > 10:
            grids[: n // 5] = grids[n // 5 : 2 * (n // 5)]  # duplicates included
        assert scoring.diversity(grids) == diversity_score(grids), f"n={n}"
    assert scoring.diversity(rng.integers(0, 5, size=(1, 7, 7))) == 0.0


def _valid():
    rng = np.random.default_rng(3)
    return rng.integers(0, 5, size=(100, 7, 7)).astype(int)


def test_a_valid_submission_passes():
    scoring.check_submission(_valid())


@pytest.mark.parametrize(
    "break_it, complaint",
    [
        (lambda a: a[:99], "shape"),
        (lambda a: a.reshape(100, 49), "shape"),
        (lambda a: a.astype(float), "dtype"),
        (lambda a: a.astype(np.int8), "dtype"),
        (lambda a: np.where(a == 0, 5, a), "values"),
        (lambda a: np.where(a == 0, -1, a), "values"),
        (lambda a: np.concatenate([a[:99], a[:1]]), "duplicate"),
    ],
)
def test_a_broken_submission_is_rejected(break_it, complaint):
    with pytest.raises(ValueError, match=complaint):
        scoring.check_submission(break_it(_valid()))
