"""The scorecard's metrics compute what docs/stage1_evaluation.md says they do.

- approved_at returns the largest approval count whose precision meets the target.
- precision_tradeoff (the "Why 98%" table in docs/stage1_evaluation.md) agrees with the scorecard at 98%, and a
  stricter precision level never approves more grids.
"""

import itertools

import numpy as np

import validity


def _synthetic(n=600, repeats=3, seed=4):
    """Labels in [0, 1], and noisy out-of-fold-style predictions for each shuffle."""
    rng = np.random.default_rng(seed)
    y = rng.random(n)
    preds = np.stack([y + rng.normal(0, 0.08, n) for _ in range(repeats)])
    return y, preds


def test_approved_at_returns_the_largest_count_meeting_the_target():
    y, preds = _synthetic()
    p = preds[0]
    count, cutoff, hits = validity.approved_at(p, y, 0.98)
    order = np.argsort(-p, kind="stable")
    passing = y[order] >= validity.THRESHOLD
    precision = np.cumsum(passing) / np.arange(1, len(y) + 1)
    assert count > 0
    assert precision[count - 1] >= 0.98  # the returned count meets the target
    assert not np.any(precision[count:] >= 0.98)  # and no larger count does
    assert hits == passing[:count].sum()
    assert cutoff == p[order][count - 1]


def test_tradeoff_matches_the_scorecard_and_is_monotone():
    y, preds = _synthetic()
    results = {"m": {adv: {"y": y, "preds": preds} for adv in validity.ADVISORS}}
    rows = validity.precision_tradeoff(results, "m", levels=(0.90, 0.95, 0.98, 0.99))
    at_98 = next(counts for level, _, counts in rows if level == 0.98)
    assert at_98["Wellness"] == validity.score(preds, y).approved
    for (_, _, looser), (_, _, stricter) in itertools.pairwise(rows):
        assert all(stricter[a] <= looser[a] for a in validity.ADVISORS)
    for level, all_four, _ in rows:
        assert all_four == level**4
