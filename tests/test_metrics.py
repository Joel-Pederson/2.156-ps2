"""The scorecard's metrics compute what docs/stage1_evaluation.md says they do.

- approved_at returns the largest approval count whose precision meets the target, never
  stops inside a group of tied predictions, and reports no cut-off when nothing qualifies.
- A shuffle that approves nothing doesn't turn the advisor's cut-off into "none".
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


def test_approved_at_never_splits_a_tie_and_handles_no_approvals():
    pred = np.array([0.95, 0.9, 0.9, 0.9, 0.1])
    y = np.array([0.8, 0.8, 0.8, 0.1, 0.1])
    # The three grids predicted 0.9 stand or fall together: with all three, precision is 3/4.
    assert validity.approved_at(pred, y, 0.98) == (1, 0.95, 1)
    assert validity.approved_at(np.array([0.9, 0.1]), np.array([0.1, 0.9]), 0.98) == (
        0,
        np.inf,
        0,
    )


def test_cutoff_averages_only_the_shuffles_that_approve_something():
    rng = np.random.default_rng(8)
    y = rng.random(200)
    preds = np.stack(
        [y, y + 1e-3, -y]
    )  # the third "shuffle" ranks the worst grids first
    sc = validity.score(preds, y)
    assert sc.approved_per_shuffle[2] == 0 and not np.isfinite(sc.cutoff_per_shuffle[2])
    assert sc.cutoff == np.mean(sc.cutoff_per_shuffle[:2])


def test_score_works_for_a_single_shuffle():
    y, preds = _synthetic()
    one = validity.score(preds[:1], y)  # e.g. a quick one-shuffle screening run
    assert one.approved_per_shuffle == (validity.approved_at(preds[0], y)[0],)
    assert np.isfinite(one.r2_fold_sd)


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
