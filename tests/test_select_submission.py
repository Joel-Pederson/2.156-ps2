"""select_submission.py, on small synthetic grids: greedy max-min (by hand and against a slow
version), the chance of passing all four, the fallback below the cut-offs, the saved file."""

import json

import numpy as np
import pytest

import calibration
import scoring
import select_submission as sel
import surrogates
import validity
from validity import ADVISORS


def _world(monkeypatch, n=600):
    """Random grids and ratings; each advisor's stand-in chance of passing is its pred."""
    monkeypatch.setattr(calibration, "pass_probability", lambda cal, pred, spread: pred)
    rng = np.random.default_rng(7)
    grids = rng.integers(0, 5, (n, 7, 7)).astype(np.int8)
    return grids, rng.random((n, 4)).astype(np.float32), np.zeros((n, 4), np.float32)


def test_greedy_maxmin_by_hand():
    a, c, d = np.zeros((7, 7), int), np.ones((7, 7), int), np.full((7, 7), 2)
    b = a.copy()
    b[3, 3] = 1  # one cell away from a
    # a first (highest priority). c and d are both 49 cells from a; d has the higher
    # priority. Then c (49 cells from a and from d) beats b (1 cell from a).
    grids, priority = np.stack([a, b, c, d]), np.array([0.9, 0.1, 0.5, 0.6])
    assert sel.greedy_maxmin(grids, priority, 4).tolist() == [0, 3, 2, 1]


def _slow_greedy(grids, priority, n):  # the same rule, every distance recomputed
    flat = grids.reshape(len(grids), -1)
    chosen = [int(np.argmax(priority))]
    while len(chosen) < n:
        rest = [i for i in range(len(flat)) if i not in chosen]
        far = [min((flat[i] != flat[j]).sum() for j in chosen) for i in rest]
        chosen.append(max(zip(far, priority[rest], rest))[2])  # farthest, then priority
    return chosen


def test_greedy_maxmin_matches_a_slow_version(grid_zoo):
    p = np.random.default_rng(4).random(len(grid_zoo))
    assert sel.greedy_maxmin(grid_zoo, p, 12).tolist() == _slow_greedy(grid_zoo, p, 12)


def test_chance_of_passing_all_four_is_the_product(monkeypatch):
    # A stand-in calibration: an advisor's chance is cal × pred + spread.
    monkeypatch.setattr(calibration, "pass_probability", lambda c, p, s: c * p + s)
    rng = np.random.default_rng(5)
    pred, spread = rng.random((20, 4)) / 2, rng.random((20, 4)) / 10
    cals = dict(zip(ADVISORS, (0.5, 0.6, 0.7, 0.8)))
    expected = np.prod(np.array([0.5, 0.6, 0.7, 0.8]) * pred + spread, axis=1)
    np.testing.assert_allclose(sel.p_all_four(pred, spread, cals), expected)


def test_expected_score_counts_only_valid_grids():
    grids = np.random.default_rng(6).integers(0, 5, (30, 7, 7))
    all_valid = scoring.diversity(grids)
    assert sel.expected_score(grids, np.ones(30)) == pytest.approx(all_valid)
    assert sel.expected_score(grids, np.zeros(30)) == 0.0
    assert 0 < sel.expected_score(grids, np.full(30, 0.5)) < all_valid


def test_insurance_takes_candidates_safest_first(monkeypatch):
    grids, pred, spread = _world(monkeypatch)
    cuts, cals = dict.fromkeys(ADVISORS, 0.3), dict.fromkeys(ADVISORS)
    idx, stats = sel.insurance(pred, spread, cuts, cals, grids=grids)
    candidates = np.flatnonzero((pred >= 0.3).all(1))
    p = sel.p_all_four(pred, spread, cals)
    assert (stats["candidates"], stats["filled_below_cutoffs"]) == (len(candidates), 0)
    assert len(set(idx)) == 100 and set(idx) <= set(candidates)
    assert idx[0] == candidates[np.argmax(p[candidates])]
    assert stats["expected_valid"] == pytest.approx(p[idx].sum())


def test_too_few_candidates_are_topped_up_by_smallest_shortfall(monkeypatch):
    grids, pred, spread = _world(monkeypatch)
    cuts, cals = dict.fromkeys(ADVISORS, 0.75), dict.fromkeys(ADVISORS)
    short = sel.shortfall(pred, cuts)
    safest = np.argsort(short, kind="stable")[:3]
    grids[safest[1]] = grids[safest[0]]  # a repeated grid: one copy only may go in
    allowed = np.arange(len(grids)) != safest[2]  # say, safest[2] is a rated grid
    idx, stats = sel.insurance(pred, spread, cuts, cals, grids=grids, allowed=allowed)
    scoring.check_submission(grids[idx].astype(np.int64))  # 100 distinct grids
    assert (safest[0] in idx) != (safest[1] in idx) and safest[2] not in idx
    assert stats["candidates"] == (short[allowed] == 0).sum() < 100
    assert stats["filled_below_cutoffs"] == (short[idx] > 0).sum() > 0
    rest = np.setdiff1d(np.flatnonzero(allowed), [*idx, *safest[:2]])
    assert short[idx].max() <= short[rest].min()


def test_reference_takes_all_when_exactly_100_are_approved(monkeypatch):
    grids, pred, _ = _world(monkeypatch)
    pred[:], pred[200:300] = 0.1, 0.9  # only grids 200-299 are approved
    cuts = dict.fromkeys(ADVISORS, 0.5)
    reference = sel.reference_selection(pred, cuts, grids=grids)
    assert reference == pytest.approx(scoring.diversity(grids[200:300]))  # in any order


def test_cli_saves_a_valid_submission_and_its_stats(tmp_path, monkeypatch):
    grids, pred, spread = _world(monkeypatch)
    info = {"cutoffs": dict.fromkeys(ADVISORS, 0.3), "commit": "c" * 40}
    dev = np.where(np.arange(len(grids))[:, None] < 50, 0.9, np.nan)  # never picked
    monkeypatch.setattr(validity, "load_dev", lambda: (grids, dev))
    monkeypatch.setattr(validity, "_git_state", lambda: ("d" * 40, []))
    monkeypatch.setattr(surrogates, "load_ratings", lambda: (pred, spread, info))
    monkeypatch.setattr(calibration, "fit_all", lambda: dict.fromkeys(ADVISORS))
    monkeypatch.setattr(sel, "SUBMISSIONS_DIR", str(tmp_path))
    sel.main()
    [path] = tmp_path.glob("insurance_*.npy")
    saved = np.load(path)
    scoring.check_submission(saved)  # 100 distinct 7×7 int64 grids
    record = json.loads(path.with_suffix(".json").read_text())
    assert (saved == grids[record["rows"]]).all() and min(record["rows"]) >= 50
    assert record["commit"] == "d" * 40 and record["ratings"] == info
    assert {"expected_score", "reference_diversity", "assumes"} <= set(record)
