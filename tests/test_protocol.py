"""Plan tests 3-4 (and the cache): the protocol can't quietly leak or drift.

3. The lockbox never changes: its membership matches a stored fingerprint (risk R13).
4. No leakage (risk R13):
   - the lockbox and the dev set never overlap, and load_dev() hides every lockbox label;
   - no lockbox grid hides in the dev set as a rotated or mirrored copy;
   - augmentation touches only the training fold, after the split.
Plus: the data counts quoted in the documentation are what the data holds, the prediction
cache key changes when any setting that affects predictions changes, the lockbox refuses to
open unconfirmed, and the declared variants respect the cap.
"""

import numpy as np
import pytest

import features
import validity

# Fingerprint of the lockbox drawn on 2026-10-25 (seed 2156). If this test fails, the
# recipe changed. Do NOT update this number to make the test pass: every result measured
# against the old lockbox would silently stop meaning anything.
EXPECTED_LOCKBOX = "3723ff0021dbb688ac97e6afe7128169a51c9cf44838993defdaf776125a4b5e"


def test_lockbox_is_frozen(needs_data):
    _, ratings = validity._load_raw()
    mask = validity.lockbox_mask(ratings)
    assert mask.sum(0).tolist() == [1000, 1000, 1000, 1000]
    assert validity.lockbox_fingerprint(mask) == EXPECTED_LOCKBOX


def test_every_co_rated_grid_is_in_the_lockbox(needs_data):
    _, ratings = validity._load_raw()
    mask = validity.lockbox_mask(ratings)
    labelled = ~np.isnan(ratings)
    co_rated = labelled.sum(1) >= 2
    assert co_rated.sum() == 286
    assert np.array_equal(mask[co_rated], labelled[co_rated])


def test_data_counts_match_the_documentation(needs_data):
    # docs/stage1_evaluation.md ("Data") quotes these; `python validity.py --data` prints them.
    s = validity.data_summary()
    assert s["grids"] == 500_000
    assert list(s["ratings_per_advisor"].values()) == [5000] * 4
    # Grids rated by 0, 1, 2, 3 and 4 advisors: 283 + 3 = 286 co-rated, none by all four.
    assert s["grids_by_raters"] == [480_289, 19_425, 283, 3, 0]
    assert min(s["pairs"].values()) == 36 and max(s["pairs"].values()) == 64
    assert list(s["test_set_per_advisor"].values()) == [1000] * 4


def test_dev_data_hides_every_lockbox_label_and_nothing_else(needs_data):
    _, raw = validity._load_raw()
    _, dev = validity.load_dev()
    mask = validity.lockbox_mask(raw)
    assert np.isnan(dev[mask]).all()
    assert np.array_equal(~np.isnan(dev), ~np.isnan(raw) & ~mask)


def _canonical(grids):
    """One key per grid that's identical for all 8 of its orientations."""
    turned = features.orientations(grids).reshape(8, len(grids), 49)
    return {min(turned[s, i].tobytes() for s in range(8)) for i in range(len(grids))}


def test_no_lockbox_grid_hides_in_dev_as_a_rotated_copy(needs_data):
    grids, raw = validity._load_raw()
    mask = validity.lockbox_mask(raw)
    for a in range(4):
        dev_rows = np.flatnonzero(~np.isnan(raw[:, a]) & ~mask[:, a])
        lock_rows = np.flatnonzero(mask[:, a])
        assert not _canonical(grids[dev_rows]) & _canonical(grids[lock_rows])


def test_augmentation_only_touches_the_training_fold():
    rng = np.random.default_rng(1)
    grids = rng.integers(0, 5, size=(60, 7, 7)).astype(np.int8)
    y = np.arange(60.0)
    for _, _, train, val in validity.cv_splits(60, repeats=1):
        g_tr, _, g_val, y_val = validity.fold_data(grids, y, train, val, augment=True)
        assert len(g_tr) == 8 * len(train)
        assert np.array_equal(y_val, y[val]) and np.array_equal(g_val, grids[val])
        in_training = {g.tobytes() for g in g_tr}
        for g in g_val:
            for o in features.orientations(g):
                assert o.tobytes() not in in_training


def test_cache_key_changes_whenever_predictions_could(monkeypatch):
    v = validity.VARIANTS["gbm_engineered"]
    y, rows = np.linspace(0, 1, 40), np.arange(40)
    base = validity.cache_key(v, y, rows)
    assert validity.cache_key(v, y, rows) == base  # deterministic
    assert validity.cache_key(v, y[::-1], rows) != base  # different labels
    assert validity.cache_key(v, y, rows + 1) != base  # different rows
    assert validity.cache_key(validity.VARIANTS["gbm_onehot"], y, rows) != base
    monkeypatch.setattr(features, "FEATURE_VERSION", "something-new")
    assert validity.cache_key(v, y, rows) != base  # engineered features redefined


def test_cache_key_records_every_other_setting(monkeypatch):
    """Ridge's penalty, the one-hot version and the shuffle seeds all reach the key."""
    y, rows = np.linspace(0, 1, 40), np.arange(40)
    ridge = validity.VARIANTS["ridge_onehot"]
    base = validity.cache_key(ridge, y, rows)
    monkeypatch.setattr(validity, "RIDGE_ALPHA", 2.0)
    assert validity.cache_key(ridge, y, rows) != base
    monkeypatch.setattr(validity, "RIDGE_ALPHA", 1.0)
    monkeypatch.setattr(features, "ONEHOT_VERSION", "something-new")
    assert validity.cache_key(ridge, y, rows) != base
    monkeypatch.undo()
    monkeypatch.setattr(validity, "CV_SEEDS", (5, 6, 7))
    assert validity.cache_key(ridge, y, rows) != base


def test_cache_key_records_versions_gbm_settings_and_course_pin(monkeypatch):
    # Library versions, the GBM settings and the course-data version all reach the key.
    import lightgbm
    import sklearn

    y, rows = np.linspace(0, 1, 40), np.arange(40)
    gbm = validity.VARIANTS["gbm_engineered"]
    base = validity.cache_key(gbm, y, rows)
    for module in (sklearn, lightgbm):
        monkeypatch.setattr(module, "__version__", "0.0-test")
        assert validity.cache_key(gbm, y, rows) != base, module.__name__
        monkeypatch.undo()
    monkeypatch.setattr(
        validity, "GBM_PARAMS", {**validity.GBM_PARAMS, "num_leaves": 15}
    )
    assert validity.cache_key(gbm, y, rows) != base
    monkeypatch.undo()
    monkeypatch.setattr(validity, "course_pin", lambda: "0" * 40)
    assert validity.cache_key(gbm, y, rows) != base


def test_lockbox_refuses_to_open_unconfirmed():
    with pytest.raises(PermissionError):
        validity.open_lockbox("just looking")


def test_declared_variants_respect_the_cap():
    roles = [v.role for v in validity.VARIANTS.values()]
    for role in set(roles):
        assert roles.count(role) <= validity.MAX_VARIANTS_PER_ROLE, role
