"""Plan tests 3-4 (and the cache): the protocol can't quietly leak or drift.

3. The lockbox never changes: its membership matches a stored fingerprint (risk R13).
4. No leakage (risk R13):
   - the lockbox and the dev set never overlap, and load_dev() hides every lockbox label;
   - no lockbox grid hides in the dev set as a rotated or mirrored copy;
   - augmentation touches only the training fold, after the split.
Plus: the data counts quoted in the documentation are what the data holds; out_of_fold never
predicts a grid with a model that trained on it, augmented or not; the prediction cache key
changes when any input or setting that affects predictions changes; the lockbox refuses to
open unconfirmed or with anything uncommitted, before loading or logging anything, and a
confirmed opening returns only test-set ratings and logs itself; and Stage 1's declared
variants respect its cap.
"""

import dataclasses
import subprocess

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


def test_canonical_form_treats_rotations_and_mirrors_as_one_grid():
    g = np.random.default_rng(9).integers(0, 5, size=(7, 7)).astype(np.int8)
    for other in (np.rot90(g), np.fliplr(g), np.rot90(np.fliplr(g), 3)):
        assert _canonical(g[None]) == _canonical(np.ascontiguousarray(other)[None])


@pytest.mark.parametrize("augment", [False, True], ids=["plain", "augmented"])
def test_out_of_fold_never_trains_on_the_grids_it_predicts(
    augment, tmp_path, monkeypatch
):
    """A spy model records every grid it trains on and refuses to predict any of them.

    Grids are compared in canonical form, so a rotated or mirrored copy of a predicted grid
    in the training data counts too: that's how augmentation would leak. Each spy keeps
    everything it has ever trained on, so one model reused across rounds is caught as well.
    """
    rng = np.random.default_rng(5)
    grids = rng.integers(0, 5, size=(100, 7, 7)).astype(np.int8)
    y, rows = rng.random(100), np.arange(100)
    name = "spy_target"  # a temporary variant on one-hot inputs, which decode back to grids
    variant = validity.Variant("onehot", "ridge", "baseline", augment=augment)
    monkeypatch.setitem(validity.VARIANTS, name, variant)
    models, fits = [], []

    def as_grids(X):
        return X.reshape(-1, 7, 7, 5).argmax(-1).astype(np.int8)

    class Spy:
        def __init__(self):
            self.seen = set()
            models.append(self)

        def fit(self, X, y_train):
            self.seen |= _canonical(as_grids(X))
            fits.append(len(X))
            return self

        def predict(self, X):
            leaked = _canonical(as_grids(X)) & self.seen
            assert not leaked, "predicted a trained-on grid"
            return np.zeros(len(X))

    monkeypatch.setattr(validity, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(validity.Variant, "make_model", lambda self: Spy())
    preds = validity.out_of_fold(name, grids, y, rows)
    assert preds.shape == (validity.CV_REPEATS, 100) and not np.isnan(preds).any()
    # A fresh model each round, trained on 4 of the 5 folds (in all 8 orientations if
    # augmented).
    rounds = validity.CV_FOLDS * validity.CV_REPEATS
    assert len(models) == rounds
    assert fits == [(8 if augment else 1) * 80] * rounds
    validity.out_of_fold(name, grids, y, rows)
    assert len(fits) == rounds  # nothing changed: the saved predictions are reused
    validity.out_of_fold(name, grids, y[::-1].copy(), rows)
    assert len(fits) == 2 * rounds  # different ratings: recomputed


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


def _key_inputs():
    """Small stand-ins for a cache key's inputs: grids, model inputs X, ratings y, row numbers."""
    grids = (np.arange(40 * 49) % 5).reshape(40, 7, 7).astype(np.int8)
    X = np.arange(120, dtype=np.float32).reshape(40, 3)
    return grids, X, np.linspace(0, 1, 40), np.arange(40)


def test_cache_key_changes_whenever_predictions_could(monkeypatch):
    v = validity.VARIANTS["gbm_engineered"]
    grids, X, y, rows = _key_inputs()
    key = validity.cache_key
    base = key(v, grids, X, y, rows)
    assert key(v, grids, X, y, rows) == base  # deterministic
    assert key(v, grids, X, y[::-1], rows) != base  # different labels
    assert key(v, grids, X, y, rows + 1) != base  # different rows
    rotated = np.ascontiguousarray(np.rot90(grids, axes=(1, 2)))
    assert key(v, rotated, X, y, rows) != base  # different grid contents, same rows
    edited = X.copy()
    edited[0, 0] += 1
    assert key(v, grids, edited, y, rows) != base  # an input changed (e.g. a feature)
    assert key(v, grids, X.reshape(30, 4), y, rows) != base  # same values, new shape
    assert key(validity.VARIANTS["gbm_onehot"], grids, X, y, rows) != base
    monkeypatch.setattr(features, "FEATURE_VERSION", "something-new")
    assert key(v, grids, X, y, rows) != base  # engineered features relabelled


def test_cache_key_records_every_other_setting(monkeypatch):
    """Ridge's penalty, the one-hot version, the shuffle seeds, the number of folds and the
    augmentation flag all reach the key."""
    args = _key_inputs()
    ridge = validity.VARIANTS["ridge_onehot"]
    base = validity.cache_key(ridge, *args)
    monkeypatch.setattr(validity, "RIDGE_ALPHA", 2.0)
    assert validity.cache_key(ridge, *args) != base
    monkeypatch.setattr(validity, "RIDGE_ALPHA", 1.0)
    monkeypatch.setattr(features, "ONEHOT_VERSION", "something-new")
    assert validity.cache_key(ridge, *args) != base
    monkeypatch.undo()
    monkeypatch.setattr(validity, "CV_SEEDS", (5, 6, 7))
    assert validity.cache_key(ridge, *args) != base
    monkeypatch.undo()
    monkeypatch.setattr(validity, "CV_FOLDS", 4)
    assert validity.cache_key(ridge, *args) != base
    monkeypatch.undo()
    assert validity.cache_key(dataclasses.replace(ridge, augment=True), *args) != base


def test_cache_key_records_versions_gbm_settings_and_course_pin(monkeypatch):
    # Library versions, the GBM settings and the course-data version all reach the key.
    import lightgbm
    import scipy
    import sklearn

    args = _key_inputs()
    gbm = validity.VARIANTS["gbm_engineered"]
    base = validity.cache_key(gbm, *args)
    for module in (sklearn, lightgbm, np, scipy):
        monkeypatch.setattr(module, "__version__", "0.0-test")
        assert validity.cache_key(gbm, *args) != base, module.__name__
        monkeypatch.undo()
    monkeypatch.setattr(
        validity, "GBM_PARAMS", {**validity.GBM_PARAMS, "num_leaves": 15}
    )
    assert validity.cache_key(gbm, *args) != base
    monkeypatch.undo()
    monkeypatch.setattr(validity, "course_pin", lambda: "0" * 40)
    assert validity.cache_key(gbm, *args) != base


def _no_real_ratings():
    raise AssertionError("the lockbox loaded ratings before its checks passed")


def test_lockbox_refuses_to_open_unconfirmed(tmp_path, monkeypatch):
    # The refusal comes before any rating is loaded or the log is written. Git reports a clean
    # tree here, so only the confirm guard can refuse.
    log = tmp_path / "lockbox_log.md"
    monkeypatch.setattr(validity, "_load_raw", _no_real_ratings)
    monkeypatch.setattr(validity, "LOCKBOX_LOG", str(log))
    monkeypatch.setattr(validity, "_git_state", lambda: ("0" * 40, []))
    with pytest.raises(PermissionError, match="confirm=True"):
        validity.open_lockbox("just looking")
    assert not log.exists()


def test_lockbox_stays_shut_while_anything_is_uncommitted(tmp_path, monkeypatch):
    # Rule 2.7: the logged commit must hold the code that opens the lockbox.
    log = tmp_path / "lockbox_log.md"
    monkeypatch.setattr(validity, "_load_raw", _no_real_ratings)
    monkeypatch.setattr(validity, "LOCKBOX_LOG", str(log))
    monkeypatch.setattr(validity, "_git_state", lambda: ("0" * 40, ["new_script.py"]))
    with pytest.raises(PermissionError, match="new_script.py"):
        validity.open_lockbox("final grade", confirm=True)
    assert not log.exists()


def _git(repo, *args):
    """Git in a temporary test repository, with a fixed identity and no commit signing."""
    settings = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    settings += ["-c", "commit.gpgsign=false"]
    subprocess.run(
        ["git", "-C", str(repo), *settings, *args], check=True, capture_output=True
    )


def test_git_state_lists_every_uncommitted_change(tmp_path, monkeypatch):
    """Untracked, edited and staged files all count, even where git is set to hide untracked
    files (``git describe --dirty`` would miss an untracked file altogether)."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "status.showUntrackedFiles", "no")
    (tmp_path / "pipeline.py").write_text("x = 1\n")
    _git(tmp_path, "add", "pipeline.py")
    _git(tmp_path, "commit", "-q", "-m", "first")
    monkeypatch.setattr(validity, "PS2_ROOT", str(tmp_path))
    commit, uncommitted = validity._git_state()
    assert len(commit) == 40 and uncommitted == []
    (tmp_path / "new_script.py").write_text("opens the lockbox\n")
    assert validity._git_state()[1] == ["new_script.py"]
    (tmp_path / "pipeline.py").write_text("x = 2\n")
    assert "pipeline.py" in validity._git_state()[1]  # edited
    _git(tmp_path, "add", "pipeline.py")
    assert "pipeline.py" in validity._git_state()[1]  # staged, still not committed


def test_git_state_needs_the_folder_s_own_repository(tmp_path, monkeypatch):
    """A folder without its own repository must not borrow the commit of a repository above
    it that ignores it (the course folder above ps2 is one)."""
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("inner/\n")
    _git(tmp_path, "add", ".gitignore")
    _git(tmp_path, "commit", "-q", "-m", "outer")
    (tmp_path / "inner").mkdir()
    monkeypatch.setattr(validity, "PS2_ROOT", str(tmp_path / "inner"))
    with pytest.raises(subprocess.CalledProcessError):
        validity._git_state()


def test_a_confirmed_opening_returns_only_test_set_ratings_and_logs_it(
    tmp_path, monkeypatch
):
    """Uses made-up data and a temporary log: the real test set is never touched."""
    rng = np.random.default_rng(6)
    n = 12_000
    grids = np.zeros((n, 7, 7), np.int8)
    ratings = np.full((n, 4), np.nan)
    for a in range(4):
        rated = rng.choice(n, 1500, replace=False)
        ratings[rated, a] = rng.random(1500)
    log = tmp_path / "lockbox_log.md"
    monkeypatch.setattr(validity, "_load_raw", lambda: (grids, ratings))
    monkeypatch.setattr(validity, "LOCKBOX_LOG", str(log))
    monkeypatch.setattr(validity, "_git_state", lambda: ("f" * 40, []))  # all committed
    _, lock = validity.open_lockbox("synthetic check", confirm=True)
    mask = validity.lockbox_mask(ratings)
    # Nothing outside the test set comes back, and the test set comes back unchanged.
    assert np.array_equal(~np.isnan(lock), mask)
    assert np.array_equal(lock[mask], ratings[mask])
    assert "synthetic check" in log.read_text() and "f" * 40 in log.read_text()
    with pytest.raises(ValueError):
        validity.open_lockbox("   ", confirm=True)


def test_declared_variants_respect_the_cap():
    """Stage 1's cap: at most six variants per role. Stage 2's procedure cap (per advisor;
    docs/decisions.md) is checked in tests/test_search.py."""
    roles = [v.role for v in validity.VARIANTS.values()]
    for role in set(roles):
        assert roles.count(role) <= validity.MAX_VARIANTS_PER_ROLE, role
