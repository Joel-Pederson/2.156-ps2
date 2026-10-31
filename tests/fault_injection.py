"""Checks the checks: inject seven realistic faults and confirm a test catches each one.

A test that has never failed proves little. This script breaks the code in seven ways that
would corrupt results without any crash. For each fault it first runs the matching test on the
unchanged code (it must pass), then applies the fault and runs the test again (it must fail).
Faults are applied in memory only; no file is changed.

    python tests/fault_injection.py      # needs the course data (./setup_data.sh)

Prints CAUGHT or MISSED per fault, and exits with code 1 if any fault is missed. pytest does
not collect this file (its name does not start with test_); run it by hand after changing a
test or the code a test guards.
"""

import os
import pathlib
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]  # the repository root and tests/

import conftest
import test_features
import test_protocol

import features
import validity


def leaky_fold_data(grids, y, train_idx, val_idx, augment=False):
    """The classic leak: augment the whole set first, then draw training rows from it."""
    all_g, all_y = features.augment(grids, y)
    pick = np.random.default_rng(0).permutation(len(all_g))[: 8 * len(train_idx)]
    return all_g[pick], all_y[pick], grids[val_idx], y[val_idx]


def leaky_out_of_fold(name, grids, y, rows, use_cache=True):
    """A leak: each round's model trains on every grid, its own validation fold included."""
    variant = validity.VARIANTS[name]
    X = variant.featurize(grids)
    preds = np.full((validity.CV_REPEATS, len(y)), np.nan)
    for r, _, _, val in validity.cv_splits(len(y)):
        preds[r, val] = variant.make_model().fit(X, y).predict(X[val])
    return preds


def augmented_round_trains_on_everything(name, grids, y, rows, use_cache=True):
    """A leak in the augmentation path only: each round trains on every grid's 8 orientations."""
    variant = validity.VARIANTS[name]
    preds = np.full((validity.CV_REPEATS, len(y)), np.nan)
    for r, _, train, val in validity.cv_splits(len(y)):
        if variant.augment:
            train = np.arange(len(y))
        g_tr, y_tr, g_val, _ = validity.fold_data(grids, y, train, val, variant.augment)
        model = variant.make_model().fit(variant.featurize(g_tr), y_tr)
        preds[r, val] = model.predict(variant.featurize(g_val))
    return preds


def one_model_for_every_round(name, grids, y, rows, use_cache=True):
    """One model refitted every round. Harmless for Ridge or LightGBM, whose fit starts over;
    a leak for a model that keeps what it learned, such as a network trained further each round."""
    variant = validity.VARIANTS[name]
    X = variant.featurize(grids)
    preds = np.full((validity.CV_REPEATS, len(y)), np.nan)
    model = variant.make_model()
    for r, _, train, val in validity.cv_splits(len(y)):
        preds[r, val] = model.fit(X[train], y[train]).predict(X[val])
    return preds


def unrotated_orientations(grids):
    """A broken helper: eight copies of each grid as it is, none rotated or mirrored."""
    grids = np.asarray(grids)
    return np.stack([grids] * 8)


def run_out_of_fold_spy_test(augment=False):
    """Runs the spy test outside pytest, supplying its temporary folder and monkeypatch."""
    with tempfile.TemporaryDirectory() as tmp, pytest.MonkeyPatch.context() as mp:
        test_protocol.test_out_of_fold_never_trains_on_the_grids_it_predicts(
            augment, pathlib.Path(tmp), mp
        )


ZOO = conftest.grid_zoo.__wrapped__()  # the grids the feature tests use

# (description, (module, attribute, faulty value), the test that should catch it)
FAULTS = [
    (
        "The test-set recipe changes (seed 2156 -> 1)",
        (validity, "LOCKBOX_SEED", 1),
        lambda: test_protocol.test_lockbox_is_frozen(None),
    ),
    (
        "Augmentation happens before the split",
        (validity, "fold_data", leaky_fold_data),
        test_protocol.test_augmentation_only_touches_the_training_fold,
    ),
    (
        "One feature's definition changes (absent-district distance 13 -> 12)",
        (features, "ABSENT_DISTANCE", 12.0),
        lambda: test_features.test_fast_features_match_the_slow_reference(ZOO),
    ),
    (
        "A model trains on the grids it then predicts",
        (validity, "out_of_fold", leaky_out_of_fold),
        run_out_of_fold_spy_test,
    ),
    (
        "With augmentation, each round trains on every grid",
        (validity, "out_of_fold", augmented_round_trains_on_everything),
        lambda: run_out_of_fold_spy_test(augment=True),
    ),
    (
        "One model is reused across rounds",
        (validity, "out_of_fold", one_model_for_every_round),
        run_out_of_fold_spy_test,
    ),
    (
        "The orientation helper stops rotating",
        (features, "orientations", unrotated_orientations),
        test_features.test_orientations_are_the_eight_rotations_and_mirrors,
    ),
]


def main():
    missed = 0
    for description, (module, attribute, faulty), run_test in FAULTS:
        run_test()  # must pass on the unchanged code; an error here stops the script
        original = getattr(module, attribute)
        setattr(module, attribute, faulty)
        try:
            run_test()
            print(f"  MISSED  {description}")
            missed += 1
        except AssertionError:
            print(f"  CAUGHT  {description}")
        finally:
            setattr(module, attribute, original)
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
