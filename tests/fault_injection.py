"""Checks the checks: inject three realistic faults and confirm a test catches each one.

A test that has never failed proves little. This script breaks the code in three ways that
would corrupt results without any crash. For each fault it first runs the matching test on the
unchanged code (it must pass), then applies the fault and runs the test again (it must fail).
Faults are applied in memory only; no file is changed.

    python tests/fault_injection.py      # needs the course data (./setup_data.sh)

Prints CAUGHT or MISSED per fault, and exits with code 1 if any fault is missed. pytest does
not collect this file (its name does not start with test_); run it by hand after changing a
test or the code a test guards.
"""

import os
import sys

import numpy as np

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
