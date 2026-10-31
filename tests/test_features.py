"""Plan tests 1-2: the features mean what we say they mean.

1. Every engineered feature is unchanged by all 8 rotations and mirrors (risk R14).
2. The fast vectorized code agrees with a slow, obviously-correct version (risk R14).
Plus: orientations() really gives the 8 distinct rotations and mirrors (other tests rely on
it), and an empty batch gives empty feature arrays.
"""

import numpy as np
import pytest
from reference_features import engineered_reference

import features


def test_there_are_75_uniquely_named_features_in_5_groups(grid_zoo):
    assert features.engineered(grid_zoo).shape == (len(grid_zoo), 75)
    assert len(set(features.ENGINEERED_NAMES)) == 75
    grouped = sorted(i for cols in features.FEATURE_GROUPS.values() for i in cols)
    assert grouped == list(range(75))  # every column in exactly one group


def test_every_feature_is_unchanged_by_rotations_and_mirrors(grid_zoo):
    base = features.engineered(grid_zoo)
    for name, turned in zip(features.SYMMETRIES, features.orientations(grid_zoo)):
        assert np.array_equal(features.engineered(turned), base), f"changed when {name}"


def test_fast_features_match_the_slow_reference(grid_zoo):
    fast = features.engineered(grid_zoo)
    for i, grid in enumerate(grid_zoo):
        ref = engineered_reference(grid)
        assert set(ref) == set(features.ENGINEERED_NAMES)
        for j, name in enumerate(features.ENGINEERED_NAMES):
            assert fast[i, j] == pytest.approx(ref[name], abs=1e-5), f"grid {i}: {name}"


def test_onehot_holds_exactly_the_grid(grid_zoo):
    oh = features.onehot(grid_zoo)
    assert oh.shape == (len(grid_zoo), 245)
    assert np.array_equal(oh.reshape(-1, 7, 7, 5).argmax(-1), grid_zoo)
    assert np.all(oh.sum(1) == 49)  # exactly one district per cell


def test_orientations_are_the_eight_rotations_and_mirrors():
    g = np.random.default_rng(7).integers(0, 5, size=(7, 7))
    expected = {np.rot90(h, k).tobytes() for h in (g, np.fliplr(g)) for k in range(4)}
    got = {np.ascontiguousarray(o).tobytes() for o in features.orientations(g)}
    assert len(expected) == 8 and got == expected
    turned, labels = features.augment(g[None], [0.5])
    assert len({t.tobytes() for t in turned}) == 8 and np.all(labels == 0.5)


def test_an_empty_batch_gives_empty_features():
    empty = np.zeros((0, 7, 7), np.int8)
    assert features.engineered(empty).shape == (0, 75)
    assert features.onehot(empty).shape == (0, 245)


def test_augment_keeps_each_grid_with_its_own_label(grid_zoo):
    grids, labels = features.augment(grid_zoo[:5], np.arange(5.0))
    assert grids.shape == (40, 7, 7) and labels.shape == (40,)
    for g, y in zip(grids, labels):  # each copy is an orientation of its source
        source = grid_zoo[int(y)]
        assert any(np.array_equal(g, o) for o in features.orientations(source))
