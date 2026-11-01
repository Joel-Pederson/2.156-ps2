"""Slot 3's lab features mean what features_lab.py says they mean."""

import numpy as np
import pytest

import features
import features_lab as fl


def _one_home_grid():
    """All Office, except a home in the top-left corner, a park beside it and a shop in the
    bottom-right corner."""
    g = np.full((1, 7, 7), fl.OFFICE, np.int8)
    g[0, 0, 0], g[0, 0, 1], g[0, 6, 6] = fl.HOME, fl.PARK, fl.SHOP
    return g


def test_148_uniquely_named_features_in_14_groups(grid_zoo):
    assert fl.lab(grid_zoo).shape == (len(grid_zoo), 148)
    assert len(set(fl.LAB_NAMES)) == 148 and len(fl.LAB_GROUPS) == 14
    assert sorted(i for cols in fl.LAB_GROUPS.values() for i in cols) == list(
        range(148)
    )


def test_every_group_but_halves_is_unchanged_by_rotations_and_mirrors(grid_zoo):
    keep = [i for g, cols in fl.LAB_GROUPS.items() if g != "halves" for i in cols]
    base = fl.lab(grid_zoo)[:, keep]
    for name, turned in zip(features.SYMMETRIES, features.orientations(grid_zoo)):
        assert np.allclose(fl.lab(turned)[:, keep], base), f"changed when {name}"


def test_hand_checked_values():
    x = dict(zip(fl.LAB_NAMES, fl.lab(_one_home_grid())[0]))
    assert x["coverage Park r1"] == 1  # the only home is 1 step from the park
    assert x["crowding Park r1 busiest"] == 1 and x["crowding Park r1 share empty"] == 0
    assert x["crowding Commercial r3 share empty"] == 1  # the shop is 12 steps away
    assert x["worst home->Commercial"] == 12
    assert x["Park per home"] == 1
    assert x["cluster access Park average"] == 1
    assert x["commute homes per job"] == pytest.approx(1 / 47)  # 46 offices + 1 shop
    assert x["commute average trip"] == 1 and x["commute busiest job"] == 1
    assert x["commute idle jobs"] == pytest.approx(46 / 47)
    assert x["ratio Office/Residential"] == 23  # 46 / (1 + 1)
    assert x["top Residential"] == 1 and x["bottom Residential"] == 0
    assert x["ring 3 Residential"] == 1 and x["ring 0 Office"] == 1
    # A corner home has 3 neighbours, one of them the park.
    assert x["home neighbours Park"] == pytest.approx(1 / 3)
    assert x["parks of 3+ cells"] == 0 and x["largest park per home"] == 1


def test_a_grid_with_no_homes_and_an_empty_batch():
    g = np.full((1, 7, 7), fl.PARK, np.int8)
    assert np.isfinite(fl.lab(g)).all()
    assert fl.lab(np.zeros((0, 7, 7), np.int8)).shape == (0, 148)
