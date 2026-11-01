"""generate.py: the hot-and-cold climb and the near-copy gate, on a fake rating with a known
answer, so no real models or data are needed.

The fake: every advisor's prediction is the share of the 49 cells that are parks. With every
cut-off at 0.5 and a margin of 0.03, the bar is 0.53, so a grid passes at 26 parks (26/49 is
0.531; 25/49 is 0.510). Each step of the climb can add one park.
"""

import json

import numpy as np
import pytest

import generate
import surrogates
import validity
from validity import ADVISORS

PARK = 3
CUTS = dict.fromkeys(ADVISORS, 0.5)
MARGIN = 0.03
# run()'s fake dataset: 50 random grids (about 10 parks each), the first all parks.
GRIDS = np.random.default_rng(1).integers(0, 5, (50, 7, 7)).astype(np.int8)
GRIDS[0] = PARK
GRIDS.flags.writeable = False  # like the real data: the climb must not edit it


def parks_rating(models, grids):
    """Every advisor rates a grid by its share of park cells; the three models agree."""
    share = (np.asarray(grids) == PARK).mean((1, 2))
    pred = np.repeat(share[:, None], len(ADVISORS), axis=1)
    return pred, np.zeros_like(pred)


@pytest.fixture
def parks(monkeypatch):
    monkeypatch.setattr(surrogates, "rate", parks_rating)


@pytest.fixture
def fake_run(parks, monkeypatch, tmp_path):
    """run() on GRIDS, with no real git state or models, saving to tmp_path."""
    monkeypatch.setattr(validity, "_git_state", lambda: ("abc123", []))
    monkeypatch.setattr(validity, "load_dev", lambda: (GRIDS, None))
    monkeypatch.setattr(surrogates, "fit_models", lambda: None)
    monkeypatch.setattr(surrogates, "check_refits", lambda models: None)
    monkeypatch.setattr(surrogates, "cutoffs", lambda: CUTS)
    monkeypatch.setattr(generate, "DESIGNS_DIR", str(tmp_path))


def test_one_cell_changes_are_the_196_single_cell_edits():
    grid = np.random.default_rng(0).integers(0, 5, (7, 7)).astype(np.int8)
    variants = generate.one_cell_changes(grid)
    assert variants.shape == (196, 7, 7) and variants.dtype == grid.dtype
    assert ((variants != grid).sum((1, 2)) == 1).all()
    assert len(np.unique(variants.reshape(196, -1), axis=0)) == 196


def test_a_kick_changes_exactly_three_cells_of_a_copy():
    grid = np.zeros((7, 7), np.int8)
    kicked = generate.kick(grid, np.random.default_rng(0))
    assert (kicked != grid).sum() == 3 and not grid.any()


def test_the_climb_adds_parks_until_it_clears_the_bar(parks):
    start = np.zeros((7, 7), np.int8)  # no parks yet
    grid, info = generate.design(start, None, CUTS, MARGIN, np.random.default_rng(0))
    assert (grid == PARK).sum() == 26
    assert info["steps"] == 26 and info["kicks"] == 0 and info["shortfall"] == 0
    assert (info["pred"] >= 0.5 + MARGIN).all()
    # Every new park ties: drawn at random, not cells 0-25 (the top-left, first in order).
    assert not (grid.flat[:26] == PARK).all()


def test_when_no_change_helps_it_kicks_until_max_steps(monkeypatch):
    def flat_rating(models, grids):  # every grid rated 0: no change is ever warmer
        return np.zeros((len(grids), 4)), np.zeros((len(grids), 4))

    monkeypatch.setattr(surrogates, "rate", flat_rating)
    start, rng = np.zeros((7, 7), np.int8), np.random.default_rng(0)
    grid, info = generate.design(start, None, CUTS, MARGIN, rng, max_steps=5)
    assert info["steps"] == info["kicks"] == 5 and info["shortfall"] > 0
    assert (grid != start).any()


def test_the_near_copy_gate_compares_with_pooled_designs_only():
    a = np.zeros((7, 7), np.int8)
    b, c = a.copy(), a.copy()
    b.flat[:9] = 1  # 9 cells from a: a near-copy, dropped
    c.flat[:10] = 1  # 10 cells from a: kept, though only 1 from the dropped b
    assert generate.near_copy_gate(np.stack([a, b, c])) == [0, 2]


def test_run_saves_the_pool_of_passing_designs(fake_run, tmp_path):
    # 16 moves: a start with fewer than 10 parks runs out before reaching 26.
    pool = generate.run(6, MARGIN, workers=1, seed=0, max_steps=16)
    saved = np.load(tmp_path / "pool_0.npz")
    summary = json.loads((tmp_path / "pool_0.json").read_text())["summary"]
    assert 0 < summary["succeeded"] < summary["tried"] == 6
    pooled = summary["succeeded"] - summary["near_copies"]
    assert summary["pooled"] == len(saved["grids"]) == pooled
    assert np.array_equal(saved["grids"], pool["grids"])
    assert saved["pred"].shape == (pooled, 4)
    # Only designs that cleared the bar are pooled: one step per park added to the start.
    assert ((saved["grids"] == PARK).sum((1, 2)) == 26).all()
    start_parks = (GRIDS[saved["start"]] == PARK).sum((1, 2))
    assert np.array_equal(saved["steps"], 26 - start_parks)


def test_an_empty_pool_is_saved_and_never_overwritten(fake_run, tmp_path):
    # 2 moves can't reach 26 parks. Only the all-parks start passes, and it isn't a design.
    generate.run(50, MARGIN, workers=1, seed=0, max_steps=2)
    summary = json.loads((tmp_path / "pool_0.json").read_text())["summary"]
    assert [summary[k] for k in ("already_passing", "succeeded", "pooled")] == [1, 0, 0]
    assert np.load(tmp_path / "pool_0.npz")["grids"].shape == (0, 7, 7)
    with pytest.raises(FileExistsError):  # same seed, new margin: refused, not replaced
        generate.run(50, MARGIN + 0.01, workers=1, seed=0, max_steps=2)


def test_run_refuses_an_uncommitted_tree(monkeypatch):
    monkeypatch.setattr(validity, "_git_state", lambda: ("abc123", ["generate.py"]))
    with pytest.raises(PermissionError):
        generate.run(1)
