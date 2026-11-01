"""surrogates.py: the refits are the saved outer models, and rate() summarizes them correctly.

- fit_models() refits round (shuffle s, fold 0) on its training grids with its saved choice;
  check_refits() passes for those models and stops when a saved prediction differs.
- cutoffs() is the mean over the three shuffles of the cut-off at 98% precision.
- rate() gives each advisor's mean and standard deviation over its three models, in ADVISORS
  order, the same across chunk boundaries and when the chunks go to parallel processes.
- The CLI saves float32 ratings and a record that load_ratings() reads back, unless stale.
"""

import numpy as np
import pytest

import search
import surrogates
import validity
from validity import ADVISORS


class Toy:
    """A procedure whose model is its choice plus the mean training rating."""

    def fit(self, choice, grids, y):
        return choice + float(np.mean(y))

    def predict(self, model, grids):
        return np.full(len(grids), model)


class Scaled:
    """A procedure that rates a grid as its total of district codes times the model."""

    def predict(self, model, grids):
        return grids.reshape(len(grids), -1).sum(1) * model


def _saved_run(monkeypatch, n=40):
    """A Toy incumbent for every advisor, with outer results saved as search.py saves them."""
    rng = np.random.default_rng(0)
    grids, y = rng.integers(0, 5, (n, 7, 7)).astype(np.int8), rng.random(n)
    preds, rounds = np.full((3, n), np.nan), []
    for s, f, train, val in validity.cv_splits(n):
        choice = 10 * s + f  # every round makes its own choice
        preds[s, val] = Toy().fit(choice, grids[train], y[train])
        rounds.append({"shuffle": s, "fold": f, "choice": choice})
    saved = (Toy(), grids, y, preds, {"rounds": rounds})
    monkeypatch.setattr(surrogates, "_incumbent", lambda advisor: saved)
    return grids, y, preds


def test_refits_are_the_saved_fold_zero_models(monkeypatch):
    _, y, preds = _saved_run(monkeypatch)
    models = surrogates.fit_models(["Tax"])
    fold_zero = [(s, tr, va) for s, f, tr, va in validity.cv_splits(len(y)) if f == 0]
    refits = [model for _, model in models["Tax"]]
    assert refits == [10 * s + y[tr].mean() for s, tr, _ in fold_zero]
    surrogates.check_refits(models)
    preds[1, fold_zero[1][2][0]] += 1e-9  # one saved prediction of round (1, 0) changes
    with pytest.raises(AssertionError, match="shuffle 1"):
        surrogates.check_refits(models)


def test_cutoff_is_the_mean_over_shuffles(monkeypatch):
    rng = np.random.default_rng(1)
    y = rng.random(400)
    preds = y + rng.normal(0, 0.05, (3, 400))
    saved = (None, None, y, preds, None)  # cutoffs() reads only y and preds
    monkeypatch.setattr(surrogates, "_incumbent", lambda advisor: saved)
    each = [validity.approved_at(p, y)[1] for p in preds]
    assert np.isfinite(each).all()
    assert surrogates.cutoffs(["Tax"]) == {"Tax": np.mean(each)}


def test_rate_gives_each_advisors_mean_and_spread(monkeypatch):
    monkeypatch.setattr(surrogates, "CHUNK", 7)  # five chunks, the last one short
    grids = np.random.default_rng(2).integers(0, 5, (30, 7, 7))
    models = {
        "Business": [(Scaled(), k) for k in (1.0, 2.0, 6.0)],
        "Wellness": [(Scaled(), 1.0)] * 3,
    }
    pred, spread = surrogates.rate(models, grids)
    total = grids.reshape(30, -1).sum(1)
    assert pred.shape == spread.shape == (30, 4)
    np.testing.assert_allclose(pred[:, 3], 3 * total)
    np.testing.assert_allclose(spread[:, 3], np.std([1, 2, 6]) * total)
    np.testing.assert_allclose(pred[:, 0], total)
    assert (spread[:, 0] == 0).all() and np.isnan(pred[:, 1:3]).all()
    # The five chunks rated in two processes give the same numbers.
    parallel = surrogates.rate(models, grids, workers=2)
    np.testing.assert_array_equal(parallel[0], pred)
    np.testing.assert_array_equal(parallel[1], spread)


def test_cli_saves_ratings_that_load_back(tmp_path, monkeypatch):
    grids, _, _ = _saved_run(monkeypatch)
    monkeypatch.setattr(surrogates, "RATINGS_DIR", str(tmp_path))
    monkeypatch.setattr(validity, "load_dev", lambda: (grids, None))  # rates only grids
    monkeypatch.setattr(validity, "_git_state", lambda: ("c" * 40, []))
    surrogates.main()
    pred, spread, info = surrogates.load_ratings()
    assert pred.dtype == spread.dtype == np.float32 and pred.shape == (len(grids), 4)
    assert (spread > 0).all()  # the three fold-zero models differ
    assert info["commit"] == "c" * 40 and not info["uncommitted_changes"]
    assert info["incumbents"] == {a: search.INCUMBENT[a] for a in ADVISORS}
    assert set(info["approved"]) == {*ADVISORS, "all four"}
    monkeypatch.setitem(search.INCUMBENT, "Wellness", "cnn")  # a new incumbent since
    with pytest.raises(SystemExit, match="stale"):
        surrogates.load_ratings()
