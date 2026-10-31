"""Stage 2's harness (search.py): nested cross-validation can't quietly leak or drift.

- No procedure trains on, or searches with, a grid it then predicts: a spy procedure records
  every grid it is given in every outer round, and each round gets a fresh copy of it.
- Inner folds split each outer training set in four, the same way for every procedure.
- Stage 1's GBM run through the harness gives exactly Stage 1's predictions, one round at a
  time or in parallel processes (so one-thread fits change nothing).
- Outer predictions need a fully committed repository and the committed definition, and
  record the commit. Saved results are never replaced; a procedure changed after its outer
  results (its settings, code or inputs, for any advisor) needs a new name; every name with
  results counts toward the cap, each in its own slot.
- A round must return finite predictions and an inner score; finished advisors stay saved
  when a later round fails.
- The registry respects the cap and the owner's six slots.
- hp_recall, the corrected t, the adoption test and rule 2.2 follow their definitions.
"""

import dataclasses
import json
import re

import numpy as np
import pytest
import scipy.stats
from test_protocol import _canonical

import features
import search
import validity


def _isolate(tmp_path, monkeypatch):
    """Save everything under tmp_path, with the repository reported fully committed and the
    procedures in memory taken as the committed ones."""
    monkeypatch.setattr(search, "NESTED_DIR", str(tmp_path / "nested"))
    monkeypatch.setattr(validity, "CACHE_DIR", str(tmp_path / "oof"))
    monkeypatch.setattr(validity, "_git_state", lambda: ("c" * 40, []))
    monkeypatch.setattr(
        search,
        "_definition_on_disk",
        lambda name, advisor: search._definition(search._lookup(name, advisor)),
    )


def _toy_data(n=60, seed=7):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, 5, size=(n, 7, 7)).astype(np.int8),
        rng.random(n),
        np.arange(n),
    )


@dataclasses.dataclass(frozen=True)
class Toy:
    """A procedure that predicts its training ratings' mean, times `scale`."""

    scale: float = 1.0
    slot: int = 3
    summary: str = "toy"
    advisors: tuple = validity.ADVISORS

    def spec(self):
        return {"kind": "toy", "scale": self.scale}

    def model_inputs(self, grids):
        return features.onehot(grids)

    def search(self, grids, y, inner):
        return None, {"inner_hp_recall": 0.0}

    def fit(self, choice, grids, y):
        return float(np.mean(y)) * self.scale

    def predict(self, model, grids):
        return np.full(len(grids), model)


def test_no_procedure_sees_the_grids_it_predicts(tmp_path, monkeypatch):
    """A spy procedure records every grid it is given to learn from and refuses to predict
    any of them. Its search fits on each inner fold, as a real search does. Its memory lives
    in the instance, so a copy reused across rounds would be caught too. Grids are compared
    in canonical form, so a rotated or mirrored copy counts as well."""
    _isolate(tmp_path, monkeypatch)
    rng = np.random.default_rng(5)
    grids = rng.integers(0, 5, size=(100, 7, 7)).astype(np.int8)
    y, rows = rng.random(100), np.arange(100)
    rounds = []

    class Spy:
        slot, summary, advisors = 3, "spy", validity.ADVISORS

        def __init__(self):
            self.learned = set()  # every grid this instance was given to learn from

        def spec(self):
            return {"kind": "spy"}

        def model_inputs(self, g):
            return features.onehot(g)

        def search(self, g, y_train, inner):
            assert len(g) == len(y_train)
            self.learned |= _canonical(g)
            rounds.append(
                {"given": _canonical(g), "trained": set(), "predicted": set()}
            )

            def fit_predict(g_fit, y_fit, g_new):
                assert not _canonical(g_fit) & _canonical(g_new), "inner fold leaked"
                rounds[-1]["trained"] |= _canonical(g_fit)
                return np.zeros(len(g_new))

            search.inner_predictions(fit_predict, g, y_train, inner)
            return None, {"inner_hp_recall": 0.0}

        def fit(self, choice, g, y_train):
            rounds[-1]["trained"] |= _canonical(g)

        def predict(self, model, g):
            assert not _canonical(g) & self.learned, "predicted a grid it learned from"
            rounds[-1]["predicted"] |= _canonical(g)
            return np.zeros(len(g))

    monkeypatch.setitem(search.REGISTRY, "spy", Spy())
    preds = search.outer_predictions("spy", "Wellness", grids, y, rows, workers=1)
    assert preds.shape == (validity.CV_REPEATS, 100) and not np.isnan(preds).any()
    splits = list(validity.cv_splits(100))
    assert len(rounds) == len(splits)  # one search per outer round
    for record, (_, _, train, val) in zip(rounds, splits):
        # Given exactly the outer training set, trained on all of it and nothing else, and
        # asked to predict exactly the outer-validation fold.
        assert record["given"] == record["trained"] == _canonical(grids[train])
        assert record["predicted"] == _canonical(grids[val])


def test_inner_folds_split_each_outer_training_set_the_same_way_for_every_procedure():
    first_folds = set()
    for r, f, train, _ in validity.cv_splits(4000):
        inner = search.inner_splits(len(train), r, f)
        assert len(inner) == search.INNER_FOLDS
        validated = np.sort(np.concatenate([val for _, val in inner]))
        assert np.array_equal(
            validated, np.arange(len(train))
        )  # each grid validated once
        for fit, val in inner:
            assert len(fit) == 2400 and len(val) == 800 and not set(fit) & set(val)
        again = search.inner_splits(
            len(train), r, f
        )  # as a second procedure would get them
        assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(inner, again))
        first_folds.add(inner[0][1].tobytes())
    assert len(first_folds) == search.ROUNDS  # each outer round has its own inner folds


def test_the_harness_reproduces_stage1_one_round_at_a_time_or_in_parallel(
    grid_zoo, tmp_path, monkeypatch
):
    """Stage 1's GBM through the harness (one-thread fits) gives exactly what
    validity.out_of_fold gives (LightGBM's default threads)."""
    _isolate(tmp_path, monkeypatch)
    y = np.random.default_rng(6).random(len(grid_zoo))
    rows = np.arange(len(grid_zoo))
    stage1 = validity.out_of_fold("gbm_engineered", grid_zoo, y, rows)
    one_at_a_time = search.outer_predictions(
        "gbm_engineered", "Wellness", grid_zoo, y, rows, workers=1
    )
    assert np.array_equal(one_at_a_time, stage1)
    in_parallel = search.outer_predictions(
        "gbm_engineered", "Wellness", grid_zoo, y, rows, workers=2, use_cache=False
    )
    assert np.array_equal(in_parallel, stage1)


def test_outer_predictions_need_everything_committed_and_record_the_commit(
    tmp_path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    grids, y, rows = _toy_data()
    monkeypatch.setitem(search.REGISTRY, "toy", Toy())
    monkeypatch.setattr(validity, "_git_state", lambda: ("a" * 40, ["search.py"]))
    with pytest.raises(PermissionError, match="committed"):
        search.outer_predictions("toy", "Wellness", grids, y, rows, workers=1)
    assert not (
        tmp_path / "nested"
    ).exists()  # refused before computing or saving anything
    monkeypatch.setattr(validity, "_git_state", lambda: ("b" * 40, []))
    monkeypatch.setattr(search, "_definition_on_disk", lambda name, advisor: {})
    with pytest.raises(PermissionError, match="differs from the committed"):
        search.outer_predictions("toy", "Wellness", grids, y, rows, workers=1)
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(validity, "_git_state", lambda: ("b" * 40, []))
    with pytest.raises(ValueError, match="at least 1"):
        search.outer_predictions("toy", "Wellness", grids, y, rows, workers=0)
    search.outer_predictions("toy", "Wellness", grids, y, rows, workers=1)
    (record,) = (tmp_path / "nested").glob("toy-wellness-*.json")
    assert json.loads(record.read_text())["commit"] == "b" * 40
    # Saved predictions can be read back with uncommitted changes: nothing new is made.
    monkeypatch.setattr(validity, "_git_state", lambda: ("a" * 40, ["notes.md"]))
    search.outer_predictions("toy", "Wellness", grids, y, rows, workers=1)


def test_saved_results_are_never_replaced_and_a_changed_procedure_needs_a_new_name(
    tmp_path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    grids, y, rows = _toy_data()
    monkeypatch.setitem(search.REGISTRY, "toy", Toy())
    saved = search.outer_predictions("toy", "Wellness", grids, y, rows, workers=1)
    (record,) = (tmp_path / "nested").glob("toy-wellness-*.json")
    before = record.read_bytes()
    again = search.outer_predictions(
        "toy", "Wellness", grids, y, rows, workers=1, use_cache=False
    )
    assert (
        np.array_equal(again, saved) and record.read_bytes() == before
    )  # a check only
    # Changed settings, for this advisor or another, are refused under the same name...
    monkeypatch.setitem(search.REGISTRY, "toy", Toy(scale=0.5))
    for advisor in ("Wellness", "Tax"):
        with pytest.raises(ValueError, match="new name"):
            search.outer_predictions("toy", advisor, grids, y, rows, workers=1)
    # ...and so are changed model inputs on the same data (here, a feature edited).
    monkeypatch.setitem(search.REGISTRY, "toy", Toy())
    with monkeypatch.context() as m:
        m.setattr(features, "onehot", lambda g: np.ones((len(g), 3), np.float32))
        with pytest.raises(ValueError, match="new name"):
            search.outer_predictions("toy", "Wellness", grids, y, rows, workers=1)
    # A new name in a free slot runs; the same definition on new data is simply computed.
    monkeypatch.setitem(search.REGISTRY, "toy_v2", Toy(scale=0.5, slot=4))
    search.outer_predictions("toy_v2", "Wellness", grids, y, rows, workers=1)
    search.outer_predictions("toy", "Wellness", grids, y[::-1].copy(), rows, workers=1)


def test_every_name_with_outer_results_counts_toward_the_cap_in_its_own_slot(
    tmp_path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    grids, y, rows = _toy_data()
    for slot in range(1, search.MAX_PROCEDURES + 1):
        monkeypatch.setitem(search.REGISTRY, f"toy{slot}", Toy(scale=slot, slot=slot))
        search.outer_predictions(f"toy{slot}", "Wellness", grids, y, rows, workers=1)
    monkeypatch.setitem(search.REGISTRY, "renamed", Toy(scale=9, slot=3))
    with pytest.raises(ValueError, match="slot 3 already has outer results"):
        search.outer_predictions("renamed", "Wellness", grids, y, rows, workers=1)
    monkeypatch.setitem(search.REGISTRY, "seventh", Toy(scale=7, slot=7))
    with pytest.raises(ValueError, match="more than 6"):
        search.outer_predictions("seventh", "Wellness", grids, y, rows, workers=1)
    search.outer_predictions(
        "seventh", "Tax", grids, y, rows, workers=1
    )  # another advisor


def test_a_round_must_return_finite_predictions_and_an_inner_score(
    tmp_path, monkeypatch
):
    _isolate(tmp_path, monkeypatch)
    grids, y, rows = _toy_data()

    @dataclasses.dataclass(frozen=True)
    class Broken(Toy):
        record: tuple = ()
        value: float = 0.5

        def search(self, grids, y, inner):
            return None, dict(self.record)

        def predict(self, model, grids):
            return np.full(len(grids), self.value)

    cases = {
        "no inner score": Broken(record=()),
        "a harness key": Broken(record=(("inner_hp_recall", 0.0), ("seconds", 1))),
        "NaN predictions": Broken(record=(("inner_hp_recall", 0.0),), value=np.nan),
    }
    for problem, procedure in cases.items():
        monkeypatch.setitem(search.REGISTRY, "broken", procedure)
        with pytest.raises(ValueError):
            search.outer_predictions("broken", "Wellness", grids, y, rows, workers=1)
        assert not (tmp_path / "nested").exists(), problem
    numpy_values = (("inner_hp_recall", np.float64(0.5)), ("trees", np.int64(3)))
    monkeypatch.setitem(search.REGISTRY, "broken", Broken(record=numpy_values))
    search.outer_predictions(
        "broken", "Wellness", grids, y, rows, workers=1
    )  # saves fine


def test_finished_advisors_stay_saved_when_a_later_round_fails(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)

    @dataclasses.dataclass(frozen=True)
    class FailsOnSmallData(Toy):
        def fit(self, choice, grids, y):
            if len(y) < 40:
                raise RuntimeError("out of memory")
            return super().fit(choice, grids, y)

    monkeypatch.setitem(search.REGISTRY, "fragile", FailsOnSmallData())
    data = {"Wellness": _toy_data(60), "Tax": _toy_data(45, seed=8)}  # Tax trains on 36
    with pytest.raises(RuntimeError, match="out of memory"):
        search._outer("fragile", data, workers=1, use_cache=True, run=True)
    assert len(list((tmp_path / "nested").glob("fragile-wellness-*.npy"))) == 1
    assert not list((tmp_path / "nested").glob("fragile-tax-*"))


def test_only_registered_procedures_run():
    grids, y, rows = _toy_data()
    with pytest.raises(KeyError, match="not registered"):
        search.outer_predictions("made_up", "Wellness", grids, y, rows, workers=1)


def test_the_registry_respects_the_cap_and_the_owner_s_slots():
    for advisor in validity.ADVISORS:
        procedures = search.procedures_for(advisor)
        assert len(procedures) <= search.MAX_PROCEDURES
        slots = [p.slot for p in procedures.values()]
        assert len(set(slots)) == len(slots) and set(slots) <= set(search.SLOTS)
        assert search.INCUMBENT[advisor] in procedures
    assert len(search.SLOTS) == search.MAX_PROCEDURES
    assert all(re.fullmatch(r"[a-z0-9_]+", name) for name in search.REGISTRY)


def _slow_hp_recall(pred, y):
    """hp_recall written out plainly: for each level, the most passing grids approvable."""
    order = np.argsort(-pred, kind="stable")
    passes = y[order] >= validity.THRESHOLD
    recalls = []
    for level in search.HP_LEVELS:
        best = 0
        for k in range(1, len(y) + 1):
            if passes[:k].mean() >= level:
                best = passes[:k].sum()
        recalls.append(best / passes.sum())
    return np.mean(recalls)


def test_hp_recall_follows_its_definition():
    # Ranked by prediction: pass, pass, pass, fail, pass, pass, fail, fail, pass, fail. Precision
    # stays at 96% or more only through the first three, which are 3 of the 6 passing grids.
    passes = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0], bool)
    assert search.hp_recall(np.linspace(1, 0, 10), np.where(passes, 0.9, 0.1)) == 0.5
    rng = np.random.default_rng(3)
    for _ in range(20):  # no tied predictions here, so the two definitions must agree
        y = rng.random(300)
        pred = y + rng.normal(0, 0.1, 300)
        assert np.isclose(search.hp_recall(pred, y), _slow_hp_recall(pred, y))


def test_corrected_t_widens_the_ordinary_t_for_overlapping_training_sets():
    d = np.random.default_rng(4).normal(0.01, 0.02, 15)
    ordinary = d.mean() / (d.std(ddof=1) / np.sqrt(15))
    # 800 validation and 3,200 training grids per fold: (1/15 + 0.25) / (1/15) = 4.75.
    assert np.isclose(search.corrected_t(d, 3200, 800), ordinary / np.sqrt(4.75))
    assert np.isclose(search.T_CRITICAL, scipy.stats.t.ppf(0.95, 14), atol=5e-4)


def test_the_adoption_test_needs_both_conditions():
    rng = np.random.default_rng(8)
    y = rng.random(4000)
    noisy = np.stack([y + rng.normal(0, 0.1, 4000) for _ in range(validity.CV_REPEATS)])
    same = search.compare_predictions(noisy, noisy, y, draws=50)
    assert same.t == 0 and same.gain == 0 and not same.adopt
    assert same.hp_recall_interval == same.approved_interval == (0.0, 0.0)
    sharper = np.stack(
        [y + rng.normal(0, 0.03, 4000) for _ in range(validity.CV_REPEATS)]
    )
    better = search.compare_predictions(sharper, noisy, y, draws=50)
    assert better.passes_test and better.passes_margin and better.adopt
    assert better.folds_better == search.ROUNDS
    assert better.hp_recall_interval[0] > 0 and better.approved_interval[0] > 0
    worse = search.compare_predictions(noisy, sharper, y, draws=50)
    assert not worse.passes_test and not worse.passes_margin and not worse.adopt


def test_reorient_turns_each_grid_the_same_way_in_every_fold(grid_zoo):
    turned = search.reorient(grid_zoo, seed=1)
    for grid, t in zip(grid_zoo, turned):
        others = features.orientations(grid)[
            1:
        ]  # its 7 orientations other than "as is"
        assert any(np.array_equal(t, o) for o in others)
    # A grid is turned the same way whichever batch (fold) it arrives in.
    every_third = np.arange(0, len(grid_zoo), 3)
    assert np.array_equal(
        search.reorient(grid_zoo[every_third], 1), turned[every_third]
    )
    assert not np.array_equal(search.reorient(grid_zoo, 2), turned)  # each draw differs


def test_rule_2_2_compares_errors_fold_by_fold():
    rng = np.random.default_rng(9)
    y = rng.random(4000)
    true = np.stack([y + rng.normal(0, 0.05, 4000) for _ in range(validity.CV_REPEATS)])
    controls = [
        np.stack([y + rng.normal(0, 0.1, 4000) for _ in range(validity.CV_REPEATS)])
        for _ in search.REORIENT_SEEDS
    ]
    t, better = search.orientation_effect(true, controls, y)
    assert better == search.ROUNDS and t > search.T_CRITICAL
    assert search.orientation_effect(controls[0], [controls[0]] * 5, y) == (0.0, 0)


def test_the_label_free_check_finds_position_patterns_and_only_those():
    base = np.random.default_rng(10).integers(0, 5, size=(2000, 7, 7)).astype(np.int8)
    every_orientation = features.orientations(base).reshape(-1, 7, 7)
    chi2, _, p, gap = search.orientation_patterns(every_orientation)
    assert chi2 < 1e-6 and p > 0.999 and gap == 0
    corner_park = base.copy()
    corner_park[:, 0, 0] = 3  # a park in the same corner of every grid
    _, _, p, gap = search.orientation_patterns(corner_park)
    assert p < 1e-6 and gap > 0.5


def test_diagnostics_are_outside_the_cap_and_never_adopted():
    for advisor in validity.ADVISORS:
        assert not set(search.DIAGNOSTICS) & set(search.procedures_for(advisor))
    with pytest.raises(ValueError, match="diagnostic"):
        search.compare("rule22_reoriented_1")
