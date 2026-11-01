"""Stage 2's harness (search.py): nested cross-validation can't quietly leak or drift.

- No recipe searches with, or trains on, a grid it then predicts, and each round gets a
  fresh copy of the recipe (the spy test). Inner folds are the same for every recipe.
- Stage 1's GBM through the harness gives exactly Stage 1's predictions, serial or parallel.
- Outer predictions need a clean, committed tree; saved results are loaded, never
  replaced; a changed recipe, data or folds needs a new name; old files migrate.
- hp_recall, the corrected t, the adoption decision, its bootstrap and rule 2.2 follow
  their definitions.
"""

import dataclasses
import json

import numpy as np
import pytest
import scipy.stats
from test_protocol import _canonical

import features
import migrate_results
import search
import validity


def _isolate(tmp_path, monkeypatch):
    """Save under tmp_path, with the repository reported clean at commit ccc..."""
    monkeypatch.setattr(search, "NESTED_DIR", str(tmp_path / "nested"))
    monkeypatch.setattr(validity, "CACHE_DIR", str(tmp_path / "oof"))
    monkeypatch.setattr(validity, "_git_state", lambda: ("c" * 40, []))


def _toy_data(n=60, seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 5, size=(n, 7, 7)).astype(np.int8), rng.random(n)


def _run(name, grids, y, advisor="Wellness", **options):
    """One advisor's outer predictions, one round at a time."""
    data = {advisor: (grids, y)}
    return search.outer_predictions(name, data, workers=1, **options)[advisor]


@dataclasses.dataclass(frozen=True)
class Toy:
    """A recipe that predicts its training ratings' mean, times `scale`."""

    scale: float = 1.0
    slot: int = 3
    summary: str = "toy"
    advisors: tuple = validity.ADVISORS

    def spec(self):
        return {"kind": "toy", "scale": self.scale}

    def search(self, grids, y, inner):
        return None, {"inner_hp_recall": 0.0}

    def fit(self, choice, grids, y):
        return float(np.mean(y)) * self.scale

    def predict(self, model, grids):
        return np.full(len(grids), model)


def test_no_recipe_sees_the_grids_it_predicts(tmp_path, monkeypatch):
    """A spy recipe records every grid it learns from, in its inner fits and its refit,
    and refuses to predict any of them. Its memory lives in the instance, so a copy reused
    across rounds is caught. Grids are compared in canonical form (rotations count)."""
    _isolate(tmp_path, monkeypatch)
    grids, y = _toy_data(100, seed=5)
    rounds = []

    class Spy(Toy):
        def __init__(self):
            object.__setattr__(self, "learned", set())  # this copy's memory

        def search(self, g, y_train, inner):
            assert not self.learned, "a round got another round's copy of the recipe"
            self.learned.update(_canonical(g))
            rounds.append({"given": _canonical(g), "trained": set()})

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
            rounds[-1]["predicted"] = _canonical(g)
            return np.zeros(len(g))

    monkeypatch.setitem(search.REGISTRY, "spy", Spy())
    assert _run("spy", grids, y).shape == (validity.CV_REPEATS, 100)
    splits = list(validity.cv_splits(100))
    assert len(rounds) == len(splits)  # one search per outer round
    for record, (_, _, train, val) in zip(rounds, splits):
        assert record["given"] == record["trained"] == _canonical(grids[train])
        assert record["predicted"] == _canonical(grids[val])


def test_every_recipe_gets_the_same_inner_folds():
    first_folds = set()
    for r, f, train, _ in validity.cv_splits(4000):
        inner = search.inner_splits(len(train), r, f)
        validated = np.sort(np.concatenate([val for _, val in inner]))
        assert np.array_equal(validated, np.arange(len(train)))  # each grid once
        for fit, val in inner:
            assert len(fit) == 2400 and len(val) == 800 and not set(fit) & set(val)
        again = search.inner_splits(len(train), r, f)  # as a second recipe gets them
        assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(inner, again))
        first_folds.add(inner[0][1].tobytes())
    assert len(first_folds) == search.ROUNDS  # each outer round has its own inner folds


def test_serial_and_parallel_runs_equal_stage1(grid_zoo, tmp_path, monkeypatch):
    """One-thread fits, serial or in 2 processes, equal validity.out_of_fold exactly."""
    _isolate(tmp_path, monkeypatch)
    y = np.random.default_rng(6).random(len(grid_zoo))
    stage1 = validity.out_of_fold("gbm_engineered", grid_zoo, y, np.arange(len(y)))
    data = {"Tax": (grid_zoo, y)}
    for workers in (1, 2):  # use_cache=False: each computes, neither saves
        preds = search.outer_predictions("gbm_engineered", data, workers, False)
        assert np.array_equal(preds["Tax"], stage1)


def test_runs_need_a_clean_tree_and_record_the_commit(tmp_path, monkeypatch, capsys):
    _isolate(tmp_path, monkeypatch)
    grids, y = _toy_data()
    monkeypatch.setitem(search.REGISTRY, "toy", Toy())
    monkeypatch.setattr(validity, "_git_state", lambda: ("a" * 40, ["search.py"]))
    with pytest.raises(PermissionError, match="committed"):
        _run("toy", grids, y)
    assert not (tmp_path / "nested").exists()  # refused before computing anything
    states = iter([("a" * 40, []), ("b" * 40, [])])  # HEAD moves during the run
    monkeypatch.setattr(validity, "_git_state", lambda: next(states))
    preds = _run("toy", grids, y)
    assert "toy: starting 15 outer rounds (Wellness)" in capsys.readouterr().out
    loaded, record = search.load_outer("toy", "Wellness")
    assert np.array_equal(loaded, preds) and record["commit"] == "a" * 40
    assert record["unchanged_during_run"] is False  # saved, and flagged on loading:
    assert "commit, aaaaaaa, may not hold the code" in capsys.readouterr().out
    assert record["spec"] == Toy().spec() and record["context"] == search.run_context()
    in_order = [(r, f) for r, f, _, _ in validity.cv_splits(60)]
    assert [(r["shuffle"], r["fold"]) for r in record["rounds"]] == in_order
    monkeypatch.setattr(validity, "_git_state", lambda: ("d" * 40, ["notes.md"]))
    assert np.array_equal(_run("toy", grids, y), preds)  # loading needs no clean tree
    with pytest.raises(FileNotFoundError, match="--run toy"):
        search.load_outer("toy", "Tax")
    with pytest.raises(KeyError, match="not registered"):
        _run("made_up", grids, y)


def test_results_are_never_replaced_a_change_needs_a_new_name(
    tmp_path, monkeypatch, capsys
):
    _isolate(tmp_path, monkeypatch)
    grids, y = _toy_data()
    monkeypatch.setitem(search.REGISTRY, "toy", Toy())
    saved = _run("toy", grids, y)
    files = sorted((tmp_path / "nested").iterdir())
    before = [file.read_bytes() for file in files]
    again = _run("toy", grids, y, use_cache=False)  # recomputed as a check, not saved
    monkeypatch.setattr(search, "_execute", None)  # from here on, nothing may compute
    assert np.array_equal(again, saved) and np.array_equal(_run("toy", grids, y), saved)
    with pytest.raises(FileExistsError):
        search._save("toy", "Wellness", saved * 0, {})
    assert [file.read_bytes() for file in files] == before
    np.save(tmp_path / "nested" / "toy-tax.npy", saved)  # a save cut short: no record
    with pytest.raises(FileExistsError, match="no record"):
        _run("toy", grids, y, advisor="Tax")  # refused before any round runs
    with pytest.raises(ValueError, match="saved for 60 grids"):
        _run("toy", *_toy_data(100))  # other data under the same name
    monkeypatch.setitem(search.REGISTRY, "toy", Toy(scale=0.5))
    with pytest.raises(ValueError, match=r"\(spec\): give it a new name"):
        _run("toy", grids, y)
    with pytest.raises(ValueError, match="give it a new name"):  # for every advisor
        _run("toy", grids, y, advisor="Business")
    monkeypatch.setitem(search.REGISTRY, "toy", Toy())
    with monkeypatch.context() as m:
        m.setattr(validity, "course_pin", lambda: "another course version")
        with pytest.raises(ValueError, match=r"\(context\)"):
            search.load_outer("toy", "Wellness")
    monkeypatch.setattr(search, "_versions", lambda procedure: {"numpy": "0.1"})
    search.load_outer("toy", "Wellness")  # library versions only warn
    assert "warning: toy for Wellness ran with" in capsys.readouterr().out


def test_a_round_needs_finite_predictions_and_an_inner_score():
    grids, y = _toy_data()
    args = (grids, y, 0, 0, np.arange(45), np.arange(45, 60))

    class Broken(Toy):
        def __init__(self, record, value=0.5):
            object.__setattr__(self, "out", (record, value))

        def search(self, grids, y, inner):
            return None, self.out[0]

        def predict(self, model, grids):
            return np.full(len(grids), self.out[1])

    score = {"inner_hp_recall": 0.0}
    for broken in (Broken({}), Broken(score | {"seconds": 1}), Broken(score, np.nan)):
        with pytest.raises(ValueError):
            search._run_round(broken, *args)
    numpy = Broken({"inner_hp_recall": np.float64(0.5), "trees": np.int64(3)})
    assert search._run_round(numpy, *args)[1]["trees"] == 3  # passes the JSON check


def test_the_registry_keeps_the_cap_and_slots_and_diagnostics_out():
    for advisor in validity.ADVISORS:
        procedures = search.procedures_for(advisor)
        slots = [p.slot for p in procedures.values()]
        assert len(procedures) <= search.MAX_PROCEDURES
        assert len(set(slots)) == len(slots) and set(slots) <= set(search.SLOTS)
        assert search.INCUMBENT[advisor] in procedures
        assert not set(search.DIAGNOSTICS) & set(procedures)  # outside the cap
    with pytest.raises(ValueError, match="diagnostics"):
        search.compare("rule22_reoriented_1")  # and never adopted


def test_the_kick_off_printout_shows_what_a_run_tests():
    names = ("cnn", "cnn_with_rotations", "gbm_engineered")
    cnn, rotations, gbm = (search.describe(search.REGISTRY[n]) for n in names)
    assert "10 of 32 settings" in cnn and "augment" not in cnn
    assert "10 of 64 settings" in rotations and "augment" in rotations
    assert "nothing to search" in gbm


def test_the_advisors_default_to_the_recipe_s_own(monkeypatch):
    grids, y = _toy_data()
    shown = []
    monkeypatch.setattr(validity, "load_dev", lambda: (grids, np.tile(y[:, None], 4)))
    monkeypatch.setattr(search, "outer_predictions", lambda name, data, *_: data)
    monkeypatch.setattr(search, "load_outer", lambda name, advisor: (None, {}))
    monkeypatch.setattr(search, "compare_predictions", lambda *_: None)
    for printer in ("print_outer", "print_comparisons"):
        monkeypatch.setattr(search, printer, lambda _, res: shown.append(list(res)))
    search.main(["--run", "cnn"])
    search.main(["--compare", "cnn_with_rotations"])
    search.main(["--run", "gbm_engineered", "--advisors", "Tax"])
    assert shown == [["Wellness", "Tax", "Business"], ["Transportation"], ["Tax"]]


def test_hp_recall_follows_its_definition():
    # Ranked: pass, pass, pass, fail, pass, pass, fail, fail, pass, fail. Precision stays at
    # 96% or more only through the first three: 3 of the 6 passing grids.
    passes = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0], bool)
    assert search.hp_recall(np.linspace(1, 0, 10), np.where(passes, 0.9, 0.1)) == 0.5
    # The definition written out: at each level, the most passing grids among the top
    # k grids, over every k whose precision reaches that level.
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.random(300)
        pred = y + rng.normal(0, 0.1, 300)  # no ties, so each top k is well defined
        hits = np.cumsum(y[np.argsort(-pred)] >= validity.THRESHOLD)
        precise = [hits / np.arange(1, 301) >= level for level in search.HP_LEVELS]
        top_k = [hits[ok].max(initial=0) for ok in precise]
        assert np.isclose(search.hp_recall(pred, y), np.mean(top_k) / hits[-1])


def test_corrected_t_widens_the_ordinary_t():
    d = np.random.default_rng(4).normal(0.01, 0.02, 15)
    ordinary = d.mean() / (d.std(ddof=1) / np.sqrt(15))
    # 800 validation and 3,200 training grids per fold: (1/15 + 0.25) / (1/15) = 4.75.
    assert np.isclose(search.corrected_t(d, 3200, 800), ordinary / np.sqrt(4.75))
    assert np.isclose(search.T_CRITICAL, scipy.stats.t.ppf(0.95, 14), atol=5e-4)


def test_the_adoption_decision_needs_both_conditions_exactly():
    def decide(t, approved):
        return search.Comparison(t, 15, 0.01, (0, 0), approved, (0, 0))

    adopted = decide(1.761, (103.0, 100.0))  # both conditions met exactly
    assert adopted.passes_test and adopted.passes_margin and adopted.adopt
    assert adopted.gain == pytest.approx(0.03)
    assert not decide(1.760, (110.0, 100.0)).adopt  # t just short
    assert not decide(9.0, (102.9, 100.0)).adopt  # 2.9% more grids is short
    assert decide(9.0, (5.0, 0.0)).gain == np.inf


def test_the_bootstrap_resamples_grids_and_brackets_the_differences():
    rng = np.random.default_rng(8)
    y = rng.random(4000)
    noisy = np.stack([y + rng.normal(0, 0.1, 4000) for _ in range(3)])
    sharper = np.stack([y + rng.normal(0, 0.03, 4000) for _ in range(3)])
    same = search.compare_predictions(noisy, noisy, y, draws=50)
    assert same.t == 0 and same.gain == 0 and not same.adopt
    assert same.hp_recall_interval == same.approved_interval == (0.0, 0.0)
    better = search.compare_predictions(sharper, noisy, y, draws=50)
    assert better.adopt and better.folds_better == search.ROUNDS
    low, high = better.hp_recall_interval
    assert 0 < low <= better.hp_recall_diff <= high
    low, high = better.approved_interval
    assert 0 < low <= better.approved[0] - better.approved[1] <= high
    assert not search.compare_predictions(noisy, sharper, y, draws=50).passes_test


def test_reorient_turns_each_grid_the_same_way_in_every_fold(grid_zoo):
    turned = search.reorient(grid_zoo, seed=1)
    for grid, t in zip(grid_zoo, turned):  # one of its 7 orientations besides "as is"
        assert any(np.array_equal(t, o) for o in features.orientations(grid)[1:])
    third = np.arange(0, len(grid_zoo), 3)  # the same grids, in another batch
    assert np.array_equal(search.reorient(grid_zoo[third], 1), turned[third])
    assert not np.array_equal(search.reorient(grid_zoo, 2), turned)  # each draw differs


def test_rule_2_2_compares_errors_fold_by_fold():
    rng = np.random.default_rng(9)
    y = rng.random(4000)
    true = np.stack([y + rng.normal(0, 0.05, 4000) for _ in range(3)])
    controls = [np.stack([y + rng.normal(0, 0.1, 4000) for _ in range(3)])] * 5
    t, better = search.orientation_effect(true, controls, y)
    assert better == search.ROUNDS and t > search.T_CRITICAL
    assert search.orientation_effect(controls[0], controls, y) == (0.0, 0)
    t, better = search.orientation_effect(controls[0], [true] * 5, y)  # true is worse
    assert better == 0 and t < -search.T_CRITICAL


def test_the_label_free_check_finds_position_patterns_and_only_those():
    base = np.random.default_rng(10).integers(0, 5, size=(2000, 7, 7)).astype(np.int8)
    every_orientation = features.orientations(base).reshape(-1, 7, 7)
    chi2, dof, gap = search.orientation_patterns(every_orientation)
    assert chi2 < 1e-6 and gap == 0 and dof > 0
    corner_park = base.copy()
    corner_park[:, 0, 0] = 3  # a park in the same corner of every grid
    chi2, dof, gap = search.orientation_patterns(corner_park)
    assert chi2 > 10 * dof and gap > 0.5


def test_migration_copies_old_results_to_new_names(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setitem(search.REGISTRY, "toy", Toy())
    nested = tmp_path / "nested"
    nested.mkdir()
    # An old record's fields, as the old harness saved them (any others are kept too).
    inner_seeds = [1000 + 10 * r + f for r in range(3) for f in range(5)]
    old = {"name": "toy", "advisor": "Wellness", "spec": Toy().spec(), "rounds": []}
    old |= {"versions": search._versions(Toy()), "course_pin": validity.course_pin()}
    old |= {"lockbox_seed": 2156, "folds": [5, [0, 1, 2], 4, inner_seeds]}
    old |= {"scoring": [[0.96, 0.97, 0.98, 0.99], 0.75], "unchanged_during_run": True}

    def write_old(key, record):
        (nested / f"toy-wellness-{key}.json").write_text(json.dumps(record))
        np.save(nested / f"toy-wellness-{key}.npy", np.arange(180.0).reshape(3, 60))

    write_old("0" * 16, old)
    assert migrate_results.migrate() == [("toy", "Wellness", "0" * 16, "copied")]
    preds, record = search.load_outer("toy", "Wellness")  # passes the context check
    assert preds[2, 59] == 179 and record == {**old, "context": search.run_context()}
    assert migrate_results.migrate()[0][3] == "already there"
    assert (nested / f"toy-wellness-{'0' * 16}.npy").exists()  # old files stay
    write_old("1" * 16, old)  # a second old run of the same recipe
    with pytest.raises(ValueError, match="would both become"):
        migrate_results.migrate()
    for key in ("0", "1"):
        (nested / f"toy-wellness-{key * 16}.json").unlink()
    write_old("2" * 16, {**old, "scoring": [[0.9], 0.75]})  # other precision levels
    with pytest.raises(ValueError, match="differ from the current code"):
        migrate_results.migrate()


def test_slot_3_adds_the_group_that_carries_the_signal():
    # The rating is parks per home, which only "park access" holds directly; "mix" and
    # one-hot cells don't. Forward selection must add "park access" and stop there.
    rng = np.random.default_rng(3)
    grids = rng.integers(0, 5, (400, 7, 7)).astype(np.int8)
    parks, homes = (grids == 3).sum((1, 2)), (grids == 0).sum((1, 2))
    # As a percentile, like the real ratings, so about a quarter of grids pass.
    y = scipy.stats.rankdata(parks / np.maximum(homes, 1)) / len(grids)
    groups = ("mix", "park access", "onehot")
    slot3 = search.GroupSearchGBM(slot=3, summary="test", groups=groups, max_added=2)
    choice, record = slot3.search(grids, y, search.inner_splits(len(y), 0, 0))
    assert choice[0] == "park access" and len(record["steps"]) == len(choice) + 1
    assert slot3.predict(slot3.fit(choice, grids, y), grids[:5]).shape == (5,)
    assert "3 candidates" in search.describe(slot3)
