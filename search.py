"""Stage 2's harness: score whole model-building recipes honestly, with nested cross-validation.

Where this fits
---------------
Stage 1 (validity.py) scored fixed models. Stage 2 *searches*: it tries many options and keeps
the best, and a search's own best score flatters it. This module scores a search the honest way
(docs/stage2_models.md, Figure 1):

* ``REGISTRY``             The procedures allowed to compete. A procedure is a recipe: what may
                           vary, how the options are tried, and a refit. At most six per
                           advisor, the incumbent included, one per slot (docs/decisions.md,
                           31 October).
* ``outer_predictions()``  Runs a procedure in each of Stage 1's 15 outer rounds. In each round
                           the procedure gets only the 3,200 outer-training grids and their
                           ratings, searches on 4 inner folds of them, refits, and predicts the
                           800 outer-validation grids. Needs a fully committed repository.
* ``compare()``            The adoption test: a corrected paired t-test on per-fold hp_recall,
                           and at least 3% more approvals at 98% precision.
* ``rule_2_2()``           Does a cell's position matter? Slot 2's GBM against five copies fed
                           re-oriented grids (``DIAGNOSTICS``: never adopted, outside the cap).
* ``hp_recall()``          The search's score: recall averaged over 96-99% precision.

Reads / writes
--------------
reads   the course data, through validity.load_dev() (test-set ratings hidden)
writes  data/cache/nested/*.npy    outer predictions (gitignored)
        data/cache/nested/*.json   a record of the run that made them: commit, times, choices

How to run
----------
    python search.py --list              # each advisor's procedures, slot by slot
    python search.py --time NAME         # time NAME without predicting any outer fold
    python search.py --reproduce         # the incumbent run here equals Stage 1 (commit first)
    python search.py --rule-2-2          # does position matter, per advisor (commit first)
    python search.py --run NAME          # NAME's outer predictions (commit first)
    python search.py --compare NAME      # the adoption test against each advisor's incumbent

Surprises
---------
* A procedure is defined by its settings (spec) and its source code. Outer predictions are made
  only from a fully committed repository, when the procedure in memory is the committed one;
  the commit is saved with them, so git history shows the procedure was registered before its
  results existed. --time predicts no outer fold, so it runs with uncommitted changes.
* Once a name has outer results, a changed definition needs a new name, which counts again
  toward the cap. Saved results are never replaced: use_cache=False recomputes as a check,
  without saving.
* Each round gets a fresh copy of the procedure, so nothing learned in one round (whose training
  set holds other rounds' validation grids) reaches another.
* Each fit runs on one thread, with rounds in separate processes. On data this small, ten
  one-thread fits side by side get far more done than one fit spread over ten cores, and
  LightGBM's predictions are identical (the reproduction test).
* Each advisor's results are saved as soon as its 15 rounds are in; after an error, rounds not
  yet started are cancelled and finished advisors stay saved.
* Never reads Stage 1's saved predictions (data/cache/oof/); --reproduce recomputes them.
* Procedures reach worker processes by pickling, so one defined inside a function (as some
  tests do) runs only with workers=1.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
import scipy
import scipy.stats
import sklearn
from sklearn.model_selection import KFold

import features
import validity
from validity import ADVISORS, CV_FOLDS, CV_REPEATS

NESTED_DIR = os.path.join(validity.PS2_ROOT, "data", "cache", "nested")
NESTED_SCHEMA = 1  # bump if the saved files' layout changes

INNER_FOLDS = 4
HP_LEVELS = (0.96, 0.97, 0.98, 0.99)  # the precision levels hp_recall averages over
# The adoption test (docs/decisions.md, Stage 2 protocol): a corrected t of at least the
# one-sided 5% point of Student's t with 14 degrees of freedom (15 folds), and at least 3%
# more approvals at 98% precision.
T_CRITICAL = 1.761
MARGIN = 0.03
MAX_PROCEDURES = 6  # per advisor, the incumbent included ("Procedure cap")
BOOTSTRAP_DRAWS = 1000
ROUNDS = CV_FOLDS * CV_REPEATS  # outer rounds per advisor
WORKERS = os.cpu_count() or 1

# The owner's six slots (docs/decisions.md, 31 October). A slot is filled by registering its
# procedure in REGISTRY, before any of that procedure's outer results exist. If rule 2.4 finds a
# formula for an advisor, it takes slot 5 or 6 for that advisor.
SLOTS = {
    1: "incumbent: Stage 1's GBM on the 75 engineered features",
    2: "rule 2.2's true-orientation arm: GBM on the engineered and one-hot inputs",
    3: "GBM choosing among feature groups",
    4: "neural network: a CNN search",
    5: "tuned GBM: slot 3's search plus GBM settings",
    6: "stacker or ensemble",
}


# --------------------------------------------------------------------------------------
# inner folds, and the scores a search and a comparison use
# --------------------------------------------------------------------------------------


def inner_seed(shuffle, fold):
    """The seed of one outer round's inner folds. New seeds, used nowhere else."""
    return 1000 + 10 * shuffle + fold


def inner_splits(n, shuffle, fold):
    """The 4 inner folds of one outer round's n training grids, as (train, val) positions.

    Seeded by the outer round, so every procedure searches on the same inner folds in that
    round and their inner scores compare fairly.
    """
    splitter = KFold(INNER_FOLDS, shuffle=True, random_state=inner_seed(shuffle, fold))
    return list(splitter.split(np.arange(n)))


def inner_predictions(fit_predict, grids, y, inner):
    """One option's inner out-of-fold predictions: each grid predicted by a fit on the other
    three inner folds. fit_predict(training grids, their ratings, grids to predict) -> preds."""
    pred = np.full(len(y), np.nan)
    for train, val in inner:
        pred[val] = fit_predict(grids[train], y[train], grids[val])
    return pred


def hp_recall(pred, y, levels=HP_LEVELS):
    """The share of the truly passing grids approved, averaged over precision levels 96-99%.

    Rule 1.3's approval count as a share of the passing grids, so that folds compare, averaged
    over neighbouring precision levels so that it doesn't hinge on two or three grids
    (docs/decisions.md, Stage 2 protocol). NaN if no grid passes.
    """
    passing = int((np.asarray(y) >= validity.THRESHOLD).sum())
    if not passing:
        return np.nan
    hits = [validity.approved_at(pred, y, level)[2] for level in levels]
    return float(np.mean(hits)) / passing


def _per_fold(score, preds, y):
    """score(fold predictions, fold ratings) in each outer fold, in cv_splits order."""
    folds = validity.cv_splits(len(y), repeats=len(preds))
    return np.array([score(preds[r, val], y[val]) for r, _, _, val in folds])


def per_fold_hp_recall(preds, y):
    """hp_recall in each outer fold, from (shuffles, n) outer predictions."""
    return _per_fold(hp_recall, preds, y)


def per_fold_mse(preds, y):
    """Mean squared error in each outer fold, from (shuffles, n) outer predictions."""
    return _per_fold(lambda p, t: float(np.mean((p - t) ** 2)), preds, y)


def _fold_sizes(n):
    """The outer folds' mean training and validation sizes, for n grids."""
    folds = list(validity.cv_splits(n))
    n_train = np.mean([len(train) for _, _, train, _ in folds])
    return n_train, np.mean([len(val) for _, _, _, val in folds])


def corrected_t(diffs, n_train, n_val):
    """Nadeau and Bengio's corrected paired t, for differences from overlapping training sets.

    The ordinary t divides the variance by the number of folds J, as if the folds were
    independent. Here any two outer training sets share most of their grids, so results rise
    and fall together; the correction adds n_val / n_train to 1 / J.
    """
    d = np.asarray(diffs, float)
    var = d.var(ddof=1)
    if var == 0:  # every fold moved by the same amount
        return math.copysign(math.inf, d.mean()) if d.mean() else 0.0
    return float(d.mean() / np.sqrt((1 / len(d) + n_val / n_train) * var))


# --------------------------------------------------------------------------------------
# procedures
# --------------------------------------------------------------------------------------

INPUT_VERSIONS = {
    "engineered": features.FEATURE_VERSION,
    "onehot": features.ONEHOT_VERSION,
}


@dataclass(frozen=True)
class FixedGBM:
    """GBM with Stage 1's settings on fixed inputs: nothing to search, so the refit is the model.

    Every procedure has these methods. The harness calls the first three in each outer round:
        search(grids, y, inner) -> (choice, record)  choose, using the inner folds only; the
                                                     record must include the choice's
                                                     inner_hp_recall
        fit(choice, grids, y)   -> model             refit the choice on the outer training set
        predict(model, grids)   -> predictions       predict the outer-validation grids
        spec()                  its settings (fingerprinted with its code; see code())
        model_inputs(grids)     every model input it may use (hashed into the cache key)
        code()                  optional: the classes and functions whose source defines it
    The choice and the record are saved with the results, so they must be JSON values.
    """

    inputs: tuple  # names in validity.FEATURIZERS, joined side by side in this order
    slot: int
    summary: str
    advisors: tuple = ADVISORS

    def spec(self):
        return {
            "kind": "fixed_gbm",
            "inputs": [[name, INPUT_VERSIONS[name]] for name in self.inputs],
            "gbm_params": validity.GBM_PARAMS,
        }

    def code(self):
        return (FixedGBM,)

    def model_inputs(self, grids):
        return np.hstack([validity.FEATURIZERS[name](grids) for name in self.inputs])

    def search(self, grids, y, inner):
        # Nothing to choose. The inner score is still recorded, for the protocol's
        # "selection by inner score" check (docs/decisions.md, Stage 2 protocol).
        def fit_predict(g_fit, y_fit, g_new):
            return self.predict(self.fit(None, g_fit, y_fit), g_new)

        pred = inner_predictions(fit_predict, grids, y, inner)
        return None, {"inner_hp_recall": hp_recall(pred, y)}

    def fit(self, choice, grids, y):
        import lightgbm as lgb  # imported here so tests that never fit needn't pay for it

        # One thread per fit: see the header.
        model = lgb.LGBMRegressor(**validity.GBM_PARAMS, n_jobs=1)
        return model.fit(self.model_inputs(grids), y)

    def predict(self, model, grids):
        return model.predict(self.model_inputs(grids))


REGISTRY = {
    "gbm_engineered": FixedGBM(
        ("engineered",),
        slot=1,
        summary="Stage 1's primary model: GBM on the 75 engineered features",
    ),
    "gbm_engineered_onehot": FixedGBM(
        ("engineered", "onehot"),
        slot=2,
        summary="rule 2.2's true-orientation arm: GBM on the 75 engineered features and the "
        "245 one-hot inputs, which record each cell's position",
    ),
}

# The procedure in force for each advisor. Changes only with an adoption recorded in
# docs/decisions.md.
INCUMBENT = dict.fromkeys(ADVISORS, "gbm_engineered")


def procedures_for(advisor):
    """{name: procedure} registered for one advisor."""
    return {name: p for name, p in REGISTRY.items() if advisor in p.advisors}


# Rule 2.2's controls: slot 2's GBM, with the one-hot inputs computed on re-oriented grids.

REORIENT_SEEDS = (1, 2, 3, 4, 5)  # rule 2.2's five draws


def reorient(grids, seed):
    """Each grid turned or mirrored into one of its 7 other orientations, picked at random
    from the seed and the grid's own cells, so a grid is turned the same way in every fold."""
    turns = list(features.SYMMETRIES.values())
    grids = np.asarray(grids)
    out = np.empty_like(grids)
    for i, grid in enumerate(grids):
        digest = hashlib.sha256(bytes([seed]) + grid.tobytes()).digest()
        which = 1 + int.from_bytes(digest[:4], "little") % 7  # 0 would be "as is"
        out[i] = turns[which](grid)
    return out


@dataclass(frozen=True)
class ReorientedGBM(FixedGBM):
    """Slot 2's GBM, but its one-hot inputs (each cell's position) come from re-oriented
    grids. The engineered features ignore orientation, so they are unchanged."""

    seed: int = 1

    def spec(self):
        return {**super().spec(), "kind": "reoriented_gbm", "seed": self.seed}

    def code(self):
        return (FixedGBM, ReorientedGBM, reorient)

    def model_inputs(self, grids):
        turned = reorient(grids, self.seed)
        return np.hstack([features.engineered(grids), features.onehot(turned)])


# Diagnostics run through the harness like procedures, but can't be adopted and don't count
# toward the cap (docs/decisions.md, "Procedure cap").
DIAGNOSTICS = {
    f"rule22_reoriented_{seed}": ReorientedGBM(
        ("engineered", "onehot"),
        slot=0,
        summary=f"rule 2.2's control, draw {seed}: one-hot inputs from re-oriented grids",
        seed=seed,
    )
    for seed in REORIENT_SEEDS
}


def _lookup(name, advisor):
    """The registered procedure or diagnostic called `name`, if it may run for `advisor`."""
    procedure = procedures_for(advisor).get(name) or DIAGNOSTICS.get(name)
    if procedure is None:
        raise KeyError(f"{name!r} is not registered for {advisor} (search.REGISTRY)")
    return procedure


# --------------------------------------------------------------------------------------
# one outer round
# --------------------------------------------------------------------------------------


def _round_data(grids, y, train, val):
    """What a procedure gets in one outer round: the training grids with their ratings, and
    the validation grids without theirs."""
    return grids[train], y[train], grids[val]


def _fresh(procedure):
    """A new copy of the procedure for each round, so nothing it learns in one round (whose
    training set holds other rounds' validation grids) can reach another."""
    return copy.deepcopy(procedure)


def _json_default(value):
    """Lets NumPy numbers and arrays into a JSON record."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    raise TypeError(f"a {type(value).__name__} can't be saved in a run's record")


def _run_round(procedure, grids, y, shuffle, fold, train, val, predict=True):
    """One outer round: search on the inner folds, refit on the outer training set, then
    (unless only timing) predict the outer-validation grids. Returns (predictions, record)."""
    start = time.perf_counter()
    procedure = _fresh(procedure)
    g_train, y_train, g_val = _round_data(grids, y, train, val)
    inner = inner_splits(len(y_train), shuffle, fold)
    choice, record = procedure.search(g_train, y_train, inner)
    if "inner_hp_recall" not in record or {
        "shuffle",
        "fold",
        "choice",
        "seconds",
    } & set(record):
        raise ValueError(
            "a search's record must include inner_hp_recall, and may not use the harness's "
            "own keys (shuffle, fold, choice, seconds)"
        )
    model = procedure.fit(choice, g_train, y_train)
    pred = None
    if predict:
        pred = np.asarray(procedure.predict(model, g_val), dtype=float)
        if pred.shape != (len(val),) or not np.isfinite(pred).all():
            raise ValueError(
                f"shuffle {shuffle}, fold {fold}: expected {len(val)} finite predictions, "
                f"got an array of shape {pred.shape}"
            )
    seconds = round(time.perf_counter() - start, 3)
    full = {
        "shuffle": shuffle,
        "fold": fold,
        "choice": choice,
        **record,
        "seconds": seconds,
    }
    json.dumps(
        full, default=_json_default
    )  # fail in this round, not after the whole run
    return pred, full


def _execute(jobs, workers, done):
    """_run_round over jobs, calling done(index, result) as each finishes: here, in order, if
    workers == 1; else in fresh processes. After an error, rounds not yet started are
    cancelled."""
    if workers == 1:
        for i, job in enumerate(jobs):
            done(i, _run_round(*job))
        return
    # "spawn" starts clean processes. Forked ones would inherit this process's thread pools,
    # which can hang LightGBM on Linux.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=context) as pool:
        futures = {pool.submit(_run_round, *job): i for i, job in enumerate(jobs)}
        try:
            for future in as_completed(futures):
                done(futures[future], future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise


# --------------------------------------------------------------------------------------
# what defines a run, and its saved results
# --------------------------------------------------------------------------------------

# The harness code every procedure runs through; part of every run's identity.
HARNESS_PARTS = (
    "inner_seed",
    "inner_splits",
    "inner_predictions",
    "hp_recall",
    "_round_data",
    "_fresh",
    "_run_round",
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _source_hash(objects):
    return _sha("\n".join(inspect.getsource(obj) for obj in objects).encode())


def _plain(value):
    """value as it reads back from JSON (tuples become lists), so comparisons are fair."""
    return json.loads(json.dumps(value))


def _definition(procedure):
    """What defines a procedure: its settings (spec) and the source code of its parts."""
    parts = procedure.code() if hasattr(procedure, "code") else (type(procedure),)
    return {"spec": _plain(procedure.spec()), "code": _source_hash(parts)}


def _identity(procedure, grids, y, rows):
    """Everything that could change a procedure's outer predictions; "key" is its hash.

    As validity.cache_key: the procedure's definition, the harness code, the library
    versions, the scoring settings, the outer and inner folds, the test-set seed, the
    course-data version, the grids, the model inputs built from them, and the rows with their
    ratings. Saved with each run, so later runs can be checked against it.
    """
    import lightgbm as lgb

    grids = np.ascontiguousarray(grids)
    X = np.ascontiguousarray(_fresh(procedure).model_inputs(grids))
    versions = {
        "python": ".".join(map(str, sys.version_info[:2])),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lgb.__version__,
    }
    versions.update(getattr(procedure, "library_versions", dict)())  # e.g. PyTorch's
    rows_and_ratings = (
        np.asarray(rows, np.int64).tobytes() + np.asarray(y, np.float64).tobytes()
    )
    seeds = [inner_seed(r, f) for r in range(CV_REPEATS) for f in range(CV_FOLDS)]
    identity = {
        "schema": NESTED_SCHEMA,
        **_definition(procedure),
        "harness": _source_hash([globals()[part] for part in HARNESS_PARTS]),
        "versions": versions,
        "scoring": [list(HP_LEVELS), validity.THRESHOLD],
        "folds": [CV_FOLDS, list(validity.CV_SEEDS), INNER_FOLDS, seeds],
        "course_pin": validity.course_pin(),
        "lockbox_seed": validity.LOCKBOX_SEED,
        "data": _sha(grids.tobytes() + str(grids.shape).encode() + rows_and_ratings),
        "inputs": [_sha(X.tobytes()), list(X.shape), str(X.dtype)],
    }
    identity["key"] = _sha(json.dumps(identity, sort_keys=True).encode())
    return identity


def _path(name, advisor, key):
    return os.path.join(NESTED_DIR, f"{name}-{advisor.lower()}-{key[:16]}.npy")


def _all_saved():
    """[(file, record)] for every saved run."""
    if not os.path.isdir(NESTED_DIR):
        return []
    saved = []
    for file in sorted(os.listdir(NESTED_DIR)):
        if file.endswith(".json"):
            with open(os.path.join(NESTED_DIR, file)) as f:
                saved.append((file, json.load(f)))
    return saved


def _saved_runs(name, advisor):
    return [
        (file, record)
        for file, record in _all_saved()
        if record["name"] == name and record["advisor"] == advisor
    ]


def _load(path):
    """Saved predictions, or None. The record is written first, so predictions without one
    mean a damaged save."""
    if not os.path.exists(path):
        return None
    if not os.path.exists(path[:-4] + ".json"):
        raise RuntimeError(f"{path} has no record beside it: delete it and run again")
    return np.load(path)


def _write_whole(path, write):
    """Write a file whole or not at all: under a temporary name, then renamed into place."""
    partial = path + ".partial"
    with open(partial, "wb") as f:
        write(f)
    os.replace(partial, path)


def _save(path, preds, record):
    text = json.dumps(record, indent=1, default=_json_default).encode()
    os.makedirs(NESTED_DIR, exist_ok=True)
    _write_whole(path[:-4] + ".json", lambda f: f.write(text))
    _write_whole(path, lambda f: np.save(f, preds))


# --------------------------------------------------------------------------------------
# the checks before any outer prediction is made
# --------------------------------------------------------------------------------------


def _check_workers(workers):
    if workers < 1:
        raise ValueError(f"workers must be at least 1, not {workers}")


def _committed():
    """The commit to record with new outer predictions. Refuses while anything is uncommitted."""
    commit, uncommitted = validity._git_state()
    if uncommitted:
        raise PermissionError(
            "Outer predictions are made only from a fully committed repository, so that the "
            "commit recorded with them holds the registered procedure (docs/decisions.md, "
            f"31 October). Not committed: {', '.join(uncommitted[:5])}"
        )
    return commit


def _definition_on_disk(name, advisor):
    """The definition of `name` that a fresh Python process imports from the files on disk;
    with nothing uncommitted, the committed definition."""
    script = (
        "import json, search; "
        f"print(json.dumps(search._definition(search._lookup({name!r}, {advisor!r}))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=validity.PS2_ROOT,
        check=False,  # a failure is reported below
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise PermissionError(
            f"{name} is not registered for {advisor} in search.py on disk"
        )
    return json.loads(result.stdout)


def _check_definition(name, advisor, identity, slot):
    """Refuse to score `name` for `advisor` unless (1) its definition in memory is the
    committed one, (2) no earlier outer results of `name`, for any advisor, came from a
    different definition, and (3) the cap still holds: every name with outer results counts,
    each in its own slot (docs/decisions.md, "Procedure cap")."""
    mine = {"spec": identity["spec"], "code": identity["code"]}
    if _definition_on_disk(name, advisor) != mine:
        raise PermissionError(
            f"{name} in memory differs from the committed search.py: run it from a fresh "
            "process (python search.py ...)"
        )
    saved = _all_saved()
    for file, record in saved:
        if record["name"] != name:
            continue
        changed = {"spec": record["spec"], "code": record["code"]} != mine
        if record["advisor"] == advisor and record["data"] == identity["data"]:
            changed |= record["inputs"] != identity["inputs"]  # e.g. a feature edited
        if changed:
            raise ValueError(
                f"{name} already has outer results ({file}) from a different definition. A "
                "changed procedure takes a new name and counts again toward the cap "
                "(docs/decisions.md, Procedure cap)."
            )
    if name in DIAGNOSTICS:
        return
    slots = {
        record["name"]: record["slot"]
        for _, record in saved
        if record["advisor"] == advisor and record["name"] not in DIAGNOSTICS
    }
    rivals = [other for other, s in slots.items() if s == slot and other != name]
    if rivals:
        raise ValueError(
            f"slot {slot} already has outer results from {rivals[0]} for {advisor}: a new "
            "procedure needs a free slot"
        )
    if len(set(slots) | {name}) > MAX_PROCEDURES:
        raise ValueError(f"{advisor} would have more than {MAX_PROCEDURES} procedures")


# --------------------------------------------------------------------------------------
# outer predictions
# --------------------------------------------------------------------------------------


def outer_predictions(
    name, advisor, grids, y, rows, *, workers=WORKERS, use_cache=True, run=True
):
    """(CV_REPEATS, n) outer predictions of a registered procedure for one advisor.

    Each grid's prediction in each shuffle comes from a round whose procedure never saw that
    grid: not in its search, not in its refit. Saved predictions are reused while everything
    in the run's identity is unchanged. Otherwise, with run=True, the 15 rounds are computed
    (which needs a fully committed repository) and saved; use_cache=False recomputes them as
    a check, without saving; run=False raises instead.
    """
    return _outer(name, {advisor: (grids, y, rows)}, workers, use_cache, run)[advisor]


def _outer(name, data, workers, use_cache, run):
    """outer_predictions for several advisors ({advisor: (grids, y, rows)}), with every round
    sharing one pool of processes. Each advisor's results are saved as soon as its rounds are
    all in, so an error late in a run keeps what was finished."""
    _check_workers(workers)
    found, todo = {}, {}
    for advisor, (grids, y, rows) in data.items():
        procedure = _lookup(name, advisor)
        identity = _identity(procedure, grids, y, rows)
        path = _path(name, advisor, identity["key"])
        saved = _load(path) if use_cache else None
        if saved is not None:
            found[advisor] = saved
        elif not run:
            raise FileNotFoundError(
                f"no outer predictions of {name} for {advisor}: python search.py --run {name}"
            )
        else:
            todo[advisor] = (procedure, identity, path)
    if not todo:
        return found

    commit = _committed()
    for advisor, (procedure, identity, _) in todo.items():
        _check_definition(name, advisor, identity, procedure.slot)
    started, tic = datetime.now(UTC), time.perf_counter()
    jobs, places = [], []
    for advisor, (procedure, _, _) in todo.items():
        grids, y, _ = data[advisor]
        for r, f, train, val in validity.cv_splits(len(y)):
            jobs.append((procedure, grids, y, r, f, train, val))
            places.append((advisor, r, val))
    preds = {
        advisor: np.full((CV_REPEATS, len(data[advisor][1])), np.nan)
        for advisor in todo
    }
    rounds = {advisor: [] for advisor in todo}

    def done(i, result):
        advisor, r, val = places[i]
        preds[advisor][r, val], record = result
        rounds[advisor].append(record)
        if len(rounds[advisor]) < ROUNDS:
            return
        found[advisor] = preds[advisor]
        if not use_cache:
            return  # a check: saved results are never replaced
        procedure, identity, path = todo[advisor]
        end_commit, end_uncommitted = validity._git_state()
        record = {
            **identity,
            "name": name,
            "advisor": advisor,
            "slot": procedure.slot,
            "summary": procedure.summary,
            "n": len(data[advisor][1]),
            "commit": commit,
            "unchanged_during_run": end_commit == commit and not end_uncommitted,
            "run_from": validity._caller(),
            "started": started.isoformat(timespec="seconds"),
            "seconds_until_saved": round(time.perf_counter() - tic, 1),
            "workers": workers,
            "rounds": sorted(
                rounds[advisor], key=lambda rec: (rec["shuffle"], rec["fold"])
            ),
        }
        _save(path, preds[advisor], record)

    _execute(jobs, workers, done)
    return found


def _advisor_data(grids, dev, advisor):
    """One advisor's development-set grids, ratings and row numbers, as validity.scorecard()."""
    a = ADVISORS.index(advisor)
    rows = np.flatnonzero(~np.isnan(dev[:, a]))
    return grids[rows], dev[rows, a], rows


def run(name, advisors=ADVISORS, workers=WORKERS, use_cache=True):
    """{advisor: (ratings, outer predictions)} of procedure `name` on the development set."""
    grids, dev = validity.load_dev()
    data = {advisor: _advisor_data(grids, dev, advisor) for advisor in advisors}
    preds = _outer(name, data, workers, use_cache, run=True)
    return {advisor: (data[advisor][1], preds[advisor]) for advisor in advisors}


def time_procedure(
    name, advisor=ADVISORS[0], workers=WORKERS, rounds=None, trials=None
):
    """(wall seconds, per-round seconds, estimated seconds for all four advisors' rounds).

    Runs the first `rounds` outer rounds of one advisor (default: one per worker), `workers`
    at a time as a real run would: search and refit, but no outer-validation grid is
    predicted, so no outer result is made. The estimate scales the wall time to 4 × 15 rounds
    run `workers` at a time; process start-up dominates it for fast procedures. For a search
    with a `trials` setting, `trials` times a shorter search and scales the estimate up in
    proportion (a slight overestimate: the refit doesn't grow with the trials).
    """
    _check_workers(workers)
    procedure, scale = _lookup(name, advisor), 1.0
    if trials is not None:
        if not hasattr(procedure, "trials"):
            raise ValueError(f"{name} has no trials setting")
        scale = procedure.trials / trials
        procedure = replace(procedure, trials=trials)
    grids, dev = validity.load_dev()
    g, y, _ = _advisor_data(grids, dev, advisor)
    splits = list(validity.cv_splits(len(y)))[: rounds or min(workers, ROUNDS)]
    jobs = [(procedure, g, y, r, f, train, val, False) for r, f, train, val in splits]
    seconds = [0.0] * len(jobs)

    def done(i, result):
        seconds[i] = result[1]["seconds"]

    tic = time.perf_counter()
    _execute(jobs, workers, done)
    wall = time.perf_counter() - tic
    timed_waves = math.ceil(len(jobs) / min(workers, len(jobs)))
    all_rounds = len(ADVISORS) * ROUNDS
    full_waves = math.ceil(all_rounds / min(workers, all_rounds))
    return wall, seconds, wall / timed_waves * full_waves * scale


# --------------------------------------------------------------------------------------
# the adoption test
# --------------------------------------------------------------------------------------


@dataclass
class Comparison:
    """A challenger against the incumbent on one advisor (docs/decisions.md, Stage 2 protocol).

    Pairs are (challenger, incumbent). Approval counts are at 98% precision, means over the 3
    shuffles. Intervals are 95% paired bootstraps over grids.
    """

    t: float  # corrected paired t on the 15 per-fold hp_recall differences
    folds_better: int  # folds where the challenger's hp_recall is higher
    hp_recall_diff: float  # mean per-fold difference, challenger minus incumbent
    hp_recall_interval: tuple
    approved: tuple
    approved_interval: tuple  # of the difference in approval counts, in grids
    gain: float  # challenger's approval count / incumbent's - 1
    passes_test: bool  # t >= T_CRITICAL
    passes_margin: bool  # gain >= MARGIN
    adopt: bool  # both


def _mean_approved(preds, y):
    return float(np.mean([validity.approved_at(p, y)[0] for p in preds]))


def compare_predictions(challenger, incumbent, y, draws=BOOTSTRAP_DRAWS):
    """The adoption test on two procedures' outer predictions for one advisor, each of shape
    (CV_REPEATS, n) from the same outer folds."""
    hp_c, hp_i = per_fold_hp_recall(challenger, y), per_fold_hp_recall(incumbent, y)
    t = corrected_t(hp_c - hp_i, *_fold_sizes(len(y)))
    approved_c, approved_i = _mean_approved(challenger, y), _mean_approved(incumbent, y)
    passes_test = bool(t >= T_CRITICAL)
    passes_margin = bool(approved_c >= (1 + MARGIN) * approved_i)
    hp_interval, approved_interval = _bootstrap(challenger, incumbent, y, draws)
    return Comparison(
        t=t,
        folds_better=int((hp_c > hp_i).sum()),
        hp_recall_diff=float(np.mean(hp_c - hp_i)),
        hp_recall_interval=hp_interval,
        approved=(approved_c, approved_i),
        approved_interval=approved_interval,
        gain=approved_c / approved_i - 1 if approved_i else math.inf,
        passes_test=passes_test,
        passes_margin=passes_margin,
        adopt=passes_test and passes_margin,
    )


def _bootstrap(challenger, incumbent, y, draws, seed=0):
    """95% paired-bootstrap intervals of (the mean per-fold hp_recall difference, the
    difference in approval counts). Each draw resamples the same grids for both procedures:
    within each outer fold for hp_recall, across all the grids for the counts."""
    rng = np.random.default_rng(seed)
    folds = list(validity.cv_splits(len(y), repeats=len(challenger)))
    hp, counts = [], []
    for _ in range(draws):
        diffs = []
        for r, _, _, val in folds:
            pick = rng.choice(val, len(val))
            diffs.append(
                hp_recall(challenger[r, pick], y[pick])
                - hp_recall(incumbent[r, pick], y[pick])
            )
        hp.append(np.mean(diffs))
        pick = rng.integers(0, len(y), len(y))
        counts.append(
            _mean_approved(challenger[:, pick], y[pick])
            - _mean_approved(incumbent[:, pick], y[pick])
        )
    low_high = [2.5, 97.5]
    return (
        tuple(float(v) for v in np.percentile(hp, low_high)),
        tuple(float(v) for v in np.percentile(counts, low_high)),
    )


def compare(name, against=None, advisors=ADVISORS):
    """{advisor: (incumbent's name, Comparison)} for procedure `name`, from saved outer
    predictions only. `against` defaults to each advisor's incumbent."""
    if name in DIAGNOSTICS or against in DIAGNOSTICS:
        raise ValueError(f"{name} or {against} is a diagnostic: it cannot be adopted")
    grids, dev = validity.load_dev()
    out = {}
    for advisor in advisors:
        g, y, rows = _advisor_data(grids, dev, advisor)
        other = against or INCUMBENT[advisor]
        mine = outer_predictions(name, advisor, g, y, rows, run=False)
        theirs = outer_predictions(other, advisor, g, y, rows, run=False)
        out[advisor] = (other, compare_predictions(mine, theirs, y))
    return out


# --------------------------------------------------------------------------------------
# rule 2.2: does a cell's position matter?
# --------------------------------------------------------------------------------------


def orientation_effect(true, controls, y):
    """(corrected t, folds where the true orientation predicts better), for one advisor.

    Per outer fold: the mean of the control draws' mean squared errors, minus the
    true-orientation arm's, so a positive difference favours the true orientation. Each
    draw's error comes from its own predictions; they are not averaged (rule 2.2).
    """
    control = np.mean([per_fold_mse(c, y) for c in controls], axis=0)
    diffs = control - per_fold_mse(true, y)
    return corrected_t(diffs, *_fold_sizes(len(y))), int((diffs > 0).sum())


def rule_2_2(advisors=ADVISORS, workers=WORKERS):
    """{advisor: (corrected t, folds where the true orientation predicts better)}.

    Orientation matters for an advisor if t >= T_CRITICAL: one-sided 5%, with no correction
    across the advisors, because a finding leads to the cautious action (rule 2.2).
    """
    true = run("gbm_engineered_onehot", advisors, workers)
    names = [name for name in DIAGNOSTICS if name.startswith("rule22_")]
    controls = [run(name, advisors, workers) for name in names]
    return {
        advisor: orientation_effect(
            true[advisor][1], [c[advisor][1] for c in controls], true[advisor][0]
        )
        for advisor in advisors
    }


def orientation_patterns(grids):
    """Label-free: do the grids themselves favour some cells over their mirror images?

    The 8 symmetries map each cell onto a set of cells (the corners onto one another, and so
    on). For each set, a chi-square test of whether every district is equally common in all
    its cells; the sets' statistics are summed. Returns (chi-square, degrees of freedom,
    p-value, largest gap in one district's share between two cells of a set).
    """
    grids = np.asarray(grids)
    images = features.orientations(np.arange(49).reshape(7, 7))  # where each cell goes
    sets = {tuple(sorted(set(images[:, r, c]))) for r in range(7) for c in range(7)}
    counts = np.stack(
        [(grids == k).sum(0).ravel() for k in range(features.N_DISTRICTS)], axis=1
    )
    chi2, dof, gap = 0.0, 0, 0.0
    for cells in sets:
        if len(cells) < 2:
            continue  # the centre cell maps only onto itself
        table = counts[list(cells)]
        stat, _, df, _ = scipy.stats.chi2_contingency(table)
        chi2, dof = chi2 + stat, dof + int(df)
        shares = table / table.sum(1, keepdims=True)
        gap = max(gap, float((shares.max(0) - shares.min(0)).max()))
    return chi2, dof, float(scipy.stats.chi2.sf(chi2, dof)), gap


# --------------------------------------------------------------------------------------
# reproducing Stage 1
# --------------------------------------------------------------------------------------


def reproduce_stage1(workers=WORKERS):
    """True if Stage 1's GBM, run through this harness now, gives exactly Stage 1's
    predictions for every advisor, and the incumbent's saved predictions are the same ones.
    Both sides are recomputed: nothing is read from data/cache/oof/, and a later edit to the
    harness can't hide behind saved predictions."""
    saved = run("gbm_engineered", workers=workers)  # saved, or made and saved now
    fresh = run(
        "gbm_engineered", workers=workers, use_cache=False
    )  # recomputed, not saved
    grids, dev = validity.load_dev()
    print(
        "\nStage 1's GBM through the harness, against validity.out_of_fold (recomputed):"
    )
    same = True
    for advisor, (y, mine) in fresh.items():
        g, _, rows = _advisor_data(grids, dev, advisor)
        stage1 = validity.out_of_fold("gbm_engineered", g, y, rows, use_cache=False)
        identical = np.array_equal(mine, stage1) and np.array_equal(
            mine, saved[advisor][1]
        )
        counts = [validity.approved_at(p, y)[0] for p in mine]
        print(
            f"  {advisor:<15} {'identical' if identical else 'DIFFERENT':<10} approved at "
            f"98%: {np.mean(counts):.0f} [{'/'.join(map(str, counts))}]"
        )
        same &= identical
    print("PASS" if same else "FAIL: the harness does not reproduce Stage 1")
    return same


# --------------------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------------------


def print_registry():
    print(f"Procedures per advisor: at most {MAX_PROCEDURES}, the incumbent included")
    for slot, description in SLOTS.items():
        print(f"\nSlot {slot}  {description}")
        filled = [(n, p) for n, p in REGISTRY.items() if p.slot == slot]
        if not filled:
            print("        not registered yet")
        for name, procedure in filled:
            print(f"        {name}: {procedure.summary}")
            for advisor in procedure.advisors:
                notes = ["incumbent"] if INCUMBENT[advisor] == name else []
                if _saved_runs(name, advisor):
                    notes.append("has outer results")
                print(f"          {advisor:<15} {', '.join(notes)}")
    print("\nDiagnostics (never adopted, outside the cap):")
    for name, diagnostic in DIAGNOSTICS.items():
        print(f"        {name}: {diagnostic.summary}")


def print_outer(name, results):
    print(f"\n{name}: outer predictions on the development set, 15 rounds per advisor")
    print(f"  {'advisor':<15} {'approved at 98%: mean [per shuffle]':<36} hp_recall")
    for advisor, (y, preds) in results.items():
        counts = [validity.approved_at(p, y)[0] for p in preds]
        approved = f"{np.mean(counts):.1f} [{'/'.join(map(str, counts))}]"
        hp = per_fold_hp_recall(preds, y).mean()
        print(f"  {advisor:<15} {approved:<36} {hp:.3f}")
    where = os.path.relpath(NESTED_DIR, validity.PS2_ROOT)
    print(
        f"hp_recall is the mean of the 15 folds'. Saved in {where}/ with a record of the run."
    )


def print_comparisons(name, comparisons):
    print(
        f"\n{name} against the incumbent: adopted only if t >= {T_CRITICAL} "
        f"and at least {MARGIN:.0%} more approvals (95% intervals in brackets)"
    )
    print(
        f"  {'advisor':<15} {'incumbent':<22} {'t':>6}  {'folds better':<13}"
        f"{'hp_recall difference':<26}{'approved at 98%':<17}{'gain':>7}  "
        f"{'difference in grids':<24}adopt"
    )
    for advisor, (other, c) in comparisons.items():
        better = f"{c.folds_better} of {ROUNDS}"
        low, high = c.hp_recall_interval
        hp = f"{c.hp_recall_diff:+.3f} [{low:+.3f}, {high:+.3f}]"
        approved = f"{c.approved[0]:.1f} / {c.approved[1]:.1f}"
        low, high = c.approved_interval
        grids = f"{c.approved[0] - c.approved[1]:+.1f} [{low:+.0f}, {high:+.0f}]"
        print(
            f"  {advisor:<15} {other:<22} {c.t:>6.2f}  {better:<13}{hp:<26}{approved:<17}"
            f"{c.gain:>+7.1%}  {grids:<24}{'yes' if c.adopt else 'no'}"
        )
    print(
        "Decided at the end of the session, against the incumbent in force at its start "
        "(docs/decisions.md, Stage 2 protocol)."
    )


def print_rule_2_2(results, patterns, n_grids):
    print(
        "\nRule 2.2: does a cell's position matter? Slot 2's GBM on true orientations, "
        "against five re-oriented draws"
    )
    print(f"  {'advisor':<15} {'t':>6}  {'folds where true is better':<28}matters")
    for advisor, (t, better) in results.items():
        matters = "yes" if t >= T_CRITICAL else "no"
        print(f"  {advisor:<15} {t:>6.2f}  {f'{better} of {ROUNDS}':<28}{matters}")
    chi2, dof, p, gap = patterns
    print(
        f"Label-free, all {n_grids:,} grids: chi-square {chi2:,.0f} on {dof} degrees of "
        f"freedom (p = {p:.2g}); largest gap in a district's share between mirror-image "
        f"cells: {gap:.2%}"
    )
    print(
        f"'matters' means corrected t >= {T_CRITICAL}. Then: no augmentation, and no rotated "
        "or mirrored copies in submissions (rules 2.2 and 4.2)."
    )


def _at_least_one(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    ap = argparse.ArgumentParser(
        description="Stage 2's nested cross-validation harness (docs/stage2_models.md)."
    )
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="the procedures, by slot")
    action.add_argument(
        "--time",
        metavar="NAME",
        help="time NAME's search and refit; predicts no outer fold",
    )
    action.add_argument(
        "--reproduce",
        action="store_true",
        help="check the incumbent run here equals Stage 1",
    )
    action.add_argument(
        "--rule-2-2", action="store_true", help="does position matter? (commit first)"
    )
    action.add_argument(
        "--run", metavar="NAME", help="NAME's outer predictions (commit first)"
    )
    action.add_argument(
        "--compare",
        metavar="NAME",
        help="the adoption test: NAME against each incumbent",
    )
    ap.add_argument("--advisors", nargs="+", choices=ADVISORS, default=list(ADVISORS))
    ap.add_argument(
        "--workers", type=_at_least_one, default=WORKERS, help=f"default {WORKERS}"
    )
    ap.add_argument(
        "--trials",
        type=_at_least_one,
        help="with --time: time this many search trials, then scale up",
    )
    args = ap.parse_args()

    if args.list:
        print_registry()
    elif args.time:
        advisor = args.advisors[0]
        wall, per_round, estimate = time_procedure(
            args.time, advisor, args.workers, trials=args.trials
        )
        print(
            f"{args.time} on {advisor}: {len(per_round)} outer rounds, {args.workers} at a "
            "time, search and refit only (no outer fold predicted)"
        )
        print(
            f"  wall time {wall:.1f} s; rounds took {min(per_round):.1f}-{max(per_round):.1f} s"
        )
        scaled = f", scaled up from {args.trials} trials" if args.trials else ""
        rounds = len(ADVISORS) * ROUNDS
        print(f"  estimate for all {rounds} rounds: {estimate / 60:.1f} min{scaled}")
    elif args.reproduce:
        sys.exit(0 if reproduce_stage1(args.workers) else 1)
    elif args.rule_2_2:
        grids, _ = validity.load_dev()
        results = rule_2_2(args.advisors, args.workers)
        print_rule_2_2(results, orientation_patterns(grids), len(grids))
    elif args.run:
        print_outer(args.run, run(args.run, args.advisors, args.workers))
    else:
        print_comparisons(args.compare, compare(args.compare, advisors=args.advisors))


if __name__ == "__main__":
    main()
