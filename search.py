"""Stage 2's harness: score whole model-building recipes honestly, with nested cross-validation.

Where it fits: a search (try options, keep the best) flatters its own best score. So each
recipe runs in each of Stage 1's 15 outer rounds: it gets only that round's 3,200 training
grids, searches on 4 inner folds of them, refits, and predicts the 800 grids it never saw
(docs/stage2_models.md, Figure 1). The adoption test compares recipes on those predictions.
In: the course data via validity.load_dev(). Out: data/cache/nested/{name}-{advisor}.npy
(outer predictions, 3 shuffles x 4,000 grids) and .json (the run's record).

    python search.py --list | --run NAME | --compare NAME | --rule-2-2 | --reproduce

Surprises
* Saved results are never replaced: to change a recipe, give it a new name. Loading results
  whose recipe, data or folds changed raises. use_cache=False recomputes, without saving.
* Making outer predictions needs a clean, committed tree (commit before --run, --rule-2-2
  or --reproduce). The commit is saved with them: git history shows the recipe came first.
* Each round gets a fresh copy of the recipe: nothing learned in one round (whose training
  set holds other rounds' validation grids) reaches another.
* One thread per fit, rounds in separate processes: fastest here, identical predictions.
  Recipes reach workers by pickling, so one defined inside a function needs workers=1.
* Run time: the progress bar estimates it after the first wave of rounds (one per worker).
  An advisor's results are saved once its 15 rounds are in; to budget a run, read the
  estimate and stop it (Ctrl-C) before then if it is too slow.
"""

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import scipy.stats
import sklearn
from sklearn.model_selection import KFold

import features
import features_lab
import models_nn
import tables
import validity
from validity import ADVISORS, CV_FOLDS, CV_REPEATS

NESTED_DIR = os.path.join(validity.PS2_ROOT, "data", "cache", "nested")
INNER_FOLDS = 4
HP_LEVELS = (0.96, 0.97, 0.98, 0.99)  # the precision levels hp_recall averages over
# The adoption test (docs/decisions.md, Stage 2 protocol): corrected t at least the
# one-sided 5% point of Student's t on 14 degrees of freedom (15 folds), and at least 3%
# more approvals at 98% precision.
T_CRITICAL = 1.761
MARGIN = 0.03
MAX_PROCEDURES = 6  # recipes per advisor, the incumbent included ("Procedure cap")
BOOTSTRAP_DRAWS = 1000
ROUNDS = CV_FOLDS * CV_REPEATS  # outer rounds per advisor
WORKERS = os.cpu_count() or 1

# The owner's six slots (docs/decisions.md, 31 October), each filled by registering its
# recipe before any of its outer results exist. A rule 2.4 formula would take 5 or 6.
SLOTS = {
    1: "incumbent: Stage 1's GBM on the 75 engineered features",
    2: "rule 2.2's true-orientation arm: GBM on the engineered and one-hot inputs",
    3: "GBM choosing among feature groups",
    4: "neural network: a CNN search",
    5: "tuned GBM: slot 3's search plus GBM settings",
    6: "stacker or ensemble",
}

# ---- inner folds, and the scores a search and a comparison use -----------------------


def inner_seed(shuffle, fold):
    """The seed of one outer round's inner folds. New seeds, used nowhere else."""
    return 1000 + 10 * shuffle + fold


def inner_splits(n, shuffle, fold):
    """One outer round's 4 inner folds of its n training grids, as (train, val) positions.
    Seeded by the round, so every recipe searches on the same folds there."""
    seed = inner_seed(shuffle, fold)
    return list(KFold(INNER_FOLDS, shuffle=True, random_state=seed).split(np.arange(n)))


def inner_predictions(fit_predict, grids, y, inner):
    """Each grid predicted by a fit on the other inner folds.
    fit_predict(training grids, their ratings, grids to predict) -> predictions."""
    pred = np.full(len(y), np.nan)
    for train, val in inner:
        pred[val] = fit_predict(grids[train], y[train], grids[val])
    return pred


def hp_recall(pred, y, levels=HP_LEVELS):
    """The share of truly passing grids approved, averaged over precision levels 96-99%:
    a share, so folds compare; averaged, so it doesn't hinge on two or three grids
    (docs/decisions.md, Stage 2 protocol). NaN if no grid passes."""
    passing = int((np.asarray(y) >= validity.THRESHOLD).sum())
    if not passing:
        return np.nan
    hits = [validity.approved_at(pred, y, level)[2] for level in levels]
    return float(np.mean(hits)) / passing


def _per_fold(score, preds, y):
    """score(pred, y) in each outer fold, from (shuffles, n) outer predictions."""
    folds = validity.cv_splits(len(y), repeats=len(preds))
    return np.array([score(preds[r, val], y[val]) for r, _, _, val in folds])


def _mse(pred, y):
    return np.mean((pred - y) ** 2)


def _fold_sizes(n):
    """The outer folds' mean training and validation sizes: each shuffle's folds split the
    n grids, so these are exact."""
    return n - n / CV_FOLDS, n / CV_FOLDS


def corrected_t(diffs, n_train, n_val):
    """Nadeau and Bengio's corrected paired t. Any two outer training sets share most of
    their grids, so fold results rise and fall together; the ordinary t would treat them as
    independent. The correction adds n_val / n_train to 1 / (number of folds)."""
    d = np.asarray(diffs, float)
    var = d.var(ddof=1)
    if var == 0:  # every fold moved by the same amount
        return math.copysign(math.inf, d.mean()) if d.mean() else 0.0
    return float(d.mean() / np.sqrt((1 / len(d) + n_val / n_train) * var))


# ---- recipes -------------------------------------------------------------------------

INPUT_VERSIONS = {
    "engineered": features.FEATURE_VERSION,
    "onehot": features.ONEHOT_VERSION,
}


@dataclass(frozen=True)
class FixedGBM:
    """GBM with Stage 1's settings on fixed inputs: nothing to search, so the refit is the
    model. Every recipe has slot, summary, advisors and these methods:
        search(grids, y, inner) -> (choice, record)  inner folds only; JSON values; the
                                                     record includes inner_hp_recall
        fit(choice, grids, y) -> model; predict(model, grids) -> predictions
        spec() -> its settings, saved with its results and checked when they are loaded
    """

    inputs: tuple  # names in validity.FEATURIZERS, joined side by side in this order
    slot: int
    summary: str
    advisors: tuple = ADVISORS

    def spec(self):
        inputs = [[name, INPUT_VERSIONS[name]] for name in self.inputs]
        params = validity.GBM_PARAMS
        return {"kind": "fixed_gbm", "inputs": inputs, "gbm_params": params}

    def model_inputs(self, grids):
        return np.hstack([validity.FEATURIZERS[name](grids) for name in self.inputs])

    def search(self, grids, y, inner):
        # Nothing to choose; the inner score is still recorded, for the protocol's
        # "selection by inner score" check (docs/decisions.md, Stage 2 protocol).
        def fit_predict(g_fit, y_fit, g_new):
            return self.predict(self.fit(None, g_fit, y_fit), g_new)

        pred = inner_predictions(fit_predict, grids, y, inner)
        return None, {"inner_hp_recall": hp_recall(pred, y)}

    def fit(self, choice, grids, y):
        import lightgbm as lgb  # here, so tests that never fit needn't pay for it

        # One thread per fit: see the header.
        model = lgb.LGBMRegressor(**validity.GBM_PARAMS, n_jobs=1)
        return model.fit(self.model_inputs(grids), y)

    def predict(self, model, grids):
        return model.predict(self.model_inputs(grids))


def _gbm(X, y):
    import lightgbm as lgb

    return lgb.LGBMRegressor(**validity.GBM_PARAMS, n_jobs=1).fit(
        X, y
    )  # see the header


@dataclass(frozen=True)
class GroupSearchGBM:
    """Slot 3: Stage 1's GBM on the 75 features plus up to max_added candidate groups,
    chosen by forward selection on the inner folds: add the group that raises inner
    hp_recall most, while it raises it by at least min_gain. Same interface as FixedGBM."""

    slot: int
    summary: str
    groups: tuple  # candidates: "onehot", or a features_lab.LAB_GROUPS name
    max_added: int = 3
    min_gain: float = 0.005  # smaller gains are within inner-fold noise
    advisors: tuple = ADVISORS

    def spec(self):
        return {
            "kind": "group_search_gbm",
            "versions": [
                features.FEATURE_VERSION,
                features.ONEHOT_VERSION,
                features_lab.LAB_VERSION,
            ],
            "groups": list(self.groups),
            "max_added": self.max_added,
            "min_gain": self.min_gain,
            "gbm_params": validity.GBM_PARAMS,
        }

    def blocks(self, grids):
        """{"base" or group: its columns} for these grids."""
        lab = features_lab.lab(grids)
        out = {g: lab[:, idx] for g, idx in features_lab.LAB_GROUPS.items()}
        return {
            "base": features.engineered(grids),
            "onehot": features.onehot(grids),
        } | out

    def search(self, grids, y, inner):
        blocks = self.blocks(grids)

        def score(chosen):
            X = np.hstack([blocks[g] for g in ("base", *chosen)])
            pred = inner_predictions(lambda a, b, c: _gbm(a, b).predict(c), X, y, inner)
            return hp_recall(pred, y)

        chosen, best = [], score([])
        steps = [{"added": None, "inner_hp_recall": best}]
        while len(chosen) < self.max_added:
            tried = {g: score([*chosen, g]) for g in self.groups if g not in chosen}
            group = max(tried, key=tried.get)  # a tie goes to the earlier group
            if tried[group] < best + self.min_gain:
                break
            chosen.append(group)
            best = tried[group]
            steps.append({"added": group, "inner_hp_recall": best, "tried": tried})
        return chosen, {"inner_hp_recall": best, "steps": steps}

    def fit(self, choice, grids, y):
        blocks = self.blocks(grids)
        return choice, _gbm(np.hstack([blocks[g] for g in ("base", *choice)]), y)

    def predict(self, model, grids):
        choice, gbm = model
        blocks = self.blocks(grids)
        return gbm.predict(np.hstack([blocks[g] for g in ("base", *choice)]))


# Slot 3's candidates, in tie-break order: the owner's ideas by place first, then the
# groups the 1 Nov screen suggested (docs/decisions.md), then one-hot cells.
SLOT3_GROUPS = (
    *features_lab.LAB_GROUPS,
    "onehot",
)

# Slot 5's settings search (Optuna). Ranges are log-scaled where marked. The first trial is
# Stage 1's settings (validity.GBM_PARAMS), so tuning starts from what is known to work.
SLOT5_SPACE = {
    "library": ["lightgbm", "xgboost"],
    "n_estimators": [200, 2000],  # log
    "learning_rate": [0.01, 0.1],  # log
    "leaves": [8, 128],  # LightGBM's tree size, log
    "depth": [3, 10],  # XGBoost's tree size
    "subsample": [0.5, 1.0],  # share of grids per tree
    "colsample": [0.5, 1.0],  # share of features per tree
}
STAGE1_SETTINGS = {"library": "lightgbm", "n_estimators": 600, "learning_rate": 0.03}
STAGE1_SETTINGS |= {"leaves": 31, "subsample": 0.8, "colsample": 0.8}


def _suggest(trial):
    """One trial's settings, drawn by Optuna from SLOT5_SPACE."""
    space, out = SLOT5_SPACE, {}
    out["library"] = trial.suggest_categorical("library", space["library"])
    for key in ("n_estimators", "learning_rate"):
        kind = trial.suggest_int if key == "n_estimators" else trial.suggest_float
        out[key] = kind(key, *space[key], log=True)
    if out["library"] == "lightgbm":
        out["leaves"] = trial.suggest_int("leaves", *space["leaves"], log=True)
    else:
        out["depth"] = trial.suggest_int("depth", *space["depth"])
    for key in ("subsample", "colsample"):
        out[key] = trial.suggest_float(key, *space[key])
    return out


def _model(settings):
    """An unfitted GBM with these settings; one thread (see the header)."""
    common = {"n_estimators": settings["n_estimators"], "random_state": 0, "n_jobs": 1}
    common |= {
        "learning_rate": settings["learning_rate"],
        "subsample": settings["subsample"],
    }
    common |= {"colsample_bytree": settings["colsample"]}
    if settings["library"] == "lightgbm":
        import lightgbm as lgb

        leaves = settings["leaves"]
        return lgb.LGBMRegressor(
            **common, num_leaves=leaves, subsample_freq=1, verbose=-1
        )
    import xgboost as xgb

    return xgb.XGBRegressor(**common, max_depth=settings["depth"], tree_method="hist")


@dataclass(frozen=True)
class TunedGroupGBM(GroupSearchGBM):
    """Slot 5: slot 3's group search, then `trials` Optuna trials of GBM settings on the
    chosen groups, each scored by inner hp_recall; the best settings are refit. `start`
    groups join the 75 features before the search, so an advisor keeps its current inputs
    (Wellness: one-hot cells)."""

    trials: int = 40
    start: tuple = ()

    def spec(self):
        extra = {
            "space": SLOT5_SPACE,
            "first_trial": STAGE1_SETTINGS,
            "trials": self.trials,
            "start": list(self.start),
        }
        return super().spec() | {"kind": "tuned_group_gbm"} | extra

    def blocks(self, grids):
        blocks = super().blocks(grids)
        blocks["base"] = np.hstack([blocks["base"], *(blocks[g] for g in self.start)])
        return blocks

    def library_versions(self):
        import optuna
        import xgboost

        return {"optuna": optuna.__version__, "xgboost": xgboost.__version__}

    def _inputs(self, grids, groups):
        blocks = self.blocks(grids)
        return np.hstack([blocks[g] for g in ("base", *groups)])

    def search(self, grids, y, inner):
        import optuna

        groups, record = super().search(grids, y, inner)
        X = self._inputs(grids, groups)

        def score(trial):
            settings = _suggest(trial)
            fit = lambda a, b, c: _model(settings).fit(a, b).predict(c)
            return hp_recall(inner_predictions(fit, X, y, inner), y)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=0)  # same draws for the same scores
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.enqueue_trial(STAGE1_SETTINGS)
        study.optimize(score, n_trials=self.trials)
        tried = [t.params | {"inner_hp_recall": t.value} for t in study.trials]
        record = record | {"groups_inner_hp_recall": record["inner_hp_recall"]}
        record |= {"inner_hp_recall": study.best_value, "tried": tried}
        return {"groups": groups, "settings": study.best_params}, record

    def fit(self, choice, grids, y):
        X = self._inputs(grids, choice["groups"])
        return choice, _model(choice["settings"]).fit(X, y)

    def predict(self, model, grids):
        choice, gbm = model
        return gbm.predict(self._inputs(grids, choice["groups"]))


@dataclass(frozen=True)
class Ensemble:
    """Slot 6: the mean of several recipes' predictions, each searched and fitted exactly as
    it is alone (same inner folds, so the same choices). Same interface as FixedGBM."""

    members: tuple  # REGISTRY names
    slot: int
    summary: str
    advisors: tuple = ADVISORS

    def spec(self):
        return {
            "kind": "ensemble",
            "members": {m: REGISTRY[m].spec() for m in self.members},
        }

    def library_versions(self):
        found = (getattr(REGISTRY[m], "library_versions", dict)() for m in self.members)
        return {key: value for versions in found for key, value in versions.items()}

    def search(self, grids, y, inner):
        choices, records = {}, {}
        for m in self.members:
            choices[m], records[m] = REGISTRY[m].search(grids, y, inner)
        # The mean's own inner score would refit every member on every inner fold, and it
        # decides nothing, so only the members' records are kept.
        return choices, {"inner_hp_recall": float("nan"), "members": records}

    def fit(self, choice, grids, y):
        return {m: REGISTRY[m].fit(choice[m], grids, y) for m in self.members}

    def predict(self, model, grids):
        return np.mean(
            [REGISTRY[m].predict(model[m], grids) for m in self.members], axis=0
        )


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
    # Slot 3, registered 1 November (owner's choice: forward selection, up to 3 groups).
    "gbm_groups": GroupSearchGBM(
        slot=3,
        summary="GBM on the 75 features plus up to 3 feature groups, chosen per round",
        groups=SLOT3_GROUPS,
    ),
    # Slot 5, registered 1 November: slot 3's search, then Optuna (owner's choice). Each
    # advisor starts from its current inputs: Wellness keeps its one-hot cells; for the
    # others, the 75 features (Business's current model is slot 3's search itself).
    "gbm_tuned": TunedGroupGBM(
        slot=5,
        summary="slot 3's group search, then 40 Optuna trials of GBM library and settings",
        groups=SLOT3_GROUPS,
        advisors=("Tax", "Transportation", "Business"),
    ),
    "gbm_tuned_onehot": TunedGroupGBM(
        slot=5,
        summary="as gbm_tuned, but one-hot cells always join the 75 features",
        groups=tuple(g for g in SLOT3_GROUPS if g != "onehot"),
        advisors=("Wellness",),
        start=("onehot",),
    ),
    # Slot 6, registered 4 November (owner's choice): the mean of the advisor's GBM recipes.
    # The CNN stays out: it is the independent check model, so it can't be in the primary one.
    "gbm_ensemble": Ensemble(
        ("gbm_engineered", "gbm_engineered_onehot", "gbm_groups", "gbm_tuned"),
        slot=6,
        summary="the mean of slots 1, 2, 3 and 5's predictions",
        advisors=("Tax", "Transportation", "Business"),
    ),
    "gbm_ensemble_onehot": Ensemble(
        ("gbm_engineered", "gbm_engineered_onehot", "gbm_groups", "gbm_tuned_onehot"),
        slot=6,
        summary="as gbm_ensemble, with Wellness's slot 5 (gbm_tuned_onehot)",
        advisors=("Wellness",),
    ),
    # Slot 4, registered 31 October after rule 2.2: position matters for Wellness, Tax and
    # Business, so their CNN may not train on rotated copies; Transportation's may.
    "cnn": models_nn.CNNSearch(
        slot=4,
        summary="CNN search; no rotated training copies",
        advisors=("Wellness", "Tax", "Business"),
        augment_allowed=False,
    ),
    "cnn_with_rotations": models_nn.CNNSearch(
        slot=4,
        summary="CNN search; rotated training copies are one of its options",
        advisors=("Transportation",),
        augment_allowed=True,
    ),
}

# Each advisor's recipe in force; changes only with an adoption in docs/decisions.md.
INCUMBENT = dict.fromkeys(ADVISORS, "gbm_engineered")
# Adoptions (docs/decisions.md): 1 Nov, at the end of Sessions 1 and 2.
INCUMBENT["Wellness"] = "gbm_engineered_onehot"
INCUMBENT["Business"] = "gbm_groups"


def procedures_for(advisor):
    """{name: recipe} registered for one advisor."""
    return {name: p for name, p in REGISTRY.items() if advisor in p.advisors}


REORIENT_SEEDS = (1, 2, 3, 4, 5)  # rule 2.2's five draws


def reorient(grids, seed):
    """Each grid turned or mirrored into one of its 7 other orientations, picked from the
    seed and the grid's own cells, so a grid turns the same way in every fold."""
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
    """Rule 2.2's control: slot 2's GBM with its one-hot inputs (each cell's position)
    taken from re-oriented grids. The engineered features ignore orientation."""

    seed: int = 1

    def spec(self):
        return {**super().spec(), "kind": "reoriented_gbm", "seed": self.seed}

    def model_inputs(self, grids):
        turned = reorient(grids, self.seed)
        return np.hstack([features.engineered(grids), features.onehot(turned)])


# Diagnostics run through the harness like recipes, but can't be adopted and don't count
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


def recipe(name, advisor=None):
    """The registered recipe or diagnostic `name`; with `advisor`, only if it is for them."""
    found = REGISTRY.get(name) or DIAGNOSTICS.get(name)
    if found is None or advisor not in (None, *found.advisors):
        raise KeyError(f"{name!r} is not registered for {advisor or 'any advisor'}")
    return found


def describe(procedure):
    """What a run tests, for its kick-off printout: a GBM's inputs or a CNN's settings."""
    lines = [f"  {procedure.summary}"]
    if isinstance(procedure, FixedGBM):
        inputs = " + ".join(procedure.inputs)
        lines.append(f"  GBM, Stage 1's settings, inputs {inputs}: nothing to search")
    elif isinstance(procedure, GroupSearchGBM):
        lines.append(
            f"  Forward selection on the inner folds: add the best group while inner "
            f"hp_recall rises by {procedure.min_gain}+, at most {procedure.max_added}. "
            f"{len(procedure.groups)} candidates:"
        )
        lines.append("    " + ", ".join(procedure.groups))
        if isinstance(procedure, TunedGroupGBM):
            if procedure.start:
                lines.append(f"  Always included: {', '.join(procedure.start)}")
            lines.append(
                f"  Then {procedure.trials} Optuna trials on the chosen groups:"
            )
            lines += [f"    {key:<14} {values}" for key, values in SLOT5_SPACE.items()]
    elif isinstance(procedure, Ensemble):
        lines.append(f"  The mean of: {', '.join(procedure.members)}")
    elif isinstance(procedure, models_nn.CNNSearch):
        allowed = procedure.augment_allowed
        n = len(models_nn.settings(allowed))
        lines.append(f"  Each round tries {min(procedure.trials, n)} of {n} settings:")
        for key, values in models_nn.SPACE.items():
            if key != "augment" or allowed:
                lines.append(f"    {key:<9} {', '.join(map(str, values))}")
        stop = f"{models_nn.PATIENCE} epochs without gain, or at {models_nn.MAX_EPOCHS}"
        lines.append(f"  Inner fits stop after {stop}. Best inner hp_recall wins.")
    return "\n".join(lines)


# ---- one outer round -----------------------------------------------------------------


def _round_data(grids, y, train, val):
    """A recipe's data in one round: training grids with ratings, validation grids only."""
    return grids[train], y[train], grids[val]


def _fresh(procedure):
    """A new copy of the recipe for each round, so nothing it learns in one round (whose
    training set holds other rounds' validation grids) can reach another."""
    return copy.deepcopy(procedure)


def _json_default(value):
    """Lets NumPy numbers and arrays into a JSON record."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    raise TypeError(f"a {type(value).__name__} can't be saved in a run's record")


HARNESS_KEYS = {"shuffle", "fold", "choice", "seconds"}  # set by _run_round itself


def _run_round(procedure, grids, y, shuffle, fold, train, val):
    """One outer round: search on the inner folds, refit on the outer training set,
    predict the outer-validation grids. Returns (predictions, record)."""
    start = time.perf_counter()
    procedure = _fresh(procedure)
    g_train, y_train, g_val = _round_data(grids, y, train, val)
    inner = inner_splits(len(y_train), shuffle, fold)
    choice, record = procedure.search(g_train, y_train, inner)
    if "inner_hp_recall" not in record or HARNESS_KEYS & set(record):
        raise ValueError(f"a record needs inner_hp_recall, and none of {HARNESS_KEYS}")
    model = procedure.fit(choice, g_train, y_train)
    pred = np.asarray(procedure.predict(model, g_val), dtype=float)
    if pred.shape != (len(val),) or not np.isfinite(pred).all():
        raise ValueError(f"round {shuffle}.{fold}: not {len(val)} finite predictions")
    full = {"shuffle": shuffle, "fold": fold, "choice": choice, **record}
    full["seconds"] = round(time.perf_counter() - start, 3)
    json.dumps(full, default=_json_default)  # fail in this round, not after the run
    return pred, full


def _execute(jobs, workers, done):
    """_run_round over jobs, calling done(index, result) as each finishes: in order here if
    workers == 1, else in fresh processes. An error cancels the rounds not yet started."""
    if workers == 1:
        for i, job in enumerate(jobs):
            done(i, _run_round(*job))
        return
    # "spawn" starts clean processes. Forked ones would inherit this process's thread
    # pools, which can hang LightGBM on Linux.
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


# ---- saved results: {name}-{advisor}.npy and .json, never replaced -------------------


def run_context():
    """The data and fold settings all results rest on: saved, and checked on loading. (The
    lockbox seed decides which grids are development grids.)"""
    seeds = [inner_seed(r, f) for r in range(CV_REPEATS) for f in range(CV_FOLDS)]
    data = {"course_pin": validity.course_pin(), "lockbox_seed": validity.LOCKBOX_SEED}
    outer = {"cv_folds": CV_FOLDS, "cv_seeds": list(validity.CV_SEEDS)}
    inner = {"inner_folds": INNER_FOLDS, "inner_seeds": seeds}
    scoring = {"hp_levels": list(HP_LEVELS), "threshold": validity.THRESHOLD}
    return data | outer | inner | scoring


def _versions(procedure):
    """Library versions. A CNN recipe adds PyTorch's (CNNSearch.library_versions)."""
    import lightgbm

    libraries = {"numpy": np, "scipy": scipy, "sklearn": sklearn, "lightgbm": lightgbm}
    versions = {name: library.__version__ for name, library in libraries.items()}
    versions["python"] = ".".join(map(str, sys.version_info[:2]))
    return {**versions, **getattr(procedure, "library_versions", dict)()}


def _spec(procedure):
    """The recipe's settings as they read back from JSON (tuples become lists)."""
    return json.loads(json.dumps(procedure.spec()))


def _path(name, advisor):
    """Where a recipe's results for one advisor live, without the .npy or .json ending."""
    return os.path.join(NESTED_DIR, f"{name}-{advisor.lower()}")


def _save(name, advisor, preds, record):
    """Predictions, then the record: results exist once the record does. Open mode "x"
    refuses to replace either file."""
    text = json.dumps(record, indent=1, default=_json_default)  # before opening a file
    os.makedirs(NESTED_DIR, exist_ok=True)
    with open(_path(name, advisor) + ".npy", "xb") as f:
        np.save(f, preds)
    with open(_path(name, advisor) + ".json", "x") as f:
        f.write(text)


def load_outer(name, advisor):
    """(predictions, record) saved for recipe `name` and one advisor; predictions are
    (3 shuffles, n grids), in _advisor_data's order. Raises if the recipe's settings or
    its data and folds have changed since; only warns if library versions have."""
    path = _path(name, advisor)
    if not os.path.exists(path + ".json"):
        raise FileNotFoundError(f"no results of {name} for {advisor}: --run {name}")
    with open(path + ".json") as f:
        record = json.load(f)
    procedure = recipe(name, advisor)
    now = {"spec": _spec(procedure), "context": run_context()}
    changed = " and ".join(key for key, value in now.items() if record[key] != value)
    if changed:
        raise ValueError(
            f"{name} for {advisor}: recipe changed since its results were saved "
            f"({changed}): give it a new name"
        )
    then, versions = record["versions"], _versions(procedure)
    if then != versions:  # rarely changes predictions: say so, go on
        print(f"warning: {name} for {advisor} ran with {then}; now {versions}")
    if not record["unchanged_during_run"]:
        print(f"warning: the repository changed while {name} ran for {advisor}, so its")
        print(f"  saved commit, {record['commit'][:7]}, may not hold the code that ran")
    return np.load(path + ".npy"), record


# ---- outer predictions ---------------------------------------------------------------


def outer_predictions(name, data, workers=WORKERS, use_cache=True):
    """{advisor: (3, n) outer predictions} of recipe `name`; data is {advisor: (grids, y)}.

    Each prediction comes from a round whose recipe never saw that grid. Saved results are
    loaded; the rest are computed from a clean, committed tree, and each advisor's saved
    once its 15 rounds are in. use_cache=False recomputes them as a check, without saving.
    """
    saved = [a for a in data if use_cache and os.path.exists(_path(name, a) + ".json")]
    found = {advisor: load_outer(name, advisor)[0] for advisor in saved}
    for advisor, preds in found.items():  # results of other grids, under the same name?
        if preds.shape != (CV_REPEATS, len(data[advisor][1])):
            raise ValueError(f"{name} for {advisor}: saved for {preds.shape[1]} grids")
    todo = {advisor: recipe(name, advisor) for advisor in data if advisor not in found}
    if not todo:
        return found
    commit, uncommitted = validity._git_state()
    if uncommitted:
        raise PermissionError(
            "Outer predictions are made only from a clean, committed tree, so that the "
            f"saved commit holds the recipe. Not committed: {', '.join(uncommitted[:5])}"
        )
    # Checked before any round runs: a save cut short would make saving fail at the end,
    # and a changed recipe needs a new name for every advisor, not only those run now.
    for advisor in recipe(name).advisors if use_cache else ():
        path = _path(name, advisor)
        if advisor in todo and os.path.exists(path + ".npy"):
            raise FileExistsError(f"{path}.npy: no record (save cut short?); delete it")
        if advisor not in data and os.path.exists(path + ".json"):
            load_outer(name, advisor)  # raises if the recipe changed since
    jobs, places = [], []
    for advisor, procedure in todo.items():
        grids, y = data[advisor]
        for r, f, train, val in validity.cv_splits(len(y)):
            jobs.append((procedure, grids, y, r, f, train, val))
            places.append((advisor, r, val))
    preds = {a: np.full((CV_REPEATS, len(data[a][1])), np.nan) for a in todo}
    rounds = {a: [] for a in todo}
    started, tic = datetime.now(UTC), time.perf_counter()
    check = "" if use_cache else " (a check: nothing is saved)"
    print(f"{name}: starting {len(jobs)} outer rounds ({', '.join(todo)}){check}")
    print(f"  from commit {commit[:7]}, {min(workers, len(jobs))} at a time")
    print(describe(recipe(name)), flush=True)

    def save(advisor):
        procedure = todo[advisor]
        head, uncommitted = validity._git_state()  # again, just before saving
        unchanged = head == commit and not uncommitted
        record = {"name": name, "advisor": advisor, "slot": procedure.slot}
        record |= {"summary": procedure.summary, "spec": _spec(procedure)}
        record |= {"context": run_context(), "versions": _versions(procedure)}
        record |= {"commit": commit, "unchanged_during_run": unchanged}
        record |= {"run_from": validity._caller(), "workers": workers}
        record |= {"started": started.isoformat(timespec="seconds")}
        record |= {"seconds_until_saved": round(time.perf_counter() - tic, 1)}
        ordered = sorted(rounds[advisor], key=lambda rec: (rec["shuffle"], rec["fold"]))
        _save(name, advisor, preds[advisor], record | {"rounds": ordered})

    def done(i, result):
        advisor, r, val = places[i]
        preds[advisor][r, val], record = result
        rounds[advisor].append(record)
        if len(rounds[advisor]) == ROUNDS:
            found[advisor] = preds[advisor]
            if use_cache:
                save(advisor)  # before the progress line, which could fail
        _progress(name, sum(map(len, rounds.values())), len(jobs), tic)

    _execute(jobs, workers, done)
    return found


def _progress(name, finished, total, tic):
    """A bar, the time so far and a rough time left (rough at first: the first rounds
    run side by side and all finish late)."""
    elapsed = time.perf_counter() - tic
    left = elapsed / finished * (total - finished)
    bar = f"[{'#' * round(20 * finished / total):<20}] {finished}/{total} rounds"
    times = f"{elapsed / 60:.0f} min so far, about {left / 60:.0f} min left"
    try:  # a closed terminal must not stop a long run
        print(f"{name}: {bar}, {times}", flush=True)
    except OSError:
        pass


def _advisor_data(grids, dev, advisor):
    """One advisor's development grids, ratings and row numbers (as validity.scorecard)."""
    a = ADVISORS.index(advisor)
    rows = np.flatnonzero(~np.isnan(dev[:, a]))
    return grids[rows], dev[rows, a], rows


def run(name, advisors=None, workers=WORKERS, use_cache=True):
    """{advisor: (ratings, outer predictions)} of recipe `name` on the development set,
    for `advisors` (default: the recipe's own)."""
    grids, dev = validity.load_dev()
    advisors = advisors or recipe(name).advisors
    data = {advisor: _advisor_data(grids, dev, advisor)[:2] for advisor in advisors}
    preds = outer_predictions(name, data, workers, use_cache)
    return {advisor: (data[advisor][1], preds[advisor]) for advisor in data}


# ---- the adoption test ---------------------------------------------------------------


@dataclass
class Comparison:
    """A challenger against the incumbent on one advisor (docs/decisions.md, Stage 2
    protocol). Pairs are (challenger, incumbent); approval counts are at 98% precision,
    means over the 3 shuffles; intervals are 95% paired bootstraps over grids."""

    t: float  # corrected paired t on the 15 per-fold hp_recall differences
    folds_better: int  # folds where the challenger's hp_recall is higher
    hp_recall_diff: float  # mean per-fold difference, challenger minus incumbent
    hp_recall_interval: tuple
    approved: tuple
    approved_interval: tuple  # of the difference in approval counts, in grids

    def __post_init__(self):  # the decision
        mine, theirs = self.approved
        self.gain = mine / theirs - 1 if theirs else math.inf
        self.passes_test = bool(self.t >= T_CRITICAL)
        self.passes_margin = bool(mine >= (1 + MARGIN) * theirs)
        self.adopt = self.passes_test and self.passes_margin


def _mean_approved(preds, y):
    return float(np.mean([validity.approved_at(p, y)[0] for p in preds]))


def compare_predictions(challenger, incumbent, y, draws=BOOTSTRAP_DRAWS):
    """The adoption test on two recipes' (3, n) outer predictions for one advisor."""
    hp_c, hp_i = _per_fold(hp_recall, challenger, y), _per_fold(hp_recall, incumbent, y)
    t = corrected_t(hp_c - hp_i, *_fold_sizes(len(y)))
    better, diff = int((hp_c > hp_i).sum()), float(np.mean(hp_c - hp_i))
    approved = (_mean_approved(challenger, y), _mean_approved(incumbent, y))
    hp_interval, approved_interval = _bootstrap(challenger, incumbent, y, draws)
    return Comparison(t, better, diff, hp_interval, approved, approved_interval)


def _bootstrap(challenger, incumbent, y, draws, seed=0):
    """95% paired-bootstrap intervals of (the mean per-fold hp_recall difference, the
    difference in approval counts). Each draw resamples the n grids once, with
    replacement, and computes both numbers for both recipes on that same resample."""
    rng, n = np.random.default_rng(seed), len(y)
    folds = list(validity.cv_splits(n, repeats=len(challenger)))
    hp, counts = [], []
    for _ in range(draws):
        w = np.bincount(rng.integers(0, n, n), minlength=n)  # times each grid was drawn
        diffs = []
        for r, _, _, val in folds:
            pick = np.repeat(val, w[val])
            mine, theirs = challenger[r, pick], incumbent[r, pick]
            diffs.append(hp_recall(mine, y[pick]) - hp_recall(theirs, y[pick]))
        hp.append(np.mean(diffs))
        pick = np.repeat(np.arange(n), w)  # every fold's picks together
        mine, theirs = challenger[:, pick], incumbent[:, pick]
        counts.append(_mean_approved(mine, y[pick]) - _mean_approved(theirs, y[pick]))
    intervals = [np.percentile(values, [2.5, 97.5]) for values in (hp, counts)]
    return tuple(tuple(float(v) for v in interval) for interval in intervals)


def compare(name, against=None, advisors=None):
    """{advisor: (incumbent's name, Comparison)} for recipe `name`, from saved results.
    `against` defaults to each advisor's incumbent, `advisors` to the recipe's own."""
    if {name, against} & set(DIAGNOSTICS):
        raise ValueError("rule 2.2's diagnostics can't be compared: never adopted")
    grids, dev = validity.load_dev()
    out = {}
    for advisor in advisors or recipe(name).advisors:
        y, other = _advisor_data(grids, dev, advisor)[1], against or INCUMBENT[advisor]
        if other == name:  # already adopted: nothing to compare it with
            continue
        mine, theirs = load_outer(name, advisor)[0], load_outer(other, advisor)[0]
        out[advisor] = (other, compare_predictions(mine, theirs, y))
    return out


def group_report(name, advisors):
    """Slot 3's extras, printed and returned as sheets: how many rounds chose each group, and
    the approvals on shuffle 1's folds 4-5 alone, the grids its candidate screens never read
    (docs/decisions.md, "Slot 3 registered"). Advisors that adopted the recipe are left out
    of the folds 4-5 check: it was made against their model from before the switch."""
    grids, dev = validity.load_dev()
    counts, clean = {}, []
    for advisor in advisors:
        # Slot 3's choice is a list of groups; slot 5's a dict that also holds settings.
        choices = [r["choice"] for r in load_outer(name, advisor)[1]["rounds"]]
        picks = [
            g for c in choices for g in (c["groups"] if isinstance(c, dict) else c)
        ]
        counts[advisor] = {g: picks.count(g) for g in recipe(name).groups}
        if INCUMBENT[advisor] == name:
            continue
        y = _advisor_data(grids, dev, advisor)[1]
        folds = validity.cv_splits(len(y))
        val = np.concatenate([v for r, f, _, v in folds if r == 0 and f >= 3])
        mine, theirs = (
            load_outer(n, advisor)[0][0, val] for n in (name, INCUMBENT[advisor])
        )
        n_mine, n_theirs = (validity.approved_at(p, y[val])[0] for p in (mine, theirs))
        gain = round(100 * (n_mine / max(n_theirs, 1) - 1), 1)
        clean.append({"Advisor": advisor, "Grids": len(val), "Approved": n_mine})
        clean[-1] |= {"Approved, current model": n_theirs, "Gain %": gain}
    print("\nGroups chosen, rounds of 15:")
    for advisor, n in counts.items():
        top = sorted((g for g in n if n[g]), key=n.get, reverse=True)
        print(f"  {advisor:<15} {', '.join(f'{g} {n[g]}' for g in top) or 'none'}")
    print("Shuffle 1, folds 4-5 only (never screened), approved at 98%:")
    for r in clean:
        print(
            f"  {r['Advisor']:<15} {r['Approved']} vs {r['Approved, current model']}, {r['Gain %']:+.1f}%"
        )
    grid = [
        {"Group": g} | {a: counts[a][g] for a in advisors} for g in recipe(name).groups
    ]
    return {"Groups chosen (rounds)": grid, "Folds 4-5 only": clean}


# ---- rule 2.2: does a cell's position matter? ----------------------------------------


def orientation_effect(true, controls, y):
    """(corrected t, folds where the true orientation predicts better), for one advisor.
    Per fold: the control draws' mean squared errors, averaged, minus the true arm's, so a
    positive difference favours the true orientation. Predictions aren't averaged."""
    control = np.mean([_per_fold(_mse, c, y) for c in controls], axis=0)
    diffs = control - _per_fold(_mse, true, y)
    return corrected_t(diffs, *_fold_sizes(len(y))), int((diffs > 0).sum())


def rule_2_2(advisors=ADVISORS, workers=WORKERS):
    """{advisor: orientation_effect}. Position matters if t >= T_CRITICAL: one-sided 5%,
    no correction across advisors, because a finding leads to the cautious action."""
    true = run("gbm_engineered_onehot", advisors, workers)
    names = [name for name in DIAGNOSTICS if name.startswith("rule22_")]
    controls = [run(name, advisors, workers) for name in names]
    return {
        a: orientation_effect(true[a][1], [c[a][1] for c in controls], true[a][0])
        for a in advisors
    }


def orientation_patterns(grids):
    """Label-free: do the grids favour some cells over the cells that a rotation or mirror
    maps them onto? Per set of such cells (the 4 corners, ...), a chi-square statistic of
    how district counts differ, summed: (chi-square, degrees of freedom, largest gap in a
    district's share within a set). No p-value: a grid's cells aren't independent."""
    grids = np.asarray(grids)
    images = features.orientations(np.arange(49).reshape(7, 7))  # where each cell goes
    sets = {tuple(sorted(set(images[:, r, c]))) for r in range(7) for c in range(7)}
    districts = range(features.N_DISTRICTS)
    counts = np.stack([(grids == k).sum(0).ravel() for k in districts], axis=1)
    chi2, dof, gap = 0.0, 0, 0.0
    for cells in sets:
        if len(cells) < 2:
            continue  # the centre cell maps only onto itself
        table = counts[list(cells)]
        stat, _, df, _ = scipy.stats.chi2_contingency(table)
        chi2, dof = chi2 + stat, dof + int(df)
        shares = table / table.sum(1, keepdims=True)
        gap = max(gap, float((shares.max(0) - shares.min(0)).max()))
    return chi2, dof, gap


def reproduce_stage1(workers=WORKERS):
    """True if Stage 1's GBM through this harness gives exactly Stage 1's predictions,
    and the incumbent's saved results equal them. Both are recomputed (not read from
    data/cache/oof/), so a later harness edit can't hide behind saved predictions."""
    saved = run("gbm_engineered", workers=workers)  # loaded, or made and saved now
    fresh = run("gbm_engineered", workers=workers, use_cache=False)  # not saved
    grids, dev = validity.load_dev()
    print("\nStage 1's GBM through the harness, against validity.out_of_fold:")
    same = True
    for advisor, (y, mine) in fresh.items():
        g, _, rows = _advisor_data(grids, dev, advisor)
        stage1 = validity.out_of_fold("gbm_engineered", g, y, rows, use_cache=False)
        identical = all(np.array_equal(mine, p) for p in (stage1, saved[advisor][1]))
        counts = "/".join(str(validity.approved_at(p, y)[0]) for p in mine)
        verdict = "identical" if identical else "DIFFERENT"
        print(f"  {advisor:<15} {verdict:<10} approved at 98%: {counts} per shuffle")
        same &= identical
    print("PASS" if same else "FAIL: the harness does not reproduce Stage 1")
    return same


# ---- printing ------------------------------------------------------------------------


def print_registry():
    print(f"Recipes per advisor: at most {MAX_PROCEDURES}, the incumbent included")
    for slot, description in SLOTS.items():
        print(f"\nSlot {slot}  {description}")
        filled = {n: p for n, p in REGISTRY.items() if p.slot == slot}
        if not filled:
            print("        not registered yet")
        for name, procedure in filled.items():
            print(f"        {name}: {procedure.summary}")
            for advisor in procedure.advisors:
                notes = ["incumbent"] if INCUMBENT[advisor] == name else []
                if os.path.exists(_path(name, advisor) + ".json"):
                    notes.append("has outer results")
                print(f"          {advisor:<15} {', '.join(notes)}")
    print("\nDiagnostics (never adopted, outside the cap):")
    for name, diagnostic in DIAGNOSTICS.items():
        print(f"        {name}: {diagnostic.summary}")


def print_outer(name, results):
    print(f"\n{name}: outer predictions, 15 rounds per advisor; approvals per shuffle")
    rows = []
    for advisor, (y, preds) in results.items():
        counts = "/".join(str(validity.approved_at(p, y)[0]) for p in preds)
        hp = _per_fold(hp_recall, preds, y).mean()
        print(f"  {advisor:<15} approved at 98%: {counts}; hp_recall {hp:.3f}")
        rows.append({"Advisor": advisor, "Approved at 98%, per shuffle": counts})
        rows[-1]["hp_recall"] = hp
    print("Saved", tables.save(f"run_{name}", {"Outer results": rows}))


def print_comparisons(name, comparisons):
    rule = f"adopted if t >= {T_CRITICAL} and at least {MARGIN:.0%} more approved"
    print(f"\n{name} against the incumbent: {rule} at 98% precision [95% intervals]")
    rows = []
    for advisor, (other, c) in comparisons.items():
        (low, high), (fewest, most) = c.hp_recall_interval, c.approved_interval
        hp = f"{c.hp_recall_diff:+.3f} [{low:+.3f}, {high:+.3f}]"
        more = f"{c.approved[0] - c.approved[1]:+.1f} [{fewest:+.0f}, {most:+.0f}]"
        approved = f"{c.approved[0]:.1f} vs {c.approved[1]:.1f}, {c.gain:+.1%}"
        better = f"higher in {c.folds_better} of {ROUNDS} folds"
        print(f"  {advisor} against {other}: adopt {'yes' if c.adopt else 'no'}")
        print(f"    t {c.t:.2f}; hp_recall {hp}, {better}")
        print(f"    approved {approved}; {more} grids")
        row = {
            "Advisor": advisor,
            "Against": other,
            "Adopt": "yes" if c.adopt else "no",
        }
        row |= {"t": round(c.t, 2), "Folds better (of 15)": c.folds_better}
        row |= {
            "Approved": round(c.approved[0], 1),
            "Approved, current model": round(c.approved[1], 1),
        }
        row |= {
            "Gain %": round(100 * c.gain, 1),
            "Extra grids": round(c.approved[0] - c.approved[1], 1),
        }
        row |= {
            "Extra grids, 95% low": round(fewest),
            "Extra grids, 95% high": round(most),
        }
        row |= {"hp_recall difference": round(c.hp_recall_diff, 3)}
        rows.append(
            row
            | {
                "hp_recall, 95% low": round(low, 3),
                "hp_recall, 95% high": round(high, 3),
            }
        )
    for advisor in set(recipe(name).advisors) - set(comparisons):
        print(f"  {advisor}: {name} is its current model (adopted), so not compared")
    print("Decided at the end of the session, against the incumbent at its start.")
    sheets = {"Adoption test": rows}
    if isinstance(recipe(name), GroupSearchGBM):
        sheets |= group_report(name, list(recipe(name).advisors))
    # A dated name: rerunning after an adoption mustn't overwrite the record of it.
    print(
        "Saved", tables.save(f"compare_{name}_{datetime.now():%Y-%m-%d_%H%M}", sheets)
    )


def print_rule_2_2(results, patterns, n_grids):
    print("\nRule 2.2: does position matter? Slot 2's GBM, true against re-oriented")
    for advisor, (t, better) in results.items():
        verdict = "matters" if t >= T_CRITICAL else "doesn't matter"
        print(f"  {advisor:<15} t {t:5.2f}, better in {better} of {ROUNDS}: {verdict}")
    chi2, dof, gap = patterns  # no p-value: a grid's cells aren't independent
    print(f"Label-free, {n_grids:,} grids: chi-square {chi2:,.0f}, {dof} degrees of")
    print("freedom. Largest gap in a district's share between cells that a rotation or")
    print(f"mirror maps onto each other: {gap:.2%}. Where position matters: no rotated")
    print("or mirrored copies in training or in submissions (rules 2.2 and 4.2).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    do = ap.add_mutually_exclusive_group(required=True)
    do.add_argument("--list", action="store_true", help="the recipes, by slot")
    do.add_argument("--run", metavar="NAME", help="outer predictions (commit first)")
    do.add_argument("--compare", metavar="NAME", help="the adoption test")
    do.add_argument("--rule-2-2", action="store_true", help="does position matter?")
    do.add_argument("--reproduce", action="store_true", help="the harness = Stage 1?")
    ap.add_argument("--advisors", nargs="+", choices=ADVISORS, help="default: its own")
    ap.add_argument("--workers", type=int, default=WORKERS, help=f"default {WORKERS}")
    args = ap.parse_args(argv)
    if args.list:
        print_registry()
    elif args.run:
        print_outer(args.run, run(args.run, args.advisors, args.workers))
    elif args.compare:
        print_comparisons(args.compare, compare(args.compare, advisors=args.advisors))
    elif args.rule_2_2:
        grids, _ = validity.load_dev()
        results = rule_2_2(args.advisors or ADVISORS, args.workers)
        print_rule_2_2(results, orientation_patterns(grids), len(grids))
    else:
        sys.exit(0 if reproduce_stage1(args.workers) else 1)


if __name__ == "__main__":
    main()
