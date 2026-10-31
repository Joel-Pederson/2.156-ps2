"""The model scorecard: how often is a model right when it says a grid passes?

Everything about measuring models honestly lives here, and
every later stage imports from it:

* ``load_dev()``         The labels we're allowed to learn from. Test-set (lockbox) labels come
                         back as NaN ("missing"), so code that loads data this way can't see them.
* ``open_lockbox()``     The ONLY way to reach the lockbox labels. It refuses unless you confirm
                         and everything is committed, and logs every opening in
                         docs/lockbox_log.md.
* ``cv_splits()``        5-fold cross-validation, repeated 3 times with different shuffles.
* ``fold_data()``        One fold's train and validation data. Rotation augmentation, when asked
                         for, touches only the training part, and only after the split.
* ``VARIANTS``           Stage 1's model variants, declared up front. (Stage 2 compares
                         "procedures" under a cap of 6 per advisor; docs/decisions.md.)
* ``scorecard()``        Out-of-fold predictions for every variant and advisor, and the numbers
                         we choose models by.

Why not just R²? (docs/stage1_evaluation.md, "Model scorecard") R² is the average error over the whole
0–1 rating range, and an error at 0.2 versus 0.3 never matters here: both fail. The scorecard
leads with the **approval count at 98% precision**: the number of development-set grids a
model can approve while at least 98% of its approvals are truly rated ≥ 0.75. Precision protects
the score; the count is room to choose varied grids.

Reads / writes
--------------
reads   data/cp2-repo/datasets/*.npy   the course data (fetch with ./setup_data.sh)
        course_commit.txt              the pinned course version; part of the cache key
writes  data/cache/oof/*.npy, *.json   cached out-of-fold predictions (gitignored)
        docs/lockbox_log.md            only when the lockbox is opened

How to run
----------
    python validity.py                          # the full scorecard (~8 min first time, seconds after)
    python validity.py --variants gbm_engineered
    python validity.py --no-cache               # recompute everything from scratch
    python validity.py --grid 12                # one grid's predictions against each advisor's cut-off
    python validity.py --data                   # counts of which grids each advisor rated

"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features

PS2_ROOT = os.path.dirname(os.path.abspath(__file__))
COURSE_ROOT = os.path.join(PS2_ROOT, "data", "cp2-repo")
CACHE_DIR = os.path.join(PS2_ROOT, "data", "cache", "oof")
LOCKBOX_LOG = os.path.join(PS2_ROOT, "docs", "lockbox_log.md")
PIN_FILE = os.path.join(PS2_ROOT, "course_commit.txt")

ADVISORS = ("Wellness", "Tax", "Transportation", "Business")
# A grid passes an advisor that rates it at least this; it counts only if all four pass.
THRESHOLD = 0.75
# Per-advisor precision used to compare models. 0.98 ** 4 ≈ 0.92: the chance a grid approved
# for all four advisors is valid, if their errors are unrelated (docs/stage1_evaluation.md, "Why 98%").
TARGET_PRECISION = 0.98
TRADEOFF_LEVELS = (0.90, 0.95, 0.98, 0.99)  # docs/stage1_evaluation.md, "Why 98%"
# True ratings near the 0.75 pass mark: where errors flip pass and fail.
# Reports one extra accuracy number, measured only on grids whose true rating is between the BAND values.
BAND = (0.6, 0.9)

LOCKBOX_SEED = 2156
LOCKBOX_SIZE = 1000  # per advisor
CV_FOLDS = 5
CV_SEEDS = (0, 1, 2)  # one random seed per shuffle
CV_REPEATS = len(CV_SEEDS)
RIDGE_ALPHA = 1.0  # Ridge's penalty on large weights
# Bump if the cache file layout changes (2: the key hashes the model inputs).
CACHE_SCHEMA = 2
# Stage 1's cap on compared variants per role. Stage 2 caps decision-bearing procedures per
# advisor instead (docs/decisions.md, "Procedure cap").
MAX_VARIANTS_PER_ROLE = 6


# --------------------------------------------------------------------------------------
# data, and the lockbox
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_raw():
    """All grids and ALL labels, lockbox included. Private: use load_dev() or open_lockbox()."""
    datasets = os.path.join(COURSE_ROOT, "datasets")
    if not os.path.isdir(datasets):
        raise FileNotFoundError(
            f"course data missing at {COURSE_ROOT}; run ./setup_data.sh"
        )
    grids = np.concatenate(
        [np.load(os.path.join(datasets, f"grids_{i}.npy")) for i in range(5)]
    ).astype(np.int8)
    ratings = np.load(os.path.join(datasets, "scores.npy"))
    grids.flags.writeable = False  # cached and shared: nobody gets to edit it in place
    ratings.flags.writeable = False
    return grids, ratings


def lockbox_mask(ratings):
    """(500000, 4) bool: True where a label belongs to the lockbox.

    Reads only *which* grids each advisor labelled, never the ratings themselves. Every grid
    that two or more advisors rated goes in, because those are our only evidence about whether
    models fail on the same grids. The rest is a seeded random fill up to 1,000 per advisor.
    """
    labelled = ~np.isnan(ratings)
    co_rated = labelled.sum(1) >= 2
    rng = np.random.default_rng(LOCKBOX_SEED)
    mask = np.zeros(ratings.shape, bool)
    for a in range(ratings.shape[1]):
        idx = np.flatnonzero(labelled[:, a])
        always = idx[co_rated[idx]]
        rest = idx[~co_rated[idx]]
        fill = rng.choice(rest, size=LOCKBOX_SIZE - len(always), replace=False)
        mask[always, a] = True
        mask[fill, a] = True
    return mask


def lockbox_fingerprint(mask):
    """A short hash of which labels are in the lockbox. Any change to the recipe changes it."""
    h = hashlib.sha256()
    for a in range(mask.shape[1]):
        h.update(np.flatnonzero(mask[:, a]).astype(np.int64).tobytes())
    return h.hexdigest()


def load_dev():
    """All 500k grids, plus the labels we may learn from: lockbox labels are NaN."""
    grids, ratings = _load_raw()
    dev = ratings.copy()
    dev[lockbox_mask(ratings)] = np.nan
    return grids, dev


def open_lockbox(reason, *, confirm=False):
    """The only route to the lockbox labels. Returns (grids, ratings with ONLY lockbox labels).

    The lockbox is the final exam: opened once, after the final pipeline is fixed and its
    expected precision is recorded (rule 2.7 in docs/decisions.md). This refuses unless
    confirm=True and git shows nothing uncommitted in this repository, untracked files
    included. Every opening is appended to docs/lockbox_log.md with the time, the full commit
    hash, the script that called it and your reason. Code outside the repository, or in paths
    git ignores, escapes the check, so rule 2.7 also requires a committed grading script.
    """
    if not confirm:
        raise PermissionError(
            "The lockbox is opened once, for the final pipeline (rule 2.7 in docs/decisions.md). "
            "Pass confirm=True deliberately; every opening is logged in docs/lockbox_log.md."
        )
    if not reason or not reason.strip():
        raise ValueError("say why you are opening the lockbox")
    commit, uncommitted = _git_state()
    if uncommitted:
        raise PermissionError(
            "Commit everything first (rule 2.7), so the logged commit holds the code. "
            f"Not committed: {', '.join(uncommitted[:5])}"
        )
    grids, ratings = _load_raw()
    lock = np.where(lockbox_mask(ratings), ratings, np.nan)
    _log_lockbox_opening(reason.strip(), commit)
    return grids, lock


def _git_state():
    """(full hash of the current commit, paths that git lists as uncommitted or untracked).

    Only this folder's own repository counts: git may not look in the folders above it (the
    course folder above is a repository that ignores this one). Untracked files are listed
    whatever git's settings say; ignored paths (data/, caches) are not. Raises if git can't
    answer, so the lockbox never opens without a commit to log.
    """
    env = {**os.environ, "GIT_CEILING_DIRECTORIES": os.path.realpath(f"{PS2_ROOT}/..")}

    def git(*args):
        return subprocess.run(
            ["git", "-C", PS2_ROOT, *args],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout

    commit = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain", "--untracked-files=all")
    return commit, [line[3:] for line in status.splitlines()]


def _caller():
    """The running script: its path inside the repository, or its full path if outside."""
    script = os.path.realpath(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    root = os.path.realpath(PS2_ROOT)
    if script.startswith(root + os.sep):
        return os.path.relpath(script, root)
    return script or "an interactive session"


def _log_lockbox_opening(reason, commit):
    os.makedirs(os.path.dirname(LOCKBOX_LOG), exist_ok=True)
    new = not os.path.exists(LOCKBOX_LOG)
    with open(LOCKBOX_LOG, "a") as f:
        if new:
            f.write(
                "# Lockbox openings\n\nWritten by validity.open_lockbox(). "
                "Every line is one look at the final exam.\n\n"
            )
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        f.write(f"- {stamp} · commit {commit} · run from {_caller()} · {reason}\n")


def course_pin():
    """The course-repo commit we build against, read from course_commit.txt."""
    with open(PIN_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise ValueError(f"no commit found in {PIN_FILE}")


# --------------------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------------------


def cv_splits(n, folds=CV_FOLDS, repeats=CV_REPEATS):
    """Yield (shuffle, fold, train_idx, val_idx) for repeated k-fold cross-validation.

    Shuffle r assigns grids to folds with seed CV_SEEDS[r]; shuffle 0 (called shuffle 1 in the
    documents) is exactly the split the planning experiments used. Running under several
    shuffles shows how much a result moves just from which grids land in which fold: the
    noise threshold (docs/stage1_evaluation.md, "Shuffle procedure").
    """
    for r in range(repeats):
        splitter = KFold(folds, shuffle=True, random_state=CV_SEEDS[r])
        for f, (train, val) in enumerate(splitter.split(np.arange(n))):
            yield r, f, train, val


def fold_data(grids, y, train_idx, val_idx, augment=False):
    """One fold's (train grids, train labels, validation grids, validation labels).

    With augment=True, the training part gets all 8 rotations/mirrors, *after* the split.
    Rotating first would put rotated copies of validation grids into training.
    """
    g_train, y_train = grids[train_idx], y[train_idx]
    if augment:
        g_train, y_train = features.augment(g_train, y_train)
    return g_train, y_train, grids[val_idx], y[val_idx]


# --------------------------------------------------------------------------------------
# the declared model variants
# --------------------------------------------------------------------------------------

GBM_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 0,
    "verbose": -1,
}

FEATURIZERS = {"onehot": features.onehot, "engineered": features.engineered}


@dataclass(frozen=True)
class Variant:
    """One model configuration we compare. Declared in VARIANTS, never invented on the fly."""

    features: str  # "onehot" or "engineered"
    model: str  # "ridge" or "gbm"
    role: str  # "baseline", "main" or "second_opinion"
    augment: bool = False  # train on all 8 orientations (training fold only)

    def featurize(self, grids):
        return FEATURIZERS[self.features](grids)

    def make_model(self):
        if self.model == "ridge":
            if self.features == "engineered":
                # Engineered features live on different scales (counts vs distances). The
                # scaler is part of the model, so it's refit on each training fold only.
                return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
            return Ridge(alpha=RIDGE_ALPHA)  # one-hot columns are already all 0/1
        import lightgbm as lgb  # imported here so the tests needn't pay for it

        return lgb.LGBMRegressor(**GBM_PARAMS)

    def spec(self):
        """Everything that changes this variant's predictions, for the cache key."""
        spec = {"features": self.features, "model": self.model, "augment": self.augment}
        if self.model == "ridge":
            spec.update(ridge_alpha=RIDGE_ALPHA, scaled=self.features == "engineered")
        if self.model == "gbm":
            import lightgbm as lgb

            spec.update(gbm_params=GBM_PARAMS, lightgbm=lgb.__version__)
        if self.features == "engineered":
            spec["feature_version"] = features.FEATURE_VERSION
        else:
            spec["onehot_version"] = features.ONEHOT_VERSION
        spec["sklearn"] = sklearn.__version__
        spec["numpy"] = np.__version__
        spec["scipy"] = scipy.__version__
        return spec


# The planning experiments already compared these four; Stage 1 reproduces them. Stage 2 adds the
# CNN second-opinion candidates. Stage 1 capped variants per role; Stage 2 caps decision-bearing
# procedures per advisor instead (docs/decisions.md, "Procedure cap").
VARIANTS = {
    "ridge_onehot": Variant("onehot", "ridge", role="baseline"),
    "gbm_onehot": Variant("onehot", "gbm", role="baseline"),
    "ridge_engineered": Variant("engineered", "ridge", role="baseline"),
    "gbm_engineered": Variant("engineered", "gbm", role="main"),
}

# R² from the planning experiments (development set only, 2026-10-25): one-off scripts run
# before the Stage 1 scorecard was added to this module. They used one shuffle, with the same
# folds as shuffle 0 here. Stage 1's completion check requires the scorecard's three-shuffle
# mean to match them within 0.01 (docs/decisions.md).
PLANNING_R2 = {
    "ridge_onehot": (0.483, 0.758, 0.121, 0.483),
    "gbm_onehot": (0.504, 0.707, 0.291, 0.500),
    "ridge_engineered": (0.697, 0.868, 0.649, 0.602),
    "gbm_engineered": (0.887, 0.968, 0.935, 0.971),
}


# --------------------------------------------------------------------------------------
# out-of-fold predictions, cached under a fingerprint of everything that produced them
# --------------------------------------------------------------------------------------


def cache_key(variant, grids, X, y, rows):
    """A hash of everything that could change a variant's out-of-fold predictions.

    It covers the grids and the model inputs built from them (X: so editing a feature
    definition changes the key even if no version label was bumped), the rows and their
    ratings, the variant's settings and library versions (Variant.spec), the number of folds
    and the shuffle seeds, the test-set seed and the course-data version. If any of them
    changes, the cache misses and the predictions are recomputed.

    Not covered: edits to the code that leave every setting unchanged: Variant.make_model,
    the loop in out_of_fold, cv_splits beyond its fold count and seeds, and, for a variant
    with augmentation, fold_data, features.augment and the orientations it uses
    (features.orientations, features.SYMMETRIES); and the Python version (pinned in
    environment.yml). Run with --no-cache after such an edit.
    """
    grids, X = np.ascontiguousarray(grids), np.ascontiguousarray(X)
    payload = {
        "schema": CACHE_SCHEMA,
        "variant": variant.spec(),
        "folds": CV_FOLDS,
        "cv_seeds": list(CV_SEEDS),
        "course_pin": course_pin(),
        "lockbox_seed": LOCKBOX_SEED,
        "grids": [hashlib.sha256(grids.tobytes()).hexdigest(), list(grids.shape)],
        "inputs": [
            hashlib.sha256(X.tobytes()).hexdigest(),
            list(X.shape),
            str(X.dtype),
        ],
        "rows": hashlib.sha256(
            np.asarray(rows, np.int64).tobytes() + np.asarray(y, np.float64).tobytes()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def out_of_fold(name, grids, y, rows, use_cache=True):
    """(CV_REPEATS, n) predictions; each made by a model that never saw that grid."""
    variant = VARIANTS[name]
    X = variant.featurize(grids)  # the model inputs; also fingerprinted by cache_key
    key = cache_key(variant, grids, X, y, rows)
    path = os.path.join(CACHE_DIR, f"{name}-{key[:16]}.npy")
    if use_cache and os.path.exists(path):
        return np.load(path)

    preds = np.full((CV_REPEATS, len(y)), np.nan)
    for r, _, train, val in cv_splits(len(y)):
        if variant.augment:
            g_train, y_train, g_val, _ = fold_data(grids, y, train, val, augment=True)
            X_train, X_val = variant.featurize(g_train), variant.featurize(g_val)
        else:
            X_train, y_train, X_val = X[train], y[train], X[val]
        model = variant.make_model()
        model.fit(X_train, y_train)
        preds[r, val] = model.predict(X_val)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, preds)
    # A readable record of what produced this file.
    with open(path.replace(".npy", ".json"), "w") as f:
        json.dump(
            {"name": name, "key": key, "spec": variant.spec(), "n": len(y)}, f, indent=1
        )
    return preds


# --------------------------------------------------------------------------------------
# the numbers we choose models by
# --------------------------------------------------------------------------------------


def approved_at(pred, y, target=TARGET_PRECISION):
    """How many grids can we approve, highest predictions first, staying `target` precise?

    Walk down the grids from the highest predicted rating to the lowest. After each one, ask
    what fraction of those approved so far are truly rated ≥ 0.75. Return the largest count
    where that fraction is still ≥ target, the approval cut-off (the predicted rating of the
    last grid approved), and how many of the approved grids truly pass. (0, inf, 0) if no
    count reaches the target.

    The count only ever stops at the end of a group of tied predictions. The rule "approve at
    or above the cut-off" approves every grid that shares the cut-off's predicted rating, so a
    count that split a tie group would not be the count that rule approves.
    """
    order = np.argsort(-pred, kind="stable")
    ranked = pred[order]
    hits = np.cumsum(y[order] >= THRESHOLD)
    precision = hits / np.arange(1, len(y) + 1)
    ends_tie_group = np.r_[ranked[1:] != ranked[:-1], True]
    ok = np.flatnonzero((precision >= target) & ends_tie_group)
    if not len(ok):
        return 0, np.inf, 0
    last = ok[-1]
    return last + 1, float(ranked[last]), int(hits[last])


def mean_cutoff(cutoffs):
    """The mean of the per-shuffle cut-offs that exist (a shuffle that approves nothing has
    none, shown as inf). inf if no shuffle approves anything."""
    finite = [c for c in cutoffs if np.isfinite(c)]
    return float(np.mean(finite)) if finite else np.inf


@dataclass
class Score:
    """One variant on one advisor. Values are means over the three shuffles unless noted."""

    r2: float  # 1 - MSE / variance of the true ratings (1 = perfect, 0 = predicting the mean)
    r2_fold_sd: float  # s.d. of the 15 single-fold R² values: the R² noise threshold
    band_rmse: float  # RMSE for grids with true rating 0.6–0.9
    approved: float  # approval count at 98% precision
    approved_per_shuffle: tuple  # the three per-shuffle counts behind that mean
    cutoff: float  # approval cut-off: predicted rating of the last grid approved (mean_cutoff)
    cutoff_per_shuffle: tuple  # each shuffle's cut-off; inf where it approves nothing
    recall: float  # share of the advisor's truly passing grids that are approved


def score(preds, y):
    """Summarize (shuffles, n) out-of-fold predictions against labels y.

    Works for any number of shuffles from 1 to CV_REPEATS; shuffle r must use the folds of
    cv_splits' shuffle r.
    """
    fold_r2 = [
        r2_score(y[val], preds[r, val])
        for r, _, _, val in cv_splits(len(y), repeats=len(preds))
    ]
    band = (y >= BAND[0]) & (y <= BAND[1])
    per_repeat = [approved_at(p, y) for p in preds]
    passing = (y >= THRESHOLD).sum()
    return Score(
        r2=float(np.mean([r2_score(y, p) for p in preds])),
        r2_fold_sd=float(np.std(fold_r2)),
        band_rmse=float(
            np.mean([np.sqrt(np.mean((p[band] - y[band]) ** 2)) for p in preds])
        ),
        approved=float(np.mean([k for k, _, _ in per_repeat])),
        approved_per_shuffle=tuple(int(k) for k, _, _ in per_repeat),
        cutoff=mean_cutoff([c for _, c, _ in per_repeat]),
        cutoff_per_shuffle=tuple(float(c) for _, c, _ in per_repeat),
        recall=float(np.mean([hits for _, _, hits in per_repeat])) / passing,
    )


def scorecard(names=None, use_cache=True, verbose=True):
    """{variant: {advisor: dict(y, preds, score)}} for the dev set."""
    names = list(names or VARIANTS)
    grids, dev = load_dev()
    results = {}
    for name in names:
        results[name] = {}
        for a, advisor in enumerate(ADVISORS):
            rows = np.flatnonzero(~np.isnan(dev[:, a]))
            y = dev[rows, a]
            if verbose:
                print(f"  {name:<17} {advisor:<15}", end="", flush=True)
            preds = out_of_fold(name, grids[rows], y, rows, use_cache=use_cache)
            results[name][advisor] = {"y": y, "preds": preds, "score": score(preds, y)}
            if verbose:
                print(" done")
    return results


def print_scorecard(results):
    """Print the scorecard: approval count, approval cut-off and share approved (the tables in
    docs/stage1_evaluation.md, "Results"), then R² and the RMSE for true ratings 0.6–0.9."""
    head = "".join(f"{a:>22}" for a in ADVISORS)
    some = next(iter(results.values()))
    n_dev = len(next(iter(some.values()))["y"])
    sc_any = next(iter(some.values()))["score"]
    print(
        f"\nApproval count at 98% precision, of {n_dev:,} development-set grids per advisor:"
        f"\nmean over the {len(sc_any.approved_per_shuffle)} shuffles [the per-shuffle counts]"
    )
    print(f"  {'variant':<18}{head}")
    for name, by_adv in results.items():
        cells = ""
        for a in ADVISORS:
            sc = by_adv[a]["score"]
            each = "/".join(str(k) for k in sc.approved_per_shuffle)
            cells += f"{sc.approved:>7.0f} [{each:>12}]"
        print(f"  {name:<18}{cells}")
    print(
        "\nApproval cut-off: the predicted rating of the last grid approved at 98% precision,"
        "\nmean over the 3 shuffles. A grid is approved for an advisor at or above it."
    )
    print(f"  {'variant':<18}{head}")
    partial = False
    for name, by_adv in results.items():
        cells = ""
        for a in ADVISORS:
            sc = by_adv[a]["score"]
            missing = sum(not np.isfinite(c) for c in sc.cutoff_per_shuffle)
            if not np.isfinite(sc.cutoff):
                cell = "none"
            else:
                cell = f"{sc.cutoff:.3f}" + ("*" if missing else "")
                partial = partial or bool(missing)
            cells += f"{cell:>22}"
        print(f"  {name:<18}{cells}")
    if partial:
        print("  * mean over the shuffles that approve at least one grid")
    print(
        "\nShare of the advisor's truly passing grids (true rating >= 0.75) that are approved"
    )
    print(f"  {'variant':<18}{head}")
    for name, by_adv in results.items():
        print(
            f"  {name:<18}"
            + "".join(f"{by_adv[a]['score'].recall:>22.0%}" for a in ADVISORS)
        )

    print(
        "\nR² (1 = perfect, 0 = no better than predicting the mean rating): mean over the"
        "\n3 shuffles, each on all out-of-fold predictions ± s.d. of the 15 single-fold R²"
        " values (the R² noise threshold)"
    )
    print(f"  {'variant':<18}{head}")
    for name, by_adv in results.items():
        cells = "".join(
            f"{by_adv[a]['score'].r2:>15.3f} ±{by_adv[a]['score'].r2_fold_sd:.3f}"
            for a in ADVISORS
        )
        print(f"  {name:<18}{cells}")

    print("\nRMSE for grids with true rating 0.6–0.9 (lower is better)")
    print(f"  {'variant':<18}{head}")
    for name, by_adv in results.items():
        print(
            f"  {name:<18}"
            + "".join(f"{by_adv[a]['score'].band_rmse:>22.3f}" for a in ADVISORS)
        )

    diffs = [
        (abs(by_adv[a]["score"].r2 - PLANNING_R2[name][i]), name, a)
        for name, by_adv in results.items()
        if name in PLANNING_R2
        for i, a in enumerate(ADVISORS)
    ]
    if not diffs:
        return
    worst, worst_name, worst_advisor = max(diffs)
    verdict = "PASS" if worst <= 0.01 else "CHECK"
    print(
        "\nStage 1 completion check: largest difference between the three-shuffle mean R² and"
        f"\nthe planning experiments' R² (first shuffle only) = {worst:.4f}"
        f" ({worst_name}, {worst_advisor}) -> {verdict} (tolerance 0.01)"
    )


def precision_tradeoff(results, name, levels=TRADEOFF_LEVELS):
    """Approval count at several precision levels, for one variant (docs/stage1_evaluation.md, "Why 98%").

    For each per-advisor precision level p, returns (p, valid_all_four, {advisor: count}):
    the approval count is averaged over the three shuffles, exactly as in the scorecard, and
    valid_all_four = p ** 4 is the chance a grid approved for all four advisors is valid,
    *if* two assumptions hold: the four models' errors are unrelated, and the precision
    measured here holds for the grids actually submitted (both unverified; risks R1 and R4
    in the plan).
    """
    rows = []
    for level in levels:
        counts = {
            adv: float(np.mean([approved_at(p, r["y"], level)[0] for p in r["preds"]]))
            for adv, r in results[name].items()
        }
        rows.append((level, level ** len(counts), counts))
    return rows


def print_precision_tradeoff(rows, name):
    """The "Why 98%" table of docs/stage1_evaluation.md, as printed by `python validity.py`."""
    advisors = list(rows[0][2])
    print(f"\nPrecision trade-off for {name} (docs/stage1_evaluation.md, 'Why 98%')")
    print(
        f"  {'precision':>9} {'valid, all 4':>13} {'invalid/100':>12}"
        + "".join(f"{a:>16}" for a in advisors)
    )
    for level, all_four, counts in rows:
        print(
            f"  {level:>9.0%} {all_four:>13.0%} {100 * (1 - all_four):>12.0f}"
            + "".join(f"{counts[a]:>16,.0f}" for a in advisors)
        )
    print(
        "  ('valid, all 4' = precision ** 4. It assumes the four models' errors are unrelated"
        "\n  and that this precision holds for the grids submitted.)"
    )


def grid_report(index, name="gbm_engineered"):
    """One grid's predicted rating for every advisor, against that advisor's approval cut-off.

    Returns [(advisor, predicted, cutoff, true_or_None)]. `cutoff` is the variant's approval
    cut-off for that advisor, exactly as in the scorecard (mean over the three shuffles).
    Where the advisor rated the grid in the development set, the prediction is out of fold
    (the mean over the three shuffles of predictions from models that did not train on it), so
    it can be compared with the true rating. Otherwise no true rating exists in the data
    available to us, and a model trained on that advisor's whole development set makes the
    prediction. Test-set ratings are never shown: load_dev() hides them, so a grid sealed in
    the test set for an advisor is reported like an unrated one. Uses the scorecard's cached
    predictions, computing them first if they are missing.
    """
    grids, dev = load_dev()
    # The row lookup below needs the plain row number.
    index = _grid_number(index, len(grids))
    variant = VARIANTS[name]
    x = variant.featurize(grids[index][None])
    report = []
    for a, advisor in enumerate(ADVISORS):
        rows = np.flatnonzero(~np.isnan(dev[:, a]))
        y = dev[rows, a]
        preds = out_of_fold(name, grids[rows], y, rows)
        cutoff = mean_cutoff([approved_at(p, y)[1] for p in preds])
        if not np.isnan(dev[index, a]):
            pos = int(np.searchsorted(rows, index))
            assert rows[pos] == index
            pred, true = float(preds[:, pos].mean()), float(dev[index, a])
        else:
            model = variant.make_model()
            model.fit(variant.featurize(grids[rows]), y)
            pred, true = float(model.predict(x)[0]), None
        report.append((advisor, pred, cutoff, true))
    return report


def _grid_number(index, n):
    """A grid's row number, 0 to n - 1. Negative numbers count from the end, as in Python
    indexing (-1 is the last grid); anything else outside the range raises IndexError."""
    if not -n <= index < n:
        raise IndexError(f"there is no grid {index}: grids are numbered 0 to {n - 1:,}")
    return int(index) % n


def print_grid_report(index, name="gbm_engineered"):
    index = _grid_number(index, len(_load_raw()[0]))  # show grid -1 as 499999
    print(
        f"Grid {index}: predicted rating for each advisor ({name}),"
        "\nagainst that advisor's approval cut-off at 98% precision"
    )
    report = grid_report(index, name)
    for advisor, pred, cutoff, true in report:
        verdict = "approved" if pred >= cutoff else "not approved"
        truth = (
            f"true rating {true:.3f}"
            if true is not None
            else "no development-set rating (unrated, or sealed in the test set)"
        )
        print(
            f"  {advisor:<15} predicted {pred:.3f}, cut-off {cutoff:.3f}:"
            f" {verdict:<12}   {truth}"
        )
    refused = [advisor for advisor, pred, cutoff, _ in report if pred < cutoff]
    print(
        "  Predicted valid (approved for all four advisors): "
        + ("yes" if not refused else "no; not approved for " + ", ".join(refused))
    )


def data_summary():
    """Counts describing the course data. Reads only which grids are rated, never a rating.

    Returns a dict: the number of grids; ratings per advisor; the number of grids rated by
    0, 1, 2, 3 and 4 advisors; the number rated by both advisors of each pair; and the
    test-set size per advisor.
    """
    grids, ratings = _load_raw()
    labelled = ~np.isnan(ratings)
    pairs = {
        (ADVISORS[i], ADVISORS[j]): int((labelled[:, i] & labelled[:, j]).sum())
        for i in range(len(ADVISORS))
        for j in range(i + 1, len(ADVISORS))
    }
    return {
        "grids": len(grids),
        "ratings_per_advisor": dict(zip(ADVISORS, labelled.sum(0).tolist())),
        "grids_by_raters": np.bincount(labelled.sum(1), minlength=5).tolist(),
        "pairs": pairs,
        "test_set_per_advisor": dict(
            zip(ADVISORS, lockbox_mask(ratings).sum(0).tolist())
        ),
    }


def print_data_summary():
    s = data_summary()
    n, per = s["grids"], s["ratings_per_advisor"]
    print("Course data: which grids each advisor rated (no rating values are read)")
    print(f"  grids in the dataset: {n:,}")
    print("  ratings per advisor: " + ", ".join(f"{a} {k:,}" for a, k in per.items()))
    print(
        "  grids by number of advisors who rated them: "
        + ", ".join(f"{i}: {k:,}" for i, k in enumerate(s["grids_by_raters"]))
    )
    # If each advisor's grids were an independent random sample of the dataset, two advisors
    # would share about k1 * k2 / n grids: 50 for 5,000 each out of 500,000.
    print(
        "  grids rated by both advisors of a pair (expected if the samples were independent):"
    )
    for (a, b), shared in s["pairs"].items():
        print(f"    {a}–{b}: {shared} ({per[a] * per[b] / n:.0f})")
    print(
        "  test set (lockbox) per advisor: "
        + ", ".join(f"{a} {k:,}" for a, k in s["test_set_per_advisor"].items())
        + "; it holds every grid rated by two or more advisors"
    )


def main():
    ap = argparse.ArgumentParser(
        description="The model scorecard (development set only)."
    )
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    ap.add_argument(
        "--no-cache", action="store_true", help="recompute every prediction"
    )
    ap.add_argument(
        "--grid", type=int, help="instead: show one grid's predictions and cut-offs"
    )
    ap.add_argument(
        "--data", action="store_true", help="instead: count which grids are rated"
    )
    args = ap.parse_args()
    if args.grid is not None:
        try:
            index = _grid_number(args.grid, len(_load_raw()[0]))
        except IndexError as e:
            ap.error(str(e))
        print_grid_report(index)
        return
    if args.data:
        print_data_summary()
        return
    print(
        f"Scorecard: {CV_FOLDS}-fold cross-validation × {CV_REPEATS} shuffles, "
        "development set only (test set untouched)"
    )
    results = scorecard(args.variants, use_cache=not args.no_cache)
    print_scorecard(results)
    for name, variant in VARIANTS.items():
        if variant.role == "main" and name in results:
            print_precision_tradeoff(precision_tradeoff(results, name), name)


if __name__ == "__main__":
    main()
