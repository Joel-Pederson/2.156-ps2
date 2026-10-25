"""Validity harness: how often is a surrogate right when it says a grid is good?

R2 is the wrong metric for CP2. We do not care about average prediction error --
we care about one question: when the model says a grid clears 0.75, is it right?
A model can have a fine R2 and still be worthless (or worse than random) at the
threshold, which is exactly what the starter notebook's linear model does to the
Transportation advisor.

So this measures precision at the threshold instead, honestly:

  stage 1  out-of-fold predictions on the labeled grids, so every prediction
           comes from a model that never saw that grid. For each confidence
           level, how many candidates survive and what fraction are truly valid?

  stage 2  fit on all labels, screen all 500k grids, and count how many clear
           every advisor at once. Then greedily pick 100 spread-out ones and
           score them, so each threshold gets an estimated final score.

Two things this cannot tell us, both worth keeping in view:

  - No grid in the dataset carries all four advisors' labels (the four labeled
    subsets are near-disjoint), so joint validity is never observed. The
    four-advisor numbers here multiply the marginals, which assumes the models
    fail independently. Unverifiable, possibly wrong.
  - Everything is measured on dataset grids, which are random. Grids you
    *optimize* are selected for being where the model is most enthusiastic, and
    so disproportionately where it is wrong. These numbers are an optimistic
    bound for generated designs, not a forecast.

    python validity.py                  # both stages, LightGBM
    python validity.py --model linear   # the starter's model, for comparison
    python validity.py --stage 1        # just the precision tables
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

PS2_ROOT = os.path.dirname(os.path.abspath(__file__))
COURSE_ROOT = os.path.join(PS2_ROOT, "data", "cp2-repo")
CACHE_DIR = os.path.join(PS2_ROOT, "data", "cache")

ADVISORS = ["Wellness", "Tax", "Transportation", "Business"]
THRESHOLD = 0.75
LEVELS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

if not os.path.isdir(COURSE_ROOT):
    sys.exit(f"course data missing at {COURSE_ROOT} -- run ./setup_data.sh")
sys.path.insert(0, COURSE_ROOT)

# Their diversity_score, not a reimplementation: it is the graded objective, and
# course_commit.txt pins the version we are measuring against.
from utils_public import diversity_score  # noqa: E402


# --------------------------------------------------------------------------
# data and features
# --------------------------------------------------------------------------

def load_data():
    grids = np.concatenate(
        [np.load(os.path.join(COURSE_ROOT, "datasets", f"grids_{i}.npy")) for i in range(5)]
    ).astype(np.int8)
    ratings = np.load(os.path.join(COURSE_ROOT, "datasets", "scores.npy"))
    return grids, ratings


def onehot(grids):
    """(n, 7, 7) ints -> (n, 245) float32. One binary feature per cell per district."""
    oh = (np.arange(5, dtype=np.int8) == grids[..., None])
    return oh.reshape(len(grids), -1).astype(np.float32)


def counts(grids):
    """The starter notebook's features: the 49 raw cells plus 5 district counts."""
    flat = grids.reshape(len(grids), -1).astype(np.float32)
    tallies = np.stack([(flat == k).sum(1) for k in range(5)], axis=1)
    return np.hstack([flat, tallies])


def make_model(kind):
    if kind == "linear":
        return LinearRegression()
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1
    )


FEATURES = {"linear": counts, "lightgbm": onehot}


# --------------------------------------------------------------------------
# stage 1: out-of-fold precision
# --------------------------------------------------------------------------

def out_of_fold(kind, X, y, n_splits=5, seed=0):
    """Predict every row from a model fitted without it. No leakage, by construction."""
    oof = np.empty(len(y), dtype=np.float64)
    for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        model = make_model(kind)
        model.fit(X[train], y[train])
        oof[test] = model.predict(X[test])
    return oof


def precision_table(oof, y, base_rate):
    """For each confidence level: survivors, and the fraction of them truly valid."""
    rows = []
    truly = y >= THRESHOLD
    for level in LEVELS:
        sel = oof >= level
        n = int(sel.sum())
        prec = float(truly[sel].mean()) if n else float("nan")
        rows.append((level, n, prec, prec / base_rate if n else float("nan")))
    return rows


def print_precision_table(rows):
    print(f"    {'require >=':>10} | {'candidates':>10} | {'truly valid':>11} | vs base")
    print(f"    {'-' * 10}-+-{'-' * 10}-+-{'-' * 11}-+--------")
    for level, n, prec, lift in rows:
        if not n:
            print(f"    {level:>10.2f} | {0:>10} | {'-':>11} |       -")
        else:
            print(f"    {level:>10.2f} | {n:>10} | {prec:>10.1%} | {lift:>6.1f}x")


# --------------------------------------------------------------------------
# stage 2: screen all 500k and estimate a final score
# --------------------------------------------------------------------------

def screen_all(kind, grids, ratings, chunk=50_000):
    """Fit each advisor on all its labels, then predict every grid. Cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"preds_{kind}.npy")
    if os.path.exists(cache):
        preds = np.load(cache)
        if preds.shape == (len(grids), 4):
            print(f"  (reusing cached predictions: {cache})")
            return preds

    featurize = FEATURES[kind]
    models = []
    for a in range(4):
        mask = ~np.isnan(ratings[:, a])
        model = make_model(kind)
        model.fit(featurize(grids[mask]), ratings[mask, a])
        models.append(model)

    preds = np.empty((len(grids), 4), dtype=np.float32)
    for start in range(0, len(grids), chunk):  # chunked: 500k x 245 float32 at once is ~0.5 GB
        stop = min(start + chunk, len(grids))
        X = featurize(grids[start:stop])
        for a, model in enumerate(models):
            preds[start:stop, a] = model.predict(X)

    # Where an advisor actually rated a grid, trust the label over the model.
    known = ~np.isnan(ratings)
    preds[known] = ratings[known]

    np.save(cache, preds)
    return preds


def greedy_diverse(grids, n=100, seed=0):
    """Farthest-point selection: repeatedly take the grid furthest from those picked.

    A cheap stand-in for real max-diversity optimization -- good enough to tell
    whether a candidate pool *can* yield a spread-out 100.
    """
    flat = grids.reshape(len(grids), -1).astype(np.int8)
    if len(flat) <= n:
        return np.arange(len(flat))

    rng = np.random.default_rng(seed)
    picked = [int(rng.integers(len(flat)))]
    # min normalized Hamming distance from each grid to the picked set, updated incrementally
    best = (flat != flat[picked[0]]).mean(axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(best))
        picked.append(nxt)
        np.minimum(best, (flat != flat[nxt]).mean(axis=1), out=best)
    return np.array(picked)


def estimate_scores(preds, grids, marginals):
    """For each threshold: how many grids pass all four, and what would we score?"""
    rows = []
    for level in LEVELS:
        passing = np.flatnonzero((preds >= level).all(axis=1))
        n = len(passing)
        joint = float(np.prod([marginals[a][level] for a in range(4)]))
        if n >= 2:
            pick = passing[greedy_diverse(grids[passing], n=100)]
            div = diversity_score(grids[pick])
        else:
            div = float("nan")
        # score ~ (fraction of the 100 that survive) x (diversity among them)
        rows.append((level, n, joint, div, joint * div if n >= 100 else float("nan")))
    return rows


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["lightgbm", "linear"], default="lightgbm")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None,
                    help="run only one stage (default: both)")
    args = ap.parse_args()

    kind = args.model
    featurize = FEATURES[kind]
    grids, ratings = load_data()
    print(f"model: {kind}   grids: {grids.shape}   labels/advisor: 5000\n")

    marginals = {}
    if args.stage != 2:
        print("=" * 68)
        print("STAGE 1 -- out-of-fold precision on labeled grids")
        print("=" * 68)
    for a in range(4):
        mask = ~np.isnan(ratings[:, a])
        g, y = grids[mask], ratings[mask, a]
        base = float((y >= THRESHOLD).mean())
        oof = out_of_fold(kind, featurize(g), y)
        rows = precision_table(oof, y, base)
        marginals[a] = {level: prec for level, _, prec, _ in rows}
        if args.stage != 2:
            print(f"\n  ADVISOR {a} ({ADVISORS[a]})  -- base rate {base:.1%} "
                  f"(hit rate if you pick blindfolded)")
            print_precision_table(rows)

    if args.stage == 1:
        return

    print("\n" + "=" * 68)
    print("STAGE 2 -- screening all 500,000 grids")
    print("=" * 68)
    preds = screen_all(kind, grids, ratings)

    print("\n  per advisor, how many of the 500k clear each level:")
    print(f"    {'level':>7} | " + " | ".join(f"{n:>14}" for n in ADVISORS))
    print(f"    {'-' * 7}-+-" + "-+-".join("-" * 14 for _ in ADVISORS))
    for level in LEVELS:
        cells = " | ".join(f"{int((preds[:, a] >= level).sum()):>14,}" for a in range(4))
        print(f"    {level:>7.2f} | {cells}")

    rows = estimate_scores(preds, grids, marginals)
    print("\n  all four at once, and what it would be worth:")
    print(f"    {'level':>7} | {'pass all 4':>11} | {'est. valid':>10} | "
          f"{'diversity':>9} | {'est. score':>10}")
    print(f"    {'-' * 7}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 9}-+-----------")
    for level, n, joint, div, score in rows:
        flag = "" if n >= 100 else "   <-- cannot fill 100"
        d = f"{div:>9.4f}" if div == div else f"{'-':>9}"
        s = f"{score:>10.4f}" if score == score else f"{'-':>10}"
        print(f"    {level:>7.2f} | {n:>11,} | {joint:>9.1%} | {d} | {s}{flag}")

    print("\n  est. valid multiplies the four marginal precisions, which assumes the")
    print("  models fail independently -- unverifiable here, since no grid carries")
    print("  all four labels. Diversity is from a greedy farthest-point pick of 100,")
    print("  a lower bound on what real diversity optimization would reach.")


if __name__ == "__main__":
    main()
