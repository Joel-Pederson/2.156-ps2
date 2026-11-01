"""Surrogate ratings: every dataset grid rated by each advisor's incumbent, three times over.

Where this fits
---------------
Stages 3 and 4 need each grid's predicted rating from every advisor, and how far to trust it.
``fit_models()`` refits each advisor's incumbent (search.INCUMBENT) on the training grids of
outer round (shuffle s, fold 0), s = 0, 1, 2, with that round's saved choice: 3 of the 15 outer
models whose out-of-fold predictions calibration.py learns from, so their spread on an unrated
grid is comparable (not on development grids, which they mostly trained on). ``rate()`` gives
the mean (pred) and standard deviation (spread) of the three, each (n, 4) in ADVISORS order.

Reads   the course data (validity.load_dev); the incumbents' outer results (search.load_outer)
Writes  data/cache/ratings/ratings.npz: pred and spread, float32, (500000, 4); ratings.json:
        commit, incumbents, cut-offs, seconds, approval counts. load_ratings() reads both.
Run     python surrogates.py   (fits 12 models, then rates 500,000 grids: ~100 s on 10 cores)

Surprises
---------
* The models train on 3,200 grids, not all 4,000: a refit on 4,000 would be a different model
  from the ones calibration describes.
* Each refit must predict 20 of its round's validation grids exactly as saved, or it stops.
* load_ratings() stops if search.INCUMBENT has changed since the ratings were made.
"""

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import search
import validity
from validity import ADVISORS

RATINGS_DIR = os.path.join(validity.PS2_ROOT, "data", "cache", "ratings")
CHUNK = 50_000  # grids rated at a time per process: ~1 GB peak (an (n, 49, 49) distance array)
CHECKED = 20  # validation grids per round compared with the saved outer predictions


def _incumbent(advisor):
    """(procedure, development grids, ratings, saved outer predictions, saved record)."""
    grids, dev = validity.load_dev()
    name = search.INCUMBENT[advisor]
    g, y, _ = search._advisor_data(grids, dev, advisor)
    preds, record = search.load_outer(name, advisor)
    return search.REGISTRY[name], g, y, preds, record


def _fold_zero(n):
    """(shuffle, train, val) of outer rounds (s, fold 0), s = 0, 1, 2."""
    return [(s, train, val) for s, f, train, val in validity.cv_splits(n) if f == 0]


def fit_models(advisors=ADVISORS):
    """{advisor: [(procedure, model)] for shuffles 0, 1, 2}: refits of rounds (s, 0)."""
    models = {}
    for advisor in advisors:
        procedure, g, y, _, record = _incumbent(advisor)
        choice = {(r["shuffle"], r["fold"]): r["choice"] for r in record["rounds"]}
        models[advisor] = [
            (procedure, procedure.fit(choice[s, 0], g[train], y[train]))
            for s, train, _ in _fold_zero(len(y))
        ]
    return models


def check_refits(models, k=CHECKED):
    """Stop unless each refit predicts k of its round's validation grids exactly as the saved
    outer predictions do, i.e. it is the model calibration learned from."""
    for advisor, pairs in models.items():
        _, g, y, preds, _ = _incumbent(advisor)
        for (s, _, val), (procedure, model) in zip(_fold_zero(len(y)), pairs):
            again = procedure.predict(model, g[val[:k]])
            same = np.array_equal(again, preds[s, val[:k]])
            assert same, f"{advisor}, shuffle {s}: refit differs from saved model"


def rate(models, grids, workers=1):
    """(pred, spread), each (n, 4) in ADVISORS order: the mean and standard deviation of each
    advisor's three ratings; NaN for an advisor not in models. Uses up to `workers` processes."""
    if workers > 1 and len(grids) > CHUNK:
        # Each process rates one chunk at a time, with this function and workers=1.
        chunks = [grids[i : i + CHUNK] for i in range(0, len(grids), CHUNK)]
        spawn = multiprocessing.get_context("spawn")  # a forked one can hang LightGBM
        with ProcessPoolExecutor(workers, mp_context=spawn) as pool:
            parts = list(pool.map(rate, [models] * len(chunks), chunks))
        return tuple(np.concatenate(x) for x in zip(*parts))
    pred, spread = np.full((2, len(grids), len(ADVISORS)), np.nan)
    for start in range(0, len(grids), CHUNK):
        part = slice(start, start + CHUNK)
        for advisor, pairs in models.items():
            each = np.stack([proc.predict(model, grids[part]) for proc, model in pairs])
            a = ADVISORS.index(advisor)
            pred[part, a], spread[part, a] = each.mean(0), each.std(0)
    return pred, spread


def cutoffs(advisors=ADVISORS):
    """{advisor: approval cut-off}: the mean over the three shuffles of the cut-off at 98%
    precision on the incumbent's saved outer predictions."""
    cuts = {}
    for advisor in advisors:
        _, _, y, preds, _ = _incumbent(advisor)
        each = [validity.approved_at(p, y)[1] for p in preds]
        cuts[advisor] = validity.mean_cutoff(each)
    return cuts


def load_ratings():
    """(pred, spread, info), as saved by `python surrogates.py`; stops if they are stale."""
    with np.load(os.path.join(RATINGS_DIR, "ratings.npz")) as saved:
        pred, spread = saved["pred"], saved["spread"]
    with open(os.path.join(RATINGS_DIR, "ratings.json")) as f:
        info = json.load(f)
    if info["incumbents"] != search.INCUMBENT:
        raise SystemExit("ratings are stale (other incumbents): run surrogates.py")
    return pred, spread, info


def main():
    tic = time.perf_counter()
    cuts = cutoffs()
    print("Approval cut-offs: " + ", ".join(f"{a} {c:.3f}" for a, c in cuts.items()))
    models = fit_models()
    check_refits(models)
    print(f"Each refit predicts {CHECKED} of its round's grids exactly as saved.")
    grids, _ = validity.load_dev()
    print(f"Rating {len(grids):,} grids with {3 * len(models)} models...", flush=True)
    pred, spread = (x.astype(np.float32) for x in rate(models, grids, search.WORKERS))
    # Counted on the float32 values as saved, so select_submission.py finds the same grids.
    ok = pred >= np.array([cuts[a] for a in ADVISORS])
    approved = dict(zip(ADVISORS, ok.sum(0).tolist()))
    approved["all four"] = int(ok.all(1).sum())
    commit, uncommitted = validity._git_state()
    info = {"commit": commit, "uncommitted_changes": bool(uncommitted)}
    info |= {"incumbents": search.INCUMBENT, "cutoffs": cuts, "approved": approved}
    info["seconds"] = round(time.perf_counter() - tic, 1)
    os.makedirs(RATINGS_DIR, exist_ok=True)
    np.savez(os.path.join(RATINGS_DIR, "ratings.npz"), pred=pred, spread=spread)
    with open(os.path.join(RATINGS_DIR, "ratings.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"Of {len(grids):,} grids, approved (predicted at or above the cut-off):")
    print("  " + ", ".join(f"{a} {k:,}" for a, k in approved.items()))
    print(f"Saved ratings.npz and .json in {RATINGS_DIR} ({info['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
