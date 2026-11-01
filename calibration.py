"""Calibration: turn an advisor's predicted rating into the chance that the grid passes.

Where this fits
---------------
Stage 4 needs each grid's chance of passing (rated ≥ 0.75), not its rating. Calibrated means the
chances come true as often as they say: of grids given 90%, about 9 in 10 pass. Reads the
incumbent's saved outer predictions (search.load_outer: 3 shuffles × 4,000 development grids) as
pred (their mean) and spread (their standard deviation: how much the models disagree), and the
true ratings (validity.load_dev). Writes nothing. Two methods compete on held-out log loss (−log
of the chance given to what happened) over grids predicted ≥ 0.6, in 5 folds (docs/decisions.md,
Stage 2 protocol, "Calibration"):
* isotonic: a staircase pass rate that never falls as pred rises;
* probit: chance = Φ((pred − c) / (s0 + s1·spread)): 50% at pred = c, wider if models disagree.

How to run
----------
    python calibration.py   # per advisor: both log losses, the choice, a reliability table

Surprises
---------
* select_submission.py multiplies the four advisors' chances: that assumes their errors are
  unrelated (rule 2.8 measures it).
* The reliability table uses held-out chances: isotonic always looks calibrated on grids it saw.
* Not done: the protocol's widening on grids approved for all four (docs/decisions.md,
  "Calibration", third bullet). Isotonic (Transportation's pick) ignores spread; its top step can
  be 1.0, i.e. 1 − 1e-6 whatever the disagreement. So widen pred first, or the owner picks probit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

import search
import validity
from validity import ADVISORS, THRESHOLD

FOLDS = 5
FOLD_SEED = 2000  # a new seed, used nowhere else
SCORED_FROM = 0.6  # log loss scores only these grids; lower ones are never approved
CLIP = 1e-6  # chances stay off 0 and 1: a certain miss costs infinite log loss
BINS = (0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1)  # finer near 1, where approvals sit


@dataclass
class Calibration:  # what fit() returns, for one advisor
    method: str  # "isotonic" or "probit": the lower cross-fitted log loss
    models: dict  # {method: fitted on all grids}; the probit's is (c, s0, s1)
    log_loss: dict  # {method: total over held-out grids predicted ≥ SCORED_FROM}
    table: list  # the chosen method's reliability table on those grids


def fit_isotonic(pred, spread, passed):
    """A staircase pass rate that never falls as pred rises (spread unused)."""
    return IsotonicRegression(increasing=True, out_of_bounds="clip").fit(pred, passed)


def _z(params, pred, spread):
    c, s0, s1 = params
    return (pred - c) / (s0 + s1 * spread)


def fit_probit(pred, spread, passed):
    """(c, s0, s1) of the widening probit, by maximum likelihood, with s0 > 0 and s1 ≥ 0."""

    def minus_log_likelihood(params):
        z = _z(params, pred, spread)
        return -norm.logcdf(np.where(passed, z, -z)).sum()  # chance of failing is Φ(−z)

    bounds = [(None, None), (1e-4, None), (0, None)]
    start = (THRESHOLD, 0.05, 1.0)
    result = minimize(minus_log_likelihood, start, method="L-BFGS-B", bounds=bounds)
    return tuple(result.x.tolist())


def _chance(method, model, pred, spread):
    """Pass chances from a fitted model, kept CLIP away from 0 and 1."""
    isotonic = method == "isotonic"
    p = model.predict(pred) if isotonic else norm.cdf(_z(model, pred, spread))
    return np.clip(p, CLIP, 1 - CLIP)


def _log_loss(p, passed):
    """−log of the chance given to what happened, summed over grids."""
    return float(-np.log(np.where(passed, p, 1 - p)).sum())


def fit(pred, spread, y):
    """Pick isotonic or probit by 5-fold cross-fitted log loss; refit both on all grids."""
    fitters = {"isotonic": fit_isotonic, "probit": fit_probit}
    passed, scored = y >= THRESHOLD, pred >= SCORED_FROM
    held_out = {m: np.empty(len(y)) for m in fitters}
    for train, val in KFold(FOLDS, shuffle=True, random_state=FOLD_SEED).split(pred):
        for m, fitter in fitters.items():
            model = fitter(pred[train], spread[train], passed[train])
            held_out[m][val] = _chance(m, model, pred[val], spread[val])
    loss = {m: _log_loss(p[scored], passed[scored]) for m, p in held_out.items()}
    method = min(loss, key=loss.get)
    models = {m: fitter(pred, spread, passed) for m, fitter in fitters.items()}
    table = reliability(held_out[method][scored], passed[scored])
    return Calibration(method, models, loss, table)


def pass_probability(cal, pred, spread):
    """Each grid's calibrated chance of passing this advisor, in [0, 1]."""
    return _chance(cal.method, cal.models[cal.method], pred, spread)


def fit_all(advisors=ADVISORS):
    """{advisor: Calibration}, from the mean and spread of the incumbent's 3 saved shuffles."""
    grids, dev = validity.load_dev()
    cals = {}
    for advisor in advisors:
        _, y, _ = search._advisor_data(grids, dev, advisor)  # same row order as preds
        preds, _ = search.load_outer(search.INCUMBENT[advisor], advisor)
        cals[advisor] = fit(preds.mean(0), preds.std(0), y)
    return cals


def reliability(p, passed):
    """(low, high, grids, mean chance, pass rate) per bin. Calibrated: the last two agree."""
    which = np.digitize(p, BINS[1:-1])
    rows = []
    for b in range(len(BINS) - 1):  # empty ones too, so no row looks missing
        i = which == b
        means = (p[i].mean(), passed[i].mean()) if i.any() else (None, None)
        rows.append((BINS[b], BINS[b + 1], int(i.sum()), *means))
    return rows


def report(advisor, cal):
    """Print both log losses, the choice, the probit's parameters and the reliability table."""
    c, s0, s1 = cal.models["probit"]
    n = sum(row[2] for row in cal.table)
    losses = ", ".join(f"{m} {v:.1f}" for m, v in cal.log_loss.items())
    print(f"\n{advisor}: held-out log loss on {n:,} grids predicted ≥ {SCORED_FROM}")
    print(f"  {losses} (lower is better) -> uses {cal.method}")
    print(f"  probit: chance = Φ((pred − {c:.3f}) / ({s0:.3f} + {s1:.2f} × spread))")
    print("  held-out chance   grids  mean chance  observed pass rate")
    for low, high, count, chance, rate in cal.table:
        means = f"{chance:11.3f}  {rate:18.3f}" if count else f"{'-':>11}  {'-':>18}"
        print(f"  {low:.2f} to {high:.2f}     {count:6,}  {means}")


if __name__ == "__main__":
    for advisor, cal in fit_all().items():
        report(advisor, cal)
