"""Stage 3's pilot: do models that never saw a design's training data still approve it?

Where this fits
---------------
generate.py changes a grid until the models like it, so a design can please the models without
pleasing the advisor. Before the final pick (select_submission.py --designs), checks fixed in
advance (docs/decisions.md) measure how often that happens:

* Rule 3.3 (--margin M): each advisor's development grids are split in half three times (seeds
  31-33). Model A (the incumbent recipe, searched and fitted on one half) designs 100 grids per
  split at margin M; model B, fitted on the other half, judges them. If B rejects designs more
  often than unrated dataset grids that A approves just as confidently (same split, same 0.01
  bin of A's margin above its cut-off), by a one-sided Cochran-Mantel-Haenszel test at 5%, that
  advisor's margin must rise. The largest excess rejection rate r discounts every design's
  chance of passing by max(0, 1 - 2r) until a design submission is scored (rule 4.4).
  Rule 3.6: 100 of the 300 designs must beat the dataset-only reference on diversity.
* Rule 3.5 (--check M): the CNN, a model with different inputs, refitted on all development
  grids, judges 300 pooled designs at margin M against unrated dataset grids that the
  incumbents approve for all four, matched on the incumbents' margin.

Reads   the course data; the incumbents' and CNNs' saved outer results; data/designs/pool_*
Writes  data/designs/pilot_m<M>.json and .npz, or check_m<M>.json (never overwritten);
        results/pilot_m<M>.xlsx or check_m<M>.xlsx
Run     python pilot.py --margin 0.03    then    python pilot.py --check 0.03

Surprises
---------
* B and the CNN reject a grid below their own 98%-precision cut-off: B's is set on A's half,
  the CNN's on its saved outer predictions.
* Designs in a margin bin with no dataset grid can't be matched: counted, not tested.
"""

import argparse
import copy
import glob
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import KFold

import generate
import scoring
import search
import select_submission as sel
import surrogates
import tables
import validity
from validity import ADVISORS

SPLITS = (31, 32, 33)  # rule 3.3's half-split seeds
PER_SPLIT = 100  # designs each split's A models make
CHECKED = 300  # rule 3.5: pooled designs the CNN judges
BIN = 0.01  # margin bins that designs and dataset grids are matched in
ALPHA = 0.05
CHECK_MODEL = {a: "cnn" for a in ADVISORS} | {"Transportation": "cnn_with_rotations"}
CHECK_SEED = 35  # the CNN's inner folds for its final search: a seed used nowhere else


def _dev(advisor):
    grids, dev = validity.load_dev()
    return search._advisor_data(grids, dev, advisor)[:2]


def _searched_and_fitted(name, g, y, seed):
    """(procedure, model): a fresh copy of the recipe, searched on 4 inner folds, then fitted."""
    procedure = copy.deepcopy(search.REGISTRY[name])
    inner = list(KFold(search.INNER_FOLDS, shuffle=True, random_state=seed).split(y))
    choice, _ = procedure.search(g, y, inner)
    return procedure, procedure.fit(choice, g, y)


def _half_model(job):
    """(procedure, model, cut-off): the incumbent recipe on one half of an advisor's
    development grids; its cut-off at 98% precision on the other half."""
    advisor, seed, side = job
    g, y = _dev(advisor)
    halves = np.array_split(np.random.default_rng(seed).permutation(len(y)), 2)
    mine, other = halves[side], halves[1 - side]
    name = search.INCUMBENT[advisor]
    procedure, model = _searched_and_fitted(name, g[mine], y[mine], seed)
    cut = validity.approved_at(procedure.predict(model, g[other]), y[other])[1]
    return procedure, model, cut


def _check_model(advisor):
    """(procedure, model, cut-off): the CNN recipe refitted on all development grids; its
    cut-off at 98% precision on its saved outer predictions (mean over the 3 shuffles)."""
    g, y = _dev(advisor)
    name = CHECK_MODEL[advisor]
    procedure, model = _searched_and_fitted(name, g, y, CHECK_SEED)
    preds, _ = search.load_outer(name, advisor)
    return (
        procedure,
        model,
        validity.mean_cutoff([validity.approved_at(p, y)[1] for p in preds]),
    )


def _in_parallel(fn, jobs, workers):
    spawn = multiprocessing.get_context("spawn")  # a forked one can hang LightGBM
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=spawn) as pool:
        return list(pool.map(fn, jobs))


def matched_test(rejected, is_design, stratum):
    """Are designs rejected more often than dataset grids in the same stratum? The matched
    dataset rate weights each stratum by its designs; z and p are a one-sided, continuity-
    corrected Cochran-Mantel-Haenszel test. Strata without dataset grids are left out."""
    hits = expected = var = base = 0.0
    matched = 0
    for s in np.unique(stratum[is_design]):
        d, b = (stratum == s) & is_design, (stratum == s) & ~is_design
        n1, n2 = d.sum(), b.sum()
        if n2 == 0:
            continue
        n, m1 = n1 + n2, rejected[d | b].sum()
        hits += rejected[d].sum()
        expected += n1 * m1 / n
        var += n1 * n2 * m1 * (n - m1) / (n**2 * (n - 1))
        matched += n1
        base += n1 * rejected[b].mean()
    z = (hits - expected - 0.5) / np.sqrt(var) if var > 0 else 0.0
    rate, base = (hits / matched, base / matched) if matched else (np.nan, np.nan)
    return {
        "designs": int(is_design.sum()),
        "designs matched": matched,
        "design rejection rate": rate,
        "matched dataset rate": base,
        "excess rejection rate r": rate - base,
        "CMH z": float(z),
        "p (one-sided)": float(norm.sf(z)),
        "significant": bool(norm.sf(z) < ALPHA),
    }


def rule_3_3(margin, workers):
    """(rows per advisor, rule 3.6 and 4.4 summary, pilot arrays)."""
    grids, dev = validity.load_dev()
    unrated = grids[np.isnan(dev).all(1)]
    jobs = [(a, seed, side) for seed in SPLITS for a in ADVISORS for side in (0, 1)]
    print(f"fitting {len(jobs)} half-models ...", flush=True)
    fitted = dict(zip(jobs, _in_parallel(_half_model, jobs, workers)))
    split, design, a_margin, b_reject, made = [], [], [], [], []
    for i, seed in enumerate(SPLITS):
        A = {a: [fitted[a, seed, 0][:2]] for a in ADVISORS}
        B = {a: [fitted[a, seed, 1][:2]] for a in ADVISORS}
        cut_a = {a: float(fitted[a, seed, 0][2]) for a in ADVISORS}
        cuts = np.array(list(cut_a.values()))
        cut_b = np.array([fitted[a, seed, 1][2] for a in ADVISORS])
        print(
            f"split {seed}: designing {PER_SPLIT} grids with the A models ...",
            flush=True,
        )
        rows = np.random.default_rng(seed).choice(len(grids), PER_SPLIT, replace=False)
        setup = (A, cut_a, margin, generate.MAX_STEPS)
        results = generate._design_all(grids[rows], setup, seed, workers)
        ok = np.stack(
            [g for g, info in results if info["shortfall"] == 0 and info["steps"]]
        )
        approved = unrated[(surrogates.rate(A, unrated, workers)[0] >= cuts).all(1)]
        both = np.concatenate([ok, approved])
        split.append(np.full(len(both), i))
        design.append(np.arange(len(both)) < len(ok))
        a_margin.append(surrogates.rate(A, both)[0] - cuts)
        b_reject.append(surrogates.rate(B, both)[0] < cut_b)
        made.append(ok)
    split, design = np.concatenate(split), np.concatenate(design)
    a_margin, b_reject = np.concatenate(a_margin), np.concatenate(b_reject)
    rows = []
    for j, advisor in enumerate(ADVISORS):
        stratum = split * 1000 + np.floor(a_margin[:, j] / BIN).astype(int)
        rows.append(
            {"advisor": advisor} | matched_test(b_reject[:, j], design, stratum)
        )
    made = np.concatenate(made)
    picked = sel.greedy_maxmin(
        made, a_margin[design].min(1), min(scoring.N_GRIDS, len(made))
    )
    reference = _insurance_record()["reference_diversity"]
    r = float(np.nanmax([0.0] + [row["excess rejection rate r"] for row in rows]))
    summary = {
        "margin": margin,
        "designs made": len(made),
        "rule 3.6: diversity of 100 pilot designs": scoring.diversity(made[picked]),
        "rule 3.6: dataset-only reference": reference,
        "rule 4.4: largest excess rejection rate r": r,
        "rule 4.4: discount max(0, 1 - 2r)": max(0.0, 1 - 2 * r),
    }
    arrays = {"grids": made, "split": split[design], "a_margin": a_margin[design]}
    return rows, summary, arrays | {"b_reject": b_reject[design]}


def rule_3_5(margin, workers):
    """Rows per advisor: the CNN's rejections of pooled designs against matched dataset grids."""
    grids, dev = validity.load_dev()
    pred, _, info = surrogates.load_ratings()
    cuts = np.array([info["cutoffs"][a] for a in ADVISORS])
    base = np.flatnonzero(np.isnan(dev).all(1) & (pred >= cuts).all(1))
    pool, pool_pred, _ = sel.load_pools(margin, info["cutoffs"])
    pick = np.random.default_rng(0).choice(
        len(pool), min(CHECKED, len(pool)), replace=False
    )
    both = np.concatenate([pool[pick], grids[base]])
    margins = np.concatenate([pool_pred[pick], pred[base]]) - cuts
    design = np.arange(len(both)) < len(pick)
    print("refitting the 4 CNN check models on all development grids ...", flush=True)
    rows = []
    for j, (advisor, (procedure, model, cut)) in enumerate(
        zip(ADVISORS, _in_parallel(_check_model, list(ADVISORS), workers))
    ):
        rejected = procedure.predict(model, both) < cut
        stratum = np.floor(margins[:, j] / BIN).astype(int)
        row = {"advisor": advisor, "check model": CHECK_MODEL[advisor], "cut-off": cut}
        rows.append(row | matched_test(rejected, design, stratum))
    return rows


def _insurance_record():
    """The newest insurance record, which holds rule 3.6's reference diversity."""
    with open(
        max(glob.glob(os.path.join(sel.SUBMISSIONS_DIR, "insurance_*.json")))
    ) as f:
        return json.load(f)


def _print(rows):
    for r in rows:
        verdict = (
            "SIGNIFICANT: raise the margin" if r["significant"] else "not significant"
        )
        print(
            f"  {r['advisor']:<15} designs rejected {r['design rejection rate']:.3f} vs matched "
            f"dataset {r['matched dataset rate']:.3f} (r {r['excess rejection rate r']:+.3f}, "
            f"{r['designs matched']}/{r['designs']} matched), p {r['p (one-sided)']:.3f}: {verdict}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    rule = ap.add_mutually_exclusive_group(required=True)
    rule.add_argument(
        "--margin", type=float, help="rule 3.3 (and 3.6): pilot at this margin"
    )
    rule.add_argument(
        "--check", type=float, metavar="MARGIN", help="rule 3.5 at this margin"
    )
    ap.add_argument("--workers", type=int, default=generate.WORKERS)
    args = ap.parse_args(argv)
    margin = args.margin if args.margin is not None else args.check
    name = f"{'pilot' if args.margin is not None else 'check'}_m{margin:.2f}"
    path = os.path.join(generate.DESIGNS_DIR, name)
    if os.path.exists(path + ".json"):  # never overwrite a result
        raise SystemExit(f"{path}.json exists")
    commit, uncommitted = validity._git_state()
    if uncommitted:  # the saved commit must hold the code that made the result
        raise SystemExit(f"Commit first. Not committed: {uncommitted[:5]}")
    record = {"commit": commit, "margin": margin}
    if args.margin is not None:
        rows, summary, arrays = rule_3_3(margin, args.workers)
        print(
            f"Rule 3.3 at margin {margin}: B models judge {summary['designs made']} designs"
        )
        _print(rows)
        for key, value in summary.items():
            print(
                f"  {key:<42} {value:.3f}"
                if isinstance(value, float)
                else f"  {key}: {value}"
            )
        np.savez(path + ".npz", **arrays)
        record |= {"rule 3.3": rows, "summary": summary}
        sheets = {"Rule 3.3": rows, "Rules 3.6 and 4.4": [summary]}
    else:
        rows = rule_3_5(margin, args.workers)
        print(
            f"Rule 3.5 at margin {margin}: the CNN judges {rows[0]['designs']} pooled designs"
        )
        _print(rows)
        record["rule 3.5"] = rows
        sheets = {"Rule 3.5": rows}
    with open(path + ".json", "w") as f:
        json.dump(record, f, indent=1, default=search._json_default)
    print(f"Saved {path}.json; table {tables.save(name, sheets)}")


if __name__ == "__main__":
    main()
