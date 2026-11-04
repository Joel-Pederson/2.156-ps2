"""The final exam: grade the final models once on the hidden test set (rules 2.8 and 2.7).

Where this fits
---------------
Since 25 October, 1,000 rated grids per advisor have been hidden in validity.py's lockbox.
The models became final on 4 November, so they are graded on those grids, once:

1. ``--expect`` uses no test data. For each model: its expected precision (the share of the
   grids it approves that truly pass), from its out-of-fold predictions at its cut-off; and
   rule 2.8: does that precision hold on grids another advisor's model also approves? If not,
   "passes all four" is less likely than the product of the four chances. Writes
   docs/test_set_expected.json: commit it before step 2.
2. ``--open`` opens the lockbox (logged in docs/lockbox_log.md) and grades the surrogate models
   (surrogates.py, the ones that rate designs) on the test grids at the same cut-offs. More than
   3 points below expected: investigate (rule 2.7). Writes docs/test_set_result.json.

Both also write results/test_set_*.xlsx.
Run     python test_set.py --expect    then commit    then    python test_set.py --open
"""

import argparse
import json
import os

import numpy as np

import search
import surrogates
import tables
import validity
from validity import ADVISORS, THRESHOLD

EXPECTED = os.path.join(validity.PS2_ROOT, "docs", "test_set_expected.json")
RESULT = os.path.join(validity.PS2_ROOT, "docs", "test_set_result.json")
WORSE_BY = 0.03  # rule 2.7: investigate a test precision this far below expected


def _precision(approved, y):
    """(approvals, share of them truly rated ≥ 0.75); the share is NaN if none is approved."""
    n = int(approved.sum())
    return n, float((y[approved] >= THRESHOLD).mean()) if n else np.nan


def _mean(results):
    """Mean approvals and precision over the three shuffles' (approvals, precision) pairs."""
    return float(np.mean([n for n, _ in results])), float(
        np.nanmean([p for _, p in results])
    )


def expect(grids, dev, pred, cuts):
    """(expected precision per advisor, rule 2.8 rows), from development grids only. A model's
    own grids are judged by its out-of-fold predictions; the other advisors' models never
    trained on them (no development grid has two raters)."""
    expected, pairs = [], []
    for advisor in ADVISORS:
        _, y, rows = search._advisor_data(grids, dev, advisor)
        oof, _ = search.load_outer(
            search.INCUMBENT[advisor], advisor
        )  # (3 shuffles, n)
        mine = oof >= cuts[advisor]
        n, precision = _mean([_precision(m, y) for m in mine])
        expected.append(
            {
                "advisor": advisor,
                "cut-off": cuts[advisor],
                "approvals": n,
                "expected precision": precision,
            }
        )
        for b, other in enumerate(ADVISORS):
            if other != advisor:
                also = pred[rows, b] >= cuts[other]
                n_both, p_both = _mean([_precision(m & also, y) for m in mine])
                pairs.append(
                    {
                        "advisor": advisor,
                        "also approved by": other,
                        "approvals": n_both,
                        "precision": p_both,
                        "drop": precision - p_both,
                    }
                )
    return expected, pairs


def grade(pred, ratings, cuts, expected):
    """(rows per advisor, joint row): the surrogate ratings `pred` judged against `ratings`
    (the test labels; NaN elsewhere) at the cut-offs."""
    rows = []
    for a, advisor in enumerate(ADVISORS):
        test = ~np.isnan(ratings[:, a])
        y = ratings[test, a]
        n, precision = _precision(pred[test, a] >= cuts[advisor], y)
        e = expected[advisor]
        rows.append(
            {
                "advisor": advisor,
                "test grids": int(test.sum()),
                "truly passing": int((y >= THRESHOLD).sum()),
                "approved": n,
                "precision": precision,
                "expected precision": e,
                "investigate": bool(precision < e - WORSE_BY),
            }
        )
    # Grids two or more advisors rated: approved by every rater's model, do they pass all?
    rated = ~np.isnan(ratings)
    approved = (~rated | (pred >= np.array([cuts[a] for a in ADVISORS]))).all(1)
    approved &= rated.sum(1) >= 2
    passed = (~rated | (np.nan_to_num(ratings) >= THRESHOLD)).all(1)
    joint = {
        "co-rated test grids": int((rated.sum(1) >= 2).sum()),
        "approved by every rater's model": int(approved.sum()),
        "of those, pass every rater": float(passed[approved].mean())
        if approved.any()
        else np.nan,
    }
    return rows, joint


def _show(title, rows):
    print(title)
    for row in rows:
        print(
            "  "
            + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                for k, v in row.items()
            )
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    step = ap.add_mutually_exclusive_group(required=True)
    step.add_argument("--expect", action="store_true", help="step 1: no test data")
    step.add_argument(
        "--open", action="store_true", help="step 2: open the test set, once"
    )
    args = ap.parse_args(argv)
    pred, _, info = surrogates.load_ratings()  # stops if the incumbents changed
    cuts = info["cutoffs"]
    if args.expect:
        if os.path.exists(EXPECTED):  # recorded before opening; never rewritten
            raise SystemExit(f"{EXPECTED} exists")
        grids, dev = validity.load_dev()
        expected, pairs = expect(grids, dev, pred, cuts)
        _show("Expected precision at each cut-off (development grids):", expected)
        _show(
            "Rule 2.8: precision on grids another advisor's model also approves:", pairs
        )
        with open(EXPECTED, "w") as f:
            json.dump({"expected": expected, "rule 2.8": pairs}, f, indent=1)
        sheets = {"Expected": expected, "Rule 2.8": pairs}
        print(f"Saved {EXPECTED}; table {tables.save('test_set_expected', sheets)}")
        print("Commit it, then: python test_set.py --open")
        return
    if os.path.exists(RESULT):
        raise SystemExit("The test set has been graded already (rule 2.7: once).")
    with open(
        EXPECTED
    ) as f:  # must exist, and be committed: open_lockbox needs a clean tree
        expected = {
            r["advisor"]: r["expected precision"] for r in json.load(f)["expected"]
        }
    _, ratings = validity.open_lockbox(
        "rule 2.7: grade the final models once (test_set.py --open)", confirm=True
    )
    rows, joint = grade(pred, ratings, cuts, expected)
    _show("Test set: the final models at their cut-offs (rule 2.7):", rows)
    _show("Grids rated by two or more advisors:", [joint])
    with open(RESULT, "w") as f:
        json.dump({"advisors": rows, "joint": joint}, f, indent=1)
    sheets = {"Test set": rows, "Joint": [joint]}
    print(f"Saved {RESULT}; table {tables.save('test_set_result', sheets)}")


if __name__ == "__main__":
    main()
