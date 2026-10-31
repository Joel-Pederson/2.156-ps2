# Stage 2: Models

Part of the documentation for 2.156 Challenge Problem 2. Stage 1's write-up
([`stage1_evaluation.md`](stage1_evaluation.md)) defines the terms used here: folds, shuffles,
out-of-fold predictions, approval cut-offs, precision and the approval count at 98% precision. The
rules this stage follows were recorded before it began ([`decisions.md`](decisions.md), entry of
28 October on the Stage 2 protocol).

*Status: the harness (`search.py`) is built and tested; no outer result exists yet. Run order:
`--time` → budget in the decision log → commit → `--reproduce` → `--run` → `--compare`.*

## What Stage 2 does

Stage 1 measured four fixed models. The best, a GBM on the 75 engineered features, approves 534
Wellness grids at 98% precision, against 744–836 for the other three advisors, so Wellness limits
how many grids pass all four (`python validity.py`). Stage 2 tries to do better, mainly by
*searching*: trying many options (new features, model settings, neural networks) and keeping
whichever scores best. Searching creates a measuring problem, and most of this section is about it.

**Terms.**
- A *procedure* is a complete recipe for producing a model:
  - what may vary (the *search space*, such as "any subset of these eight groups of features");
  - how the options are tried (the *search method*, such as "add whichever group helps most, one at
    a time");
  - the *refit*: train the chosen option on all the training data the procedure was given.

  Stage 1's GBM is a procedure with nothing to search.
- The *incumbent* is the procedure in use for an advisor. It starts as Stage 1's GBM. A
  *challenger* is a procedure being compared with it.
- The *registry* lists the procedures allowed to compete for each advisor: at most six, the
  incumbent included (decision log, "Procedure cap").

## Why searching needs a second layer of folds

**Searching flatters the winner.** Even a fixed model's result moves with luck. The same GBM
approves 550, 523 or 528 Wellness grids depending only on how the grids were dealt into folds (its
three shuffles; `python validity.py`). Now suppose a search tries 50 options that are all truly
equally good. Their scores still differ by that kind of luck, and the search keeps the highest:
the luckiest. Report that same score as the procedure's accuracy and it overstates how the model
will do on new grids. The more options searched, the larger the overstatement.

**The fix: score the search on grids it never saw.** This is *nested cross-validation*. Stage 1's
folds become the *outer* folds, and the search runs inside each outer training set, on *inner*
folds of its own (Figure 1).

```text
ONE ADVISOR · 4,000 development-set grids
│
│  OUTER: Stage 1's folds, 3 shuffles × 5 folds = 15 rounds. In each round:
│
│  ┌── outer training set: 3,200 grids with ratings (all the procedure may see) ──┐
│  │                                                                              │
│  │   INNER: 4 folds of these 3,200 (train on 2,400, check on 800, four times)   │
│  │     for each option: train on 2,400 → predict the other 800 → inner score    │
│  │     keep the option with the best inner score                                │
│  │   refit the chosen option on all 3,200                                       │
│  │                                                                              │
│  └────────────────────────────────────┬─────────────────────────────────────────┘
│                                       │ predicts
│                                       ▼
│     outer validation fold: 800 grids (ratings used only afterwards, to score)
│
▼
After 15 rounds every grid has 3 predictions, one per shuffle, each from a run of
the recipe whose search never saw that grid. These give the procedure's score.
```

**Figure 1. Nested cross-validation for one advisor.** The outer rounds are Stage 1's, so every
procedure is scored on the same 15 folds as Stage 1's GBM and the two can be compared fold by fold.
Inside each round the search sees only the outer training set. The outer validation fold is
predicted once, by the refitted winner, and its ratings are used only to score that prediction. The
inner folds are fixed for each round and shared by every procedure.

**What the outer score measures: the recipe, not one model.** The search may choose differently in
different rounds: feature groups A, B and D in one round, A, B and C in another. That is expected.
The outer score says how well the recipe does when it is run on 3,200 grids. The final model comes
from running the recipe once more, on all 4,000 development-set grids.

## The inner score: high-precision recall

The search needs one number per option. It uses *hp_recall* ("high-precision recall"):
- *Recall* at a precision level is the share of the truly passing grids that the model approves
  while keeping that precision. Stage 1's GBM approves 52% of the truly passing Wellness grids at
  98% precision (`python validity.py`).
- hp_recall averages recall at 96%, 97%, 98% and 99% precision, computed on the inner out-of-fold
  predictions of all 3,200 outer-training grids at once.

It measures what Stage 2 is judged by (how many grids can be approved at high precision), averaged
over neighbouring levels so that it doesn't hinge on two or three grids. Two steps use the smoother
mean squared error instead: abandoning a hopeless option early, and adding feature groups one at a
time.

## How a challenger replaces the incumbent

Both conditions must hold, comparing the two procedures' outer predictions:

1. **It wins fold by fold.** In each of the 15 outer folds, compute both procedures' hp_recall and
   the difference (challenger minus incumbent). A one-sided paired t-test on the 15 differences must
   give a *corrected* t of at least 1.761, the 5% significance level for 15 folds. *Corrected*
   because the 15 rounds reuse the same grids. Any two outer training sets share most of their
   grids, so their results rise and fall together, and an ordinary t-test would be overconfident.
   The Nadeau–Bengio correction widens the uncertainty by a factor of
   √((1/15 + 800/3,200) ÷ (1/15)) = √4.75 ≈ 2.2.
2. **It wins by a useful margin.** Its approval count at 98% precision, averaged over the three
   shuffles, is at least 3% higher. For Wellness the incumbent's average is (550 + 523 + 528) ÷ 3 =
   533.67, so a challenger needs at least 533.67 × 1.03 = 549.7.

A challenger is tested at the end of the session in which its 15 outer folds are complete. If
several pass, the one with the highest approval count is adopted.

## What the harness enforces

`search.py` runs Figure 1 and refuses to cut corners:
- **Registry and cap.** It makes outer predictions only for registered procedures, at most six per
  advisor.
- **Registered before results.** It makes outer predictions only when everything is committed, and
  records the commit with the results, so git history shows that each procedure was registered
  before its results existed. This is the same guard Stage 1 put on the test set.
- **It reproduces Stage 1.** Stage 1's GBM run through the harness, with nothing to search, must
  give exactly Stage 1's predictions, and so 534 / 828 / 744 / 836 approvals.
- **No peeking.** A spy test gives the harness a fake model that records every grid it is trained
  on, and fails if an outer-validation grid appears in any inner or final fit. A new injected fault,
  "the search sees outer-validation grids", must be caught.
- **No reuse of Stage 1's predictions.** The harness never reads `data/cache/oof/`. Each of those
  predictions came from a model trained on the other outer folds, so using them inside a round
  would leak.

## Cost

A search over *T* options costs 4*T* + 1 GBM fits per outer round (four inner fits per option, then
the refit), and a procedure has 60 outer rounds (15 per advisor). LightGBM's default spreads one fit
across all ten cores; on data this small, ten single-core fits run side by side get through far
more work, with identical predictions (the reproduction test checks this). `python search.py --time` will
measure the real cost, and the budget goes in the decision log before any outer result.
