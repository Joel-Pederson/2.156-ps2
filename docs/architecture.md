# How the code fits together

**Goal:** submit 100 city grids that all four advisors rate ≥ 0.75, as varied as possible.
We can't ask the advisors, so we train models to predict them, design grids the models like,
check that the models aren't being fooled, then pick the 100 most varied.

## The pipeline

```
course data: 500,000 grids; 5,000 rated per advisor     validity.py loads it and hides
        │                                                1,000 per advisor (the test set)
        ▼
features.py        a grid → 75 numbers the models read   (features_lab.py: 148 extra, optional)
        │
        ▼
search.py          per advisor, find the best model recipe, scored honestly   ── DONE
        │          result: GBMs (decision-tree models); Transportation averages 4
        ▼
surrogates.py      refit those 4 models; rate all 500,000 grids               ── REDO (4 Nov)
        │
        ├──► calibration.py   predicted rating → chance of passing
        ▼
generate.py        edit grids until all 4 models predict "pass" + a margin    ── pilot.py runs it
        │
        ▼
pilot.py           second opinions: do models that didn't make a design      ── REDO (4 Nov)
        │          still pass it? sets the safety margin
        ▼
select_submission.py   pick the 100 most varied designs → submissions/*.npy  ── NEXT
```

## Models in force (final as of 4 Nov)

| Advisor | Model | Inputs |
|---|---|---|
| Wellness | GBM | 75 features + every cell's district |
| Tax | GBM | 75 features |
| Transportation | mean of 4 GBMs (the other rows' recipes + a tuned one) | as each member |
| Business | GBM | 75 features + up to 3 extra feature groups |

The CNN (`models_nn.py`) lost to the GBM for every advisor. It survives only as the **check
model**: it reads the raw map instead of our features, so it can catch designs that game a feature.

## Commands, in order

| Step | Command | Status |
|---|---|---|
| Compare model recipes | `python search.py --run NAME`, `--compare NAME` | done |
| Rate every grid | `python surrogates.py` | redo: Transportation's model changed (4 Nov) |
| Final exam on hidden grids | `python test_set.py --expect`, commit, `--open` | next |
| Margin sweep, final pool, CNN check | `python pilot.py` (runs generate.py itself) | overnight |
| Pick the submission | `python select_submission.py --designs M` | last |

## Where things are saved

| Folder | Holds |
|---|---|
| `data/cache/nested/` | each recipe's honest test predictions (search.py) |
| `data/cache/ratings/` | all 500,000 grids' predicted ratings (surrogates.py) |
| `data/designs/` | design pools `pool_<seed>`; pilot results `pilot_m*`, `check_m*` |
| `submissions/` | submission files (`.npy`) with their stats (`.json`) |
| `results/` | Excel tables of every comparison and check (not in git) |
| `docs/decisions.md` | every decision, dated, and the checks we fixed before seeing results |

**Other files:**
- `scoring.py` computes the course's diversity score.
- `tables.py` writes the Excel files.
- `plots.py` and `make_figures.py` draw the figures in `docs/`.
- `migrate_results.py` was a one-off rename and can be deleted.
- `tests/` holds the automated checks.

## The checks, in plain words (rule numbers from `docs/decisions.md`)

| Rule | Check | If it fails |
|---|---|---|
| 3.3 | Two copies of each model, trained on different halves of the data: one designs, the other grades. Does the grader fail designs more often than ordinary grids rated just as high? | Raise the margin |
| 3.5 | The same, with the CNN as grader | Raise the margin |
| 3.6 | Are 100 designs more varied than 100 existing grids? | Force more varied designs |
| 4.4 | — | Discount predicted pass chances by twice the extra failure rate from 3.3 |
| 2.8, 2.7 | Look once at the hidden test grids: an honest grade of each model | Investigate if > 3 points worse than expected |

## Does it work? What is known and what isn't

- **Known:** each model's accuracy on grids it never trained on (nested cross-validation), and
  that the test grids never leaked into training (automated tests).
- **Not yet known:** whether designs pass the **real** advisors. The pilot estimates it, and the
  test set grades the models; only a scoreboard submission measures it directly.

## Words

- **Margin:** how far above its pass line a design's predicted rating must be (0.03, 0.06 or 0.10).
- **Cut-off:** a model's pass line, set so that 98% of the grids it passes truly pass.
- **Recipe / slot:** a way of building a model; each advisor could try at most 6 recipes (slots).
- **Incumbent:** the recipe currently in use for an advisor.
- **Nested cross-validation:** training and scoring on separate slices, many times, so a score
  never comes from grids the model saw.
