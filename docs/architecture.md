# Models, checks and findings

The code flow and the code layout are in the [README](../README.md#code-flow). This page covers
which models we use, how we checked them, and what we learned.

## Models

| Advisor | Final models (test set, 4 Nov) | Try F's models (7 Nov; `upgraded.py`) |
|---|---|---|
| Wellness | GBM on 75 features + one-hot cells | new mean (below) + per-cell network, weight 0.3 |
| Tax | GBM on 75 features | new mean + per-cell network, weight 0.3 |
| Transportation | mean of 4 GBMs | new mean |
| Business | GBM + up to 3 extra feature groups | new mean |

- **New mean:** the average of two models, trained on the bell-curve scale (norm.ppf of the
  rating):
  - a GBM on the 75 features, the 148 lab features, a neighbourhood library and one-hot cells;
  - ridge regression plus a GBM on its errors.
- **Neighbourhood library:** 600 counts. Pairs of districts within 1–3 cells, and cells with at
  least k neighbours of a district.
- **Per-cell network:** scores each cell from its surroundings and sums the scores. The earlier CNN
  flattened the grid instead, and it lost to the GBMs for every advisor.
- **Gain:** approvals at 98% precision went from 575 to 718 for Wellness and 828 to 900 for Tax.
  (Source: [`decisions.md`](decisions.md), 7 Nov evening.)
- **Since 8 Nov evening:** 5 networks at weight 0.5 instead of 2 at 0.3, so 730 for Wellness and
  916 for Tax. Not yet tried on the scoreboard. F's own models: `data/cache/upgraded/models_tryF.pkl`.

## Checks

| Check | Script | Question | Result |
|---|---|---|---|
| Final exam | `test_set.py` | Precision on the 1,000 locked grids per advisor | 0.983–0.995 for the final models (4 Nov) |
| Fooling check | `pilot.py --margin` | Do models trained on other data still approve the designs? | Passed at cushion 0.15 for Wellness, 0.10 for the others; too cautious, as the scoreboard showed |
| CNN check | `pilot.py --check` | The same, with the CNN as judge | A false alarm: it flagged designs that then passed |
| Scoreboard | by hand, ≤ 5 a day | The real score, and so the share of grids that truly passed | [`scoreboard.csv`](scoreboard.csv) |

## What we learned

Letters are the scoreboard tries in [README Figure 2](../README.md#where-we-are).

- **Our models can be trusted with a cushion.** At cushion 0.06, every grid passed, with both the
  old models (try E) and the new (F). At no cushion, 3–9% failed (C, G).
- **The search was the first limit.** Picking from a pool gave 0.68–0.72, annealing 0.734, and
  better annealing with the new models 0.757.
- **Then the approval rule was the limit.**
  - With no rule, the same annealing reaches 0.774. Trusting our models fully gives 0.755.
  - Removing any one advisor's rule gains at most 0.004.
  - Approved grids that are far apart are rare.
- **Better models paid off.** The old models would have failed 62 of F's 100 grids, yet all 100
  passed.
- **Limits:** no 100 grids can score above 0.808. The leaderboard's best is 0.772.

## Where things are saved

| Path | Holds | In git |
|---|---|---|
| `data/cache/` | model predictions and ratings (recomputed on demand); `upgraded/` holds F's models | no |
| `data/designs/` | design pools, pilot results; `scratch_7nov/` holds the 7–8 Nov test runs | no |
| `submissions/` | every submitted `.npy`, with a `.json` of settings where a script made it | yes |
| `results/` | Excel tables of every comparison and check | no |
| `docs/decisions.md` | every decision, dated, and the checks fixed before their results | yes |

## Words

- **Cut-off:** a model's approval line. Of the grids above it, 98% truly pass.
- **Cushion (margin):** how far above the cut-off a design must be. We use 0.06.
- **Approved:** above the cut-off plus the cushion, for all four models.
- **Nested cross-validation:** every score comes from grids the model never trained on, including
  the grids used to choose its settings.
- **Simulated annealing:** a search that sometimes accepts a worse change early on, so it can leave
  a local peak, and accepts fewer as it "cools".
