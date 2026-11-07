# 2.156 ps2 — Challenge Problem 2: City Zoning

MIT 2.156, Fall 2026 · Fatak Borhani, Joel Pederson, Leif Akerley · due 13 November 2026

**Goal.** Submit 100 city layouts (*grids*: 7 × 7 cells, each one of five districts:
Residential, Industrial, Commercial, Park, Office).
- **Advisors.** Four hidden scoring functions (Wellness, Tax, Transportation, Business) each rate a
  grid 0–1.
- **Valid.** A grid is valid if all four advisors rate it at least 0.75.
- **Score.** Each valid grid adds its mean distance to its 10 nearest valid grids, divided by 100.
  Distance is the fraction of the 49 cells that differ.
- **What we have.** 5,000 example ratings per advisor, and up to 5 scoreboard tries a day.

## Where we are

![Each scoreboard try: real score against the score if all 100 grids had passed](docs/figures/stage4_scoreboard.png)

**Figure 1. Our scoreboard tries (7 Nov).**
- **Reading it:** the gap between the open circle (the score if every grid passed) and the dot
  (the real score) is the grids that failed.
- **Best so far:** E, at 0.734, with all 100 passing.
- **The limits:** no submission can score above 0.808 (dotted line). With all 5 districts spread
  evenly over 100 grids, at most 80% × 100/99 of grid pairs can differ in any one cell.

*Source: `docs/scoreboard.csv`; `python make_figures.py --stage 4`.*

## How it works

![The pipeline, one box per script, with the checks beside the steps they test](docs/figures/pipeline.png)

**Figure 2. The pipeline.** Steps run top to bottom, and the checks on the right test the step
beside them. [`docs/architecture.md`](docs/architecture.md) explains each step and what each check
showed.

**Building blocks:**
- **Models:** we can't query the advisors, so we train one model per advisor (gradient-boosted
  trees on 75 features).
- **Designs:** we edit grids until all four models approve them, with a *cushion* above each
  approval cut-off.
- **The final 100:** we spread the designs apart by simulated annealing.

**What decides the score (Figure 3):**
- **Validity:** a failed grid adds nothing.
- **Variety:** a grid adds little if a few others are close to it, however far it is from the rest.

![How the diversity score is computed for one grid](docs/figures/readme_diversity.png)

**Figure 3. How the score is computed.**
- **Left:** two grids, with the 30 cells in which they differ outlined (distance 0.61).
- **Right:** one grid's sorted distances to the other 99. Only the 10 nearest (blue) count toward
  its contribution.

*Source: `python make_figures.py --stage 0`.*

## Quick start

```bash
git clone git@github.com:Joel-Pederson/2.156-ps2.git && cd 2.156-ps2
./setup_data.sh                       # course data at the pinned commit (course_commit.txt)
mamba env create -f environment.yml && conda activate ps2
pytest                                # the test suite, under a minute
```

## Running the pipeline

Every script that saves results refuses to run with uncommitted changes, so each result records
the code that made it.

| Step | Command | Time |
|---|---|---|
| Compare model recipes | `python search.py --run NAME`, then `--compare NAME` | hours |
| Rate all 500,000 grids | `python surrogates.py` | ~3 min |
| Final exam (done once, 4 Nov) | `python test_set.py --expect`, commit, `--open` | 1 min |
| Cushion checks and design pool | `python pilot.py` (runs `generate.py`) | overnight |
| Pick 100 designs | `python select_submission.py --designs 0.15,0.1,0.1,0.1` | 1 min |
| Spread them apart | `python anneal.py --start submissions/<file>.npy --margin 0.06` | 30 min–3 h |
| Figures | `python make_figures.py` | 1 min |

## Documentation

| File | What it holds |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | One page: every step, every check, what we learned |
| [`docs/decisions.md`](docs/decisions.md) | Every decision, dated, with its evidence; the source for the report |
| [`docs/scoreboard.csv`](docs/scoreboard.csv) | Every scoreboard try: file, method, cushion, score |
| [`docs/stage1_evaluation.md`](docs/stage1_evaluation.md), [`docs/stage2_models.md`](docs/stage2_models.md) | How models are evaluated and chosen |
| [`docs/plan.md`](docs/plan.md) | The original plans (25 and 28 October) |

## Collaboration

Each team member develops an independent approach. The best one, judged by scoreboard score, goes
to `main` at the end. Joel's work is on `joel_dev`. The team shares 5 scoreboard tries a day.

## Deliverables (each team member)

- **Submission:** one `.npy` file, a 100 × 7 × 7 integer array with values 0–4 (`scoring.check_submission`
  checks the format).
- **Code:** a zip of this repository.
- **AI-use statement.**
- **Reflection report (PDF):** drawn from [`docs/decisions.md`](docs/decisions.md).
