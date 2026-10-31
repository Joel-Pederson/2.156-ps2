# Project plans

This file keeps the approved project plans in the repository, so that the task, rule and risk
numbers cited in `docs/decisions.md` and in code comments (for example "task 2.7" or "risk R13")
can be looked up.

- **Part 1** is the original plan, approved on 25 October 2026, kept verbatim except that its title
  sits one heading level lower. Its Stage 2 section is superseded by Part 2.
- **Part 2** is the Stage 2 plan, approved on 28 October 2026.

Neither part is edited after approval. Changes to rules are recorded, with reasons, in
`docs/decisions.md`.

---

# Part 1: the project plan of 25 October 2026

## CP2 plan

> Can't read this nicely? Open this file in VS Code and press **Cmd+Shift+V** for a rendered
> preview with tables.

## The plan in five sentences

1. Build four models, one per advisor, that predict that advisor's score for any grid. Rough
   versions already exist and are accurate.
2. Measure honestly how often those models are right, without letting test data leak into training
   or decisions.
3. Use the models to **design new grids** that all four advisors should approve, with a safety
   margin for model error.
4. From thousands of designs, **pick the 100 that are most spread out**, preferring safer ones.
5. Use the scarce scoreboard submissions as **experiments** that tell us what the models can't.

Throughout, an **analysis notebook** (`analysis.ipynb`) lets you look at performance visually. It
gains a section with each stage.

---

## Team setup and branches

Each teammate takes their own approach, and the best one goes to `main` at the end.

```
main  <─────────────── at the end: the team's best solution
  ├── joel_dev  <───── merged in after each stage passes your review
  │     ├── joel_stage1   (groundwork)
  │     ├── joel_stage2   (models)
  │     ├── joel_stage3   (designing grids)
  │     └── joel_stage4   (selection + scoreboard)
  ├── fatak's branches
  └── leif's branches
```

- **How "best" is judged.** By the scoreboard score of each person's best submission. Offline
  numbers aren't comparable across different methods and data splits; the scoreboard is.
- **The shared budget.** The scoreboard allows 5 submissions a day *per group*, so the three of
  you need to agree a split, such as 2 / 2 / 1 rotating. **This is an action item for you to settle
  with Fatak and Leif.**
- **How code gets onto a branch.** I create each stage branch and write code on it, **without
  committing**. You review it, commit, and merge it into `joel_dev`. Tests run on GitHub for every
  push.
- **Before the first branch,** decide what to do with the three uncommitted items now in your
  working tree:
  - `validity.py` (it goes into Stage 1);
  - your re-run `starter_notebook.ipynb`;
  - the baseline `submissions/submission_20261025_071656.npy`.

## How you'll learn: the loop for every stage

1. **Explainer first.** I write a README section with figures explaining the stage's ideas, before
   the code.
2. **Build.** I write the code. Every file opens with a detailed header, and comments explain *why*.
3. **You run it** yourself and look at the outputs and figures.
4. **Explain it back.** I ask you 3–5 questions about what the stage does and why, and fill any
   gaps. This is how you'll be ready to explain the code to anyone.
5. **You commit** and merge into `joel_dev`.

## The sprint: 7 hours over the next few days, to get far ahead

These 7 hours are a head start, not the whole budget. The goal is **the whole pipeline working
end to end by Wednesday**, with a first generated submission ready before the scoreboard even
opens.

| Session | Stage | You'll be able to explain |
|---|---|---|
| **Today, 2h** | **1. Groundwork:** data splits, features, model scorecard, tests, GitHub checks | How the data is split and why; what features are; what leakage the tests catch |
| **Mon, 2h** | **2. Models:** robust feature study, CNN variants, final models, calibration, one-time lockbox grade, re-rating the 500k | Why GBM-on-features wins; what each advisor cares about; what "90% confident" means |
| **Wed, 3h** | **3–4. Designing grids + choosing 100 (first version)** | How a design is searched for; the three safety checks; what the Monte Carlo is doing |

Today starts with **Stage 0**: 15 minutes of your own tasks (see the development plan).

**After the sprint**, over the following weeks with more time:
- scoreboard experiments as soon as it opens;
- improving whichever part the results show is weakest;
- the rotation trick, if its test passes;
- Stage 5 (report material) before **Nov 13**.

If a session runs over, the next one picks up where it left off.

## How high to aim

You're competing against classmates who are also using AI tools, so the plan aims well past a
safe baseline.

| Reference point | Score |
|---|---|
| Our dataset-only baseline (estimate) | ~0.36 |
| 100 random dataset grids, *if all were valid* | 0.69 |
| Theoretical ceiling: every cell evenly spread over all 5 districts across the 100 grids | ~0.81 |

The levers that move us from 0.36 toward 0.7 are designing new grids, the rotation trick, and
choosing the 100 by expected score. All three are in this plan.

---

## Development plan, with risk mitigations built in

Each task is tagged with the risks it reduces (**R1–R18**, listed in the risk register at the
bottom). Each stage ends with an **exit check**: what must be true before the next stage starts.

Lines marked **Rule** are decisions made *now*, so we don't make them up later under pressure.
The thresholds are starting values. Changing one needs a line in the decision log saying why.

### Stage 0 — Before we start (your tasks, ~15 min)
- **0.1** Decide what to do with the three uncommitted items: `validity.py`, the re-run notebook,
  and the baseline `.npy`.
- **0.2** Email staff the questions under "Scoreboard plan". (R5, R10, R11)
- **0.3** Agree the scoreboard split with Fatak and Leif. (R17)

### Stage 1 — Groundwork (today, 2h) · branch `joel_stage1`
- **1.1** `features.py`: one-hot plus the 75 engineered features. (R14)
- **1.2** `tests/`: the 7 automated checks, including the leakage checks. (R13, R14, R15)
- **1.3** `validity.py`:
  - the lockbox recipe and one logged way to open it;
  - repeated cross-validation;
  - the "approved at 98% precision" scorecard;
  - a cache that can't go stale.

  (R13, R14)
- **1.4** `docs/decisions.md`, the decision log. Every model comparison gets a dated line. The log
  enforces the 6-variant cap and becomes the report. (R13, R18)
- **1.5** `setup_data.sh --check`. GitHub runs the tests on every push and the course-update check
  daily. (R14, R15)
- **1.6** A README section with figures: the splits diagram, and one grid next to its features.
  (R18)

**Exit check.**
- All tests pass, locally and on GitHub.
- The scorecard reproduces the planning numbers (within ±0.01 R²).
- You can explain the splits, the features, and what the leakage tests catch.

### Stage 2 — Models (Mon, 2h) · branch `joel_stage2`
- **2.1** Feature study: ablation; permutation importance, checked for stability across folds;
  Wellness candidate features. (R2)
- **2.2** Position test. **Rule:** if position-aware features raise any advisor's R² by more than
  its fold-to-fold noise, that advisor gets those features and the rotation trick is off. (R3)
- **2.3** Residual check. **Rule:** if the CNN can predict more than 5% of the main model's
  errors, find and add the missing feature before Stage 3. (R2)
- **2.4** Formula hunt, 30 minutes. **Rule:** if a readable formula ranks an advisor's dev grids
  almost perfectly (rank correlation ≥ 0.99), use the formula for that advisor. (R2, R6)
- **2.5** CNN variants (5×5, deeper). Pick the second-opinion model and measure its own hit rate on
  dev. (R1, R2)
- **2.6** Final GBM ensembles and calibration. (R1)
- **2.7** Write the expected lockbox numbers into the decision log, *then* open the lockbox once.
  **Rule:** if precision at our operating point is more than 3 points below cross-validation,
  investigate on dev only and disclose it. (R13, R16)
- **2.8** Check the 286 grids that two advisors both rated (all in the lockbox): are the two
  models wrong on the same grids? This is the only offline evidence about failures clustering.
  (R4)
- **2.9** Refit on all 5,000 and re-rate the 500k. Save the **fallback submission** (dataset-only,
  conservative margin), pre-registered. (R9, R11)
- **2.10** README and analysis notebook: the scorecard, "what each advisor cares about", and the
  PCA map. (R18)

**Exit check.**
- The lockbox grade is within tolerance, or investigated and disclosed.
- The fallback submission is saved and format-tested.
- You can explain why GBM-on-features wins and what calibration means.

### Stage 3 — Designing grids (Wed, first ~2h) · branch `joel_stage3`
- **3.1** Build in small steps: one search on one grid, checked by eye, then batched. **Rule:** if
  it isn't producing designs by the end of Wednesday, the fallback becomes the first submission
  and work continues. (R9)
- **3.2** Search behaviour:
  - satisfice with the margin;
  - start from dataset grids;
  - score with the most pessimistic member of each ensemble;
  - push away from designs already in the pool.

  (R1, R8)
- **3.3** Margin experiment. **Rule:** raise each advisor's margin until model B rejects under 2%
  of model A's designs. (R1)
- **3.4** Familiar-ground gate, with its threshold set from dev data only. (R1, R2, R13)
- **3.5** Second-opinion veto. **Rule:** if more than 10% of designs are vetoed, the search is
  fooling the main model. Tighten the margin and the familiar-ground gate before scaling up.
  (R1, R2)
- **3.6** A 300-design pilot before the full pool. **Rule:** if the pilot's best 100 aren't more
  diverse than the dataset baseline (0.45), switch on "fill different district mixes" before
  scaling up. (R8)
- **3.7** Record the margin experiment's failure rate on designed grids. It becomes a discount on
  designed grids' "% chance of passing" in Stage 4. (R1, R4)

**Exit check.**
- A pool of about 3,000 designs.
- A/B failure under 2%; veto rate under 10%.
- Pool variety reported; explain-back done.

### Stage 4 — Choosing 100 + the scoreboard (Wed, last ~1h, then ongoing) · branch `joel_stage4`
- **4.1** Expected-score selection, with one dice roll per family of related grids. (R7)
- **4.2** **Family cap.** No rotated copies until the rotation test passes. After that, never more
  than 4 copies of any one design, which limits how much one bad design can take down with it.
  (R3, R7, R10)
- **4.3** **Hedge the four-way estimate.** Score each candidate set assuming failures are
  independent, and again assuming they cluster. Pick the set whose *worse* score is highest. (R4)
- **4.4** **Hedge against fooled models.** Until the first generated submission comes back,
  discount designed grids' "% chance of passing" by twice the failure rate from 3.7. (R1)
- **4.5** The frontier, pass-zone and progress plots. (R12, R18)
- **4.6** Scoreboard experiments in order, each pre-registered: dataset-only → rotated → mirrored →
  first generated. Update the discounts after each. (R1, R3, R4, R5, R10, R17)
- **4.7** Always keep a best **no-rotation** submission on hand. (R10)
- **4.8** Watch the leaderboard, and keep improving whatever the frontier plot shows is the
  weakest link. (R12)

**Exit check.**
- The final `.npy` passes the format tests.
- Every submission is logged next to its prediction.
- Explain-back done.

### Stage 5 — Report (before Nov 13)
- The decision log becomes the reflection. It covers the lockbox numbers, the disclosure of the
  early peeks, which risks materialized, and which mitigations held. (R13, R18)

---

## Glossary

*Reference only — no need to study it now. Each term is explained, with a picture, in the stage
that first uses it.*

**Data and evaluation**
- **Label**: an advisor's actual score for a grid (5,000 per advisor).
- **Lockbox (test set)**: labels sealed away and opened **once** at the end for an honest grade.
  Think of it as the final exam.
- **Dev set**: the labels we're allowed to learn from and compare models on. Think of it as the
  practice problems.
- **Cross-validation (CV)**: split the dev set into 5 parts, train on 4, check on the 5th, and
  rotate. The part being checked is the **validation fold**.
- **Leakage**: test information sneaking into training or decisions, so results look better than
  they really are.
- **R²**: the share of score variation a model explains. 1.0 is perfect; 0 means no better than
  guessing the average.
- **Typical error near the cutoff**: how far off the model typically is, in score units, *for
  grids truly scoring 0.6–0.9*. That's where errors flip pass to fail.
- **Precision**: of the grids the model says pass, the fraction that truly do.
- **Recall**: of the grids that truly pass, the fraction the model finds.
- **Calibration**: making the model's confidence honest. "90% likely to pass" should come true
  about 90% of the time.

**Models and features**
- **Feature**: a number computed from a grid that a model takes as input.
- **One-hot**: 245 yes/no inputs such as "is cell (2,3) a park?". It describes each cell but
  nothing about relationships between cells.
- **Engineered features**: 75 hand-designed numbers about relationships, such as "how many
  park–home pairs are neighbours" or "how far homes are from the nearest office".
- **Ridge**: linear regression, which can only add up feature effects.
- **GBM (gradient-boosted trees, the LightGBM library)**: hundreds of small decision trees, each
  fixing the previous trees' mistakes. It's the usual winner on table-like data.
- **CNN**: a neural net that slides small pattern-detectors over the grid.
- **Receptive field**: how much of the grid one CNN output can "see". Stacking layers grows it.
- **Augmentation**: making extra training examples by rotating or mirroring grids, keeping the
  same label.
- **The 8 symmetries**: the 4 rotations, each with or without a mirror flip.
- **Invariant**: unchanged by those 8 symmetries.
- **Ensemble**: several copies of a model trained with different seeds or slices of data. Their
  disagreement shows how uncertain a prediction is.
- **Ablation**: hide one group of features at a time and measure how much worse the model gets.
- **Permutation importance**: scramble one feature's values in the validation fold and measure how
  much worse the model gets. That reveals how much the model relies on it.
- **PCA**: compresses many inputs into a few directions of greatest *input* variation. It's
  unsupervised, meaning it never looks at the scores.

**Designing grids**
- **Surrogate**: our model, standing in for an advisor we can't ask.
- **Margin**: how far above 0.75 we require the *predicted* score to be, to absorb model error.
- **Satisfice**: stop improving a design once it's good enough, instead of maximizing.
- **Simulated annealing**: improve a grid one small change at a time, occasionally accepting a
  slightly worse change early on so the search doesn't get stuck.
- **Pool**: the thousands of candidate designs we generate before choosing 100.
- **Expected score**: the average score once we account for the chance that each grid fails.
- **Monte Carlo**: estimating an average by simulating many random scenarios.

**Workflow**
- **Pre-registering**: writing down our prediction before seeing the scoreboard result.
- **Tests / pytest / CI**: automated checks that the code still does what it should. CI means
  GitHub runs them on every push.
- **Pin**: the course-repo version we build against, recorded in `course_commit.txt`.

---

## Data splits

**The dataset** is 500,000 grids, each 7×7: 49 cells, and each cell is one of the 5 districts.
About 19,700 grids carry at least one advisor's score; the other ~480,000 carry none.

Each advisor labelled a *different* set of 5,000 grids, so each advisor gets its own split:

```
5,000 labelled grids (per advisor)
│
├── LOCKBOX — 1,000 (20%) ── the test set. Opened once, after every decision is frozen.
│                             Includes all 286 grids that 2+ advisors rated (our only evidence
│                             of whether models fail together).
│
└── DEV — 4,000 (80%) ── everything else happens here:
      │
      ├── 5-fold cross-validation, repeated 3× with different shuffles
      │      train on 3,200 → validate on 800 → rotate
      │      → compares models, features and settings
      │
      ├── margin experiment: two halves of 2,000 (model A designs, model B checks)
      │
      └── calibration ("% chance of passing"): fitted on the cross-validation's
          held-out predictions, so no extra split is needed

After the lockbox grade: refit the final models on all 5,000 for designing grids.
The lockbox's job (an honest grade) is done by then, and 25% more data helps.

480,000 unlabelled grids: starting points for the design search. They have no labels, so
there's nothing to leak.
```

The lockbox recipe (seed 2156) is the one the planning probes already respected. It is frozen,
and a test guarantees it can never silently change.

**Validation vs test: which is which?**
- **Training set**: what a model learns from. Here, the 3,200 grids in each cross-validation
  round.
- **Validation set**: what we check against *while making decisions*, such as which model, which
  features and which settings. Here, that's the rotating 800-grid slice inside cross-validation,
  and we look at it many times.
- **Test set**: what we check against *once*, after every decision is frozen. That's the lockbox.

The difference matters because a validation set gets used up. Every decision made by looking at it
nudges us toward choices that happen to suit those particular grids, so its numbers drift
optimistic. The test set stays honest because nothing was ever decided on it.

---

## Where we are

The dataset-only baseline is about **0.30–0.36**. My earlier "about 0.15" came from a weak model
that missed most passing grids.

| Require predicted ≥ | Pass all four | Est. truly valid | Diversity of best 100 | Est. score |
|---|---|---|---|---|
| 0.75 | 743 | 80.5% | 0.446 | **0.36** |
| 0.80 | 466 | 95.1% | 0.317 | 0.30 |
| 0.85 | 246 | 98.9% | 0.284 | 0.28 |

Stricter requirements mean fewer failures, but fewer and more similar grids to choose from.

**The advisors fight each other.** These are the correlations between district counts and each
advisor's score on the dev set. Wellness wants parks (+.64) and Tax hates them (−.69). Tax likes
commercial (+.29) and Business doesn't (−.58). Passing grids have to settle both fights at once,
which is why they're rare and look alike.

---

## Which models, and why

R² on the dev set, using cross-validation, with the lockbox untouched (1.0 is perfect):

| Model, and what it's given | Wellness | Tax | Transportation | Business |
|---|---|---|---|---|
| Ridge (linear), one-hot cells | 0.48 | 0.76 | 0.12 | 0.48 |
| GBM, one-hot cells | 0.50 | 0.71 | 0.29 | 0.50 |
| CNN, raw cells + rotate/mirror augmentation | 0.81 | 0.97 | 0.87 | 0.95 |
| **GBM, 75 engineered features** | **0.89** | **0.97** | **0.94** | **0.97** |

1. **The input matters more than the model.** The same GBM goes from 0.29 to 0.94 on Transportation.
2. **Advisors care about relationships, not location.** The features don't change when a grid is
   rotated, yet they explain 89–97%.
3. **The winner is accurate where it counts.** Its typical error near the cutoff is 0.05–0.10, and
   when it predicts ≥ 0.80 it's right 96–100% of the time. Wellness is the weakest.

**Decisions.**
- **Main model:** GBM on engineered features, one ensemble per advisor.
- **Second-opinion model:** the CNN. It's a separately built model that checks finished designs.
  This is **not** reinforcement learning.
- **Not using:**
  - one-hot-only models;
  - nearest-neighbour methods;
  - transformers or graph nets;
  - AutoML (automatic "try hundreds of models").
- **No "try everything".** Every extra variant compared on the same folds is a small peek, and
  luck starts picking winners. The cap is ≤ 6 variants per advisor, declared in advance.

### How the 75 features were designed

Hand-designing features means turning ideas about what the advisors *might* care about into
numbers a computer can calculate from a grid. The ideas come from the advisors' descriptions in
the notebook, such as "accessibility of green spaces" or "commute distance". Five questions cover
them:

| Question | Feature family | How many |
|---|---|---|
| How much of each district is there? | Counts | 5 |
| What sits next to what? | Neighbour-pair counts, side-by-side and diagonal, for every pair of districts | 30 |
| How far is X from the nearest Y? | Average distance in steps from each district to the nearest cell of each other district | 20 |
| Clumped or scattered? | Number of separate blobs of each district, and the biggest blob's size | 10 |
| Edge or centre? | How many of each district sit on the border; their average distance from the centre | 10 |

A worked example on a small grid (P = park, R = home, C = commercial, O = office, I = industry):

```
P R C
R P R
O R I
```

- **Park count** = 2.
- **Park–home side-by-side neighbour pairs** = 6. The corner park touches 2 homes and the centre
  park touches 4.
- **Average distance from a home to the nearest park** = 1.0 step, since every home touches a park.
- **Park blobs** = 2, the biggest of size 1. The two parks only touch diagonally, and a blob only
  counts side-by-side neighbours.

Every family was chosen to be **rotation-proof**: rotate the grid, and the counts, pairs, distances
and blobs all stay the same.

Hand-design proposes candidates; the data decides which ones matter. That's the feature study.

### Why the CNN uses 3×3, and the variants we'll test (Stage 2)

3×3 is the size of each pattern-detector, **not** the limit of what the CNN sees:

- Each layer looks at a cell plus its 8 neighbours.
- Stacking three layers grows the view from 3×3 to 5×5 to 7×7, which is the whole grid.
- The final layer then combines every position.

Small stacked detectors are the standard choice. They need fewer parameters than one big
detector, which matters with only 3,200 training grids. Stage 2 still tests this rather than
assuming it, with two declared variants:

- 5×5 detectors;
- a deeper network.

### Feature study: the robust version (Stage 2)

**Why not PCA for this.** PCA is unsupervised: it finds the directions in which the *inputs* vary
most, without ever looking at the scores. The features that vary most across grids aren't
necessarily the ones the advisors care about. PCA also blends features into mixtures, which makes
them harder to interpret. And GBMs already pick out useful features themselves, so 75 features
and 4,000 examples don't need compressing.

We'll still **use PCA for a picture**: the 500k grids projected to 2D and coloured by score, to
*see* where passing grids live.

**What the study does**, with everything inside dev CV:

1. **Ablation by group.** Remove each of the 5 groups (counts, adjacency, distances, clusters,
   edge/centre) in turn. How much worse does each advisor's model get?
2. **Permutation importance, per feature.** Measured fold by fold, with a check that the same
   features matter in *every* fold. Stable importance is real; importance that varies from fold to
   fold is noise.
3. **New candidate features**, aimed mainly at Wellness:
   - the share of homes within 1–2 steps of a park;
   - the number of separate park clusters;
   - the worst-case home-to-park distance;
   - two-step neighbourhood counts.

   Each one is kept only if it helps in CV.
4. **Position test.** Add a few position-aware features, such as each district's count in the top,
   bottom, left and right halves. If any advisor improves (Tax is the one hint so far), the advisors
   aren't fully rotation-blind, and the rotation trick is off.
5. **Residual check.** Train the CNN to predict the main model's *errors*. If it can, the main
   model is missing an ingredient the CNN can see, and that's exactly the blind spot a search would
   exploit.
6. **Formula hunt, time-boxed to about 30 minutes.** The scores look like rankings of a simple
   count-based formula (Transportation has only 241 distinct values across 5,000 labels). Try
   simple, readable formulas. If one ranks an advisor's grids almost perfectly, use the formula
   instead of a model for that advisor, and its errors drop to nearly zero.
7. **Output.** A figure of "what each advisor cares about", for the README and the report.

---

## Accuracy or precision?

**Neither on its own.** We tune models for *how many grids they can approve while staying right
about 98% of the time*:

- **Precision** protects the score, since each failed grid costs about 1%.
- **Recall** is room to choose varied grids.
- **R²** is a sanity check only.

The models predict the actual score, not just pass/fail, because 0.74 and 0.10 are both "fail" but
very different. **Calibration** then turns each prediction into an honest "% chance of passing".

## The constraints, and what each forces

| Constraint | What we do |
|---|---|
| A grid counts only if all four give ≥ 0.75 | **Satisfice** with a margin; don't push scores higher |
| No grid has all four labels | Only the scoreboard can test our four-way estimate |
| No access to the advisors; 5 submissions/day shared | Measure blind spots offline (margin experiment); pre-register; early submissions are experiments |
| Searches find model blind spots | Margin, "stay on familiar ground", second-opinion model |
| Advisors conflict | Watch the variety of district mixes in our designs |
| Diversity uses each grid's 10 nearest, ÷100 | Choose 100 with the course's exact scoring function; no near-copies |

## Leakage protocol

- The lockbox is opened once, through one logged function. **If its grade comes out clearly worse
  than cross-validation predicted,** we investigate on dev data only. Any second lockbox look is
  then contaminated, and the report says so.
- **Everything learned from data happens inside the training folds:** augmentation, calibration,
  and when to stop training. *Example:* rotating grids *before* splitting puts a rotated copy of a
  test grid into training, so the model has effectively seen the answer.
- At most 6 model variants per advisor, declared in advance.
- **The three early analyses that saw all labels are void.** No number from them is used anywhere.
  A new lockbox wouldn't be cleaner, because those runs saw every label. The report gets one
  sentence about it.

---

## Designing new grids (Stage 3)

**The idea is a game of "hot and cold".** Start with a random city. Try every small edit, and ask
our four models of each one: *warmer or colder?* That means: is the city closer to all four
advisors approving it? Keep the warmest edit, and repeat until all four approve with room to
spare. Save that city, then start again from a different random city.

```
  ① Take a random city from the 500k dataset
                    │
                    ▼
  ② Try every small edit to it
     (change one cell, or swap two cells)
                    │
                    ▼
  ③ Our 4 models score every edited version  (takes milliseconds)
                    │
                    ▼
  ④ Keep the edit that gets closest to
     "all 4 advisors pass, with a safety margin"
                    │
                    ▼
  ⑤ Do all 4 pass with the margin?
        no  → back to ②, and keep editing this city
        yes → continue
                    │
                    ▼
  ⑥ Safety gates. Throw the city away if:
        • it looks unlike anything we trained on
        • the second-opinion model says it fails
        • it's nearly a copy of a city already kept
                    │ survived
                    ▼
  ⑦ Add it to the pool
                    │
                    ▼
  ⑧ Pool has ~3,000 cities yet?
        no  → back to ① with a new random city
        yes → Stage 4: choose the best 100
```

**What "closest" means, with made-up numbers.** Say the margin is 0.05, so the bar is a predicted
0.80.

| | Wellness | Tax | Transport | Business | Total shortfall |
|---|---|---|---|---|---|
| Before the edit | 0.62 (0.18 short) | 0.91 | 0.80 | 0.77 (0.03 short) | **0.21** |
| After turning one office into a park | 0.70 (0.10 short) | 0.88 | 0.81 | 0.78 (0.02 short) | **0.12**, so keep it |

Notice that Tax *dropped* when a park was added. That's the parks fight in action. The search keeps
going until the total shortfall reaches 0.

**Two details:**
- Early on, the search sometimes keeps a slightly worse edit, so it doesn't get stuck in a dead
  end. That's what "simulated annealing" means.
- Hundreds of these searches run side by side, which is fast because the models score edits in
  big batches.

**Three safety checks**, because a search will find the models' blind spots:

1. **Margin, chosen by experiment.** Model A (trained on one half of dev) designs grids, and model
   B (the other half) checks them. If B rejects more than 2% for any advisor, raise that advisor's
   margin.
2. **Stay on familiar ground.** Reject designs unlike anything in training, because tree models
   can't extrapolate. "Unlike" is measured on dev data only.
3. **Second-opinion model.** The CNN vetoes finished designs it's confident will fail. It's never
   used during the search, so the search can't learn to fool it. Its veto rate is our alarm. Before
   trusting its vetoes, we measure its own hit rate on dev data, since it's weaker than the main
   model on Wellness.

## Is this a Pareto-frontier problem? Partly.

- **Not for the four advisor scores.** In ps1 the objectives were things to push as far as
  possible, so the Pareto frontier was the goal, and hypervolume measured how much of it you
  covered. Here the four scores are **pass/fail gates**: a grid at 0.76 counts exactly as much as
  one at 0.99. Chasing the frontier of the four scores would push every design to the same extreme
  edges, giving near-copies and low diversity. So we spread designs **throughout** the pass zone
  rather than along its edge. Methods built for that are called *quality-diversity*: many
  different solutions that all clear a bar.
- **Yes for safety vs diversity.** A bigger margin means more of our 100 survive, but leaves fewer,
  more alike candidates to choose from. Each margin and method gives one point (estimated %
  valid, diversity), and those points trace a frontier. The final score ≈ % valid × diversity, so
  the best point is where that product peaks. The baseline table above is three points on the
  dataset-only curve.

**Visuals for this** (Stage 4, in the analysis notebook and the README):
1. **The safety–diversity frontier**, the CP2 version of your ps1 Pareto plot.
   - x = diversity of the chosen 100, y = estimated % valid.
   - One curve per method: dataset-only, generated, and generated + rotations.
   - Dashed contours of expected score, so the best point is visible.
   - Each curve's 2-D hypervolume, the same calculation as in ps1, compares methods in one number.
2. **Pass-zone map.** Predicted Wellness vs Tax, and the other advisor pairs, with the 0.75 lines
   drawn in. The pass zone is the top-right box, so you can see the parks fight and where our 100
   sit.
3. **Spread inside the pass zone.** The PCA map with the chosen 100 highlighted.
4. **Progress curve**, like ps1's hypervolume-over-time plots. It shows the expected score of the
   best 100 as the pool grows, which tells us when more generation stops helping.

The "% valid" axis is an estimate until the scoreboard calibrates it.

## Choosing the final 100 (Stage 4)

**What the Monte Carlo does.**
- Each candidate has a "% chance of passing", say 97%, 85%, 99% and so on.
- To score one possible set of 100, roll dice: grid 1 passes if a random number comes up below
  0.97, grid 2 if below 0.85, and so on. That gives one possible "world" of which grids passed.
- Score that world with the course's `diversity_score`.
- Repeat 500 times and average. That average is the set's **expected score**.
- **Related grids share one roll.** Rotated copies of the same design, and near-identical designs,
  pass or fail *together*, so each family gets one dice roll. Rolling them separately would hide
  all-or-nothing risk.

**Why not just a formula?** When a grid fails it also stops being its neighbours' neighbour, which
changes *their* contributions too, and there's no neat equation for that.

**Searching for the best set.**
- Start with a greedy pick: keep adding the candidate furthest from those already picked.
- Then try swapping one chosen grid for an unchosen one, keeping swaps that raise the expected
  score. This automatically prefers a safe grid over a risky one that adds similar variety.

## The rotation trick (only after a scoreboard test)

- **Why it could help.** If advisors only care about relationships, a rotated design scores
  identically, but it counts as very different under the cell-by-cell distance (76% of cells
  change on average, 66% for a mirror). Every good design could yield 7 more.
- **The risk.** If an advisor cares about orientation, all 8 copies fail together, and our models
  can't warn us.
- **The test.** Submit a set, then the same set rotated, then mirrored. Rotating every grid leaves
  all distances between them unchanged, so any score change means some grid flipped between pass
  and fail.

## Scoreboard plan

- **Ask staff:**
  - How many decimals it shows.
  - Whether final grading is identical.
  - Whether experiment-style submissions are OK.
  - Whether the advisors are fixed formulas (the same grid always gets the same score).
  - Whether rotated or mirrored copies of a design count as distinct grids, and whether the
    diversity formula will stay as published.
  - Whether each teammate may submit the team's best `.npy` on Canvas.
- **Pre-register every submission** in `submissions/log.csv`: the fingerprint, method, predicted
  score and predicted number of failures.
- **First submissions are experiments:**
  1. A dataset-chosen set.
  2. The same set rotated.
  3. The same set mirrored.
  4. The first generated set.
- **Keep about 10 in reserve** for the last two days, coordinated with your teammates' share.

---

## README as we go

Each stage adds an explanation with figures to the README. The figures are generated by script into
`docs/figures/`, so anyone can regenerate them.

| Stage | Figures |
|---|---|
| 1 | The splits diagram; one grid shown next to its feature values |
| 2 | The model scorecard; "what each advisor cares about"; the PCA map of where passing grids live; a CNN receptive-field sketch |
| 3 | One design's search, step by step as a strip of city pictures; the safety-check results |
| 4 | The safety–diversity frontier; the pass-zone map; the final 100 as pictures; a Monte Carlo illustration; predicted vs actual scoreboard results |

## How long things take on your laptop

| Step | Time |
|---|---|
| Features for the ~20k labelled grids | ~2 s |
| Train one GBM / one CNN | ~1–2 s / ~8 s |
| Full model scorecard (4 advisors, repeated CV) | ~5–10 min |
| Feature study + CNN variants | ~20–30 min |
| Rate all 500k grids | ~2 min |
| Generate a pool of a few thousand designs | ~10–60 min (measured in Stage 3) |
| Choose the final 100 | a few minutes |
| All tests | < 1 min |

**Apple Silicon.** PyTorch already uses the Mac's GPU through Apple's "MPS" backend; the CNN probe
ran on it. Apple's own MLX library isn't needed for models this small, and GitHub's Linux test
machines couldn't run it.

## Files

| File | Purpose |
|---|---|
| `setup_data.sh` (exists) | Downloads the course data at the pinned version. **New:** `--check` for staff updates. |
| `validity.py` (exists) | **The model scorecard.** How often a model is right when it says "pass", measured without leakage. Stage 1 adds the lockbox recipe and fixes a stale-prediction-cache bug. |
| `features.py` | Turns grids into model inputs: one-hot plus engineered features. |
| `surrogates.py` | Trains and saves the final models, the lockbox grade, and ratings for all 500k. |
| `generate.py` | Designs new grids. |
| `select_submission.py` | Picks the final 100, checks the format, saves and logs it. |
| `plots.py` | All plotting code in one place, so the notebook and the README figures never disagree. |
| `make_figures.py` | Saves the README figures, using `plots.py`. |
| `analysis.ipynb` | Your interactive analysis notebook. It covers predicted-vs-actual plots, calibration curves, "how many grids certified at each confidence", feature importance, the PCA map, the safety–diversity frontier, design pictures and scoreboard predicted-vs-actual. It gains a section per stage. |
| `submissions/log.csv` | Pre-registered predictions next to actual scoreboard results. |
| `tests/` | Automated checks (below). |
| `.github/workflows/` | Tests on every push, plus a daily course-update check. |

**Every file opens with a detailed header:** what it's for, where it fits, what it reads and
writes, how to run it, and any surprises. Functions get short docstrings, and comments say *why*.

## Tests (run with `pytest`, and on GitHub for every push)

1. Features are unchanged under all 8 rotations and mirrors.
2. The fast feature code matches a slow, obviously-correct version.
3. The lockbox never changes; it's checked against a stored fingerprint.
4. **No leakage.**
   - Lockbox and dev never overlap.
   - After augmentation, no validation grid (nor any rotated copy of it) is in its training fold.
5. Our diversity calculation equals the course's `diversity_score` exactly.
6. The submission checker rejects a wrong shape, wrong type, values outside 0–4, and duplicates.
7. The downloaded course data matches the pin.

## Course-repo updates: check automatically, update deliberately

Auto-fetching would defeat the pin: a staff change to `diversity_score` would silently change what
our numbers mean. Instead:

- `./setup_data.sh --check` takes about a second.
- A daily GitHub job emails you if staff push something.
- Then `./setup_data.sh --update` shows exactly what changed, and you decide.

## How to run everything (once built)

```bash
./setup_data.sh                        # course data, pinned version
mamba env update -f environment.yml --prune && conda activate ps2

./setup_data.sh --check                # have staff pushed updates?
pytest                                 # all automated checks (<1 min)

python validity.py                     # Stage 1: model scorecard
python surrogates.py study             # Stage 2: feature study + CNN variants
python surrogates.py fit               #          train final models
python surrogates.py lockbox           #          the one-time honest grade (asks to confirm)
python surrogates.py screen            #          rate all 500k grids
python generate.py --pool 3000         # Stage 3: design candidates
python select_submission.py            # Stage 4: pick 100, save .npy, log it
python make_figures.py                 # refresh the README figures
# then open analysis.ipynb in VS Code (kernel: "Python (ps2)") to explore the results visually
```

## Risk register

Every risk has mitigations built into specific tasks; "(3.2)" means Stage 3, task 2 of the
development plan. *How likely* is my honest guess today.

| ID | Risk | How likely | Mitigations (where in the plan) | Warning sign |
|---|---|---|---|---|
| | **Models and data** | | | |
| R1 | Designed grids fool our models: the models are less accurate on grids we design than on random ones | **High** | Satisfice; start from dataset grids; most pessimistic ensemble member (3.2). Margin experiment (3.3). Familiar-ground gate (3.4). Second-opinion veto (3.5). Discounted pass chances when choosing (4.4). | A/B failures; veto rate; first generated submission |
| R2 | A missing ingredient: the advisors use something our features don't capture, so no model can see it | Medium | Feature study (2.1); residual check (2.3); formula hunt (2.4); the CNN veto, which reads raw cells (3.5) | The CNN can predict the main model's errors; high veto rate |
| R3 | Advisors care about orientation | Low–medium | Position test (2.2); no rotated copies until the scoreboard test passes (4.2, 4.6) | Position features help; the rotated set scores differently |
| R4 | Our "all four pass" estimate is wrong, and no grid passing all four has ever been seen | **High**, direction unknown | Pairwise check on the multi-labelled lockbox grids (2.8); worst-case selection (4.3); the first submission recalibrates it (4.6) | First submission's failures vs prediction |
| R5 | Advisors are noisy, or the scoreboard uses different advisors from the labels | Low | Ask staff (0.2); conservative margins until calibrated (3.3, 4.4) | Scoreboard vs prediction on dataset grids |
| R6 | We approximate an advisor that could have been recovered exactly | Medium | Formula hunt (2.4) | — |
| | **Search and selection** | | | |
| R7 | Chosen grids fail together | Medium | One dice roll per family (4.1); family cap (4.2) | — |
| R8 | The pass zone is too narrow for 100 different designs | Medium | Push-away from the pool (3.2); a 300-design pilot that can switch to "fill different district mixes" (3.6); the rotation trick, if allowed | Pilot diversity not above 0.45 |
| R9 | The generator slips | Medium | Build in small steps (3.1); fallback submission saved Monday (2.9) | No designs by end of Wednesday |
| | **External and process** | | | |
| R10 | Rotated copies are disallowed, or the scoring changes | Medium | Ask staff (0.2); always keep a no-rotation submission (4.7); daily course-update check (1.5) | Staff answer; update alert |
| R11 | The scoreboard opens late or shows too few decimals | Unknown | Fallback ready Monday (2.9); offline estimates good enough to choose blind; ask staff (0.2) | — |
| R12 | We don't know what a winning score is | — | Track the leaderboard; the frontier plot shows where gains remain (4.5, 4.8) | Our rank |
| R13 | Leakage creeps in during development | Medium | Logged lockbox access (1.3); leakage tests on every push (1.2); decision log + 6-variant cap (1.4); familiar-ground threshold from dev only (3.4) | A test fails; lockbox clearly worse than CV |
| R14 | A silent bug corrupts results | Medium | Tests against slow reference versions (1.2); a cache that can't go stale (1.3); CI (1.5) | Test failures |
| R15 | Staff change the data or the scoring code | Low | The pin; daily check (1.5); updates only through the diff-showing `--update` | Update alert |
| R16 | The lockbox grade disagrees with cross-validation | Low | Predictions written down first, plus the 3-point rule (2.7) | Lockbox vs prediction |
| R17 | The shared submission budget runs short | Medium | Agreed split (0.3); pre-registered log; experiments ordered by what they teach (4.6) | — |
| R18 | You can't explain the code when it counts | Low | Explainer-first README; file headers; explain-back every stage; decision log (1.4, 5) | Gaps in the explain-back |

### The blind spot worth understanding

**Model A and model B share blind spots.** The margin experiment (3.3) trains both halves on the
same 75 features. It catches a model that's overconfident from too little data, but **not a
missing ingredient**: if neither half can see something, neither can flag it. Only the CNN, which
reads raw cells, and the scoreboard can. That's why the residual check (2.3) and a trustworthy
second-opinion model (2.5, 3.5) are in the plan.

**A correction to an earlier version of this plan.** It said that if the models fail together,
multiplying their hit rates is "too rosy". That had the direction wrong. Failures landing on the
*same* grids make the multiplied estimate *pessimistic*, and other effects push the other way. It
could be off in either direction, which is why R4 is handled by a worst-case selection (4.3)
rather than by trusting the number.

---

# Part 2: the Stage 2 plan of 28 October 2026

## Stage 2 plan: models

## Context
- Stage 1 is done and audited, with no leakage. The primary model is GBM on 75 engineered features.
  Wellness is the limiting advisor (534 approvals vs 744–836 for the others).
- Your choices: nested cross-validation; about 3 sessions, with compute running overnight on the
  M1 Max; all audit fixes first; add Optuna and FLAML.
- New lead: ratings look like percentile ranks of a hidden score. They are uniform, about 25% pass,
  and up to 61 grids share an identical rating. That suggests a formula hunt.
- Goal, per advisor: the best honestly measured model, a calibrated chance of passing, an uncertainty
  signal, one test-set grade, and ratings for all 500,000 grids.

## Session 0: preparation (about 1 hour)
- You commit Stage 1; branch `joel_stage2`; save the 25 October plan as `docs/plan.md`.
- Audit fixes:
  - the cache fingerprint hashes the model inputs;
  - missing tests: `out_of_fold`, `orientations`, a confirmed lockbox opening, the pin;
  - six small bugs;
  - the starter notebook hides test-set ratings;
  - comment fixes;
  - delete `preds_lightgbm.npy`;
  - add `optuna` and `flaml`.
- Record the Stage 2 protocol and the revised rules in the decision log before any result.

## Honesty protocol: nested cross-validation
- **Outer:** Stage 1's 5 folds (shuffle 1) per advisor. These score each whole procedure, search
  included, honestly.
- **Inner:** the search runs inside each outer training set: 3 folds for GBM, one 80/20 split for
  neural networks.
- **Search target and decision:** the search optimizes RMSE for true ratings 0.6–0.9. Procedures are
  compared by approval count at 98%, paired per fold (corrected t ≥ 2).
- **Cap:** at most 6 declared procedures per advisor; unlimited configurations inside each.
- **Checks:** the no-search procedure must reproduce Stage 1 exactly; leakage tests for the harness.
- An independent design review of this harness is still running; its findings are folded in before
  the protocol is logged.

## Session 1: feature trade study
- You lead the domain research: walkability, green-space access, land-use mix, jobs–housing balance,
  industrial buffers, commuting.
- New feature families in `features_lab.py`:
  - access and coverage;
  - neighbourhood mix;
  - conflicts and buffers;
  - park quality;
  - economy;
  - commute (including matched commute);
  - rings and shape;
  - position-aware.
- **Formula hunt** (1 hour, Transportation first): tied grids share a hidden score, so find what they
  have in common.
- **Orientation test:** position features on true versus randomly re-oriented grids, paired.
- Under nested CV, compare three procedures: GBM with family selection, GBM on all families, and the
  Stage 1 baseline. Make a figure of what each advisor cares about.

## Session 2: neural networks
- Speed check: GPU versus CPU.
- CNN search with Optuna: depth, width, filter size, position channels, and augmentation (only where
  orientation doesn't matter).
- Multi-task network: one shared body with 4 heads; each grid trains only its own advisor's head.
- Residual check (rule 2.3), and choose the check model.
- Stretch: pretrain on the 480,289 unrated grids by predicting their engineered features.

## Session 3: combine and finish
- Optuna-tuned GBM; FLAML benchmark (stretch).
- Network of networks (stretch): stacking per advisor, and other advisors' predictions as inputs.
- Ensembles, which give Stage 3 its uncertainty signal.
- Pick the final procedure per advisor; calibrate predictions into pass probabilities; measure rule
  2.8 on the development set.
- The test set, once: record the expected precision first.
- Refit on all 5,000 ratings and rate all 500,000 grids (`surrogates.py`); save a fallback submission.
- Docs, notebook and explain-back; you commit.

## Compute (M1 Max; a GBM fit measured at 1.8–4.6 s)
- Overnight runs: feature selection about 4–5 h, Optuna GBM about 3–4 h.
- Daytime runs: CNN search about 1.5 h, multi-task about 1 h, FLAML about 1.7 h.
- PyTorch's MPS backend uses the Apple GPU; MLX and Core ML aren't needed. `caffeinate` keeps the Mac
  awake overnight.

## Files
- **New:** `features_lab.py`, `search.py`, `models_nn.py`, `surrogates.py`, `docs/stage2_models.md`,
  `docs/plan.md`, and tests for each.
- **Changed:** `validity.py`, `features.py`, `plots.py`, `make_figures.py`, `setup_data.sh`, the
  environment files, the starter notebook, `decisions.md`, `README.md`, `analysis.ipynb`.
- **Reused:** `validity.load_dev`, `cv_splits`, `approved_at` and `score`; the `Variant` pattern;
  `features.engineered` and `orientations`; the `grid_zoo` test fixture.

## Verification
- `pytest` passes, including the harness leakage tests. Fault injection adds a new fault, "the inner
  search sees outer-test grids", which must be caught.
- The no-search procedure matches Stage 1, and Stage 1's numbers are unchanged after the fingerprint
  fix.
- Every documented number comes from a command in the repository. Explain-back at the end of each
  session.
