# Decision log

This log records every modelling decision, the comparison or evidence behind it, and its source.
It serves three purposes:

1. **Limiting selection bias.** Selection bias is the best of many compared options looking better
   than it is, because chance favoured it (`stage1_evaluation.md`, "Validation and test sets"). Each
   comparison made on the development set is a look at that data. Recording every comparison keeps
   the number of compared *variants* (model configurations: inputs, model type and settings) within
   the cap below, so that chance does not decide which model wins.
2. **Fixing decision rules in advance.** The thresholds below were set before the results they judge.
   Any change to a threshold requires an entry explaining why.
3. **Supporting the reflection report.** The entries document the reasoning behind each choice.

**Entry format.** Each entry gives the date, what was compared, the data used (the development set,
never the test set), the result with its source, and the decision. Entries are in date order.

**Terms** are as defined in [`stage1_evaluation.md`](stage1_evaluation.md) and the
[README](../README.md), including:
- an advisor's *rating*;
- a grid *passes* an advisor at a rating of 0.75 or higher, and is *valid* if it passes all four;
- the *development set* ("dev" in code) and the *test set* (the lockbox);
- *shuffle*, *round* and *fold*, and *out-of-fold* predictions;
- the *R² noise threshold*;
- the *approval count at 98% precision*, and the *approval cut-off*;
- the *score* (the course's diversity score).

The *primary model* is GBM (gradient boosting machine) on engineered features.

---

## Decision rules fixed in advance (project plan, 25 October 2026)

Task numbers refer to the project plan's stages: 1 = evaluation protocol, 2 = models, 3 = generating
designs, 4 = selecting the final 100. Rule 2.8 was added on 26 October, before the test set had been
opened.

*Provenance.* The plan is a local file outside this repository, so its task numbers cannot yet be
looked up here. It was last edited at 09:59 on 25 October, before the first scorecard run at 10:11
(the timestamps of the cache files), so these rules predate the results they judge. Until this log
is committed, git cannot confirm its dates.

| Task | Condition | Action |
|---|---|---|
| 1.3 | Models are compared during development | Compare them by the approval count at 98% precision for each advisor (rationale: entry of 26 October) |
| 2.2 | Position-dependent features (features that change when a grid is rotated or reflected, such as the district in the top-left cell) raise an advisor's R² by more than its R² noise threshold | Add them for that advisor, and submit no rotated or reflected copies of designs |
| 2.3 | A CNN (convolutional neural network: a model that scans the grid with small learned pattern detectors, called filters) trained to predict the primary model's errors reaches R² above 0.05 on them | Identify and add the missing feature before Stage 3 |
| 2.4 | A readable formula (a short explicit expression in a few engineered features) ranks an advisor's development-set grids with a rank correlation of at least 0.99. Rank correlation measures agreement between two orderings of the same grids; 1 means identical order | Use the formula instead of a model for that advisor |
| 2.7 | The test set is about to be opened | First record each advisor's expected test-set precision in this log; then open it once. If test-set precision at the final models' approval cut-offs is more than 3 percentage points below the cross-validation estimate, investigate on the development set only, and disclose |
| 2.8 | The test set is evaluated (task 2.7) | On the grids rated by two or more advisors, measure for each pair whether one model's errors (predicted minus true rating) rise with the other model's predicted rating, which would lower its precision on grids approved by both. Disclose the result. It is used to interpret the scoreboard results; models are not re-chosen on it |
| 3.1 | The design generator (the Stage 3 method that creates new grids) produces no designs by the end of 28 October | Submit a dataset-only selection first (100 grids chosen by the models from the unrated dataset grids, with no new designs); continue development |
| 3.3 | A model trained on one half of the development set rejects at least 2% of the designs produced with a model trained on the other half, for any advisor | Raise that advisor's safety margin (how far above 0.75 a design's predicted ratings must be) |
| 3.5 | The independent check model (a second model with different inputs, used to catch the primary model's errors; see the entry of 25 October on the primary model) rejects more than 10% of designs | Tighten the safety margin and the similarity-to-training-data requirement (how close a design must be to rated grids, so that the models are not extrapolating) before generating more |
| 3.6 | The best 100 of a 300-design pilot (a trial batch) do not exceed a score of 0.45 (the diversity score computed by `scoring.diversity`, treating all 100 as valid) | Require designs to cover different district compositions |
| 4.2 | The scoreboard test of rotated designs has not passed | Submit no rotated copies; after it passes, at most four copies per design |
| 4.4 | The first submission of generated designs has not been scored | Reduce designed grids' estimated pass probabilities by twice the failure rate found in task 3.3. A pass probability is the estimated chance that a grid passes all four advisors, derived in Stage 4 from the models' precision |

**Notes.**
- **Rotated copies (rules 2.2 and 4.2).** If the advisors ignore orientation, each valid design's
  seven rotations and reflections are also valid. Each differs from the original cell by cell, so it
  adds diversity. Rules 2.2 and 4.2 test the assumption before any copy is submitted.
- **Why rule 2.2 uses R².** The rule asks whether position explains any part of the ratings at all.
  R² measures prediction error across the whole 0–1 range, whereas the approval count reflects only
  the top of it.

## Variant cap: at most six per role

A variant counts toward the cap if a decision depends on comparing it. The role names in code
(`Variant.role` in `validity.py`) are given in brackets.

| Role | Declared | Planned |
|---|---|---|
| Baseline, for reference only (`baseline`) | `ridge_onehot`, `gbm_onehot`, `ridge_engineered` | None |
| Primary model (`main`) | `gbm_engineered` | Stage 2 feature-study candidates (up to five more) |
| Independent check model (`second_opinion`) | None | Stage 2 CNN variants: with rotation augmentation; with larger filters; with a deeper network (more layers) |

*Open question (26 October audit).* The project plan capped comparisons at six variants per advisor
in total; this table applies six per role, which allows more. Two further variants were compared in
the planning experiments and count under the rule above: GBM on engineered features plus the
one-hot inputs, and the CNN with rotation augmentation. Which cap applies must be settled before
Stage 2.

---

## Entries

### 25 October 2026: Data split and test set fixed

- **Test set (lockbox):** 1,000 ratings per advisor. It is evaluated once, after the final pipeline
  is fixed and its expected precision is recorded here (rule 2.7), through
  `validity.open_lockbox()`, which records each opening in `docs/lockbox_log.md`.
- **Development set:** the remaining 4,000 ratings per advisor. From the Stage 1 scorecard on,
  comparisons use five-fold cross-validation under three shuffles; the planning experiments of
  25 October used the first shuffle only.
- **Assignment:** random seed 2156. A seed is the starting value of the random-number generator; it
  makes the assignment reproducible. All 286 grids rated by two or more advisors are assigned to the
  test set, because they are the only evidence of whether different advisors' models are wrong on
  the same grids. The remainder is a random sample.
- **Integrity:** the assignment's fingerprint (`3723ff00…`) is checked by `tests/test_protocol.py`.

### 25 October 2026: Disclosure of four analyses that used every rating

Before the test set was fixed, four exploratory analyses used all 5,000 ratings per advisor,
including those now in the test set. The first was added to this list on 26 October, when the
audit found it missing.

1. **The starter notebook,** run locally at about 07:16. Its saved outputs and
   `submissions/submission_20261025_071656.npy` come from this run. It plotted every rating and
   fitted the course's linear regressions on each advisor's 5,000 ratings.
2. **The first version of `validity.py`** (commit 3138846).
   - It compared linear regression on the starter notebook's district counts with LightGBM (400
     trees, learning rate 0.05) on one-hot inputs, using out-of-fold predictions.
   - It then fitted both on all ratings, screened all 500,000 grids, and estimated final scores at
     several prediction cut-offs.
   - It saved `data/cache/preds_lightgbm.npy`, in which every rated grid's prediction was replaced
     by its true rating, test set included. No code has read that file since, and it should be
     deleted.
3. **An independent review of the project strategy,** which ran its own exploratory models. It
   started at 08:39, a minute after the planning script had fixed the GBM settings, the engineered
   features and the test-set recipe.
4. **The correlations between advisors' ratings** on the grids rated by two or more advisors, all of
   which are now in the test set.

**Treatment.**
- No number from these analyses is used as evidence. Every conclusion below was re-derived on the
  development set alone.
- The GBM settings, the engineered features and the approval cut-offs were not taken from them.
  Choices made after them (such as building engineered features at all) may still carry a small
  optimistic bias.
- Drawing a new test set would not remove the exposure, because those analyses saw every rating.
- The present test set has not been used by any analysis since it was fixed.

### 25 October 2026: Primary model is GBM on the 75 engineered features

Planning experiments: five-fold cross-validation on the development set, first shuffle only, with
the test set untouched. R²:

| Model | Wellness | Tax | Transportation | Business |
|---|---|---|---|---|
| Ridge regression on one-hot inputs | 0.483 | 0.758 | 0.121 | 0.483 |
| GBM on one-hot inputs | 0.504 | 0.707 | 0.291 | 0.500 |
| CNN on one-hot inputs, with rotation augmentation | 0.812 | 0.967 | 0.865 | 0.950 |
| Ridge regression on engineered features | 0.697 | 0.868 | 0.649 | 0.602 |
| **GBM on engineered features** | **0.887** | **0.968** | **0.935** | **0.971** |
| GBM on engineered features plus the one-hot inputs (which record each cell's position) | 0.891 | 0.968 | 0.932 | 0.968 |

*Source.*
- **Where the values come from.** One-off exploratory scripts, run before the Stage 1 scorecard was
  added to `validity.py`, and not kept in the repository. The values are copied from their printed
  output.
- **What is reproduced.** `python validity.py` reproduces the four Ridge regression and GBM rows: the
  largest difference is 0.0052 (see the entry of 25 October on the Stage 1 scorecard).
- **What is not yet reproducible.** The CNN row and the last row cannot be reproduced from the
  repository. Stage 2 adds the CNN and tests position-dependent features (rule 2.2).
- **GBM settings.** `validity.GBM_PARAMS` (600 trees, learning rate 0.03, 31 leaves, row and column
  subsampling 0.8) were set by hand in the planning script, not tuned on any data. They differ from
  the first `validity.py`'s (400 trees, learning rate 0.05). The planning runs set no random seed;
  `random_state=0` was added afterwards. Without it, the primary model reproduces all four planning
  values to within 0.0004 on the same folds, which also confirms that the planning experiments used
  only the development set (26 October audit).

**Decision.** GBM on engineered features is the primary model. A CNN is retained as the independent
check model, because it learns from the raw cells rather than the engineered features and is
therefore likely to make different errors.

**Basis.**
- The differences between models are far larger than the noise. For Transportation, the same GBM
  reaches R² 0.29 with one-hot inputs and 0.94 with engineered features.
- Features that ignore orientation explain 89–97% of the variance in ratings (R² 0.887–0.971).
- Adding the one-hot inputs, which record each cell's position, changed R² by −0.003 to +0.004,
  within the primary model's R² noise thresholds (0.002–0.009, measured later by the Stage 1
  scorecard). That was read as showing the advisors ignore orientation. The 26 October audit found
  the reading too strong for Wellness and Tax (see that entry).

### 25 October 2026: Colours for district figures

- **Palette:** Residential `#f48cae` (pink), Industrial `#8c2e23` (brick), Commercial `#eba002`
  (amber), Park `#528a54` (green), Office `#145786` (deep blue).
- **Choice.** The hues follow the course's district images.
- **Check.** Every pair of colours was checked for colour difference under normal vision, and under
  simulated red-green colour blindness (the Machado, Oliveira and Fernandes 2009 model).
  - *Colour difference* is ΔE: the distance between two colours in the OKLab colour space, in which
    equal distances look about equally different.
  - All pairs were checked, rather than only neighbouring ones, because any two districts can be
    adjacent on a grid.
  - The closest pair differs by ΔE 18.0 under normal vision (minimum 15) and 13.3 under simulated
    colour blindness (target 8).
- **Labels.** Every cell also carries its district letter (R, I, C, P, O), because pink and amber
  have less than 3:1 contrast with the page background. 3:1 is the minimum contrast ratio
  recommended for graphics against their background.

*Source: `tests/test_palette.py`, which recomputes these values.*

### 25 October 2026: Stage 1 scorecard reproduces the planning experiments; Wellness is the limiting advisor

*Source: `python validity.py` (development set only, five-fold cross-validation under three
shuffles, test set untouched).*

Approval count at 98% precision: the mean over the three shuffles, with the per-shuffle counts in
brackets, and the share of the advisor's truly passing grids that are approved.

| Model | Wellness | Tax | Transportation | Business |
|---|---|---|---|---|
| Ridge regression on one-hot inputs | 67 [68/66/67] (7%) | 611 [612/608/612] (63%) | 0 [0/0/0] (0%) | 40 [4/62/55] (4%) |
| GBM on one-hot inputs | 76 [31/78/118] (7%) | 543 [562/512/556] (56%) | 5 [4/7/5] (1%) | 82 [104/74/69] (8%) |
| Ridge regression on engineered features | 242 [261/241/223] (24%) | 817 [814/822/815] (84%) | 19 [1/30/27] (2%) | 182 [185/185/177] (18%) |
| **GBM on engineered features** | **534 [550/523/528] (52%)** | **828 [842/815/828] (86%)** | **744 [745/750/738] (74%)** | **836 [847/814/847] (84%)** |

- **Completion check passed.** The largest difference in R² from the planning experiments is
  0.0052, for GBM on one-hot inputs on Business (tolerance 0.01). It compares the scorecard's
  three-shuffle mean with the planning value, which used the first shuffle only.
- **R² noise threshold (for rule 2.2).** For the primary model: Wellness 0.009, Tax 0.002,
  Transportation 0.005, Business 0.004. A position-dependent feature must improve R² by more than
  these values to count.
- **A margin above 0.75 is necessary.**
  - At a predicted-rating cut-off of exactly 0.75, the primary model approves grids that truly fail:
    77 for Wellness, 34 for Tax, 47 for Transportation and 30 for Business (first shuffle; shown in
    `analysis.ipynb`, Figure 6).
  - The approval cut-offs at 98% precision are all above 0.75: 0.824, 0.772, 0.775 and 0.765 for
    the primary model.
  - The Stage 3 margin experiment (rule 3.3) sets the margin for designs.
- **Decision.** Stage 2's feature work begins with Wellness, for which the primary model approves only
  about half of the truly passing grids at 98% precision.

### 26 October 2026: Rationale for comparing models at 98% precision (rule 1.3)

The 98% level was set in the project plan on 25 October (rule 1.3) and used by the Stage 1
scorecard. This entry records the analysis behind it, made on 26 October.

A grid counts only if it passes all four advisors. If each advisor's model has precision *p*, a grid
approved for all four is valid with probability *p*⁴, provided two things hold:
- the four models' errors are unrelated;
- each model's precision holds for the grids submitted.

At 98%, related errors cannot lower the figure much: at most 4 × 2% = 8% of the approved grids fail
any advisor. Whether the precision holds for the submitted grids is the open question
(`stage1_evaluation.md`, "Why 98%").

| Precision per advisor | Valid for all four (precision⁴) | Invalid per 100 | Approval count: Wellness / Tax / Transportation / Business |
|---|---|---|---|
| 90% | 66% | 34 | 844 / 1,008 / 959 / 1,026 |
| 95% | 81% | 19 | 668 / 916 / 840 / 931 |
| **98%** | **92%** | **8** | **534 / 828 / 744 / 836** |
| 99% | 96% | 4 | 473 / 757 / 711 / 800 |

*Source: `python validity.py`, which computes this table with `validity.precision_tradeoff()` from the
primary model's out-of-fold predictions (means over the three shuffles). The table was first computed
on 26 October by a one-off command, then added to `validity.py` the same day so that it can be
reproduced.*

**Confirmed.** Models are compared by their approval count at 98% precision.
- **Alternatives.** Raising precision to 99% halves the expected number of invalid grids but
  approves 4–11% fewer. Lowering it to 95% approves 11–25% more but more than doubles the invalid
  grids.
- **Scope.** This is a criterion for comparing models only.
  - Rule 2.2 is the exception: it uses R² (see the notes under the rules).
  - The safety margin for submitted designs is set by the Stage 3 margin experiment (rule 3.3).
  - The final 100 grids are chosen in Stage 4 by expected score: the score averaged over the
    possible outcomes of which grids turn out valid (README, "Validity and diversity together").

### 26 October 2026: Each advisor's rated grids are treated as random samples

- **Evidence.** The course says each advisor rated 5,000 grids, but not how they were chosen.
  Independent random samples would give about 50 shared grids per pair of advisors. The observed
  overlaps are 36–64: Wellness–Business 36, Tax–Transportation 43, Wellness–Transportation 44,
  Wellness–Tax 51, Tax–Business 54 and Transportation–Business 64.
- **Use.** The assumption justifies applying each model's development-set precision to dataset
  grids its advisor never rated (`stage1_evaluation.md`, "What can and cannot be checked"). It does
  not extend to designed grids.

*Source: `python validity.py --data`.*

### 26 October 2026: The tests catch three injected faults

Each fault was applied in memory to otherwise unchanged code, after the matching test had passed on
the unchanged code.

| Fault | Test that caught it |
|---|---|
| The test-set seed changes from 2156 to 1 | `test_lockbox_is_frozen` |
| Augmentation happens before the split, so rotated copies of validation grids reach training | `test_augmentation_only_touches_the_training_fold` |
| One feature's definition changes (the distance used for an absent district, 13 to 12) | `test_fast_features_match_the_slow_reference` |

*Source: `python tests/fault_injection.py`.*

### 26 October 2026: Audit of the Stage 1 work

*Method.* A read-only audit by eight independent reviewers (leakage, code, tests and CI, features
and figures, methodology, two documentation reviews, and consistency across files), plus direct
checks. No test-set rating value was read: rating values came only from `validity.load_dev()`, and
the raw data was used only to see which grids are rated. No repository file was changed.

*Source.* One-off scripts run during the audit. They are not yet in the repository, so the numbers
in this entry cannot yet be reproduced from it; Stage 2 adds the checks that decide rules 2.2 and
2.8 to the code.

**Leakage: none found in Stage 1.**
- For every advisor, the development and test sets are disjoint and together make up exactly the
  5,000 rated grids.
- No grid appears twice, exactly or as a rotated or reflected copy, inside a development set or
  across the split.
- Trained on randomly shuffled ratings, the primary model scores R² −0.08 to −0.11 out of fold, as
  it should when there is nothing to learn. A leak would let it score well.
- Retraining the primary model by hand reproduces its saved out-of-fold predictions exactly (largest
  difference 0).
- The planning experiments reproduce on the development set alone (see the 25 October entry on the
  primary model).

**Hazards found outside the reported results.**
- `data/cache/preds_lightgbm.npy` holds every rating, test set included.
- The starter notebook reads all ratings directly.

Neither affects any reported number. The disclosure entry now lists the starter-notebook run as a
fourth analysis that used every rating.

**Corrections to how the results are read.**

| Topic | Finding |
|---|---|
| Per-grid chance of passing | Approved grids within 0.02 above a cut-off truly pass 86–91% of the time, by advisor; 0.02–0.05 above, 95–97%; more than 0.05 above, at least 99.6%. A grid just above all four cut-offs is valid with probability about 0.6, not 92% |
| Orientation | Wellness: adding the one-hot inputs in their true orientation raises R² by 0.0032 on average (better in 14 of 15 folds) and the approval count from 534 to 575. True-orientation inputs beat randomly rotated and reflected ones in 15 of 15 folds (corrected t 5.3). Tax shows the same pattern more weakly (+0.0013, better in 12 of 15 folds; 15 of 15 against re-oriented inputs, t 3.1). The dataset itself may have orientation patterns, so this is strong evidence, not proof, that these advisors care about orientation |
| Random-sample assumption | A classifier given the 75 engineered features cannot tell each advisor's rated grids from unrated ones (AUC 0.495–0.499, where 0.5 means no better than chance), and the district compositions match |
| Uncertainty of the 98% | Each advisor's 98% rests on 11–16 failures (shuffle 1). The 95% intervals for the failure rate are about 1–3.6%, so invalid grids per 100 at this level are roughly 4–13 |
| R² noise threshold | It measures how single-fold R² varies, not the uncertainty of a paired difference between two models. For the Wellness comparison above, the paired differences vary about three times less than the threshold |
| Scoreboard gap | A failure's cost depends on the set: 0.0064 among well-spread grids, less among similar ones. Removing a near-copy can even raise the score (0.6243 to 0.6296 in a constructed example) |
| Test-set resolution | About 130–210 approved test grids per advisor, so measured test-set precision is uncertain by about one percentage point |

**Open decisions before Stage 2.**
1. Variant cap: six per advisor in total (the plan) or six per role (this log).
2. Rule 2.2: replace the comparison with the R² noise threshold by a paired, fold-by-fold test
   against randomly re-oriented inputs.
3. Rule 2.8: run its measurement on the development set now, by applying each model to the other
   advisors' rated grids, rather than relying on the 286 co-rated test grids.
4. Rule 2.7: report test-set precision with a 95% interval, given its resolution.
5. Rules 2.3, 2.4, 3.3, 3.5, 3.6, 4.2 and 4.4: state exactly what is measured. Examples: whether
   rule 2.3's R² is out of fold, which rank correlation rule 2.4 uses, and what "rejects" means in
   rules 3.3 and 3.5. Rules 3.3 and 3.5 should also be set against the rates seen on ordinary
   dataset grids.
6. Stage 4: estimate each grid's chance of passing from how far above the cut-offs its predictions
   sit, not from the pooled 98%.

**Action.** The documents were corrected on 26 October. The code was left unchanged so that it could
be studied as it was; the code changes the audit suggests (bug fixes, missing tests, and a fuller
saved-prediction fingerprint) await review before Stage 2.
