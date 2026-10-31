# Decision log

This log records every modelling decision, the comparison or evidence behind it, and its source.
It serves three purposes:

1. **Limiting selection bias.** Selection bias is the best of many compared options looking better
   than it is, because chance favoured it (`stage1_evaluation.md`, "Validation and test sets"). Each
   comparison made on the development set is a look at that data. Recording every comparison keeps
   the number of compared options within the caps below (in Stage 1, *variants*: model
   configurations; from Stage 2, *procedures*), so that chance does not decide which model wins.
2. **Fixing decision rules in advance.** The thresholds below were set before the results they judge,
   with two disclosed exceptions (rule 2.2 and the Stage 2 adoption rule; see the 28 October entry on
   the Stage 2 protocol). Any change to a threshold requires an entry explaining why.
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

## Decision rules fixed in advance (project plan of 25 October 2026, revised 28 October)

Task numbers refer to the stages of the project plan in [`plan.md`](plan.md): 1 = evaluation
protocol, 2 = models, 3 = generating designs, 4 = selecting the final 100. Rule 2.8 was added on
26 October. The rows marked "revised" were made exact on 28 October, before the Stage 2 harness ran
(see that day's entry on the Stage 2 protocol). Two exceptions, disclosed in that entry: rule 2.2's
revised test had already been run in the 26 October audit, so its outcome for Wellness and Tax was
known; and rule 1.3's adoption rule was chosen knowing how rule 2.2's true-orientation arm fares
under it for Wellness. The original wording is in `plan.md`, except rule 2.8's (added 26 October),
which is in commit 8a558fd.

*Provenance.* The plan was last edited at 09:59 on 25 October, before the first scorecard run at
10:11 (the timestamps of the cache files), so the original rules predate the results they judge.
This log was first committed on 28 October (commit 8a558fd).

| Task | Condition | Action |
|---|---|---|
| 1.3, revised | Models are compared during development | Compare them by the approval count at 98% precision for each advisor (rationale: entry of 26 October). From Stage 2, a model is adopted only under the Stage 2 adoption rule: the paired test and at least 3% more approvals (28 October protocol) |
| 2.2, revised | A paired test on the same 15 folds: GBM given the engineered features plus the one-hot inputs (which record each cell's position) in their true orientation, against the same model with the one-hot inputs computed on re-oriented grids. In each of five draws (seeds 1 to 5), every grid gets one of its seven other orientations at random, kept across folds. Per-fold mean squared error; the control's is the mean over the five draws' models (their predictions are not averaged). Orientation matters for an advisor if the corrected t is at least 1.761 (one-sided 5%), with no correction across the four advisors, because a finding leads to the cautious action. A label-free test of whether the dataset itself has orientation patterns is reported alongside | If orientation matters for an advisor: no augmentation for it, and no rotated or reflected copies of designs in submissions (rule 4.2). Otherwise augmentation is one of the search's options, kept only if inner scores favour it. Position-aware features compete inside the search for every advisor either way |
| 2.3, revised | A CNN (convolutional neural network: a model that scans the grid with small learned pattern detectors, called filters), fitted inside each outer training set to the inner out-of-fold errors (predicted minus true rating) of the advisor's procedure in force, predicts that procedure's out-of-fold errors on the outer validation grids with an R² above 0.05, pooled over the 15 outer folds. The CNN's settings are the CNN procedure's inner choice, fixed before this R² is seen. Run in Session 2, and again if Session 3 changes the procedure. The squared correlation is reported too, because inner errors (from models trained on 2,400 grids) and outer errors (3,200) differ in scale | Identify and add the missing feature before Stage 3 |
| 2.4, revised | A readable formula (a short explicit expression in a few features), found using shuffle 1's folds 1–3 only and recorded here before it is scored; one formula per advisor. The hunt reads data only through a loader that returns those grids: not `scorecard()`, `out_of_fold()` or `data/cache/oof/`, and any helper model is trained on folds 1–3 only. On folds 4–5 it ranks the grids with a Spearman rank correlation of at least 0.99 (agreement between two orderings of the same grids; 1 means identical order), and approves at least as many grids at 98% precision as the incumbent's out-of-fold predictions for the same grids | Use the formula instead of a model for that advisor. It counts as one of the advisor's procedures. Folds 4–5 are its only clean test, because folds 1–3 overlap every outer fold of shuffles 2 and 3, so later challengers for that advisor are compared with it on folds 4–5 only (at least 3% more approvals) |
| 2.7, revised | The test set is about to be opened | First commit each advisor's expected test-set precision, from approval cut-offs chosen by cross-fitting on the development set, with a Wilson 95% interval. The graded model is the final procedure refit on all 4,000 development-set grids, with those cut-offs; the expectation comes from models trained on 3,200 grids, so the comparison also absorbs the effect of the larger training set. Also commit each advisor's predictions for its 1,000 test-set grids (making them needs no ratings), with the cut-offs; a committed grading script reads only those files and records their SHA-256 hashes in the log. The working tree must be clean: `validity.open_lockbox()` refuses while git shows anything uncommitted in the repository, untracked files included, and logs the calling script; code outside the repository escapes that check, hence the committed script. Then open it once, and report each advisor's test-set precision with a 95% interval. If it is more than 3 percentage points below the expected value, investigate on the development set only, and disclose |
| 2.8, revised | Measured on the development set, before the test set is opened | For each ordered pair of advisors: the correlation between one model's out-of-fold errors (predicted minus true rating) and the other model's predicted rating on the same grids, over all of them and over those the first model approves, and the first model's precision on grids the second also approves, with Wilson intervals. Disclose; use in Stage 4's chance of passing all four; models are not re-chosen on it. The co-rated test-set grids give a small supplementary check when the test set is opened |
| 3.1, revised | The design generator (the Stage 3 method that creates new grids) produces no designs by the end of 6 November | Submit a dataset-only selection first (100 grids chosen by the models from the unrated dataset grids, with no new designs); continue development |
| 3.3, revised | Three half-splits of each advisor's development set are drawn (seeds 31, 32 and 33). In each, model A trains on one half and model B on the other, each with its approval cut-off set at 98% precision on the half it did not train on; B *rejects* a grid it predicts below its cut-off. Each split's A models produce 100 of the 300 pilot designs. The baseline is unrated dataset grids that the same A models approve for all four advisors, matched to the designs on this advisor's margin above A's cut-off (bins of 0.01). For some advisor, B rejects the designs more often than the baseline, significantly at the one-sided 5% level (Cochran–Mantel–Haenszel test, with split and margin bin as strata). The *excess rejection rate* is the designs' rate minus the matched rate | Raise that advisor's safety margin (how far above 0.75 a design's predicted ratings must be) |
| 3.5, revised | The independent check model (a second model with different inputs, used to catch the primary model's errors; see the entry of 25 October on the primary model) rejects a batch of at least 100 designs more often than it rejects unrated dataset grids that the primary model approves for all four advisors, matched on the primary model's margin above its cut-off (bins of 0.01), significantly at the one-sided 5% level (Cochran–Mantel–Haenszel test). Both are the final refits that check designs, so no ratings are needed; the check model is the CNN procedure's final refit, its settings chosen by its inner search alone | Tighten the safety margin and the similarity-to-training-data requirement (how close a design must be to rated grids, so that the models are not extrapolating) before generating more |
| 3.6, revised | The 100 designs chosen from a 300-design pilot (a trial batch) by greedy max-min diversity do not exceed the diversity score of the dataset-only fallback selection: 100 grids chosen by the same greedy max-min rule from a random 300 of the dataset grids approved for all four advisors, computed and recorded in Stage 2 before the pilot (both scored by `scoring.diversity`, treating all 100 as valid) | Require designs to cover different district compositions |
| 4.2, revised | The rotation test has not passed. The test: submit a set, then the same set rotated by 90°, then the same set reflected; it passes if all three scores are equal | Submit no rotated copies; after it passes, at most four copies per design. If rule 2.2 finds that orientation matters for any advisor, no rotated or reflected copies are submitted, whatever the test shows |
| 4.4, revised | The first submission of generated designs has not been scored | Multiply each designed grid's estimated pass probability (its estimated chance of passing all four advisors, from Stage 2's calibration) by max(0, 1 − 2r), where r is the largest per-advisor excess rejection rate found by rule 3.3 (against its matched baseline), or 0 if none is positive |

**Notes.**
- **Rotated copies (rules 2.2 and 4.2).** If the advisors ignore orientation, each valid design's
  seven rotations and reflections are also valid. Each differs from the original cell by cell, so it
  adds diversity. Rules 2.2 and 4.2 test the assumption before any copy is submitted.
- **Why rule 2.2 uses mean squared error.** The rule asks whether position explains any part of the
  ratings at all. Error across the whole 0–1 range measures that, whereas the approval count reflects
  only the top of it.

## Procedure cap (from 28 October; replaces the variant cap)

At most six *decision-bearing procedures* per advisor are compared at the outer level of Stage 2's
nested cross-validation, the incumbent included (see the 28 October entry on the Stage 2 protocol).
- A procedure is a search space, a search method and a final refit. Configurations tried inside its
  search don't count.
- A procedure counts from its first prediction on an outer validation fold, if those results could
  lead to its adoption. That includes quick screens, rule 2.2's true-orientation arm and rule 2.4's
  formula. Diagnostics that cannot be adopted don't count: rule 2.2's re-oriented arm and rule 2.3's
  error model.
- A procedure changed after its outer results were seen gets a new name and counts again.
- The starting incumbent is Stage 1's primary model, GBM on the 75 engineered features. After an
  adoption, the adopted procedure is the incumbent, for the adoption rule, the quick screen and
  rule 2.4. Stage 1's three baselines (`ridge_onehot`, `gbm_onehot`, `ridge_engineered`) are
  references, not candidates.
- Part 2 lists more ideas than the five challenger slots. An idea becomes a challenger only when it
  is registered, before any of its outer results exist; a benchmark that is never registered (such
  as FLAML's) cannot be adopted.

Until 28 October, the cap was six variants: per advisor in the plan, per role in this log's table
(an open question in the 26 October audit). GBM on engineered features plus the one-hot inputs was
compared outside the scorecard (planning experiments, 26 October audit, 28 October verification);
in Stage 2 it is rule 2.2's true-orientation arm, which counts as a procedure. A CNN with rotation
augmentation was compared only in the planning experiments; it is re-examined inside the CNN
procedure.

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
     by its true rating, test set included. After the split, only the 26 October audit opened it,
     reading its header and its values at development-set positions to confirm what it held; no
     test-set position was read. It was deleted on 28 October without being opened again.
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
- **R² noise threshold.** For the primary model: Wellness 0.009, Tax 0.002, Transportation 0.005,
  Business 0.004. Until 28 October, rule 2.2 required a position-dependent feature to improve R² by
  more than these values; it now uses a paired test (see the rules).
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
  - Rule 2.2 is an exception: it uses mean squared error (see the notes under the rules).
  - From Stage 2, adopting a model also needs a paired test on per-fold `hp_recall` (see the Stage 2
    protocol).
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

### 28 October 2026: Audit fixes applied before Stage 2

The fixes came in two rounds: the first from the 26 October audit, the second from a read-only
verification of the first round by four independent reviewers the same day.

**What changed.**
- **Saved-prediction fingerprint** (`validity.cache_key`): it now covers the grids themselves, the
  model inputs built from them, and the NumPy and SciPy versions. Editing a feature definition, or
  passing different grids under the same row numbers, can no longer reuse old predictions. What it
  still does not cover is listed in `stage1_evaluation.md` ("How the code fits together").
- **Correctness:**
  - `approved_at` never stops inside a group of tied predictions;
  - a shuffle that approves nothing no longer turns an advisor's cut-off into "none";
  - `--grid` handles negative grid numbers, and a number outside the range stops with the valid
    range;
  - `score()` works for one to three shuffles (for example, a one-shuffle screen);
  - the featurizers accept an empty batch;
  - Figure 7 draws every prediction: Figures 6 and 7 widen their vertical range only when a
    prediction falls outside it;
  - Figure 4's code lists above the plot any precision level that no count reaches, or that is
    reached beyond the plotted range.
- **`setup_data.sh`:** `--update` fetches the pinned commit before comparing, which removes the
  false "utils_public.py CHANGED" alarm; a missing or malformed pin file now stops with a message
  and exit code 3.
- **Test set:** `open_lockbox()` refuses to open while git shows anything uncommitted in the
  repository, untracked files included whatever git's settings say, and refuses a folder without
  its own repository (rule 2.7). The log records the full commit hash and the calling script.
- **Starter notebook:** its ratings cell hides the test-set ratings when run locally; on Colab,
  where it cannot, it stops with a message.
- **Deleted:** `data/cache/preds_lightgbm.npy`.
- **Tests:** 44 in all. New ones cover:
  - a spy model that refuses to predict any grid it trained on, or a rotated or reflected copy of
    one, with and without augmentation, and a fresh model each round;
  - the 8 orientations;
  - a confirmed test-set opening on made-up data, and refusals (unconfirmed, or with anything
    uncommitted) that load and log nothing;
  - untracked, edited and staged files counting as uncommitted, whatever git's settings say, and a
    folder without its own repository refused;
  - ties and no approvals;
  - a single shuffle;
  - empty input;
  - a clean course clone;
  - a fingerprint that tracks grids, inputs, library versions, the number of folds and
    augmentation.
- **Fault injection:** four new faults: models trained on the grids they predict; with
  augmentation, each round trained on every grid; one model reused across rounds; an orientation
  helper that stops rotating. All seven faults are caught.
- **Environment:** Optuna 5.0.0, FLAML 2.7.0 and XGBoost 3.4.2 (needed by FLAML's AutoML) added.
  GitHub's test environment uses `matplotlib-base`.

**Verification.** The new fingerprint forced a full recompute (7 minutes). All 16 new prediction
files are bit-for-bit identical to the earlier ones, so every Stage 1 number is unchanged, for
example approval counts of 534, 828, 744 and 836. The documentation figures are byte-identical, and
so are the analysis notebook's, except Figure 7, which now draws every prediction.

*Source:* `pytest`; `python tests/fault_injection.py`; `python validity.py`. The bit-for-bit check
was a one-off comparison of the old and new cache files.

### 28 October 2026: Stage 2 protocol (recorded before the Stage 2 harness ran)

*Basis.* The Stage 2 plan (`plan.md`, Part 2, approved 28 October), refined by an independent design
review and revised after the read-only verification, both the same day. It settles the open
decisions listed in the 26 October audit entry. The Stage 2 harness had not run. The audit and the
verification had already run rule 2.2's comparison, and the verification had compared its
true-orientation arm with the incumbent; these and the other results already known are listed under
"Disclosure" below.

**Folds.**
- **Outer:** Stage 1's five folds under all three shuffles (`cv_splits`, seeds 0, 1 and 2), for each
  advisor. Every procedure's outer predictions line up fold by fold with Stage 1's, so comparisons
  are paired.
- **Inner:** four folds, one shuffle, inside each outer training set (2,400 training and 800
  validation grids). The inner folds are seeded by the outer shuffle and fold, and shared by every
  procedure.
- **Multi-task procedures:** each fold is the union of the four advisors' folds. This is valid
  because the development sets are disjoint across advisors.
- **Quick screening:** a run on shuffle 1 alone may only *drop* a procedure, when its approval count
  is below 90% of the incumbent's on shuffle 1. Adopting a procedure needs all three shuffles. A
  screened procedure counts toward the cap; a screen that uses only inner scores does not.

**What a search optimizes (inner).**
- `hp_recall`: the share of truly passing grids approved, averaged over precision levels of 96%,
  97%, 98% and 99%, on the pooled inner out-of-fold predictions.
- It is rule 1.3's quantity, averaged over nearby levels; pooled over 3,200 grids, it rests on more
  than a handful of failures.
- Stopping weak trials early, and greedy selection of feature groups, use inner mean squared error
  instead, because it is smoother.

**What decides (outer).** A challenger replaces the incumbent (the procedure in use for that
advisor; see "Procedure cap") only if both hold:
- **The paired test:** its per-fold `hp_recall` is higher, significantly at the one-sided 5% level:
  corrected t ≥ 1.761 (Nadeau–Bengio, 15 folds, 14 degrees of freedom).
- **The margin:** its approval count at 98% precision (mean over the three shuffles) is at least 3%
  higher.

Each challenger is tested on its own, with no correction across challengers (the owner's choice;
see "Disclosure"). Otherwise the incumbent stays. Effects are reported with 95% intervals from a
paired bootstrap over grids. Rule 2.4's formula is decided by that rule instead.
- **When.** A challenger is registered before any of its outer results exist, and tested at the end
  of the first session in which its 15-fold outer predictions are complete. The challengers tested
  at the end of a session are all compared with the incumbent in force at the start of that
  session; if several pass, the one with the highest approval count is adopted. Rule 2.2's
  true-orientation arm is the exception to registering first: the verification already ran it, so
  it is registered now and tested at the end of Session 1.
- **Why these values.** In 80 comparisons between identical models trained with different random
  seeds, none passed both tests. The largest corrected t was 1.01; approval counts differed with a
  standard deviation of 0.5–1.7% (by advisor), and in 1 of the 80 the challenger approved 3% more.
- **What is risked.** Without a correction across challengers, each challenger with no real gain
  keeps up to a 5% chance of passing the paired test. Such a model approves about as many grids as
  the incumbent, at about the same precision. Rule 2.7 checks precision only, so it would catch a
  large shortfall (about 3 points or more) but not a chance adoption; that cost is accepted.
- **Limitation.** Per fold (800 grids), the primary model's 98% and 99% levels rest on 0–3
  failures, so per-fold `hp_recall` is coarse. The margin, on the pooled approval count, is the
  second safeguard.

**Cap:** at most six decision-bearing procedures per advisor (see "Procedure cap" above).

**Stacking, ensembles and calibration.** Anything a procedure learns from data is learned inside
each outer training set.
- A stacker's or blender's inputs for the outer-training grids are inner out-of-fold predictions;
  for the outer validation grids, they come from the base models refit on the whole outer training
  set.
- Ensemble weights, early-stopping epochs and any calibration inside a compared procedure are
  chosen inside the outer training set.
- A stacker's base procedures are those registered for that advisor when the stacker is
  registered, or a subset chosen by inner scores alone.
- The nested harness never reads Stage 1's saved predictions (`data/cache/oof/`): each was made by a
  model that trained on the other outer folds.
- The spy test (`test_out_of_fold_never_trains_on_the_grids_it_predicts`) is extended to the harness
  and to any stacker.

**Checks built into the harness.**
- **Registry:** the harness keeps each advisor's list of procedures and refuses to score an unlisted
  one on outer folds; a test checks the list against the cap.
- **Reproduction:** the "no search" procedure must reproduce Stage 1's out-of-fold predictions.
- **Selection by inner score:** in each outer fold, choose the procedure by its inner score alone.
  This is reported alongside as an indicative check on choosing the winner from outer results. It
  favours heavily searched procedures, whose inner score is the best of many trials.

**Calibration.**
- Each advisor's predicted rating becomes a chance of passing, cross-fitted on out-of-fold
  predictions.
- The method (isotonic regression, or a probit that widens with the models' disagreement) is chosen
  by cross-fitted log loss on grids predicted at least 0.6.
- Grids approved for all four advisors are rare among rated grids. Stage 2 measures how much more
  two half-models disagree on such unrated grids than on grids approved for one advisor, and the
  calibration's uncertainty there reflects it.

**Compute.** Part 2's time estimates assumed shuffle 1 alone and fewer inner folds; under these folds
a GBM search costs about four times as much. Session 1 times the harness and records a budget here
before any outer result. If it does not fit, the cheaper options, fixed now, are:
- fewer search trials;
- neural networks choosing their settings on one inner 80/20 split, as Part 2 planned (a network
  that feeds a stacker still needs all four inner folds);
- screening on shuffle 1 first.

**Changes from Part 2, with reasons.**
- **Outer folds:** all three shuffles (15 folds), not shuffle 1 alone (5). A paired test on 5 folds
  has 4 degrees of freedom and little power.
- **Inner folds:** four, shared by every procedure, instead of three for GBM and one 80/20 split for
  neural networks. Every outer-training grid then gets an inner out-of-fold prediction, which
  stacking needs, and inner scores are comparable across procedures.
- **Search target:** `hp_recall`, instead of the error for true ratings 0.6–0.9, because it is the
  quantity the decision is about.
- **Decision:** a paired test on per-fold `hp_recall` at the exact one-sided 5% value for 15 folds
  (1.761), instead of a paired test on the approval count with corrected t ≥ 2. Per fold, the
  primary model's approval count at 98% rests on 1–3 failures. The margin keeps the approval count
  in the decision.
- **Cap:** Part 2's "declared procedures" are defined under "Procedure cap": what counts, and from
  when.

**Disclosure: results known when this protocol was set (development set only).**
- **Rule 2.2:** the 26 October audit had already run its comparison: true-orientation inputs beat
  re-oriented ones for Wellness (corrected t 5.3) and Tax (t 3.1). Single re-orientation draws give
  Tax t from 2.0 to 3.5, and five of them, combined as the rule now says, gave t 3.4. The rule was
  worded knowing this; its seeds (1 to 5) are new.
- **The adoption rule:** the verification compared GBM on engineered features plus one-hot inputs
  (rule 2.2's true-orientation arm) with the incumbent. For Wellness, the approval count rose from
  534 to 575 (+7.8%), with per-fold `hp_recall` higher in 13 of 15 folds (corrected t 1.999). That
  passes the rule as chosen, and would fail under a Holm correction (a stricter threshold for each
  of several challengers) with two or more challengers. An earlier draft of this entry used a Holm
  correction; the owner chose the present rule knowing this result.
- **The margin:** the earlier draft's 2% sat within the seed-to-seed spread above, so it was raised
  to 3%. The Wellness result clears both.
- **Rules 3.3, 3.5 and 3.6:** the verification measured, on the development set and unrated grids,
  how often a second model rejects grids at each margin above the cut-off, including grids approved
  for all four advisors, where two half-models disagree more; and the diversity of dataset-only
  selections. These shaped the revised rules; no design existed yet.
- **The formula hunt:** its lead (up to 61 Transportation grids sharing one rating) came from all
  4,000 development-set ratings. The hunt itself uses shuffle 1's folds 1–3 only (rule 2.4).

*Source.* Every number in this entry comes from one-off scripts run during the two verification
rounds, not in the repository. Session 1's harness recomputes rule 2.2's comparison, the Wellness
adoption test, the seed-to-seed comparisons and the per-fold failure counts, and this entry will then
cite it.

**Rule revisions** (marked "revised" in the rules table):
- **The audit's findings:** paired comparisons instead of the R² noise threshold (2.2), rule 2.8
  measured on the development set, exact measurements and baselines for 2.3, 2.4, 3.3, 3.5, 3.6, 4.2
  and 4.4, and an interval for the test-set grade (2.7).
- **The verification's findings:**
  - margin-matched baselines for 3.3 and 3.5, and a bounded factor in 4.4;
  - rule 2.4's formula must also approve as many grids as the incumbent;
  - an explicit threshold and consequences for 2.2, with 4.2 tied to it;
  - rule 2.7 names the graded model, and the code enforces its clean working tree;
  - exact measurements for 2.3 and 2.8, and a defined reference selection for 3.6;
  - rule 1.3 states the whole adoption rule;
  - second round: rules 3.3 and 3.5 compare with grids approved for all four advisors; when
    challengers are tested; rule 2.2's draws defined; rule 2.3's error model and rule 3.5's check
    model defined; the formula hunt's data access limited; rule 2.7 commits the graded predictions
    and the grading script.
- **The new schedule:** rule 3.1's date moves from 28 October to 6 November, chosen by the owner.

### 31 October 2026: Harness guard, the six slots, and the CNN first (owner's choices)

- **Outer results need a commit.** `search.py` makes outer predictions only from a fully committed
  repository and saves the commit with them, so git history shows each procedure was registered
  before its results existed.
- **Slots, per advisor:** 1 the incumbent (Stage 1's GBM); 2 rule 2.2's true-orientation arm;
  3 GBM choosing among feature groups; 4 a CNN search; 5 a tuned GBM (slot 3's search plus GBM
  settings); 6 a stacker or ensemble. A rule 2.4 formula would take slot 5 or 6 for its advisor.
  Each slot's exact search space is registered in `search.REGISTRY` before any of its outer results.
  Plan Part 2's multi-task network has no slot; it would need slot 5 or 6.
- **Order:** Session 1 is the harness, rule 2.2 and the CNN search, run overnight on 31 October;
  it ends with the adoption tests on 1 November. The feature groups move to Session 2. Reason: GBM
  searches take minutes, while the CNN needs overnight compute and no input from the owner. Rule
  2.2 runs first because its result decides where the CNN may train on rotated copies.

### 31 October 2026: Compute budget (recorded before any Stage 2 outer result)

- GBM procedures, all 60 outer rounds: about 0.3–0.4 min each (`python search.py --time NAME`).
  Rule 2.2's five controls cost the same.
- The CNN search (slot 4): estimated 4–5 h overnight; confirmed with
  `python search.py --time NAME --trials 1` before its run. If over 8 h, the protocol's cheaper
  options apply: fewer trials first.
