"""Regenerate the documentation's figures into docs/figures/.

Where this fits
---------------
The stage documents in docs/ explain each stage with figures. They're generated, never hand-edited,
so anyone
can rebuild them after a change: one command, same pictures. The drawing lives in plots.py;
this file only decides which figures exist and where they're saved.

Reads / writes
--------------
reads   the course data (via validity.load_dev(); dev labels only, never the lockbox)
        data/cache/oof/         Figure 4's out-of-fold predictions, via validity.scorecard()
                                (computed first if `python validity.py` has not been run)
        data/cache/nested/      Stage 2's CNN run records, via search.load_outer()
        docs/scoreboard.csv     every scoreboard try (Stage 4)
        data/cache/upgraded/    the upgraded models' held-out predictions (report figures)
writes  docs/figures/*.png      (tracked in git, because the documentation shows them)

How to run
----------
    python make_figures.py              # every stage built so far
    python make_figures.py --stage 1     # one stage only (0 = the README's own figures)
    python make_figures.py --stage 2     # the CNN settings chart, and the numbers behind it
    python make_figures.py --stage 4     # the pipeline and the scoreboard tries
    python make_figures.py --stage 5     # the report's model comparison and failure bands

Surprises
---------
Figures are saved at 150 dpi on the light chart surface, so they read correctly whether
GitHub shows the documentation in light or dark mode.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # files only, no windows

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm

import plots
import search
import upgraded
import validity

FIG_DIR = os.path.join(validity.PS2_ROOT, "docs", "figures")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, validity.PS2_ROOT)}")


def readme():
    grids, _ = validity.load_dev()
    _save(plots.fig_diversity_example(grids), "readme_diversity.png")


def stage1():
    grids, dev = validity.load_dev()
    # Ten real development-set grids (rated by Wellness) for the shuffle example.
    examples = grids[np.flatnonzero(~np.isnan(dev[:, 0]))[:10]]
    _save(plots.fig_cv_rounds(), "stage1_cv_rounds.png")
    _save(plots.fig_shuffles(examples), "stage1_shuffles.png")
    # Figure 3 shows Grid A of Figure 2, so the two figures share an example.
    _save(plots.fig_grid_features(examples[0]), "stage1_grid_features.png")
    # Figure 4 reads the primary model's cached out-of-fold predictions (python validity.py).
    results = validity.scorecard(["gbm_engineered"], verbose=False)
    _save(plots.fig_approval_curve(results), "stage1_approval_curve.png")


# Slot 4's CNN run per advisor: rule 2.2 allowed rotated training copies only for Transportation.
CNN_RUNS = {
    "Wellness": "cnn",
    "Tax": "cnn",
    "Transportation": "cnn_with_rotations",
    "Business": "cnn",
}


def _print_effects(effects):
    """The chart's numbers: per level, mean inner hp_recall ± rough 95% interval (trials)."""
    print("  Each level: mean inner hp_recall ± rough 95% interval (number of trials)")
    for advisor, by_setting in effects.items():
        for setting, levels in by_setting.items():
            cells = "   ".join(
                f"{plots.level_label(v)}: {m:.3f} ± {h:.3f} ({n})"
                for v, (m, n, h) in levels.items()
            )
            print(f"  {advisor:<15} {setting:<9} {cells}")


def stage2():
    records = [search.load_outer(name, adv)[1] for adv, name in CNN_RUNS.items()]
    effects = plots.cnn_setting_effects(records)
    _print_effects(effects)
    _save(plots.fig_cnn_settings(effects), "stage2_cnn_settings.png")


SCOREBOARD = os.path.join(validity.PS2_ROOT, "docs", "scoreboard.csv")
LEADERBOARD_BEST = 0.7749  # the course leaderboard's top score, 10 Nov 2026


def stage4():
    _save(plots.fig_pipeline(), "pipeline.png")
    with open(SCOREBOARD) as f:
        rows = [
            {
                "label": f"{r['try']}: {r['method']}, cushion {r['cushion']}",
                "variety": float(r["variety_if_all_pass"]),
                "score": float(r["score"]),
            }
            for r in csv.DictReader(f)
        ]
    _save(plots.fig_scoreboard(rows, LEADERBOARD_BEST), "stage4_scoreboard.png")


BANDS = ((0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10))  # above the cut-off


def _upgraded_held_out(advisor):
    """(probit-scale predictions (3 shuffles, n), true ratings) of the upgraded models."""
    o = np.load(os.path.join(upgraded.DIR, f"oof_{advisor}.npz"))
    w = upgraded.NETWORK.get(advisor, (0.0, 0))[0]
    return (1 - w) * o["gb"] + w * o["fc"], o["y"]


def report():
    """The report's model comparison and failure bands (held out, development ratings only)."""
    grids, dev = validity.load_dev()
    stage1 = validity.scorecard(["gbm_onehot"], verbose=False)["gbm_onehot"]
    recipes = {  # labels for readers who don't know the project's stages
        "GBM, raw cells only": None,
        "CNN, raw cells only": CNN_RUNS,
        "GBM, 75 hand-made features": dict.fromkeys(validity.ADVISORS, "gbm_engineered"),
        "GBM, features + per-advisor extras (checked on the test set)": search.INCUMBENT,
        "Upgraded: bell-curve target, more features; networks for Wellness, Tax (submitted)": "upgraded",
    }
    rows = {name: {} for name in recipes}
    passing, shares = {}, {}
    for a in validity.ADVISORS:
        _, y, _ = search._advisor_data(grids, dev, a)
        passing[a] = int((y >= validity.THRESHOLD).sum())
        z, y_up = _upgraded_held_out(a)
        assert np.array_equal(y_up, y), f"{a}: held-out predictions are for other grids"
        for name, recipe in recipes.items():
            if recipe is None:
                rows[name][a] = stage1[a]["score"].approved
            elif recipe == "upgraded":
                rows[name][a] = search._mean_approved(norm.cdf(z), y)
            else:
                rows[name][a] = search._mean_approved(search.load_outer(recipe[a], a)[0], y)
        margin = norm.cdf(z) - upgraded.load()["cutoffs"][a]
        fail = np.broadcast_to(y < validity.THRESHOLD, margin.shape)
        shares[a] = {(lo, hi): fail[(margin >= lo) & (margin < hi)].mean() for lo, hi in BANDS}
    for name, counts in rows.items():
        print(f"  {name}: " + ", ".join(f"{a} {v:.0f}" for a, v in counts.items()))
    print("  truly passing: " + ", ".join(f"{a} {v}" for a, v in passing.items()))
    for a, s in shares.items():
        print(f"  {a} failing by band: " + ", ".join(f"{100 * v:.1f}%" for v in s.values()))
    _save(plots.fig_model_approvals(rows, passing), "report_models.png")
    _save(plots.fig_failure_bands(shares), "report_cushion.png")


STAGES = {0: readme, 1: stage1, 2: stage2, 4: stage4, 5: report}  # 0 = the README's own figures


def main():
    ap = argparse.ArgumentParser(description="Regenerate the documentation figures.")
    ap.add_argument("--stage", type=int, choices=sorted(STAGES), default=None)
    args = ap.parse_args()
    for stage in [args.stage] if args.stage is not None else sorted(STAGES):
        print(f"Stage {stage} figures")
        STAGES[stage]()


if __name__ == "__main__":
    main()
