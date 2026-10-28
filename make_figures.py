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
writes  docs/figures/*.png      (tracked in git, because the documentation shows them)

How to run
----------
    python make_figures.py              # every stage built so far
    python make_figures.py --stage 1     # one stage only (0 = the README's own figures)

Surprises
---------
Figures are saved at 150 dpi on the light chart surface, so they read correctly whether
GitHub shows the documentation in light or dark mode.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # files only, no windows

import matplotlib.pyplot as plt
import numpy as np

import plots
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


STAGES = {0: readme, 1: stage1}  # 0 = the README's own figures


def main():
    ap = argparse.ArgumentParser(description="Regenerate the documentation figures.")
    ap.add_argument("--stage", type=int, choices=sorted(STAGES), default=None)
    args = ap.parse_args()
    for stage in [args.stage] if args.stage is not None else sorted(STAGES):
        print(f"Stage {stage} figures")
        STAGES[stage]()


if __name__ == "__main__":
    main()
