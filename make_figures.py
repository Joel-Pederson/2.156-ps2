"""Regenerate the README's figures into docs/figures/.

Where this fits
---------------
The README explains each stage with figures. They're generated, never hand-edited, so anyone
can rebuild them after a change: one command, same pictures. The drawing lives in plots.py;
this file only decides which figures exist and where they're saved.

Reads / writes
--------------
reads   the course data (via validity.load_dev(); dev labels only, never the lockbox)
writes  docs/figures/*.png      (tracked in git, because the README shows them)

How to run
----------
    python make_figures.py              # every stage built so far
    python make_figures.py --stage 1

Surprises
---------
Figures are saved at 150 dpi on the light chart surface, so they read correctly whether
GitHub shows the README in light or dark mode.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # files only, no windows

import matplotlib.pyplot as plt

import plots
import validity

FIG_DIR = os.path.join(validity.PS2_ROOT, "docs", "figures")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, validity.PS2_ROOT)}")


def stage1():
    grids, _ = validity.load_dev()
    _save(plots.fig_splits(), "stage1_splits.png")
    _save(plots.fig_grid_features(grids[0]), "stage1_grid_features.png")


STAGES = {1: stage1}


def main():
    ap = argparse.ArgumentParser(description="Regenerate the README figures.")
    ap.add_argument("--stage", type=int, choices=sorted(STAGES), default=None)
    args = ap.parse_args()
    for stage in [args.stage] if args.stage else sorted(STAGES):
        print(f"Stage {stage} figures")
        STAGES[stage]()


if __name__ == "__main__":
    main()
