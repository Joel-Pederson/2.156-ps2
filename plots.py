"""Every figure in this project, in one place.

Where this fits
---------------
The README figures (made by make_figures.py) and the interactive analysis.ipynb both call
these functions, so a picture in the README and the same picture in the notebook can never
disagree. Each function returns a matplotlib Figure; saving and showing it is the caller's job.

Stage 1 figures:
* ``fig_splits()``              How the labels are divided: lockbox, dev set, the 5 folds.
* ``fig_grid_features(grid)``   One city, and the numbers our models read off it.
* ``fig_scorecard(results)``    "Approved at 98%" per model and advisor.
* ``fig_pred_vs_actual(...)``   Predicted vs true score: where a model is right and wrong.

Colours
-------
* The districts use five colours checked with the dataviz skill's palette validator in
  *all-pairs* mode, because every district can sit next to every other on a grid. The worst
  pair is ΔE 18.0 under normal vision and 13.3 under colour-blind simulation (floors 15 and 8).
  Pink and amber are below 3:1 contrast with the page, so every cell also carries its
  letter (R I C P O).
* Everything else uses the stock categorical palette in fixed order, never cycled. The splits
  diagram uses only the first three slots, which the skill validates for any combination.
* Text is always ink (near-black or grey), never a series colour.

Reads / writes
--------------
Nothing. Pure plotting.

How to run
----------
Imported. In a notebook, ``plots.fig_grid_features(grids[0])`` shows inline.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

import features

# --- palette ------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"  # secondary text
RULE = "#d9d8d3"  # recessive gridlines and axes

DISTRICT_COLOURS = ("#f48cae", "#8c2e23", "#eba002", "#528a54", "#145786")
DISTRICT_LETTERS = "RICPO"
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": RULE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": RULE,
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def _luminance(hex_colour):
    rgb = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ink_on(fill):
    """Black or white text, whichever contrasts more with the fill."""
    lum = _luminance(fill)
    return (
        INK
        if (lum + 0.05) / (_luminance(INK) + 0.05) >= 1.05 / (lum + 0.05)
        else "#ffffff"
    )


# --- grids ---------------------------------------------------------------------------------


def draw_grid(ax, grid, letters=True, fontsize=11):
    """Draw one 7x7 city: coloured cells, each with its district letter. Row 0 at the top."""
    grid = np.asarray(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            k = int(grid[r, c])
            ax.add_patch(
                Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=DISTRICT_COLOURS[k],
                    edgecolor=SURFACE,
                    linewidth=2,
                )
            )
            if letters:
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    DISTRICT_LETTERS[k],
                    ha="center",
                    va="center",
                    color=ink_on(DISTRICT_COLOURS[k]),
                    fontsize=fontsize,
                    fontweight="bold",
                )
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(grid.shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")


def district_legend(ax, **kw):
    handles = [
        Patch(facecolor=c, edgecolor=SURFACE, label=f"{l}  {n}")
        for c, l, n in zip(DISTRICT_COLOURS, DISTRICT_LETTERS, features.DISTRICTS)
    ]
    return ax.legend(handles=handles, **kw)


def _neighbour_pairs(grid, i, j):
    """Shared edges between side-by-side cells of districts i and j, as line segments."""
    segs = []
    for r in range(7):
        for c in range(7):
            if c < 6 and {grid[r, c], grid[r, c + 1]} == {i, j}:
                segs.append(((c + 1, r + 0.18), (c + 1, r + 0.82)))
            if r < 6 and {grid[r, c], grid[r + 1, c]} == {i, j}:
                segs.append(((c + 0.18, r + 1), (c + 0.82, r + 1)))
    return segs


# --- Stage 1 ---------------------------------------------------------------------------------


def fig_splits():
    """One advisor's 5,000 labels: the lockbox, the dev set, and the 5 rounds of cross-validation."""
    train, val, lock = SERIES[0], SERIES[1], SERIES[2]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    unit = 1 / 5000

    def block(x0, width, y, colour, label=None):
        ax.add_patch(
            Rectangle(
                (x0, y), width, 0.62, facecolor=colour, edgecolor=SURFACE, linewidth=2
            )
        )
        if label:
            ax.text(
                x0 + width / 2,
                y + 0.31,
                label,
                ha="center",
                va="center",
                color=ink_on(colour),
                fontsize=9,
                fontweight="bold",
            )

    # Five rounds: the validation fold moves along; the lockbox never takes part.
    for rnd in range(5):
        y = 4 - rnd
        for f in range(5):
            is_val = f == rnd
            block(
                f * 800 * unit,
                800 * unit,
                y,
                val if is_val else train,
                "validate" if is_val else "train",
            )
        block(4000 * unit, 1000 * unit, y, lock, "lockbox" if rnd == 0 else None)
        ax.text(
            -0.012,
            y + 0.31,
            f"round {rnd + 1}",
            ha="right",
            va="center",
            color=INK_2,
            fontsize=9,
        )

    ax.annotate(
        "",
        xy=(0, 5.05),
        xytext=(0.8, 5.05),
        arrowprops={"arrowstyle": "<->", "color": INK_2, "linewidth": 1},
    )
    ax.text(
        0.4,
        5.2,
        "DEV: 4,000 labels. Every comparison happens here.",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=9.5,
    )
    ax.annotate(
        "",
        xy=(0.8, 5.05),
        xytext=(1.0, 5.05),
        arrowprops={"arrowstyle": "<->", "color": INK_2, "linewidth": 1},
    )
    ax.text(
        0.9,
        5.2,
        "LOCKBOX: 1,000\nopened once, at the end",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=9.5,
    )
    ax.text(
        0.0,
        -0.05,
        "Each round trains on 3,200 grids and checks on the other 800, so every dev grid "
        "is checked by a model that never saw it.\nThe whole thing runs 3 times with different "
        "shuffles; how much scores move between runs is our 'noise' yardstick.",
        ha="left",
        va="top",
        color=INK_2,
        fontsize=8.8,
    )
    ax.set_xlim(-0.11, 1.0)
    ax.set_ylim(-0.75, 6.0)
    ax.axis("off")
    ax.set_title(
        "How one advisor's 5,000 labels are used (every advisor gets the same split)",
        pad=4,
    )
    return fig


def fig_grid_features(grid):
    """One city, and a sample of the 75 numbers our models read off it."""
    grid = np.asarray(grid)
    values = dict(zip(features.ENGINEERED_NAMES, features.engineered(grid[None])[0]))
    fig = plt.figure(figsize=(11, 5.4))
    ax_grid = fig.add_axes([0.02, 0.12, 0.38, 0.76])
    draw_grid(ax_grid, grid)
    # Mark what one feature counts: every side-by-side Park–Residential pair.
    pairs = _neighbour_pairs(grid, 3, 0)
    for (x0, y0), (x1, y1) in pairs:
        ax_grid.plot([x0, x1], [y0, y1], color=INK, linewidth=4, solid_capstyle="round")
    ax_grid.set_title(
        f"A city (dataset grid #0). Black bars mark its {len(pairs)} "
        "Park–Residential\nside-by-side pairs; that count is one feature.",
        fontsize=10,
        pad=6,
    )
    district_legend(
        ax_grid,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=8.5,
        handlelength=1.2,
        columnspacing=0.9,
    )

    # The numbers, grouped by the five questions they answer.
    ax_bar = fig.add_axes([0.50, 0.66, 0.42, 0.22])
    counts = [values[f"count {d}"] for d in features.DISTRICTS]
    ax_bar.barh(
        range(5),
        counts,
        color=DISTRICT_COLOURS,
        height=0.62,
        edgecolor=SURFACE,
        linewidth=2,
    )
    ax_bar.set_yticks(range(5), features.DISTRICTS, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xticks([])
    ax_bar.spines[["left", "bottom"]].set_visible(False)
    ax_bar.tick_params(axis="y", length=0)
    for i, v in enumerate(counts):
        ax_bar.text(v + 0.2, i, f"{v:.0f}", va="center", fontsize=9, color=INK)
    ax_bar.set_title("1  How much of each district?  (counts, 5 features)", fontsize=10)

    rows = [
        (
            "2  What sits next to what?  (30 features)",
            [
                (
                    "Park–Residential, side-by-side pairs",
                    "side-by-side Residential-Park",
                    "{:.0f}",
                ),
                (
                    "Industrial–Residential, side-by-side pairs",
                    "side-by-side Residential-Industrial",
                    "{:.0f}",
                ),
            ],
        ),
        (
            "3  How far is X from the nearest Y?  (20 features, avg steps)",
            [
                ("a home to its nearest park", "distance Residential->Park", "{:.2f}"),
                (
                    "a home to its nearest office (commute)",
                    "distance Residential->Office",
                    "{:.2f}",
                ),
            ],
        ),
        (
            "4  Clumped or scattered?  (10 features)",
            [
                ("separate park blobs", "blobs Park", "{:.0f}"),
                ("cells in the biggest park blob", "biggest blob Park", "{:.0f}"),
            ],
        ),
        (
            "5  Edge or centre?  (10 features)",
            [
                ("parks on the border", "border Park", "{:.0f}"),
                ("parks' avg steps from the centre", "centre distance Park", "{:.2f}"),
            ],
        ),
    ]
    y = 0.57
    for heading, items in rows:
        fig.text(0.50, y, heading, fontsize=10, fontweight="bold", color=INK)
        y -= 0.042
        for label, key, fmt in items:
            fig.text(0.52, y, label, fontsize=9.2, color=INK_2)
            fig.text(
                0.92, y, fmt.format(values[key]), fontsize=9.2, color=INK, ha="right"
            )
            y -= 0.038
        y -= 0.012
    fig.text(
        0.50,
        y - 0.01,
        "Every one of the 75 stays the same if you rotate or mirror the city.",
        fontsize=9,
        color=INK_2,
        style="italic",
    )
    return fig


def fig_scorecard(results):
    """Approved at 98% for each model (bars) on each advisor (panels)."""
    names = list(results)
    advisors = list(next(iter(results.values())))
    fig, axes = plt.subplots(1, len(advisors), figsize=(12, 3.4), sharey=True)
    for ax, adv in zip(axes, advisors):
        vals = [results[n][adv]["score"].approved for n in names]
        bars = ax.bar(
            range(len(names)),
            vals,
            color=SERIES[: len(names)],
            width=0.72,
            edgecolor=SURFACE,
            linewidth=2,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 8,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK,
            )
        ax.set_title(adv, fontsize=10)
        ax.set_xticks([])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("dev grids approved\nwhile ≥98% truly pass")
    handles = [Patch(facecolor=SERIES[i], label=n) for i, n in enumerate(names)]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(names),
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
    )
    fig.suptitle(
        "Approved at 98%: how many grids each model can approve and still be right 98% "
        "of the time (higher is better)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def fig_pred_vs_actual(results, name, threshold=0.75):
    """Predicted vs true score per advisor, one dot per dev grid (cross-validation, round 1)."""
    advisors = list(results[name])
    fig, axes = plt.subplots(1, len(advisors), figsize=(12, 3.5), sharey=True)
    for ax, adv in zip(axes, advisors):
        y, p = results[name][adv]["y"], results[name][adv]["preds"][0]
        # Shade the costly corner: the model says pass, the advisor says fail.
        ax.add_patch(
            Rectangle(
                (-0.05, threshold),
                threshold + 0.05,
                1.2 - threshold,
                facecolor=SERIES[7],
                alpha=0.10,
                linewidth=0,
            )
        )
        ax.scatter(y, p, s=3, color=SERIES[0], alpha=0.35, linewidths=0)
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.axhline(threshold, color=RULE, linewidth=1)
        ax.axvline(threshold, color=RULE, linewidth=1)
        wrong = ((p >= threshold) & (y < threshold)).sum()
        ax.text(
            0.02,
            1.13,
            f"says pass, truly fails: {wrong}",
            fontsize=8.3,
            color=INK_2,
            va="top",
        )
        ax.set_title(f"{adv}   R² {results[name][adv]['score'].r2:.2f}", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("true score")
    axes[0].set_ylabel("predicted score")
    fig.suptitle(
        f"{name}: every dot is a dev grid scored by a model that never saw it. "
        "Shaded = costly mistakes",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
