"""Every figure in this project, in one place.

Where this fits
---------------
The documentation figures (made by make_figures.py) and the interactive analysis.ipynb both call
these functions, so a picture in the documentation and the same picture in the notebook can never
disagree. Each function returns a matplotlib Figure; saving and showing it is the caller's job.

README figure:
* ``fig_diversity_example(grids)``  README Figure 1: how the diversity score is computed.

Stage 1 figures (numbered as in docs/stage1_evaluation.md and analysis.ipynb):
* ``fig_cv_rounds()``           Figure 1: the data split and the five cross-validation rounds.
* ``fig_shuffles(grids)``       Figure 2: the shuffle procedure, with ten example grids.
* ``fig_grid_features(grid)``   Figure 3: one grid and 13 of its 75 engineered features.
* ``fig_approval_curve(...)``  Figure 4: how the approval count at 98% precision is measured.
* ``fig_scorecard(results)``    Figure 5: approval count at 98% precision, per model and advisor.
* ``fig_pred_vs_actual(...)``   Figures 6-7: predicted vs true rating, per advisor.

Every figure carries labels only. Titles and descriptions are captions placed below the figure
(stage documents in docs/, analysis.ipynb), following standard practice for documents.

Colours
-------
* The districts use five colours chosen so that every pair can be told apart, because any
  district can sit next to any other on a grid. tests/test_palette.py checks this: the closest
  pair differs by ΔE 18.0 under normal vision (minimum 15) and 13.3 under simulated red-green
  colour blindness (target 8). Pink and amber are below 3:1 contrast with the page, so every
  cell also carries its letter (R I C P O).
* Everything else uses the stock categorical palette in fixed order, never cycled. The
  cross-validation diagram (fig_cv_rounds) uses only the first three slots.
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
from matplotlib.lines import Line2D
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

# Display names for model variants (validity.VARIANTS keys): the names the documentation uses.
MODEL_LABELS = {
    "ridge_onehot": "Ridge regression on one-hot inputs",
    "gbm_onehot": "GBM on one-hot inputs",
    "ridge_engineered": "Ridge regression on engineered features",
    "gbm_engineered": "GBM on engineered features",
}

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


def district_legend(ax, extra=(), **kw):
    """Legend of the five districts (colour and letter), plus any extra handles."""
    handles = [
        Patch(facecolor=c, edgecolor=SURFACE, label=f"{l}  {n}")
        for c, l, n in zip(DISTRICT_COLOURS, DISTRICT_LETTERS, features.DISTRICTS)
    ]
    return ax.legend(handles=handles + list(extra), **kw)


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


_FOLD_W = 800 / 5000  # each fold is 800 of the 5,000 grids, in the shared 0-1 x-axis


def _block(axis, x0, width, y, colour, label=None, h=0.62, size=9, ink=None):
    """One labelled rectangle of the split diagrams."""
    axis.add_patch(
        Rectangle((x0, y), width, h, facecolor=colour, edgecolor=SURFACE, linewidth=2)
    )
    if label:
        axis.text(
            x0 + width / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            color=ink or ink_on(colour),
            fontsize=size,
            fontweight="bold",
        )


def diversity_example(grids, index=12, pool=100):
    """The numbers behind README Figure 1, computed once so the figure and text agree.

    Uses the starter notebook's example set (dataset grids 0 to pool-1). Returns a dict with the
    grid's nearest neighbour, the cells in which they differ, its sorted distances to the other
    grids, the mean of its 10 nearest (its contribution x 100), and the set's score, which is
    checked against scoring.diversity (the course's formula).
    """
    import scoring  # local import: only this example needs the scorer

    grids = np.asarray(grids)[:pool]
    flat = grids.reshape(len(grids), -1)
    k = scoring.K_NEIGHBOURS

    def distances(g):
        others = np.array([j for j in range(len(grids)) if j != g])
        return others, (flat[others] != flat[g]).mean(1)

    others, dist = distances(index)
    nearest = int(others[np.argsort(dist, kind="stable")[0]])
    total = sum(np.sort(distances(g)[1])[:k].mean() for g in range(len(grids)))
    score = total / scoring.N_GRIDS
    assert abs(score - scoring.diversity(grids)) < 1e-12  # same as the official formula
    return {
        "index": index,
        "nearest": nearest,
        "k": k,
        "n_set": len(grids),
        "differ": grids[index] != grids[nearest],
        "sorted": np.sort(dist),
        "mean_nearest": float(np.sort(dist)[:k].mean()),
        "mean_all": float(dist.mean()),
        "score": float(score),
    }


def fig_diversity_example(grids, index=12, pool=100):
    """README Figure 1: how the diversity part of the score is computed.

    Left: a grid and its most similar grid in the starter notebook's example set, with the cells
    in which they differ outlined. Right: the grid's distances to the other grids of the set,
    sorted, with the 10 nearest highlighted. Labels only; the caption is below the figure.
    """
    ex = diversity_example(grids, index, pool)
    grids = np.asarray(grids)
    k, n_diff = ex["k"], int(ex["differ"].sum())
    fig = plt.figure(figsize=(12, 4.8))
    ax_a = fig.add_axes([0.01, 0.17, 0.2, 0.68])
    ax_b = fig.add_axes([0.24, 0.17, 0.2, 0.68])
    draw_grid(ax_a, grids[index])
    draw_grid(ax_b, grids[ex["nearest"]])
    for r, c in zip(*np.nonzero(ex["differ"])):
        ax_b.add_patch(
            Rectangle(
                (c + 0.07, r + 0.07),
                0.86,
                0.86,
                fill=False,
                edgecolor=INK,
                linewidth=2.2,
            )
        )
    ax_a.set_title(f"Grid A (dataset grid {index})", fontsize=10, loc="center")
    ax_b.set_title(
        f"Its nearest grid in the set ({ex['nearest']})", fontsize=10, loc="center"
    )
    fig.text(
        0.225,
        0.07,
        f"Outlined: the {n_diff} of 49 cells that differ, so the distance is "
        f"{n_diff}/49 = {n_diff / 49:.2f}",
        ha="center",
        fontsize=9.5,
        color=INK,
    )

    ax = fig.add_axes([0.53, 0.2, 0.45, 0.66])
    x = np.arange(1, len(ex["sorted"]) + 1)
    ax.bar(x[k:], ex["sorted"][k:], width=0.85, color="#c9c8c2")
    ax.bar(x[:k], ex["sorted"][:k], width=0.85, color=SERIES[0])
    ax.hlines(ex["mean_nearest"], 0.5, k + 0.5, color=INK, linewidth=2.2)
    # Annotations sit in the empty top-left of the chart, clear of the bars.
    ax.text(
        1,
        0.985,
        f"Blue: the {k} nearest grids. Their mean distance is {ex['mean_nearest']:.3f},\n"
        f"so grid A contributes {ex['mean_nearest']:.3f} / 100 = "
        f"{ex['mean_nearest'] / 100:.4f} to the score.",
        fontsize=9.5,
        color=INK,
        va="top",
    )
    ax.text(
        k + 4,
        ex["mean_all"] + 0.012,
        f"Mean distance to all {len(x)} = {ex['mean_all']:.2f} (not used by the score)",
        va="bottom",
        fontsize=9,
        color=INK_2,
    )
    ax.set_xlim(0, len(x) + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"The other {len(x)} grids of the set, from nearest to farthest")
    ax.set_ylabel("Distance from grid A")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.text(
        0.53,
        0.035,
        f"Summed over all {ex['n_set']} grids, the contributions give a score of "
        f"{ex['score']:.4f}, the starter notebook's value for this set.",
        fontsize=9.5,
        color=INK_2,
    )
    return fig


def fig_cv_rounds():
    """Figure 1: the development/test split and the five rounds of cross-validation.

    Labels only; the title and description are the caption below the figure in the documentation
    and analysis.ipynb.
    """
    train, val, test = SERIES[0], SERIES[1], SERIES[2]
    fig = plt.figure(figsize=(10.5, 4.6))
    ax = fig.add_axes([0.10, 0.02, 0.89, 0.96])
    for f in range(5):
        ax.text(
            (f + 0.5) * _FOLD_W,
            4.78,
            f"Fold {f + 1}",
            ha="center",
            va="bottom",
            color=INK_2,
            fontsize=9,
        )
    for rnd in range(5):
        y = 4 - rnd
        for f in range(5):
            is_val = f == rnd
            _block(
                ax,
                f * _FOLD_W,
                _FOLD_W,
                y,
                val if is_val else train,
                "validation" if is_val else "training",
            )
        _block(ax, 0.8, 0.2, y, test, "not used")
        ax.text(
            -0.012,
            y + 0.31,
            f"Round {rnd + 1}",
            ha="right",
            va="center",
            color=INK_2,
            fontsize=9,
        )
    arrows = {"arrowstyle": "<->", "color": INK_2, "linewidth": 1}
    ax.annotate("", xy=(0, 5.35), xytext=(0.8, 5.35), arrowprops=arrows)
    ax.text(
        0.4,
        5.5,
        "Development set: 4,000 rated grids\n(training and validation)",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=9.5,
    )
    ax.annotate("", xy=(0.8, 5.35), xytext=(1.0, 5.35), arrowprops=arrows)
    ax.text(
        0.9,
        5.5,
        "Test set (lockbox): 1,000\n(evaluated only after\na pipeline is final)",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=9.5,
    )
    ax.set_xlim(-0.11, 1.0)
    ax.set_ylim(-0.1, 6.3)
    ax.axis("off")
    return fig


def fig_shuffles(example_grids=None):
    """Figure 2: the shuffle procedure, with ten example grids standing in for 4,000.

    Labels only; the caption is below the figure. example_grids: ten (7, 7) grids pictured as
    grids A-J. Pass real development-set grids (make_figures.py does); if omitted, seeded
    random grids stand in. The fold assignments come from validity.cv_splits itself, so the
    example is exactly what the code does. Its x-axis matches fig_cv_rounds, so the folds line up.
    """
    import validity  # local import: only this figure needs the real CV splitter

    if example_grids is None:
        example_grids = np.random.default_rng(0).integers(0, 5, size=(10, 7, 7))
    example_grids = np.asarray(example_grids)[:10]
    fig_w, fig_h = 10.5, 4.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.10, 0.02, 0.89, 0.70])

    def to_fig_x(x):
        return 0.10 + (x + 0.11) / 1.11 * 0.89

    letters = "ABCDEFGHIJ"
    slot = (to_fig_x(0.8) - to_fig_x(0.0)) / len(letters)
    side = 0.048
    for i, letter in enumerate(letters):
        left = to_fig_x(0.0) + i * slot + (slot - side) / 2
        thumb = fig.add_axes([left, 0.845, side, side * fig_w / fig_h])
        draw_grid(thumb, example_grids[i], letters=False)
        for patch in thumb.patches:
            patch.set_linewidth(0.4)
        fig.text(
            left + side / 2,
            0.835,
            f"Grid {letter}",
            ha="center",
            va="top",
            fontsize=8.8,
            color=INK,
            fontweight="bold",
        )
    fig.text(
        to_fig_x(-0.012),
        0.905,
        "Example\ngrids",
        ha="right",
        va="center",
        fontsize=9,
        color=INK_2,
    )

    for f in range(5):
        ax.text(
            (f + 0.5) * _FOLD_W,
            2.82,
            f"Fold {f + 1}",
            ha="center",
            va="bottom",
            color=INK_2,
            fontsize=9,
        )
    for r, f, _, val_idx in validity.cv_splits(len(letters)):
        y = 2 - r
        names = "   ".join(sorted(letters[i] for i in val_idx))
        _block(ax, f * _FOLD_W, _FOLD_W, y, "#f0efec", names, h=0.72, size=11, ink=INK)
        if f == 0:
            ax.text(
                -0.012,
                y + 0.36,
                f"Shuffle {r + 1}",
                ha="right",
                va="center",
                color=INK_2,
                fontsize=9,
            )
    ax.text(
        0.9,
        1.36,
        "Test set:\nnot divided\ninto folds",
        ha="center",
        va="center",
        color=INK_2,
        fontsize=8.8,
        style="italic",
    )
    ax.set_xlim(-0.11, 1.0)
    ax.set_ylim(-0.1, 3.2)
    ax.axis("off")
    return fig


def fig_grid_features(grid):
    """Figure 3: one grid (left) and 13 of its 75 engineered feature values (right).

    The image carries labels only; its title and description are the caption.
    """
    grid = np.asarray(grid)
    values = dict(zip(features.ENGINEERED_NAMES, features.engineered(grid[None])[0]))
    fig = plt.figure(figsize=(11, 5.4))
    ax_grid = fig.add_axes([0.02, 0.14, 0.38, 0.80])
    draw_grid(ax_grid, grid)
    # Mark what one feature counts: every side-by-side Park–Residential pair.
    pairs = _neighbour_pairs(grid, 3, 0)
    for (x0, y0), (x1, y1) in pairs:
        ax_grid.plot([x0, x1], [y0, y1], color=INK, linewidth=4, solid_capstyle="round")
    bar_key = Line2D(
        [0],
        [0],
        color=INK,
        linewidth=4,
        label=f"Side-by-side Park–Residential pair ({len(pairs)})",
    )
    district_legend(
        ax_grid,
        extra=[bar_key],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.0,
    )

    # Right: the numbers, grouped by the five feature groups.
    ax_bar = fig.add_axes([0.50, 0.68, 0.42, 0.22])
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
    ax_bar.set_title("1  Counts: cells of each district  (all 5 shown)", fontsize=10)

    rows = [
        (
            "2  Adjacency: neighbouring district pairs  (2 of 30 shown)",
            [
                (
                    "Park–Residential side-by-side pairs",
                    "side-by-side Residential-Park",
                    "{:.0f}",
                ),
                (
                    "Industrial–Residential side-by-side pairs",
                    "side-by-side Residential-Industrial",
                    "{:.0f}",
                ),
            ],
        ),
        (
            "3  Distance: mean steps to the nearest district  (2 of 20 shown)",
            [
                (
                    "Residential to nearest Park (steps)",
                    "distance Residential->Park",
                    "{:.2f}",
                ),
                (
                    "Residential to nearest Office (steps)",
                    "distance Residential->Office",
                    "{:.2f}",
                ),
            ],
        ),
        (
            "4  Clusters: connected groups of cells  (2 of 10 shown)",
            [
                ("Park clusters", "clusters Park", "{:.0f}"),
                ("Largest Park cluster (cells)", "largest cluster Park", "{:.0f}"),
            ],
        ),
        (
            "5  Edge / centre: position on the grid  (2 of 10 shown)",
            [
                ("Park cells on the border", "border Park", "{:.0f}"),
                (
                    "Park mean distance from centre (steps)",
                    "centre distance Park",
                    "{:.2f}",
                ),
            ],
        ),
    ]
    y = 0.59
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
    return fig


def fig_approval_curve(
    results, name="gbm_engineered", advisor="Wellness", shuffle=0, x_max=1200
):
    """Figure 4: how the approval count is measured, for one advisor and one shuffle.

    The development-set grids are sorted by predicted rating, highest first. After approving the
    top k, the curve shows the share of those k grids that truly pass (true rating >= 0.75). The
    approval count at a precision level is the last k at which the curve is at or above that
    level, computed by validity.approved_at, the same function the scorecard uses. Each marked
    point is labelled with the number approved and how many of those truly fail. A level that
    no count reaches, or that is reached beyond x_max, is listed above the plot instead of
    marked. Labels only; the caption is below the figure.
    """
    import validity  # local import: only this figure needs the metric code

    r = results[name][advisor]
    y, pred = r["y"], r["preds"][shuffle]
    order = np.argsort(-pred, kind="stable")
    k = np.arange(1, len(y) + 1)
    share = np.cumsum(y[order] >= validity.THRESHOLD) / k
    shown = k <= x_max
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(k[shown], 100 * share[shown], color=SERIES[0], linewidth=2)
    levels = ((0.95, 1.1), (0.98, 1.1), (0.99, 1.1))
    notes = []  # levels with no point on the drawn curve
    for level, dy in levels:
        count, _, hits = validity.approved_at(pred, y, level)
        ax.axhline(
            100 * level, color=RULE, linewidth=1, linestyle=(0, (4, 3)), zorder=0
        )
        if count == 0:
            notes.append(f"{level:.0%}: no number of approvals reaches it")
            continue
        if count > x_max:
            notes.append(
                f"{level:.0%}: {count} approved, {count - hits} fail (beyond this axis)"
            )
            continue
        ax.text(
            x_max * 1.005,
            100 * level,
            f"{level:.0%}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK_2,
        )
        ax.plot(
            [count],
            [100 * share[count - 1]],
            marker="o",
            markersize=8,
            color=INK,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            zorder=3,
        )
        ax.text(
            count + 12,
            100 * share[count - 1] + dy,
            f"{count} approved, {count - hits} fail",
            fontsize=9,
            color=INK,
            va="center",
        )
    if notes:
        ax.set_title("\n".join(notes), fontsize=9, color=INK_2, loc="right")
    # Every level's line and label stay inside the plot, even where the curve stays above them.
    low = min(100 * share[shown].min(), 100 * levels[0][0])
    ax.set_xlim(0, x_max)
    ax.set_ylim(np.floor(low) - 1, 100.6)
    ax.set_xlabel(
        f"Grids approved, in order of predicted {advisor} rating (highest first)"
    )
    ax.set_ylabel(f"Share of approved grids that\ntruly pass {advisor} (%)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def fig_scorecard(results):
    """Figure 5: approval count at 98% precision for each model (bars) and advisor (panels).

    The image carries labels only; its title and description are the caption.
    """
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
    axes[0].set_ylabel("Development-set grids\napproved at 98% precision")
    handles = [
        Patch(facecolor=SERIES[i], label=MODEL_LABELS.get(n, n))
        for i, n in enumerate(names)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(names),
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def fig_pred_vs_actual(results, name, threshold=0.75):
    """Figures 6-7: predicted vs true rating per advisor, one point per development-set grid.

    Predictions are out-of-fold, from the first shuffle. The vertical range is -0.2 to 1.2,
    widened only if a prediction falls outside it, so every prediction is drawn. The image
    carries labels only; its title and description are the caption.
    """
    advisors = list(results[name])
    first = [results[name][adv]["preds"][0] for adv in advisors]
    low, high = min(p.min() for p in first), max(p.max() for p in first)
    lo = -0.2 if low >= -0.2 else low - 0.05
    hi = 1.2 if high <= 1.2 else high + 0.05
    fig, axes = plt.subplots(1, len(advisors), figsize=(12, 3.5), sharey=True)
    for ax, adv in zip(axes, advisors):
        y, p = results[name][adv]["y"], results[name][adv]["preds"][0]
        # Shade the region where the model predicts a pass but the grid truly fails.
        ax.add_patch(
            Rectangle(
                (-0.05, threshold),
                threshold + 0.05,
                hi - threshold,
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
            hi - 0.05 * (hi - lo),
            f"Predicted to pass,\ntruly fails: {wrong}",
            fontsize=8.3,
            color=INK_2,
            va="top",
        )
        ax.set_title(f"{adv}   R² {results[name][adv]['score'].r2:.2f}", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True rating")
    axes[0].set_ylabel("Predicted rating")
    fig.tight_layout()
    return fig
