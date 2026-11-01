"""The CNN settings chart: plots.cnn_setting_effects, plots.fig_cnn_settings and
make_figures.py --stage 2.

Synthetic records only, shaped like the saved CNN runs: a tiny one whose answers can be worked
out by hand, and full-size random ones (15 rounds x 10 settings, drawn as the search draws them).
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_palette import _delta_e

import make_figures
import models_nn
import plots


def _trial(hp_recall, **changes):
    """One tried setting as the search records it: the first one (depth 2, width 32, kernel 3,
    no position or augment, lr 0.001), changed where asked."""
    setting = {**models_nn.settings(False)[0], **changes}
    return {**setting, "hp_recall": hp_recall, "epochs": [9, 9, 9, 9]}


def _random_record(advisor, augment_allowed, rng):
    """15 rounds, each trying 10 random settings with random inner hp_recall."""
    options = models_nn.settings(augment_allowed)
    rounds = [
        [
            {**options[i], "hp_recall": rng.random(), "epochs": [9, 9, 9, 9]}
            for i in rng.choice(len(options), 10, replace=False)
        ]
        for _ in range(15)
    ]
    return {"advisor": advisor, "rounds": [{"tried": tried} for tried in rounds]}


def _cnn_records(seed):
    rng = np.random.default_rng(seed)
    return [_random_record(a, a == "Transportation", rng) for a in plots.ADVISOR_STYLE]


def test_effects_pool_every_round_and_match_hand_worked_values():
    rounds = [
        [_trial(0.5), _trial(0.7, depth=3, position=True)],
        [_trial(0.3, position=True), _trial(0.9, depth=3)],
    ]
    record = {"advisor": "Tax", "rounds": [{"tried": tried} for tried in rounds]}
    effects = plots.cnn_setting_effects([record])["Tax"]
    # Only depth and position vary; the others (augment included) were held fixed.
    assert list(effects) == ["depth", "position"]
    # depth 2: 0.5 and 0.3, mean 0.4, standard error 0.1, so the half-width is 1.96 x 0.1.
    assert effects["depth"][2] == pytest.approx((0.4, 2, 0.196))
    assert effects["depth"][3] == pytest.approx((0.8, 2, 0.196))
    # position off: 0.5 and 0.9, mean 0.7, standard error 0.2.
    assert effects["position"][False] == pytest.approx((0.7, 2, 0.392))
    assert effects["position"][True] == pytest.approx((0.5, 2, 0.392))


def test_each_advisor_pools_150_trials_and_only_transportation_searched_augment():
    effects = plots.cnn_setting_effects(_cnn_records(seed=0))
    for advisor, by_setting in effects.items():
        expected = list(plots.CNN_SETTINGS)
        if advisor != "Transportation":
            expected.remove("augment")
        assert list(by_setting) == expected
        for levels in by_setting.values():
            assert sum(n for _, n, _ in levels.values()) == 150


def test_figure_draws_each_advisors_means_in_one_panel_per_setting():
    effects = plots.cnn_setting_effects(_cnn_records(seed=1))
    fig = plots.fig_cnn_settings(effects)
    assert [ax.get_title(loc="left") for ax in fig.axes] == list(plots.CNN_SETTINGS)
    for ax, setting in zip(fig.axes, plots.CNN_SETTINGS):
        drawn = {c.get_label(): list(c.lines[0].get_ydata()) for c in ax.containers}
        assert drawn == {
            advisor: [mean for mean, _, _ in by_setting[setting].values()]
            for advisor, by_setting in effects.items()
            if setting in by_setting
        }
    assert {ax.get_yscale() for ax in fig.axes} == {"log"}  # one shared log axis
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert legend == list(plots.ADVISOR_STYLE)
    plt.close(fig)


def test_advisor_colours_are_stock_unlike_every_district_and_backed_by_markers():
    colours = [colour for colour, _ in plots.ADVISOR_STYLE.values()]
    assert set(colours) <= set(plots.SERIES)
    # ΔE as in test_palette.py. Skipped: amber 0.4, pink 4.5, dark green 9.5 from a district.
    assert min(_delta_e(c, d) for c in colours for d in plots.DISTRICT_COLOURS) >= 10
    assert len({marker for _, marker in plots.ADVISOR_STYLE.values()}) == 4


def test_stage_2_reads_each_advisors_cnn_run_and_saves_the_chart(monkeypatch, tmp_path):
    records = {r["advisor"]: r for r in _cnn_records(seed=2)}
    asked = []

    def load_outer(name, advisor):  # stands in for search.load_outer
        asked.append((name, advisor))
        return np.zeros((3, 4000)), records[advisor]

    monkeypatch.setattr(make_figures.search, "load_outer", load_outer)
    monkeypatch.setattr(make_figures, "FIG_DIR", str(tmp_path))
    make_figures.stage2()
    assert asked == [
        ("cnn", "Wellness"),
        ("cnn", "Tax"),
        ("cnn_with_rotations", "Transportation"),
        ("cnn", "Business"),
    ]
    assert (tmp_path / "stage2_cnn_settings.png").stat().st_size > 0
