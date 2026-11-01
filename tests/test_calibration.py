"""Calibration turns predicted ratings into chances of passing that come true as often as
they say. Synthetic data with a known answer: true rating = pred + Normal(0, 0.03), so a
grid's chance of passing is exactly Φ((pred − 0.75) / 0.03).

- The probit recovers the centre 0.75 and the width 0.03, and widens with disagreement only
  when errors really grow with it.
- Chances lie in [0, 1] and rise with pred, for both methods.
- The choice follows the data (probit for a probit, isotonic for a staircase) and is
  deterministic.
- The reliability table bins chances correctly and lists every bin (an empty one as 0 grids,
  so no row looks missing); held-out chances come true about as often as they say.
- fit_all feeds each advisor the mean and spread of its incumbent's saved predictions.
"""

from dataclasses import replace

import numpy as np
import pytest

import calibration


def _synthetic(n=4000, seed=0, widen=0.0):
    """pred, spread and true ratings, with rating = pred + Normal(0, 0.03 + widen × spread)."""
    rng = np.random.default_rng(seed)
    pred, spread = rng.uniform(0.4, 1.0, n), rng.uniform(0, 0.03, n)
    return pred, spread, pred + rng.normal(0, 1, n) * (0.03 + widen * spread)


def _staircase(n=4000, seed=0):
    """A chance of passing that jumps 0 → 0.5 → 0.97 at pred 0.7 and 0.85: no probit fits it."""
    rng = np.random.default_rng(seed)
    pred, spread = rng.uniform(0.4, 1.0, n), rng.uniform(0, 0.03, n)
    chance = np.select([pred < 0.7, pred < 0.85], [0.0, 0.5], 0.97)
    y = (rng.random(n) < chance).astype(float)  # rating 1 passes, 0 fails
    return pred, spread, y


def test_the_probit_recovers_the_known_centre_and_width():
    pred, spread, y = _synthetic()
    c, s0, s1 = calibration.fit_probit(pred, spread, y >= 0.75)
    assert abs(c - 0.75) < 0.01
    assert abs(s0 - 0.03) < 0.01
    assert s1 < 0.5  # here disagreement says nothing about the error


def test_the_probit_widens_when_errors_grow_with_disagreement():
    pred, spread, y = _synthetic(widen=2.0)
    c, _, s1 = calibration.fit_probit(pred, spread, y >= 0.75)
    assert abs(c - 0.75) < 0.01
    assert 1.0 < s1 < 3.5


@pytest.mark.parametrize("method", ["isotonic", "probit"])
def test_chances_lie_in_0_1_and_rise_with_pred(method):
    cal = replace(calibration.fit(*_synthetic()), method=method)
    pred = np.linspace(0, 1.5, 301)  # beyond the data's range (0.4 to 1.0) too
    p = calibration.pass_probability(cal, pred, np.full_like(pred, 0.01))
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) >= 0)
    assert p[0] < 0.001 and p[-1] > 0.999


def test_the_choice_follows_the_data_and_is_deterministic():
    assert calibration.fit(*_synthetic()).method == "probit"
    first, again = calibration.fit(*_staircase()), calibration.fit(*_staircase())
    assert first.method == "isotonic"
    assert first.log_loss == again.log_loss and first.table == again.table


def test_the_reliability_table_lists_every_bin_even_empty_ones(capsys):
    p = np.array([0.1, 0.15, 0.95, 0.97, 0.995])
    rows = calibration.reliability(p, np.array([False, False, True, False, True]))
    assert [row[0] for row in rows] == list(calibration.BINS[:-1])
    assert [row[2] for row in rows] == [2, 0, 0, 0, 0, 0, 2, 1]
    full = [row[3:] for row in rows if row[2]]
    assert np.allclose(full, [[0.125, 0], [0.96, 0.5], [0.995, 1]])
    # An empty bin prints as 0 grids and dashes, not as a gap in the table.
    cal = calibration.Calibration("probit", {"probit": (0.75, 0.03, 0)}, {}, rows)
    calibration.report("Tax", cal)
    printed = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["0.80", "to", "0.90", "0", "-", "-"] in printed


def test_held_out_chances_come_true_as_often_as_they_say(capsys):
    pred, spread, y = _synthetic()
    cal = calibration.fit(pred, spread, y)
    # Every grid predicted ≥ 0.6 lands in exactly one bin.
    assert sum(row[2] for row in cal.table) == (pred >= 0.6).sum()
    for low, high, count, chance, rate in [row for row in cal.table if row[2]]:
        assert low <= chance <= high
        # Within three standard errors of a pass rate measured on `count` grids, plus 0.02.
        assert abs(rate - chance) <= 3 * np.sqrt(chance * (1 - chance) / count) + 0.02
    calibration.report("Tax", cal)
    out = capsys.readouterr().out
    assert "uses probit" in out and len(out.splitlines()) == 5 + len(cal.table)


def test_fit_all_feeds_each_advisor_the_incumbents_mean_and_spread(monkeypatch):
    pred, spread, y = _synthetic(n=1000)
    dev = np.full((1500, 4), np.nan)
    dev[500:, 1] = y  # Tax rated rows 500 to 1499; the others rated nothing
    preds = pred + np.array([[-1.0], [0.0], [1.0]]) * spread  # 3 shuffles that disagree
    asked = []

    def load_outer(name, advisor):
        asked.append((name, advisor))
        return preds, {}

    grids = np.zeros((1500, 7, 7), np.int8)
    monkeypatch.setattr(calibration.validity, "load_dev", lambda: (grids, dev))
    monkeypatch.setattr(calibration.search, "load_outer", load_outer)
    cal = calibration.fit_all(["Tax"])["Tax"]
    assert asked == [(calibration.search.INCUMBENT["Tax"], "Tax")]
    assert cal.log_loss == calibration.fit(preds.mean(0), preds.std(0), y).log_loss
