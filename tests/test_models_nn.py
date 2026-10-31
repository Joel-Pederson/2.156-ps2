"""Stage 2's CNN (models_nn.py): it learns, repeats exactly, and keeps its inner folds apart.

- Augmentation is a search option only where allowed.
- Inputs: one channel per district, plus row and column channels that never turn with the grid.
- Trained twice with the same seed, a network predicts identically; it fits a rating that the
  grid determines far better than the mean does.
- The search scores each setting only on grids that setting's inner fits did not train on.
- It runs end to end through the harness, and its record can be saved.
"""

import json

import numpy as np
import pytest
from test_search import _isolate

import models_nn
import search

torch = pytest.importorskip(
    "torch"
)  # the CNN needs PyTorch; the slim CI environment lacks it

TINY = {
    "depth": 2,
    "width": 8,
    "kernel": 3,
    "position": False,
    "augment": True,
    "lr": 3e-3,
}


def test_augmentation_is_a_search_option_only_where_allowed():
    without = models_nn.settings(augment_allowed=False)
    assert without and not any(s["augment"] for s in without)
    every = models_nn.settings(augment_allowed=True)
    assert (
        len(every)
        == 2 * len(without)
        == np.prod([len(v) for v in models_nn.SPACE.values()])
    )


def test_inputs_have_a_channel_per_district_and_position_channels_that_never_turn(
    grid_zoo,
):
    x = models_nn._inputs(grid_zoo, position=False)
    assert x.shape == (len(grid_zoo), 5, 7, 7)
    assert np.array_equal(x.argmax(1), grid_zoo) and np.all(x.sum(1) == 1)
    with_position = models_nn._inputs(grid_zoo, position=True)
    turned = models_nn._inputs(np.rot90(grid_zoo, 1, axes=(1, 2)), position=True)
    assert np.array_equal(with_position[:, 5:], turned[:, 5:])
    # Channel 5 is the row and channel 6 the column, each running from -1 to +1.
    assert with_position[0, 5, 0, 0] == -1 and with_position[0, 5, 6, 0] == 1
    assert with_position[0, 6, 0, 6] == 1


def test_a_network_learns_and_repeats_exactly(grid_zoo):
    y = (grid_zoo == 3).mean((1, 2))  # the share of parks: a rating the grid determines
    net, _ = models_nn.train_network(TINY, grid_zoo, y, seed=1, epochs=30)
    again, _ = models_nn.train_network(TINY, grid_zoo, y, seed=1, epochs=30)
    pred = models_nn.predict_network(net, TINY, grid_zoo)
    assert np.array_equal(pred, models_nn.predict_network(again, TINY, grid_zoo))
    assert np.mean((pred - y) ** 2) < 0.5 * np.var(y)


def test_the_search_scores_each_setting_only_on_grids_its_fits_did_not_train_on(
    monkeypatch,
):
    rng = np.random.default_rng(2)
    grids = rng.integers(0, 5, size=(80, 7, 7)).astype(np.int8)
    y = rng.random(80)
    sizes = []

    def spy_train(setting, g, y_fit, seed, epochs=models_nn.MAX_EPOCHS, val=None):
        trained = {grid.tobytes() for grid in g}
        assert val is not None and not trained & {grid.tobytes() for grid in val[0]}
        sizes.append(len(g))
        return trained, 5

    def spy_predict(trained, setting, g):
        assert not trained & {grid.tobytes() for grid in g}, (
            "scored on a trained-on grid"
        )
        return np.zeros(len(g))

    monkeypatch.setattr(models_nn, "train_network", spy_train)
    monkeypatch.setattr(models_nn, "predict_network", spy_predict)
    cnn = models_nn.CNNSearch(4, "test", ("Wellness",), augment_allowed=False, trials=3)
    choice, record = cnn.search(grids, y, search.inner_splits(80, 0, 0))
    assert sizes == [60] * 12  # 3 settings × 4 inner folds, each trained on 3 of the 4
    assert len(record["tried"]) == 3 and choice["epochs"] == 5 and not choice["augment"]
    json.dumps([choice, record])  # saved with the outer predictions


def test_the_cnn_runs_end_to_end_through_the_harness(grid_zoo, tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(models_nn, "MAX_EPOCHS", 2)
    tiny_space = {key: (value,) for key, value in TINY.items() if key != "augment"}
    monkeypatch.setattr(models_nn, "SPACE", {**tiny_space, "augment": (False, True)})
    y = (grid_zoo == 3).mean((1, 2))
    cnn = models_nn.CNNSearch(4, "test", ("Wellness",), augment_allowed=True, trials=1)
    monkeypatch.setitem(search.REGISTRY, "cnn_test", cnn)
    rows = np.arange(len(grid_zoo))
    preds = search.outer_predictions(
        "cnn_test", "Wellness", grid_zoo, y, rows, workers=1
    )
    assert preds.shape == (3, len(grid_zoo)) and np.isfinite(preds).all()
    (saved,) = (tmp_path / "nested").glob("cnn_test-wellness-*.json")
    rounds = json.loads(saved.read_text())["rounds"]
    assert len(rounds) == search.ROUNDS and all("tried" in r for r in rounds)
