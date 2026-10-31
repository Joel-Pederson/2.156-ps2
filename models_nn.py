"""Stage 2's neural network: a CNN that reads a grid's cells directly (slot 4).

Where this fits
---------------
search.py runs it like any procedure (docs/stage2_models.md, Figure 1). In each outer round:
  search()   tries TRIALS settings drawn from SPACE. Each is trained on every inner fold and
             scored by hp_recall on the inner out-of-fold predictions; each inner fit stops once
             its validation error has not improved for PATIENCE epochs.
  fit()      retrains the best setting on the round's 3,200 training grids, for the average
             number of epochs its inner fits needed.
  predict()  rates the round's 800 validation grids.

The network sees a grid as 5 yes/no channels, one per district, optionally plus 2 channels
holding each cell's row and column. Each of its `depth` convolution layers looks at every cell
and its neighbours; a fully connected layer then sees every position, so the network can learn
effects of position if the advisors have any (rule 2.2).

Surprises
---------
* CPU only, one thread per process, many processes at once (search.py's workers). For networks
  this small that is about as fast as the Apple GPU, and a run repeats exactly.
* Rotated and mirrored training copies ("augment") are a search option only where rule 2.2
  found that orientation doesn't matter; search.REGISTRY sets that per advisor.
* PyTorch is imported only when a network is built, so the rest of the project runs without it.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
from dataclasses import dataclass

import numpy as np

import features

SPACE = {  # each combination is one setting; the search tries TRIALS of them
    "depth": (2, 3),  # convolution layers
    "width": (32, 64),  # filters per layer
    "kernel": (3, 5),  # each filter looks at a 3×3 or 5×5 patch
    "position": (False, True),  # add row and column channels
    "augment": (False, True),  # train on random rotations and mirrors
    "lr": (1e-3, 3e-3),  # learning rate
}
TRIALS = 16
MAX_EPOCHS = 60
PATIENCE = 8  # epochs without a better validation error before an inner fit stops
BATCH = 128
HIDDEN = 64  # units in the fully connected layer
WEIGHT_DECAY = 1e-4


def settings(augment_allowed):
    """Every setting in SPACE, as dicts; none with augmentation unless it is allowed."""
    every = [dict(zip(SPACE, values)) for values in itertools.product(*SPACE.values())]
    return [s for s in every if augment_allowed or not s["augment"]]


def _inputs(grids, position):
    """(n, 5 or 7, 7, 7) float32: a yes/no channel per district, then row and column if asked."""
    x = features.onehot(grids).reshape(-1, 7, 7, 5).transpose(0, 3, 1, 2)
    if position:
        rows, cols = np.mgrid[0:7, 0:7] / 3.0 - 1.0  # -1 at one edge, +1 at the other
        coords = np.broadcast_to(np.stack([rows, cols]), (len(x), 2, 7, 7))
        x = np.concatenate([x, coords], axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def _network(setting, channels):
    from torch import nn

    layers, c = [], channels
    for _ in range(setting["depth"]):
        k = setting["kernel"]
        layers += [
            nn.Conv2d(c, setting["width"], k, padding=k // 2),
            nn.BatchNorm2d(setting["width"]),
            nn.ReLU(),
        ]
        c = setting["width"]
    return nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(c * 49, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1),
    )


def _mse(net, X, t):
    import torch

    net.eval()
    with torch.no_grad():
        return float(torch.mean((net(X).squeeze(1) - t) ** 2))


def train_network(setting, grids, y, seed, epochs=MAX_EPOCHS, val=None):
    """(network, epochs) trained on (grids, y).

    With val=(grids, ratings), training stops once the validation error hasn't improved for
    PATIENCE epochs, and the network keeps its best epoch's weights; `epochs` is that epoch.
    Without val it trains for exactly `epochs`. With augmentation, each grid is shown in a
    random one of its 8 orientations each epoch; the row and column channels never turn.
    """
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    draw = torch.Generator().manual_seed(seed)
    views = (
        features.orientations(grids) if setting["augment"] else np.asarray(grids)[None]
    )
    X = torch.from_numpy(np.stack([_inputs(v, setting["position"]) for v in views]))
    t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    net = _network(setting, X.shape[2])
    opt = torch.optim.AdamW(
        net.parameters(), lr=setting["lr"], weight_decay=WEIGHT_DECAY
    )
    if val is not None:
        X_val = torch.from_numpy(_inputs(val[0], setting["position"]))
        t_val = torch.as_tensor(np.asarray(val[1]), dtype=torch.float32)
    best, best_epoch, best_state = np.inf, epochs, None
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(len(t), generator=draw)
        view = torch.randint(len(X), (len(t),), generator=draw)  # orientation per grid
        for i in range(0, len(t), BATCH):
            b = order[i : i + BATCH]
            if len(b) < 2:  # batch normalisation needs at least two grids
                continue
            loss = torch.nn.functional.mse_loss(net(X[view[b], b]).squeeze(1), t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is None:
            continue
        error = _mse(net, X_val, t_val)
        if error < best:
            best, best_epoch, best_state = error, epoch, copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_epoch


def predict_network(net, setting, grids):
    """Predicted ratings, float64, for (n, 7, 7) grids."""
    import torch

    net.eval()
    X = torch.from_numpy(_inputs(grids, setting["position"]))
    with torch.no_grad():
        parts = [
            net(X[i : i + 1024]).squeeze(1).numpy() for i in range(0, len(X), 1024)
        ]
    return np.concatenate(parts).astype(np.float64)


def _round_seed(inner):
    """A seed for one outer round, from its inner folds (each round has its own)."""
    first_val = np.asarray(inner[0][1], np.int64)
    return int.from_bytes(hashlib.sha256(first_val.tobytes()).digest()[:4], "little")


@dataclass(frozen=True)
class CNNSearch:
    """Slot 4: a random search over SPACE, scored on the inner folds (module docstring)."""

    slot: int
    summary: str
    advisors: tuple
    augment_allowed: bool
    trials: int = TRIALS

    def spec(self):
        return {
            "kind": "cnn_search",
            "space": SPACE,
            "trials": self.trials,
            "augment_allowed": self.augment_allowed,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch": BATCH,
            "hidden": HIDDEN,
            "weight_decay": WEIGHT_DECAY,
            "onehot_version": features.ONEHOT_VERSION,
        }

    def code(self):
        """Everything whose source defines this procedure (search.py fingerprints it)."""
        return (
            CNNSearch,
            settings,
            _inputs,
            _network,
            _mse,
            train_network,
            predict_network,
            _round_seed,
        )

    def library_versions(self):
        import torch

        return {"torch": torch.__version__}

    def model_inputs(self, grids):
        return features.onehot(grids)

    def search(self, grids, y, inner):
        import search as harness  # here, not at the top: search.py imports this module

        seed = _round_seed(inner)
        options = settings(self.augment_allowed)
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(options), min(self.trials, len(options)), replace=False)
        tried = []
        for i in picks:
            setting = options[i]
            pred, epochs = np.full(len(y), np.nan), []
            for k, (fit, val) in enumerate(inner):
                net, epoch = train_network(
                    setting, grids[fit], y[fit], seed + k, val=(grids[val], y[val])
                )
                pred[val] = predict_network(net, setting, grids[val])
                epochs.append(epoch)
            score = harness.hp_recall(pred, y)
            tried.append({**setting, "hp_recall": score, "epochs": epochs})
        # A setting whose training diverged scores NaN; it must never win.
        best = max(tried, key=lambda r: np.nan_to_num(r["hp_recall"], nan=-np.inf))
        choice = {key: best[key] for key in SPACE}
        choice["epochs"] = round(float(np.mean(best["epochs"])))
        return choice, {"tried": tried, "inner_hp_recall": best["hp_recall"]}

    def fit(self, choice, grids, y):
        setting = {key: choice[key] for key in SPACE}
        net, _ = train_network(setting, grids, y, seed=0, epochs=choice["epochs"])
        return net, setting

    def predict(self, model, grids):
        net, setting = model
        return predict_network(net, setting, grids)
