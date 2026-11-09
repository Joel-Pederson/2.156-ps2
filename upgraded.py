"""The upgraded advisor models (7 Nov), which try F used. A deviation: made after the test set
was spent (docs/decisions.md, 7 Nov evening).

Where this fits
---------------
The final models (surrogates.py) passed the final exam; then the scoreboard showed that the
approval rule limits the score, so better models pay. Ratings are percentile ranks of a hidden
score, so these models predict probit(rating) = norm.ppf(rating), a bell-curve scale, and map
back with norm.cdf. Per advisor:
  1. GBM mix: the mean of (a) a GBM on the 75 features, the 148 lab features, the 600
     neighbourhood counts (features_lab.neighbourhood) and the one-hot cells, and (b) ridge
     regression on the same without the one-hot cells, plus a GBM on what ridge misses;
  2. for the advisors in NETWORK, mixed with per-cell networks: four 3x3 convolutions see each
     cell's surroundings (and its row and column), a 1x1 layer scores each cell, and the grid's
     score is the sum of its cells' scores. Mean of several seeds.
Each cut-off is the 98%-precision cut-off of held-out predictions (3 shuffles x 5 folds), mean
over the shuffles, as for the final models. anneal.py --models upgraded rates with them.

Reads   development ratings (validity.load_dev)
Writes  data/cache/upgraded/oof_<advisor>.npz: held-out probit predictions gb, fc (3, n); y
        data/cache/upgraded/models.pkl: fitted models, networks, weights, cut-offs, commit
Run     python upgraded.py --oof Wellness Tax   held-out predictions (~1 h per advisor with
                                                networks on the GPU; minutes without)
        python upgraded.py --fit                fit on all 4,000 grids per advisor (minutes)

Surprises
---------
* No test-set grade: only held-out predictions and the scoreboard vouch for these models.
* Networks train on the GPU (MPS) and rate on the CPU, so annealing workers need no GPU. MPS
  training is not bit-reproducible, so a refit gives slightly different networks.
* Predictions pass through probit(norm.cdf(.)), the clip the ratings get (5e-4 from 0 and 1).
"""

import argparse
import os
import pickle

import lightgbm as lgb
import numpy as np
import torch
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from torch import nn

import features
import features_lab
import search
import validity
from validity import ADVISORS

DIR = os.path.join(validity.PS2_ROOT, "data", "cache", "upgraded")
SAVED = os.path.join(DIR, "models.pkl")
# advisor: (network weight, seeds), chosen on held-out approvals (docs/decisions.md, 8 Nov
# evening). Try F used (0.3, 2) for both: data/cache/upgraded/models_tryF.pkl.
NETWORK = {"Wellness": (0.5, 5), "Tax": (0.5, 5)}
GBM = dict(
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.3,
    verbose=-1,
)
EPOCHS, BATCH, LR, HOLD = 200, 128, 2e-3, 0.15  # HOLD: share of grids that pick the epoch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_ROW, _COL = np.meshgrid(np.arange(7) / 6, np.arange(7) / 6, indexing="ij")


def probit(y):
    return norm.ppf(np.clip(y, 5e-4, 1 - 5e-4))


def featurize(grids):
    """(E, X): ridge's features (75 + 148 lab + 600 neighbourhood) and the GBMs' (E + one-hot)."""
    E = np.hstack(
        [
            features.engineered(grids),
            features_lab.lab(grids),
            features_lab.neighbourhood(grids),
        ]
    ).astype(np.float64)
    return E, np.hstack([E, features.onehot(grids)])


def fit_gbm_mix(E, X, z):
    mu, sd = E.mean(0), E.std(0) + 1e-9
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((E - mu) / sd, z)
    g1 = lgb.LGBMRegressor(**GBM).fit(X, z).booster_
    g2 = lgb.LGBMRegressor(**GBM).fit(X, z - ridge.predict((E - mu) / sd)).booster_
    return mu, sd, ridge, g1, g2


def gbm_mix_z(model, E, X):
    """The GBM mix's probit-scale prediction."""
    mu, sd, ridge, g1, g2 = model
    one = dict(num_threads=1)
    raw = (g1.predict(X, **one) + ridge.predict((E - mu) / sd) + g2.predict(X, **one)) / 2
    return probit(norm.cdf(raw))


def _tensor(grids):
    """(n, 7, 7) -> (n, 7, 7, 7) float32: five one-hot channels, then row and column."""
    onehot = np.asarray(grids)[:, None] == np.arange(5)[None, :, None, None]
    pos = np.broadcast_to(np.stack([_ROW, _COL]), (len(onehot), 2, 7, 7))
    return torch.tensor(np.concatenate([onehot, pos], 1).astype(np.float32))


def _net(width=64):
    layers, channels = [], 7
    for _ in range(4):
        layers += [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU()]
        channels = width
    return nn.Sequential(*layers, nn.Conv2d(width, 1, 1))


def _grid_score(net, X):
    return net(X).sum((1, 2, 3))


def train_net(grids, z, seed=0):
    """(weights on the CPU, mu, sd): a per-cell network fitted to z (standardised)."""
    torch.manual_seed(seed)
    hold = np.random.default_rng(seed).random(len(z)) < HOLD
    mu, sd = z.mean(), z.std()

    def data(rows):
        y = torch.tensor((z[rows] - mu) / sd, dtype=torch.float32)
        return _tensor(grids[rows]).to(DEVICE), y.to(DEVICE)

    (Xt, yt), (Xh, yh) = data(~hold), data(hold)
    net = _net().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-4)
    steps = -(-len(yt) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, LR, total_steps=EPOCHS * steps, pct_start=0.05
    )
    best, state = np.inf, None
    for _ in range(EPOCHS):
        net.train()
        order = torch.randperm(len(yt), device=DEVICE)
        for b in range(steps):
            idx = order[b * BATCH : (b + 1) * BATCH]
            loss = ((_grid_score(net, Xt[idx]) - yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            err = float(((_grid_score(net, Xh) - yh) ** 2).mean())
        if err < best:  # keep the epoch that predicts the held-back grids best
            best, state = err, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return state, float(mu), float(sd)


_BUILT = {}  # networks rebuilt from saved weights, by id of the weights


def net_z(nets, grids):
    """The mean of the networks' probit-scale predictions, on the CPU."""
    X, out = _tensor(grids), []
    for state, mu, sd in nets:
        if id(state) not in _BUILT:
            _BUILT[id(state)] = _net()
            _BUILT[id(state)].load_state_dict(state)
            _BUILT[id(state)].eval()
        with torch.no_grad():
            out.append(_grid_score(_BUILT[id(state)], X).numpy() * sd + mu)
    return np.mean(out, 0)


def rate(saved, grids):
    """(n, 4) predicted ratings in ADVISORS order."""
    grids = np.asarray(grids)
    E, X = featurize(grids)
    cols = []
    for a in ADVISORS:
        z = gbm_mix_z(saved["models"][a], E, X)
        w = saved["weight"].get(a, 0.0)
        if w:
            z = (1 - w) * z + w * net_z(saved["nets"][a], grids)
        cols.append(norm.cdf(z))
    return np.stack(cols, 1)


def load(path=SAVED):
    with open(path, "rb") as f:
        return pickle.load(f)


def held_out(advisor):
    """Save held-out probit predictions (3 shuffles x 5 folds) of the GBM mix and, for an
    advisor in NETWORK, of the mean of its networks."""
    grids, dev = validity.load_dev()
    g, y, _ = search._advisor_data(grids, dev, advisor)
    z, (E, X) = probit(y), featurize(g)
    seeds = NETWORK.get(advisor, (0, 0))[1]
    gb, fc = np.zeros((2, validity.CV_REPEATS, len(y)))
    for s, f, tr, va in validity.cv_splits(len(y)):
        gb[s, va] = gbm_mix_z(fit_gbm_mix(E[tr], X[tr], z[tr]), E[va], X[va])
        if seeds:
            fc[s, va] = net_z([train_net(g[tr], z[tr], k) for k in range(seeds)], g[va])
        print(f"  {advisor}: shuffle {s}, fold {f} done", flush=True)
    os.makedirs(DIR, exist_ok=True)
    np.savez(os.path.join(DIR, f"oof_{advisor}.npz"), gb=gb, fc=fc, seeds=seeds, y=y)


def fit():
    """Fit every advisor on all its development grids; cut-offs from the held-out predictions."""
    grids, dev = validity.load_dev()
    commit, uncommitted = validity._git_state()
    saved = {"models": {}, "nets": {}, "weight": {}, "cutoffs": {}}
    saved |= {"commit": commit, "uncommitted": bool(uncommitted)}
    for a in ADVISORS:
        w, seeds = NETWORK.get(a, (0.0, 0))
        oof = np.load(os.path.join(DIR, f"oof_{a}.npz"))
        assert not w or oof["seeds"] == seeds, f"{a}: held-out networks != NETWORK's"
        y = oof["y"]
        each = [validity.approved_at(norm.cdf(p), y) for p in (1 - w) * oof["gb"] + w * oof["fc"]]
        saved["cutoffs"][a] = validity.mean_cutoff([r[1] for r in each])
        g, y_dev, _ = search._advisor_data(grids, dev, a)
        assert np.array_equal(y_dev, y), f"{a}: held-out predictions are for other grids"
        z, (E, X) = probit(y), featurize(g)
        saved["models"][a] = fit_gbm_mix(E, X, z)
        if w:
            saved["weight"][a] = w
            saved["nets"][a] = [train_net(g, z, k) for k in range(seeds)]
        print(
            f"{a}: held-out approvals at 98% precision "
            f"{np.mean([r[0] for r in each]):.1f}, cut-off {saved['cutoffs'][a]:.4f}",
            flush=True,
        )
    os.makedirs(DIR, exist_ok=True)
    with open(SAVED, "wb") as f:
        pickle.dump(saved, f)
    print(f"Saved {SAVED}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oof", nargs="+", choices=ADVISORS, default=[], metavar="ADVISOR")
    ap.add_argument("--fit", action="store_true", help="fit on all 4,000 grids and save")
    args = ap.parse_args()
    for a in args.oof:
        held_out(a)
    if args.fit:
        fit()


if __name__ == "__main__":
    main()
