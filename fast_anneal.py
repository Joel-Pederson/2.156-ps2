"""Faster heat-bath annealing under the approval rule: the annealer behind tries H to T,
including try R (0.7706). Moved from data/scratch_keep/scripts/fast_anneal_v5.py on 11 Nov.

Same moves as anneal.anneal_neighbours (196 one-cell changes + 196 random swaps per update), but
the models rate only the edits that could be drawn: each candidate gets key = gain / T + Gumbel
noise, and the first approved candidate in key order is exactly a draw from the heat-bath over
approved candidates (stay put if the stay option's key comes first). So only the top few are
rated, not all 392. Objective: 10-NN sum minus BETA * sum_j exp(-LAM (d_ij - 38)) (pushes close
pairs apart; BETA 0 gives the course score). Designs in one batch of WORKERS are planned from
the same state, then applied in turn.

    python fast_anneal.py START MARGIN UPDATES T0 T1 SEED LAM BETA OUT [BETA_END]   (from ps2/)

Each version added one option, off by default, so this file runs them all (v1 made try H,
v2 try I, v5 tries R, S and T; scripts/chain*.sh ran the chained passes):
v2: the push weight falls linearly from BETA to BETA_END (default BETA, i.e. constant).
v3: FAST_DOUBLES=N adds N random two-cell changes per update (default 0, as v2).
v4: FAST_TRIPLES=N also adds N random three-cell changes per update (default 0, as v3).
v5: FAST_MARGINS=w,t,tr,b sets one cushion per advisor (overrides MARGIN).
Needs the upgraded models (data/cache/upgraded/models.pkl).
"""
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import anneal
import scoring
import upgraded
from validity import ADVISORS

K, BIG, FIRST, MORE = 10, 10**6, 8, 32


def plan(i, X, D, rng, T, lam, beta):
    """(candidate grids in draw order, ahead of staying put) for design i."""
    n = len(X)
    oth = np.delete(np.arange(n), i)
    Xo, xi = X[oth], X[i]
    P = np.partition(D[np.ix_(oth, oth)] + BIG * np.eye(n - 1, dtype=np.int64), K - 1, 1)
    S9, a10 = P[:, : K - 1].sum(1), P[:, K - 1]
    delta = (Xo[None] != np.arange(5)[:, None, None]).astype(np.int64) - (Xo != xi)[None]
    base = D[i, oth]
    vals = np.array([(c, b) for c in range(49) for b in range(5) if b != xi[c]])
    pairs = anneal._PAIRS[xi[anneal._PAIRS[:, 0]] != xi[anneal._PAIRS[:, 1]]]
    pairs = pairs[rng.choice(len(pairs), min(anneal.SWAPS, len(pairs)), replace=False)]
    c1, c2 = pairs[:, 0], pairs[:, 1]
    nd = int(os.environ.get("FAST_DOUBLES", 0))
    d1 = rng.integers(49, size=nd)
    d2 = (d1 + rng.integers(1, 49, size=nd)) % 49  # a different cell
    v1 = (xi[d1] + rng.integers(1, 5, size=nd)) % 5  # a different district
    v2 = (xi[d2] + rng.integers(1, 5, size=nd)) % 5
    nt = int(os.environ.get("FAST_TRIPLES", 0))
    tc = np.argsort(rng.random((nt, 49)), 1)[:, :3]  # three different cells
    tv = (xi[tc] + rng.integers(1, 5, size=(nt, 3))) % 5
    R = np.concatenate([
        base[None] + delta[vals[:, 1], :, vals[:, 0]],
        base[None] + delta[xi[c2], :, c1] + delta[xi[c1], :, c2],
        base[None] + delta[v1, :, d1] + delta[v2, :, d2],
        base[None] + sum(delta[tv[:, j], :, tc[:, j]] for j in range(3)),
        base[None],  # stay put
    ])
    total = (S9[None] + np.minimum(R, a10[None])).sum(1) + np.partition(R, K - 1, 1)[:, :K].sum(1)
    total = total - 2 * beta * np.exp(-lam * (R - 38.0)).sum(1)
    key = (total - total[-1]) / 49000 / T + rng.gumbel(size=len(total))
    order = np.argsort(-key)
    ahead = order[: np.flatnonzero(order == len(total) - 1)[0]]
    cand = np.repeat(xi[None], len(total) - 1, 0)
    cand[np.arange(len(vals)), vals[:, 0]] = vals[:, 1]
    s = len(vals) + np.arange(len(pairs))
    cand[s, c1], cand[s, c2] = xi[c2], xi[c1]
    s = len(vals) + len(pairs) + np.arange(nd)
    cand[s, d1], cand[s, d2] = v1, v2
    s = len(vals) + len(pairs) + nd + np.arange(nt)
    cand[s[:, None], tc] = tv
    return cand[ahead]


def run(grids, pool, updates, t0, t1, seed, lam, beta, beta_end):
    X = grids.reshape(len(grids), 49).astype(np.int64).copy()
    D = (X[:, None] != X[None]).sum(2)
    rng, tic = np.random.default_rng(seed), time.perf_counter()
    done = moved = rated = 0
    while done < updates:
        T = t0 * (t1 / t0) ** (done / updates)
        b = beta + (beta_end - beta) * done / updates
        batch = rng.choice(len(X), anneal.WORKERS, replace=False)
        todo = {i: plan(i, X, D, rng, T, lam, b) for i in batch}
        pos = dict.fromkeys(batch, 0)
        pick, m = {}, FIRST
        while todo:
            jobs = [(i, c[pos[i] : pos[i] + m]) for i, c in todo.items()]
            jobs = [(i, c) for i, c in jobs if len(c)]
            for i in [i for i in todo if i not in dict(jobs)]:
                del todo[i]  # nothing approved ahead of staying put
            if not jobs:
                break
            for (i, c), ok in zip(jobs, pool.map(anneal._approved, [c.reshape(-1, 7, 7) for i, c in jobs])):
                rated += len(c)
                if ok.any():
                    pick[i] = c[ok.argmax()]
                    del todo[i]
                else:
                    pos[i] += m
            m = MORE
        for i, g in pick.items():
            X[i] = g
            D[i] = D[:, i] = (X != g).sum(1)
            moved += 1
        done += len(batch)
        if done % int(os.environ.get("FAST_REPORT", 10_000)) < len(batch):
            print(f"  {done:>9,} updates: score if all pass {scoring.diversity(X.reshape(-1, 7, 7)):.4f}, "
                  f"moves {moved:,}, rated per update {rated / done:.1f} ({time.perf_counter() - tic:.0f} s)",
                  flush=True)
    return X.reshape(grids.shape)


def main():
    start, margin, updates, t0, t1, seed, lam, beta, out = sys.argv[1:10]
    beta_end = float(sys.argv[10]) if len(sys.argv) > 10 else float(beta)
    margin, updates, seed = float(margin), int(updates), int(seed)
    anneal.WORKERS = int(os.environ.get("FAST_WORKERS", anneal.WORKERS))
    t0, t1, lam, beta = float(t0), float(t1), float(lam), float(beta)
    saved = upgraded.load()
    m = os.environ.get("FAST_MARGINS")
    bar = np.array([saved["cutoffs"][a] for a in ADVISORS]) + (np.array(m.split(","), float) if m else margin)
    print("bars above the cut-offs:", bar - np.array([saved["cutoffs"][a] for a in ADVISORS]), flush=True)
    g = np.load(start).astype(np.int64)
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(anneal.WORKERS, spawn, initializer=anneal._init,
                             initargs=("upgraded", bar, None)) as pool:
        bad = np.flatnonzero(~np.concatenate(list(pool.map(anneal._approved, np.array_split(g, anneal.WORKERS)))))
        for i, (fixed, step) in zip(bad, pool.map(anneal._repair, list(g[bad]))):
            assert step >= 0, f"design {i} could not be repaired"
            g[i] = fixed
        print(f"start {scoring.diversity(g):.4f}; {len(bad)} repaired", flush=True)
        g = run(g, pool, updates, t0, t1, seed, lam, beta, beta_end)
        ok = np.concatenate(list(pool.map(anneal._approved, np.array_split(g, anneal.WORKERS))))
        assert ok.all(), "a design is not approved"
    assert len(np.unique(g.reshape(len(g), -1), axis=0)) == len(g), "duplicate designs"
    np.save(out, g)
    print(f"Done: score if all pass {scoring.diversity(g):.4f}; all approved; saved {out}")


if __name__ == "__main__":
    main()
