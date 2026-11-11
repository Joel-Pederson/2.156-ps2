"""Which designs, if they failed, give a board score exactly? Usage: which_failed.py SCORE FILE..."""
import sys, itertools, numpy as np

target = round(float(sys.argv[1]) * 49000)
for path in sys.argv[2:]:
    g = np.load(path).reshape(100, -1)
    D = (g[:, None] != g[None]).sum(-1).astype(np.int64)  # cells that differ
    np.fill_diagonal(D, 10**6)
    total = lambda keep: np.sort(D[np.ix_(keep, keep)], 1)[:, :10].sum()
    full = total(np.arange(100))
    print(f"{path.split('/')[-1]}: all pass {full} (= {full / 49000:.10f}); board {target}; loss {full - target}")
    for k in (1, 2, 3):
        combos = np.array(list(itertools.combinations(range(100), k)))
        hits = []
        for b in range(0, len(combos), 2000):
            c = combos[b:b + 2000]
            M = np.ones((len(c), 100), bool)
            M[np.arange(len(c))[:, None], c] = False
            Dm = np.where(M[:, None, :], D[None], 10**6)  # failed grids aren't neighbours
            s = (np.sort(Dm, 2)[:, :, :10].sum(2) * M).sum(1)  # failed grids add nothing
            hits += [tuple(x) for x in c[s == target]]
        print(f"  {k} failed: {len(hits)} matching sets {hits[:5]}")
