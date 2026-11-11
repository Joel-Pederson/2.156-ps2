"""Chance each design fails, from held-out grids at the same height above the cut-off.
For each design and advisor: the share of held-out grids (upgraded models, 3 shuffles) whose
prediction sits within +-W of the design's height above the cut-off and whose true rating is
below 0.75. Advisors assumed independent. Usage: python risk.py FILE..."""
import os, sys
import numpy as np
from scipy.stats import norm
import make_figures, search, upgraded, validity
W = 0.01
saved = upgraded.load()
grids, dev = validity.load_dev()
ho = {}
for a in validity.ADVISORS:
    _, y, _ = search._advisor_data(grids, dev, a)
    z, _ = make_figures._upgraded_held_out(a)
    m = (norm.cdf(z) - saved["cutoffs"][a]).ravel()
    ho[a] = (m, np.broadcast_to(y < validity.THRESHOLD, z.shape).ravel())
for f in sys.argv[1:]:
    g = np.load(f)
    r = upgraded.rate(saved, g)
    p_fail = np.zeros((len(g), 4))
    for j, a in enumerate(validity.ADVISORS):
        m_ho, fail = ho[a]
        for i, m in enumerate(r[:, j] - saved["cutoffs"][a]):
            near = np.abs(m_ho - m) < W
            p_fail[i, j] = fail[near].mean() if near.sum() >= 30 else 0.0
    p_pass = (1 - p_fail).prod(1)
    print(f"{os.path.basename(f)}: expected failures {(1 - p_pass).sum():.2f}, chance all pass {p_pass.prod():.2f}; "
          "per advisor " + ", ".join(f"{a[:5]} {p_fail[:, j].sum():.2f}" for j, a in enumerate(validity.ADVISORS)))
