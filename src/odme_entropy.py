"""Entropy-regularised ODME, as an alternative to the non-negative least squares.

The NNLS answer reproduces the link counts almost exactly but is sparse — only
~41 percent of paths carry any flow. A real demand vector is not sparse, and
three of the four Task 4 terms compare against the reference demand rather than
against the counts:

    S_od   = 1 - sum|fhat - f*| / sum f*          0.45
    S_link = 1 - sum|A fhat - c| / sum c          0.25   (we already have 0.9999)
    S_dev  = exp(-|Dhat/Dstar - 1|)               0.15
    S_attr = 1 - 0.5*L1(a_hat, a_star)            0.15

Maximum-entropy demand estimation is the standard answer to exactly this
problem: keep f multiplicatively close to the prior while matching the counts.
Solved here by exponentiated gradient, which keeps every path strictly positive
by construction:

    f <- f * exp(-eta * A^T (A f - c))

Starting from the released weak prior, so the solution stays in its structure.
"""
import sys, time
import numpy as np
import pandas as pd

from common import ROOT, OUT, PANELS
from odme_fast import panel_inputs, fista, s_link


def eg_solve(A, c, b, iters=400, damp=1.0, tol=1e-12):
    """KL projection of the prior onto {A f = c}, by multiplicative balancing.

    Each link wants its loaded flow to equal its count, so it asks every path
    through it to scale by r_l = c_l / (A f)_l. A path sits on several links, so
    it takes the geometric mean of what they ask - which is the Bregman /
    Furness update, and the standard way to keep a demand estimate
    multiplicatively close to its prior while matching counts.
    """
    f = np.maximum(b.astype(float), 1e-9)
    n_p = np.maximum(A.sum(0), 1.0)          # measured links per path
    prev = np.inf
    for k in range(iters):
        load = A @ f
        r = np.where(load > 1e-9, c / np.maximum(load, 1e-9), 1.0)
        r = np.clip(r, 1e-6, 1e6)
        adj = (A.T @ np.log(r)) / n_p
        f = f * np.exp(damp * adj)
        f = np.maximum(f, 1e-12)
        obj = float(np.abs(A @ f - c).sum())
        if abs(prev - obj) <= tol * max(abs(prev), 1.0):
            break
        prev = obj
    return f


def describe(name, f, A, c, b):
    load = A @ f
    return dict(name=name,
                S_link=max(0.0, 1 - float(np.abs(load - c).sum()) / max(float(c.sum()), 1e-9)),
                dev=float(np.abs(f - b).sum()) / max(float(b.sum()), 1e-9),
                nonzero=float((f > 1e-6).mean()),
                sum_ratio=float(f.sum() / max(b.sum(), 1e-9)),
                corr=float(np.corrcoef(f, b)[0, 1]))


def main():
    panels = sys.argv[1:] or ["D7_I10_E", "D12_I5_S", "D7_I405_N"]
    for panel in panels:
        P = panel_inputs(panel, "validation", anchor="weak_prior")
        A, c, b = P["A"], P["counts"], P["b"]
        t = time.time()
        f_nnls = fista(A, c, b, 0.05)
        t_n = time.time() - t
        t = time.time()
        f_eg = eg_solve(A, c, b)
        t_e = time.time() - t
        rows = [describe("NNLS lam=0.05 (submitted)", f_nnls, A, c, b),
                describe("entropy / exp-gradient", f_eg, A, c, b)]
        print(f"\n=== {panel} ===  ({t_n:.0f}s nnls, {t_e:.0f}s entropy)")
        print(f"{'candidate':28s} {'S_link':>8s} {'D/sum b':>8s} {'nonzero':>8s} "
              f"{'sum/b':>7s} {'corr(f,b)':>10s}")
        for r in rows:
            print(f"{r['name']:28s} {r['S_link']:8.5f} {r['dev']:8.4f} "
                  f"{r['nonzero']:8.3f} {r['sum_ratio']:7.3f} {r['corr']:10.4f}")
        print(f"{'L1 distance between them':28s} "
              f"{np.abs(f_nnls - f_eg).sum() / max(f_eg.sum(), 1):.4f} of sum(f)")


if __name__ == "__main__":
    main()
