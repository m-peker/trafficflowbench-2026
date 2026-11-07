"""Fast non-negative prior-regularised ODME solver, so lambda can be scanned.

minimize  ||A f - c||^2 + lambda*||f - b||^2   subject to f >= 0

scipy.optimize.nnls on the stacked system takes ~60 s per panel, which makes a
lambda scan across ten panels a half-hour per value.  FISTA on the same
objective converges in a couple of seconds and is validated against nnls below.
"""
import sys, time
import numpy as np
import pandas as pd
from common import ROOT, OUT, PANELS, net_dir


def load_operator(panel):
    net = net_dir(panel)
    paths = pd.read_csv(net / "path_set.csv", dtype={"path_id": str})
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype={"path_id": str, "link_id": str})
    path_ids = paths.path_id.tolist()
    link_ids = inc.link_id.drop_duplicates().tolist()
    pi = {x: i for i, x in enumerate(path_ids)}
    li = {x: i for i, x in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(path_ids)), np.float64)
    A[inc.link_id.map(li).to_numpy(np.int64), inc.path_id.map(pi).to_numpy(np.int64)] = 1.0
    # base_od.csv was withdrawn in the 2026-09-12 rebuild. It only ever fed the
    # anchor="base_od" path, which measured 0.46 against the weak prior's 0.833
    # and has not been used since v1, so an empty frame keeps every live caller
    # working rather than crashing them on a file nobody needs.
    bp = net / "base_od.csv"
    base = paths[["path_id", "origin_zone", "destination_zone"]].copy()
    if bp.exists():
        raw = pd.read_csv(bp, dtype={"path_id": str})
        base = base.merge(raw[["path_id", "departure_time", "base_flow"]],
                          on="path_id", how="left")
    else:
        base["departure_time"] = pd.NA
        base["base_flow"] = 0.0
    base = base.set_index("path_id").reindex(path_ids).reset_index()
    base["base_flow"] = pd.to_numeric(base.base_flow, errors="coerce").fillna(0.0)
    return path_ids, link_ids, A, paths, base


def fista(A, c, b, lam, iters=4000, tol=1e-9):
    """Projected accelerated gradient on ||Af-c||^2 + lam*||f-b||^2, f >= 0."""
    AtA_norm = float(np.linalg.norm(A, 2)) ** 2
    Lip = 2.0 * (AtA_norm + lam)
    step = 1.0 / Lip
    f = np.maximum(b.copy(), 0.0)
    y = f.copy()
    t = 1.0
    Atc = A.T @ c
    prev = None
    for k in range(iters):
        grad = 2.0 * (A.T @ (A @ y) - Atc + lam * (y - b))
        f_new = np.maximum(y - step * grad, 0.0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = f_new + ((t - 1) / t_new) * (f_new - f)
        f, t = f_new, t_new
        if k % 50 == 0:
            obj = np.sum((A @ f - c) ** 2) + lam * np.sum((f - b) ** 2)
            if prev is not None and abs(prev - obj) <= tol * max(abs(prev), 1.0):
                break
            prev = obj
    return f


def panel_inputs(panel, split, anchor="weak_prior"):
    path_ids, link_ids, A, paths, base = load_operator(panel)
    cf = pd.read_csv(ROOT / "task4" / panel / split / "synthetic_link_counts.csv",
                     dtype={"link_id": str})
    counts = cf.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(float)
    meas = np.array([l in set(cf.link_id) for l in link_ids], dtype=bool)
    pr = pd.read_csv(ROOT / "task4" / panel / split / "synthetic_weak_prior.csv",
                     dtype={"path_id": str})
    weak = pr.set_index("path_id").reindex(path_ids)["path_flow"].fillna(0.0).to_numpy(float)
    b = weak if anchor == "weak_prior" else base.base_flow.to_numpy(float)
    dep = str(pr.departure_time.iloc[0])
    return dict(path_ids=path_ids, link_ids=link_ids, A=A[meas], counts=counts[meas],
                b=b, weak=weak, paths=paths, dep=dep)


def s_link(A, f, c):
    return max(0.0, 1.0 - float(np.abs(A @ f - c).sum()) / max(float(c.sum()), 1e-9))


if __name__ == "__main__":
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
    P = panel_inputs(panel, "validation")
    A, c, b, weak = P["A"], P["counts"], P["b"], P["weak"]
    print(f"{panel}: A={A.shape}  sum(c)={c.sum():,.0f}  sum(b)={b.sum():,.0f}")
    print(f"prior loaded onto the network: sum(Ab)={float((A @ b).sum()):,.0f} "
          f"-> S_link(b)={s_link(A, b, c):+.4f}")
    alpha = float(c.sum() / max((A @ b).sum(), 1e-9))
    print(f"best uniform rescale of the prior: alpha={alpha:.4f}  "
          f"S_link={s_link(A, alpha * b, c):.4f}\n")

    t = time.time()
    from scipy.optimize import nnls
    aa = np.vstack([A, np.sqrt(0.05) * np.eye(A.shape[1])])
    bb = np.concatenate([c, np.sqrt(0.05) * b])
    f_ref, _ = nnls(aa, bb)
    t_nnls = time.time() - t
    t = time.time()
    f_fast = fista(A, c, b, 0.05)
    t_fista = time.time() - t
    print(f"validation at lambda=0.05:  nnls {t_nnls:.0f}s  fista {t_fista:.1f}s")
    print(f"  S_link  nnls={s_link(A, f_ref, c):.5f}  fista={s_link(A, f_fast, c):.5f}")
    print(f"  max|diff|={np.abs(f_ref - f_fast).max():.4f}  "
          f"rel L1={np.abs(f_ref - f_fast).sum() / f_ref.sum():.5f}\n")

    print(f"{'lambda':>9s} {'S_link':>8s} {'sum f':>12s} {'D=sum|f-b|':>13s} {'D/sum(b)':>9s}")
    for lam in (0.05, 1, 5, 20, 50, 100, 300, 1000):
        f = fista(A, c, b, float(lam))
        D = float(np.abs(f - weak).sum())
        print(f"{lam:9g} {s_link(A, f, c):8.5f} {f.sum():12,.0f} {D:13,.0f} "
              f"{D / weak.sum():9.4f}")
