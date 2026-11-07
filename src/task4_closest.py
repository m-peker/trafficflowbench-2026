"""Task 4: the count-fitting answer that sits closest to the weak prior in L1.

Three scored answers, characterised on every axis:

    answer              S_ODME   D/sum b  total/b  dense  corr(f,b)
    base_od anchor       0.460     0.786    0.838  0.321      0.698
    weak-prior NNLS      0.833     0.306    0.806  0.415      0.992
    balanced dense       0.497     0.596    0.564  1.000      0.800

Neither the total nor the density explains the spread - base_od's total is 0.838
against NNLS's 0.806 and it still scores 0.46, and density is not even monotone.
The one axis that orders them exactly is distance from the weak prior. That is
also mechanically sensible: S_dev measures that distance outright, and if b is a
usable proxy for f* then S_od moves with it too, at three times the weight.

So the move is to minimise that distance while still reproducing the counts.
Ridge regularisation is the wrong tool for it - lambda shrinks the *L2* distance
and the score reads *L1*, which is why raising lambda 20000x only took D from
0.306 to 0.195 and cost S_link 0.10 doing it. The right problem is

    minimise  sum|f - b|   subject to  A f = c,  f >= 0

solved here by proximal gradient: a gradient step on the count residual, then a
soft-threshold toward b, then projection onto f >= 0. S_link stays ~1 because
the count term keeps its full weight; only the distance shrinks.
"""
import sys, time
import numpy as np
import pandas as pd

from common import OUT, PANELS
from odme_fast import panel_inputs, fista, load_operator, s_link

SPLITS = ["validation", "private"]


def closest_l1(A, c, b, mu=None, iters=4000, tau=None):
    """min sum|f-b| + mu*||Af-c||^2, f >= 0, by proximal gradient."""
    n = A.shape[1]
    L = float(np.linalg.norm(A, 2)) ** 2
    if mu is None:
        mu = 200.0 / max(float(np.abs(c).mean()), 1e-9)  # counts dominate
    step = 1.0 / (2.0 * mu * L)
    if tau is None:
        tau = step                                       # soft-threshold radius
    f = np.maximum(b.astype(float).copy(), 0.0)
    y = f.copy()
    t = 1.0
    for k in range(iters):
        g = 2.0 * mu * (A.T @ (A @ y - c))
        z = y - step * g
        d = z - b
        f_new = b + np.sign(d) * np.maximum(np.abs(d) - tau, 0.0)
        f_new = np.maximum(f_new, 0.0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = f_new + ((t - 1) / t_new) * (f_new - f)
        f, t = f_new, t_new
    return f


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "_l1"
    records, diag = [], []
    for panel in PANELS:
        path_ids, link_ids, A_full, paths, _ = load_operator(panel)
        for split in SPLITS:
            P = panel_inputs(panel, split, anchor="weak_prior")
            A, c, b = P["A"], P["counts"], P["b"]
            t0 = time.time()
            ref = fista(A, c, b, 0.05)                    # the answer that scored 0.833
            f = closest_l1(A, c, b)
            sl, sl_ref = s_link(A, f, c), s_link(A, ref, c)
            d, d_ref = (np.abs(f - b).sum() / b.sum(),
                        np.abs(ref - b).sum() / b.sum())
            # only accept it if the counts still hold; otherwise keep the ridge answer
            use, why = (f, "l1") if sl >= sl_ref - 0.005 else (ref, "fell back")
            x = paths[["path_id", "origin_zone", "destination_zone"]].copy()
            x["panel"] = panel
            x["departure_time"] = P["dep"]
            x["path_flow"] = use
            records.append(x[["panel", "departure_time", "path_id",
                              "origin_zone", "destination_zone", "path_flow"]])
            diag.append(dict(panel=panel, split=split, chose=why,
                             S_link=s_link(A, use, c), S_link_l1=sl, S_link_ref=sl_ref,
                             dev=float(np.abs(use - b).sum() / b.sum()), dev_ref=d_ref,
                             nonzero=float((use > 1e-6).mean()),
                             corr_b=float(np.corrcoef(use, b)[0, 1]),
                             total=float(use.sum() / b.sum())))
            print(f"[{panel}/{split}] {why:9s} S_link {sl_ref:.5f}->{sl:.5f}  "
                  f"D/b {d_ref:.4f}->{d:.4f}  corr {np.corrcoef(use,b)[0,1]:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    out = pd.concat(records, ignore_index=True)
    out.to_csv(OUT / f"odme_submission{tag}.csv", index=False)
    dd = pd.DataFrame(diag)
    dd.to_csv(OUT / f"odme_diag{tag}.csv", index=False)
    print(f"\nwrote {OUT / f'odme_submission{tag}.csv'}  {len(out):,} rows")
    print(f"chose l1 on {(dd.chose=='l1').sum()} of {len(dd)} blocks")
    print(f"S_link  mean {dd.S_link.mean():.5f}  min {dd.S_link.min():.5f}")
    print(f"D/sum b  {dd.dev_ref.mean():.4f} -> {dd.dev.mean():.4f}   "
          f"corr(f,b) mean {dd.corr_b.mean():.4f}")


if __name__ == "__main__":
    main()
