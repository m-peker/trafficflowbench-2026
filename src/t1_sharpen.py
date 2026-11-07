"""Does sharpening the time profile of density buy S_LWR?

S_LWR scores `sum|dN_ours - dN_true| / sum|dN_true|` - the *time differences* of
density, N = (q/v)*L - while the nets minimise squared error on the levels of
speed and flow. An L2 fit of a level shrinks its variation, and the error
decomposition says exactly that: where both ends of a transition are ours,
|dN| is 0.29 against 0.56 in the truth. Those transitions are 12% of the total
and 29% of the LWR error.

So: rescale each filled cell's density away from its own local time average by
gamma, hold speed, and put the change into flow (q = N*v/L). Observed cells keep
the truth, and singly-filled transitions are left alone by construction - they
carry 71% of the error but their error is just |N_err| at one cell, which
sharpening cannot fix.

gamma = 1 is the shipped answer. The sweep is scored on the metric's own
combination, so any S_state lost to sharper flow is priced against the S_LWR
gained.

    T1_EVAL_LAST=7 python t1_sharpen.py [net ...]
"""
import nice  # noqa: F401
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ens_search as E
from t1_phys_compare import score_fields, EVAL_LAST, W
import t1_nn as NN
import t1feat
import t1_fills

NETS = os.environ.get("T1_NETS", "f1,h1,i1,j1,m1,n1").split(",")
WEIGHT = float(os.environ.get("T1_NETW", "0.85"))
GAMMAS = [float(g) for g in os.environ.get("T1_GAMMAS", "1.0,1.1,1.2,1.35,1.5,1.75").split(",")]
SMOOTH = int(os.environ.get("T1_SMOOTH", "3"))          # half-window of the local mean


def sharpen(sp, fl, length, filled, gamma, half=SMOOTH):
    """Scale N away from its local time mean at filled cells; return new flow."""
    if gamma == 1.0:
        return fl
    N = fl / np.maximum(sp, 1.0) * length[None, :]
    k = 2 * half + 1
    pad = np.pad(N, ((half, half), (0, 0)), mode="edge")
    csum = np.cumsum(np.vstack([np.zeros((1, N.shape[1])), pad]), 0)
    local = (csum[k:] - csum[:-k]) / k                  # (nT, nL) running mean
    Nn = np.where(filled, local + gamma * (N - local), N)
    Nn = np.maximum(Nn, 1e-3)
    return np.where(filled, Nn / np.maximum(length[None, :], 1e-6) * np.maximum(sp, 1.0), fl)


def main():
    names = sys.argv[1:] or NETS
    panels = [E.Panel(p) for p in E.PANELS[:3]]
    os.environ.setdefault("T1_TOD", "abs")
    print(f"last {EVAL_LAST} days, {len(panels)} panels, "
          f"{WEIGHT} * mean({','.join(names)}) + {1 - WEIGHT:.2f} trees, smooth +-{SMOOTH}")
    print(f"{'gamma':>6s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}")

    built = t1_fills.build(names, WEIGHT, panels, EVAL_LAST, W, E, NN, t1feat)

    for g in GAMMAS:
        agg = {}
        for P in panels:
            v, q = built[P.panel]
            spg = P.obs_sp.astype(np.float64).copy()
            flg = P.obs_fl.astype(np.float64).copy()
            spg[P.ti, P.tj] = v; flg[P.ti, P.tj] = q
            filled = np.zeros(spg.shape, bool); filled[P.ti, P.tj] = True
            flg2 = sharpen(spg, flg, P.length, filled, g)
            for kk, vv in score_fields(P, v, flg2[P.ti, P.tj]).items():
                agg.setdefault(kk, []).append(vv)
        a = {kk: float(np.mean(vv)) for kk, vv in agg.items()}
        print(f"{g:6.2f} {a['S_state']:8.4f} {a['S_lwr']:7.4f} {a['partial']:8.5f}", flush=True)


if __name__ == "__main__":
    main()
