"""Pull filled densities toward the observed cells that bracket them in time.

S_LWR's error decomposition: transitions with one filled endpoint are 41% of the
total and **71% of the error**, and there the residual is exactly |N_err| at the
filled cell, because the observed endpoint cancels. Meanwhile the blank geometry
is short - mean time-run 1.4 steps, median 1 - so most filled cells sit between
two *observed* cells on the same link, whose densities are truth.

A value bracketed by two nearby truths is better estimated by interpolating them
than by a model that was trained to minimise squared error on levels across the
whole corridor. This blends the two:

    N <- (1 - w) * N_net + w * N_interp        w = WMAX / (1 + LAM * (da + db - 2))

where da, db are the distances in steps to the nearest observed cell above and
below. An isolated blank (da = db = 1) gets the full weight; a long outage gets
almost none, and a cell with no observed neighbour on one side is left alone.
Speed is held and the change goes into flow, so S_state moves only through flow.

WMAX = 0 is the shipped answer.

    T1_EVAL_LAST=7 python t1_interp.py
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
WMAXES = [float(x) for x in os.environ.get("T1_WMAX", "0,0.2,0.35,0.5,0.7,0.9").split(",")]
LAM = float(os.environ.get("T1_LAM", "1.0"))


def bracket(N, observed):
    """Nearest observed value and distance above/below, per cell, along time."""
    nT, nL = N.shape
    big = 10 ** 6
    up_v = np.zeros_like(N); up_d = np.full((nT, nL), big, np.int32)
    last_v = np.zeros(nL); last_d = np.full(nL, big, np.int32)
    for t in range(nT):
        o = observed[t]
        last_d = np.where(o, 0, np.minimum(last_d + 1, big))
        last_v = np.where(o, N[t], last_v)
        up_v[t] = last_v; up_d[t] = last_d
    dn_v = np.zeros_like(N); dn_d = np.full((nT, nL), big, np.int32)
    last_v = np.zeros(nL); last_d = np.full(nL, big, np.int32)
    for t in range(nT - 1, -1, -1):
        o = observed[t]
        last_d = np.where(o, 0, np.minimum(last_d + 1, big))
        last_v = np.where(o, N[t], last_v)
        dn_v[t] = last_v; dn_d[t] = last_d
    return up_v, up_d, dn_v, dn_d


def pull(sp, fl, length, filled, observed, wmax, lam=LAM):
    if wmax == 0.0:
        return fl
    N = fl / np.maximum(sp, 1.0) * length[None, :]
    up_v, up_d, dn_v, dn_d = bracket(N, observed)
    both = (up_d < 10 ** 5) & (dn_d < 10 ** 5) & filled
    tot = np.maximum(up_d + dn_d, 1)
    N_i = (up_v * dn_d + dn_v * up_d) / tot          # linear in time
    w = np.where(both, wmax / (1.0 + lam * np.maximum(tot - 2, 0)), 0.0)
    Nn = np.maximum((1 - w) * N + w * N_i, 1e-3)
    return np.where(filled, Nn / np.maximum(length[None, :], 1e-6) * np.maximum(sp, 1.0), fl)


def main():
    names = sys.argv[1:] or NETS
    os.environ.setdefault("T1_TOD", "abs")
    panels = [E.Panel(p) for p in E.PANELS[:3]]
    built = t1_fills.build(names, WEIGHT, panels, EVAL_LAST, W, E, NN, t1feat)

    print(f"\nlast {EVAL_LAST} days, {len(panels)} panels, lambda {LAM}")
    print(f"{'wmax':>6s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}")
    for wm in WMAXES:
        agg = {}
        for P in panels:
            v, q = built[P.panel]
            spg = P.obs_sp.astype(np.float64).copy()
            flg = P.obs_fl.astype(np.float64).copy()
            spg[P.ti, P.tj] = v; flg[P.ti, P.tj] = q
            filled = np.zeros(spg.shape, bool); filled[P.ti, P.tj] = True
            obs = np.isfinite(P.obs_sp) & ~filled
            flg2 = pull(spg, flg, P.length, filled, obs, wm)
            for kk, vv in score_fields(P, v, flg2[P.ti, P.tj]).items():
                agg.setdefault(kk, []).append(vv)
        r = {kk: float(np.mean(vv)) for kk, vv in agg.items()}
        print(f"{wm:6.2f} {r['S_state']:8.4f} {r['S_lwr']:7.4f} {r['partial']:8.5f}", flush=True)


if __name__ == "__main__":
    main()
