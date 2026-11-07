"""Does the dedicated density head buy S_LWR without costing S_state?

dn1 is f1's exact recipe plus a third output head trained toward true density
(k = q/v), with its OWN smoothness loss - decoupled from the v/q heads, unlike
p1 whose LWR term derived N_hat from q_hat/v_hat directly and so competed with
the very weights S_state depends on. Final holdout: dn1 speed RMSE 1.278 / flow
28.33 / S_state 0.9507, against f1's 1.230 / 28.43 / 0.9516 - the head did not
cost S_state.

This sweeps how much of the submitted flow should come from the density head
instead of the flow head:

    q_final = (1 - beta) * q_head + beta * (k_head * v_head * lanes)

beta=0 is dn1's plain flow head (no different from any other member). Scored on
the metric's own combination (0.35*S_state + 0.15*S_physics) on the held-out
week, for dn1 alone and for the shipped six with dn1 swapped in for m1 (the
weakest of six, and the only swap that has ever won - 2026-09-24, -m1 +v1).

    T1_EVAL_LAST=7 python t1_khead_sweep.py
"""
import nice  # noqa: F401
import os
import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import ens_search as E
from t1_phys_compare import score_fields, EVAL_LAST, W
import t1_nn as NN
import t1feat

os.environ.setdefault("T1_TOD", "const")
SIX = ["f1", "h1", "i1", "j1", "m1", "n1"]
NETW = float(os.environ.get("T1_NETW", "0.85"))
BETAS = [float(b) for b in os.environ.get("T1_BETAS", "0,0.1,0.2,0.3,0.5,0.7,1.0").split(",")]
PANELS = E.PANELS


def dn1_fields(panel):
    """dn1's speed, flow-head flow, and density-derived flow at held-out cells."""
    P = E.Panel(panel)
    nTb = P.obs_sp.shape[0]
    G0 = t1feat.load_grid(panel, "train", masked=True)
    lo = max(0, G0["nT"] - nTb - 256)
    net = NN.load_net("dn1")
    G, sp, fl, _, dens = NN.predict_grid(net, panel, "train", t_lo=lo, t_hi=G0["nT"])
    off = G["nT"] - nTb
    v = sp[off:off + nTb][P.ti, P.tj]
    q_head = fl[off:off + nTb][P.ti, P.tj]
    k = dens[off:off + nTb][P.ti, P.tj]
    lanes = P.lanes[P.tj]; cap = P.cap[P.tj]; vf = P.vf[P.tj]
    q_k = np.clip(k * v, 5.0, cap * 1.15)
    return P, v, q_head, q_k


def other_fields(panel):
    """The other five shipped members' mean, for the swap-in comparison."""
    P = E.Panel(panel)
    nTb = P.obs_sp.shape[0]
    G0 = t1feat.load_grid(panel, "train", masked=True)
    lo = max(0, G0["nT"] - nTb - 256)
    vs, qs = [], []
    for n in SIX:
        if n == "m1":
            continue
        G, sp, fl, _ = NN.predict_grid(NN.load_net(n), panel, "train", t_lo=lo, t_hi=G0["nT"])
        off = G["nT"] - nTb
        vs.append(sp[off:off + nTb][P.ti, P.tj]); qs.append(fl[off:off + nTb][P.ti, P.tj])
    v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[P.tj] * 1.05)
    k_t = P.mix(W, "density")
    q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k_t * v_t)
                  * P.lanes[P.tj], 5.0, P.cap[P.tj] * 1.15)
    return np.mean(vs, 0), np.mean(qs, 0), v_t, q_t


def main():
    print(f"last {EVAL_LAST} days, {len(PANELS)} panels")
    cache = {p: dn1_fields(p) for p in PANELS}
    print("dn1 fields ready", flush=True)
    other = {p: other_fields(p) for p in PANELS}
    print("other-five fields ready\n", flush=True)

    print(f"{'beta':>5s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}   dn1 alone")
    for beta in BETAS:
        agg = {}
        for p in PANELS:
            P, v, q_head, q_k = cache[p]
            q = (1 - beta) * q_head + beta * q_k
            for kk, vv in score_fields(P, v, q).items():
                agg.setdefault(kk, []).append(vv)
        a = {kk: float(np.mean(vv)) for kk, vv in agg.items()}
        print(f"{beta:5.2f} {a['S_state']:8.4f} {a['S_lwr']:7.4f} {a['partial']:8.5f}", flush=True)

    print(f"\n{'beta':>5s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}   "
         f"5/6 others + dn1(beta) as the sixth, weight {NETW}")
    for beta in BETAS:
        agg = {}
        for p in PANELS:
            P, v, q_head, q_k = cache[p]
            q_dn1 = (1 - beta) * q_head + beta * q_k
            v5, q5, v_t, q_t = other[p]
            v_net = (v5 * 5 + v) / 6; q_net = (q5 * 5 + q_dn1) / 6
            v_final = NETW * v_net + (1 - NETW) * v_t
            q_final = NETW * q_net + (1 - NETW) * q_t
            for kk, vv in score_fields(P, v_final, q_final).items():
                agg.setdefault(kk, []).append(vv)
        a = {kk: float(np.mean(vv)) for kk, vv in agg.items()}
        print(f"{beta:5.2f} {a['S_state']:8.4f} {a['S_lwr']:7.4f} {a['partial']:8.5f}", flush=True)


if __name__ == "__main__":
    main()
