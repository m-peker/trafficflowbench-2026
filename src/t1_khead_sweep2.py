"""Does the S_lwr gain from a density head survive averaging more than one in?

dn1 alone showed +0.0042 S_lwr at beta=0.3 with -0.0003 S_state (2026-09-24),
but folded in as ONE of six untouched members the gain vanished into noise
(0.44997-0.45001 across every beta) - diluted by ~1/6 * 1/0.85. dn2 is a second,
independently-seeded density-head member of the same recipe (also converged
clean: S_state 0.9506 against dn1's 0.9507 and f1's 0.9516).

This sweeps ONE shared beta applied to every density-head member simultaneously,
in a six-member ensemble of {i1, j1, m1, n1} (untouched) + {dn1, dn2} (density
heads, both blended at the same beta):

    q_final = (1 - beta) * q_head + beta * (k_head * v_head)   for dn1, dn2
    (i1, j1, m1, n1 contribute their plain flow head, unchanged)

If two density-head members preserve more of the individual gain than one did,
the dilution story is confirmed and the fix is training the whole six this way;
if it is still flat, the mechanism does not survive ensembling at all and the
individual-level gain is not usable.

    T1_EVAL_LAST=7 python t1_khead_sweep2.py
"""
import nice  # noqa: F401
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import ens_search as E
from t1_phys_compare import score_fields, EVAL_LAST, W
import t1_nn as NN
import t1feat

os.environ.setdefault("T1_TOD", "const")
KHEAD_MEMBERS = ["dn1", "dn2"]
PLAIN_MEMBERS = ["i1", "j1", "m1", "n1"]
NETW = float(os.environ.get("T1_NETW", "0.85"))
BETAS = [float(b) for b in os.environ.get("T1_BETAS", "0,0.1,0.2,0.3,0.4,0.5,0.7,1.0").split(",")]
PANELS = E.PANELS


def khead_fields(panel, name):
    """(v, q_head, q_from_density) at held-out cells for one density-head net."""
    P = E.Panel(panel)
    nTb = P.obs_sp.shape[0]
    G0 = t1feat.load_grid(panel, "train", masked=True)
    lo = max(0, G0["nT"] - nTb - 256)
    net = NN.load_net(name)
    G, sp, fl, _, dens = NN.predict_grid(net, panel, "train", t_lo=lo, t_hi=G0["nT"])
    off = G["nT"] - nTb
    v = sp[off:off + nTb][P.ti, P.tj]
    q_head = fl[off:off + nTb][P.ti, P.tj]
    k = dens[off:off + nTb][P.ti, P.tj]
    cap = P.cap[P.tj]
    q_k = np.clip(k * v, 5.0, cap * 1.15)
    return v, q_head, q_k


def plain_fields(panel, name):
    P = E.Panel(panel)
    nTb = P.obs_sp.shape[0]
    G0 = t1feat.load_grid(panel, "train", masked=True)
    lo = max(0, G0["nT"] - nTb - 256)
    G, sp, fl, _ = NN.predict_grid(NN.load_net(name), panel, "train", t_lo=lo, t_hi=G0["nT"])
    off = G["nT"] - nTb
    return sp[off:off + nTb][P.ti, P.tj], fl[off:off + nTb][P.ti, P.tj]


def main():
    print(f"ensemble: {PLAIN_MEMBERS} (plain) + {KHEAD_MEMBERS} (density-head, shared beta)")
    cache = {}
    for p in PANELS:
        P = E.Panel(p)
        kh = {n: khead_fields(p, n) for n in KHEAD_MEMBERS}
        pl = {n: plain_fields(p, n) for n in PLAIN_MEMBERS}
        v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[P.tj] * 1.05)
        k_t = P.mix(W, "density")
        q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k_t * v_t)
                      * P.lanes[P.tj], 5.0, P.cap[P.tj] * 1.15)
        cache[p] = (P, kh, pl, v_t, q_t)
        print(f"  {p} ready", flush=True)

    n_members = len(KHEAD_MEMBERS) + len(PLAIN_MEMBERS)
    print(f"\n{'beta':>5s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}"
         f"   {n_members}-member ensemble (shipped weight {NETW})")
    for beta in BETAS:
        agg = {}
        for p in PANELS:
            P, kh, pl, v_t, q_t = cache[p]
            vs, qs = [], []
            for n in KHEAD_MEMBERS:
                v, q_head, q_k = kh[n]
                vs.append(v); qs.append((1 - beta) * q_head + beta * q_k)
            for n in PLAIN_MEMBERS:
                v, q = pl[n]
                vs.append(v); qs.append(q)
            v_net = np.mean(vs, 0); q_net = np.mean(qs, 0)
            v_final = NETW * v_net + (1 - NETW) * v_t
            q_final = NETW * q_net + (1 - NETW) * q_t
            for kk, vv in score_fields(P, v_final, q_final).items():
                agg.setdefault(kk, []).append(vv)
        a = {kk: float(np.mean(vv)) for kk, vv in agg.items()}
        print(f"{beta:5.2f} {a['S_state']:8.4f} {a['S_lwr']:7.4f} {a['partial']:8.5f}", flush=True)


if __name__ == "__main__":
    main()
