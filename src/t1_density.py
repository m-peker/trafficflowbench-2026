"""Buy S_LWR by routing flow through a density estimate instead of predicting it.

S_LWR reads N = q*L/v, so its error compounds the speed and flow errors: at
1.2% on speed and 1.9% on flow the implied density error is ~2.2%, which gives
|N_err| ~ 0.22 veh against mean|dN_true| ~ 0.56 - a ratio of 0.39, i.e.
S_LWR ~ 0.61, which is what the harness reads. The nets emit speed and flow from
two independent heads, so nothing ties their errors together.

The tree ensemble already estimates density directly, and `ens_search.FLOW_BLEND`
mixes k*v into the tree flow for exactly this reason. The nets never see it.
This sweeps

    q(alpha) = (1 - alpha) * q_blend + alpha * k_tree * v_blend * lanes

so alpha = 0 is the shipped answer and alpha = 1 derives flow entirely from the
tree density and the (better) blended speed. Scored on the metric's own
combination, so S_state lost is priced against S_LWR gained.

    T1_EVAL_LAST=7 python t1_density.py
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
ALPHAS = [float(a) for a in
          os.environ.get("T1_ALPHAS", "0,0.1,0.2,0.3,0.5,0.75,1.0").split(",")]


def main():
    names = sys.argv[1:] or NETS
    os.environ.setdefault("T1_TOD", "abs")
    panels = [E.Panel(p) for p in E.PANELS[:3]]
    built = t1_fills.build(names, WEIGHT, panels, EVAL_LAST, W, E, NN, t1feat)
    ktree = {P.panel: t1_fills.trees(P, W, E)[2] for P in panels}

    print(f"\nlast {EVAL_LAST} days, {len(panels)} panels, "
          f"{WEIGHT} * mean({','.join(names)}) + {1 - WEIGHT:.2f} trees")
    print(f"{'alpha':>6s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}")
    for a in ALPHAS:
        agg = {}
        for P in panels:
            v, q = built[P.panel]
            k, lanes, cap = ktree[P.panel], P.lanes[P.tj], P.cap[P.tj]
            q_k = np.clip(k * v * lanes, 5.0, cap * 1.15)
            qa = (1 - a) * q + a * q_k
            for kk, vv in score_fields(P, v, qa).items():
                agg.setdefault(kk, []).append(vv)
        r = {kk: float(np.mean(vv)) for kk, vv in agg.items()}
        print(f"{a:6.2f} {r['S_state']:8.4f} {r['S_lwr']:7.4f} {r['partial']:8.5f}", flush=True)


if __name__ == "__main__":
    main()
