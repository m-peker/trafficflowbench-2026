"""Does a k-preserving free-flow FD projection buy S_FD without touching S_LWR?

Public-Code's Task 3 script does, in the free-flow regime (k <= k_crit):
    v <- 0.85*v + 0.15*v_free
    q <- v_new * k          (k = q/v held from the ORIGINAL v,q - unchanged)
so density and therefore S_LWR are untouched while pulling speed toward the
calibrated free-flow line, which is what S_FD's free-flow branch rewards. Never
applied to our own output. Deterministic, no training, no GPU.

    T1_EVAL_LAST=7 python t1_fd_project.py
"""
import nice  # noqa: F401
import os, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ens_search as E
from t1_phys_compare import score_fields, EVAL_LAST, W
import t1_nn as NN, t1feat, t1_fills

os.environ.setdefault("T1_TOD", "const")
NETS = "f1,h1,i1,j1,m1,n1".split(",")
WEIGHT = 0.85
ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]


def main():
    panels = [E.Panel(p) for p in E.PANELS[:3]]
    built = t1_fills.build(NETS, WEIGHT, panels, EVAL_LAST, W, E, NN, t1feat)
    print(f"\nlast {EVAL_LAST} days, {len(panels)} panels, free-flow-only FD projection")
    print(f"{'alpha':>6s} {'S_state':>8s} {'S_lwr':>7s} {'S_fd':>6s} {'partial':>8s}")
    for a in ALPHAS:
        agg = {}
        for P in panels:
            v, q = built[P.panel]
            k = q / np.maximum(v, 1.0)
            kcrit = P.kcrit[P.tj]
            free = k <= kcrit
            v_new = np.where(free, (1 - a) * v + a * P.vf[P.tj], v)
            v_new = np.clip(v_new, 5.0, P.vf[P.tj] * 1.05)
            q_new = np.where(free, k * v_new, q)
            q_new = np.clip(q_new, 5.0, P.cap[P.tj] * 1.15)
            r = score_fields(P, v_new, q_new)
            r["S_fd"] = 3 * r["S_physics"] - 2 * r["S_lwr"]
            for kk, vv in r.items():
                agg.setdefault(kk, []).append(vv)
        r = {kk: float(np.mean(vv)) for kk, vv in agg.items()}
        print(f"{a:6.2f} {r['S_state']:8.4f} {r['S_lwr']:7.4f} {r['S_fd']:6.4f} {r['partial']:8.5f}", flush=True)


if __name__ == "__main__":
    main()
