"""The shipped Task 1 answer, scored under inference conventions.

s51/s53 ship 0.85 * mean(f1,h1,i1,j1,m1,n1) + 0.15 * trees. t1_tod_test.py
showed a single net gains ~0.0008 of the Task 1 block when inference stops
feeding a time-of-day channel the net never saw and tiles at the training
shape. Averaging six nets may absorb the error, so this scores the shipped
combination itself - the only number that decides a submission.

    T1_EVAL_LAST=7 python t1_tod_ens.py
"""
import nice  # noqa: F401
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ens_search as E
from t1_phys_compare import score_fields, EVAL_LAST, W
import t1_nn as NN
import t1feat

NETS = os.environ.get("T1_NETS", "f1,h1,i1,j1,m1,n1").split(",")
WEIGHT = float(os.environ.get("T1_NETW", "0.85"))
# T1_CASES="abs:256:0,const:256:64" - tod mode : time tile : link tile (0 = full)
_D = [("abs", 256, 0), ("const", 256, 64), ("const", 96, 64)]
_E = os.environ.get("T1_CASES")
CASES = [(m, int(a), int(b)) for m, a, b in
         ([c.split(":") for c in _E.split(",")] if _E else _D)]


def main():
    panels = [E.Panel(p) for p in E.PANELS[:3]]
    nets = {n: NN.load_net(n) for n in NETS}
    trees = {}
    for P in panels:
        tj = P.tj
        v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[tj] * 1.05)
        k = P.mix(W, "density")
        q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v_t)
                      * P.lanes[tj], 5.0, P.cap[tj] * 1.15)
        trees[P.panel] = (v_t, q_t)
    print(f"last {EVAL_LAST} days, {len(panels)} panels, "
          f"{WEIGHT} * mean({','.join(NETS)}) + {1 - WEIGHT:.2f} trees")
    print(f"{'tod':6s} {'tileT':>5s} {'tileL':>5s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}")
    for mode, tt, tl in CASES:
        os.environ["T1_TOD"] = mode
        os.environ["T1_TILE_T"] = str(tt)
        os.environ["T1_TILE_L"] = str(tl)
        agg = {}
        for P in panels:
            nTb = P.obs_sp.shape[0]
            nT_full = t1feat.load_grid(P.panel, "train", masked=True)["nT"]
            lo = max(0, nT_full - 288 * EVAL_LAST - tt)
            vs, qs = [], []
            for n in NETS:
                G, sp, fl, _ = NN.predict_grid(nets[n], P.panel, "train", tile_t=tt,
                                               t_lo=lo, t_hi=nT_full)
                off = G["nT"] - nTb
                vs.append(sp[P.ti + off, P.tj]); qs.append(fl[P.ti + off, P.tj])
            v_t, q_t = trees[P.panel]
            v = WEIGHT * np.mean(vs, 0) + (1 - WEIGHT) * v_t
            q = WEIGHT * np.mean(qs, 0) + (1 - WEIGHT) * q_t
            for k2, v2 in score_fields(P, v, q).items():
                agg.setdefault(k2, []).append(v2)
        a = {k2: float(np.mean(v2)) for k2, v2 in agg.items()}
        print(f"{mode:6s} {tt:5d} {tl or 'full':>5} {a['S_state']:8.4f} "
              f"{a['S_lwr']:7.4f} {a['partial']:8.5f}", flush=True)


if __name__ == "__main__":
    main()
