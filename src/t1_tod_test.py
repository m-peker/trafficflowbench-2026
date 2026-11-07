"""Score one net through the real inference path under several conventions.

predict_grid feeds the clock, `(t % 288) / 288`, as the time-of-day channel,
while make_example trains a 0->1 ramp across the crop. t1_tod_probe.py showed
the mismatch costs 0.0008-0.0011 of 0.35*S_state + 0.10*S_LWR on training-shaped
crops; this checks it on the path that actually writes submissions, including
the tile shape, which GroupNorm makes part of the input distribution too.

Only the last T1_EVAL_LAST days are filled and scored, so a comparison costs a
fortieth of a full-grid pass.

    T1_EVAL_LAST=7 python t1_tod_test.py f1
"""
import nice  # noqa: F401
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ens_search as E
from t1_phys_compare import score_fields, EVAL_LAST
import t1_nn as NN

CASES = [("abs", 256, 0), ("const", 256, 0), ("tile", 96, 0), ("tile", 256, 0),
         ("const", 96, 0), ("const", 256, 64)]


def main():
    names = sys.argv[1:] or ["f1"]
    assert EVAL_LAST, "set T1_EVAL_LAST"
    panels = [E.Panel(p) for p in E.PANELS[:3]]
    print(f"scoring the last {EVAL_LAST} days of {len(panels)} panels")
    print(f"{'net':4s} {'tod':6s} {'tileT':>5s} {'tileL':>5s} {'S_state':>8s} "
          f"{'S_lwr':>7s} {'partial':>8s}")
    for n in names:
        net = NN.load_net(n)
        for mode, tt, tl in CASES:
            os.environ["T1_TOD"] = mode
            os.environ["T1_TILE_T"] = str(tt)
            os.environ["T1_TILE_L"] = str(tl)
            agg = {}
            for P in panels:
                nTb = P.obs_sp.shape[0]
                G = None
                # fill only the block the scorer looks at, plus a tile of lead-in
                import t1feat
                nT_full = t1feat.load_grid(P.panel, "train", masked=True)["nT"]
                lo = max(0, nT_full - 288 * EVAL_LAST - tt)
                G, sp, fl, _ = NN.predict_grid(net, P.panel, "train", tile_t=tt,
                                               t_lo=lo, t_hi=nT_full)
                off = G["nT"] - nTb
                for k, v in score_fields(P, sp[P.ti + off, P.tj], fl[P.ti + off, P.tj]).items():
                    agg.setdefault(k, []).append(v)
            a = {k: float(np.mean(v)) for k, v in agg.items()}
            print(f"{n:4s} {mode:6s} {tt:5d} {tl or 'full':>5} {a['S_state']:8.4f} "
                  f"{a['S_lwr']:7.4f} {a['partial']:8.5f}", flush=True)


if __name__ == "__main__":
    main()
