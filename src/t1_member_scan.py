"""Score each shipped Task 1 net on its own, same cells, same convention.

The six-net mean plus trees reads far below either f1 alone (0.4414) or the
trees alone (0.4414) on the 7-day block, which a mean of comparable members
cannot do - so this checks every member separately. Dilations do not change
parameter shapes, so a checkpoint loaded with the wrong ARCH entry loads
silently and predicts nonsense; this is what would catch that.

    T1_EVAL_LAST=7 python t1_member_scan.py f1 h1 i1 j1 m1 n1
"""
import nice  # noqa: F401
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ens_search as E
from t1_phys_compare import score_fields, EVAL_LAST
import t1_nn as NN
import t1feat

os.environ.setdefault("T1_TOD", "abs")


def main():
    names = sys.argv[1:] or ["f1", "h1", "i1", "j1", "m1", "n1"]
    sel = os.environ.get("T1_PANELS")
    names_p = sel.split(",") if sel else E.PANELS[:2]
    panels = [E.Panel(p) for p in names_p]
    print(f"last {EVAL_LAST} days, {len(panels)} panels, tod={os.environ['T1_TOD']}")
    print(f"{'net/tod':12s} {'ch':>3s} {'w':>4s} {'blocks':>6s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}")
    fills = {}
    # "name:tod" scores a net under its own convention - clock-trained members
    # need T1_TOD=abs, ramp-trained ones T1_TOD=const - so a mixed ensemble can
    # be scored the way it would actually be submitted.
    specs = [(x.split(":") + [os.environ.get("T1_TOD", "abs")])[:2] for x in names]
    for n, tod in specs:
        os.environ["T1_TOD"] = tod
        net = NN.load_net(n)
        nb = len([k for k in net.state_dict() if k.endswith("c1.weight")])
        agg = {}
        for P in panels:
            nTb = P.obs_sp.shape[0]
            nT_full = t1feat.load_grid(P.panel, "train", masked=True)["nT"]
            G, sp, fl, _ = NN.predict_grid(net, P.panel, "train",
                                           t_lo=max(0, nT_full - 288 * EVAL_LAST - 256),
                                           t_hi=nT_full)
            off = G["nT"] - nTb
            v, q = sp[P.ti + off, P.tj], fl[P.ti + off, P.tj]
            fills.setdefault(P.panel, {})[n] = (v, q)
            for k, val in score_fields(P, v, q).items():
                agg.setdefault(k, []).append(val)
        a = {k: float(np.mean(v)) for k, v in agg.items()}
        print(f"{n+'/'+tod:12s} {net.inp.weight.shape[1]:3d} {net.inp.weight.shape[0]:4d} {nb:6d} "
              f"{a['S_state']:8.4f} {a['S_lwr']:7.4f} {a['partial']:8.5f}", flush=True)
    # and the mean of them all, to confirm the drop is in the averaging
    agg = {}
    for P in panels:
        v = np.mean([fills[P.panel][n][0] for n, _ in specs], 0)
        q = np.mean([fills[P.panel][n][1] for n, _ in specs], 0)
        for k, val in score_fields(P, v, q).items():
            agg.setdefault(k, []).append(val)
    a = {k: float(np.mean(v)) for k, v in agg.items()}
    print(f"{'mean':12s} {'':3s} {'':4s} {'':6s} {a['S_state']:8.4f} {a['S_lwr']:7.4f} {a['partial']:8.5f}")


if __name__ == "__main__":
    main()
