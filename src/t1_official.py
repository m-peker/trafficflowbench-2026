"""Score Task 1 the way the official guide defines it, per regime.

`Public-Code/docs/TASK1_GUIDE.md` (the organizers' toolkit, absent from our
dataset copy):

    NRMSE_speed = RMSE(speed) / 25
    NRMSE_flow  = RMSE(flow / lanes) / 600
    regime_err  = 0.54 * NRMSE_speed + 0.46 * NRMSE_flow
    S_regime    = exp(-regime_err)
    S_state     = (S_R1 + S_R2 + S_R3) / 3        unweighted macro-mean

Our harness has always used `0.54*(1 - RMSE_v/25) + 0.46*(1 - RMSE_q/600)`
pooled over every cell. Two differences: exp(-e) against 1-e, and a macro-mean
over regimes against a cell-count-weighted pool. R1 days are 20% outage and 22%
of the scored rows but a third of the score; R3 days are 50% outage and 44% of
the rows and also a third.

    T1_EVAL_LAST=7 python t1_official.py
"""
import nice  # noqa: F401
import glob, os, re, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import ens_search as E
from t1_phys_compare import EVAL_LAST, W
from common import ROOT
import t1_nn as NN
import t1feat
import t1_fills

NETS = os.environ.get("T1_NETS", "f1,h1,i1,j1,m1,n1").split(",")
WEIGHT = float(os.environ.get("T1_NETW", "0.85"))


def day_regimes(panel):
    """day string -> R1/R2/R3, from the release's partitioned folders."""
    out = {}
    for r in ("R1", "R2", "R3"):
        for f in glob.glob(str(ROOT / "corridors" / panel / "train" /
                               "mainline_states_masked" / f"mask_regime={r}" / "*.parquet")):
            d = re.search(r"(\d{4}_\d{2}_\d{2})", f).group(1)
            out[d.replace("_", "-")] = r   # grid days use dashes
    return out


def main():
    names = sys.argv[1:] or NETS
    os.environ.setdefault("T1_TOD", "const")
    panels = [E.Panel(p) for p in E.PANELS[:3]]
    built = t1_fills.build(names, WEIGHT, panels, EVAL_LAST, W, E, NN, t1feat)

    acc = {r: [0.0, 0.0, 0] for r in ("R1", "R2", "R3")}
    pooled = [0.0, 0.0, 0]
    for P in panels:
        v, q = built[P.panel]
        G = t1feat.load_grid(P.panel, "train", masked=True)
        reg = day_regimes(P.panel)
        nTb = P.obs_sp.shape[0]
        off = len(G["days"]) * 288 - nTb
        keep = P.ok & (P.ti >= nTb - 288 * EVAL_LAST)
        day_of = ((P.ti + off) // 288)
        rr = np.array([reg.get(str(G["days"][d]), "?") for d in day_of])
        assert (rr != "?").any(), f"{P.panel}: no day matched a regime ({G['days'][0]!r})"
        lanes = P.lanes[P.tj]
        dv = (v - P.v_t) ** 2
        dq = (q / lanes - P.q_t) ** 2
        for r in acc:
            m = keep & (rr == r)
            if m.sum():
                acc[r][0] += dv[m].sum(); acc[r][1] += dq[m].sum(); acc[r][2] += int(m.sum())
        pooled[0] += dv[keep].sum(); pooled[1] += dq[keep].sum(); pooled[2] += int(keep.sum())

    print(f"last {EVAL_LAST} days, {len(panels)} panels, {WEIGHT}*mean({','.join(names)})+trees\n")
    print(f"{'regime':7s} {'cells':>9s} {'share':>6s} {'RMSE_v':>7s} {'RMSE_q/ln':>9s} "
          f"{'err':>7s} {'S=exp(-e)':>10s}")
    S = []
    tot = sum(a[2] for a in acc.values())
    for r, (sv, sq, n) in acc.items():
        if not n:
            continue
        rv, rq = (sv / n) ** 0.5, (sq / n) ** 0.5
        e = 0.54 * rv / 25.0 + 0.46 * rq / 600.0
        S.append(np.exp(-e))
        print(f"{r:7s} {n:9d} {n/tot:6.3f} {rv:7.4f} {rq:9.3f} {e:7.5f} {np.exp(-e):10.5f}")
    rv, rq = (pooled[0] / pooled[2]) ** 0.5, (pooled[1] / pooled[2]) ** 0.5
    e = 0.54 * rv / 25.0 + 0.46 * rq / 600.0
    print(f"\nofficial  S_state = mean over regimes = {np.mean(S):.5f}")
    print(f"pooled    exp(-e)                     = {np.exp(-e):.5f}")
    print(f"our old   0.54(1-rv/25)+0.46(1-rq/600)= "
          f"{0.54*(1-rv/25.0)+0.46*(1-rq/600.0):.5f}")


if __name__ == "__main__":
    main()
