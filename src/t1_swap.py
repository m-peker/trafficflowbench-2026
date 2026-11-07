"""Swap a member out of the six-net ensemble, instead of adding a seventh.

Every experiment since s51 has only ever ADDED a net (v1, w1, o1, p1 as a
seventh member) and every one of them cost the board something (s58-s60,
s56). None has tried REPLACING one of the original six with a differently
shaped net while keeping the ensemble at its measured-optimal size of six.
m1 was the weakest of the six standalone in the 2026-09-22 per-member scan
(partial 0.44775 against f1's 0.44899), so it is the first candidate to
replace; a couple of others are tried too since standalone strength does not
guarantee ensemble contribution.

Free to compute: predict_grid results for the ten existing checkpoints
(f1,h1,i1,j1,m1,n1,v1,w1,o1,p1) are reused across every combination tried, so
this costs ten net evaluations, not one per combination.

    python t1_swap.py            # all ten panels, the official per-regime metric
"""
import nice  # noqa: F401
import itertools
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import ens_search as E
from t1_residual import day_regime, REG_CODE  # reuse, don't duplicate
import t1_nn as NN
import t1feat

SIX = ["f1", "h1", "i1", "j1", "m1", "n1"]
EXTRAS = ["v1", "w1", "o1", "p1"]
DROP_CANDIDATES = ["m1", "h1", "j1"]   # weakest-standalone first
PANELS = E.PANELS
# w1 was trained with T1_TOD_TRAIN=clock (the real hour); everything else was
# trained on the crop-position ramp. Each net must be run at inference in the
# convention it was trained on, or the comparison is not apples to apples.
TOD_FOR = {n: ("abs" if n == "w1" else "const") for n in SIX + EXTRAS}


def panel_fills(panel):
    P = E.Panel(panel)
    nTb = P.obs_sp.shape[0]
    G0 = t1feat.load_grid(panel, "train", masked=True)
    lo = max(0, G0["nT"] - nTb - 256)
    fills = {}
    for n in SIX + EXTRAS:
        os.environ["T1_TOD"] = TOD_FOR[n]
        G, sp, fl, _ = NN.predict_grid(NN.load_net(n), panel, "train", t_lo=lo, t_hi=G0["nT"])
        off = G["nT"] - nTb
        fills[n] = (sp[off:off + nTb][P.ti, P.tj], fl[off:off + nTb][P.ti, P.tj])
    ndays = nTb // 288
    days = G0["days"][-ndays:]
    reg_by_day = {d: day_regime(panel, "train", d) for d in days}
    row_regime = np.array([REG_CODE.get(reg_by_day[days[t // 288]], 0) for t in P.ti])
    return dict(fills=fills, v_t=P.v_t, q_t=P.q_t, lanes=P.lanes[P.tj], vf=P.vf[P.tj],
               regime=row_regime, ok=P.ok)


def main():
    print(f"loading fills for {len(SIX)+len(EXTRAS)} nets x {len(PANELS)} panels...", flush=True)
    data = {}
    for p in PANELS:
        data[p] = panel_fills(p)
        print(f"  {p} ready", flush=True)

    def score(members):
        acc = {1: [0.0, 0.0, 0], 2: [0.0, 0.0, 0], 3: [0.0, 0.0, 0]}
        for panel, d in data.items():
            v = np.mean([d["fills"][m][0] for m in members], 0)
            q = np.mean([d["fills"][m][1] for m in members], 0)
            ok = d["ok"]
            dv = (v[ok] - d["v_t"][ok]) ** 2
            dq = (q[ok] / d["lanes"][ok] - d["q_t"][ok]) ** 2
            reg = d["regime"][ok]
            for r in (1, 2, 3):
                m = reg == r
                if m.any():
                    a = acc[r]
                    a[0] += dv[m].sum(); a[1] += dq[m].sum(); a[2] += int(m.sum())
        Ss = {}
        for r, (sv, sq, n) in acc.items():
            if n:
                rv, rq = (sv / n) ** 0.5, (sq / n) ** 0.5
                Ss[r] = float(np.exp(-(0.54 * rv / 25.0 + 0.46 * rq / 600.0)))
        return float(np.mean(list(Ss.values()))), Ss

    base_s, base_per = score(SIX)
    print(f"\n{'combo':40s} R1     R2     R3     S_state   delta")
    print(f"{'shipped six':40s} " + " ".join(f"{base_per[r]:.4f}" for r in (1, 2, 3)) +
         f"  {base_s:.5f}   +0.00000")

    best = (base_s, "shipped six")
    for drop in DROP_CANDIDATES:
        for extra in EXTRAS:
            combo = [m for m in SIX if m != drop] + [extra]
            s, per = score(combo)
            label = f"-{drop} +{extra}"
            print(f"{label:40s} " + " ".join(f"{per.get(r, float('nan')):.4f}" for r in (1, 2, 3)) +
                 f"  {s:.5f}  {s - base_s:+.5f}")
            if s > best[0]:
                best = (s, label)

    print(f"\nbest: {best[1]}  S_state={best[0]:.5f}  delta={best[0]-base_s:+.5f}")


if __name__ == "__main__":
    main()
