"""Blend weight per masking regime, scored the official way.

S_state is an unweighted macro-mean of exp(-err) over R1/R2/R3, but R1 is 21.9%
of the scored rows and R3 43.8%, so the metric values an R1 cell about 1.52x and
an R3 cell about 0.76x. Our single blend weight was fitted as if every cell
counted the same. The metric evaluates the three regimes separately, so the
net/tree weight can be chosen separately too - that follows the metric's own
structure rather than tuning against the board.

    T1_EVAL_LAST=7 python t1_regime_w.py
"""
import nice  # noqa: F401
import glob, os, re, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ens_search as E
from t1_phys_compare import EVAL_LAST, W
from common import ROOT
import t1_nn as NN
import t1feat
import t1_fills

NETS = os.environ.get("T1_NETS", "f1,h1,i1,j1,m1,n1").split(",")
WS = [float(x) for x in os.environ.get("T1_WS", "0.65,0.75,0.85,0.95,1.0").split(",")]


def day_regimes(panel):
    out = {}
    for r in ("R1", "R2", "R3"):
        for f in glob.glob(str(ROOT / "corridors" / panel / "train" /
                               "mainline_states_masked" / f"mask_regime={r}" / "*.parquet")):
            out[re.search(r"(\d{4}_\d{2}_\d{2})", f).group(1).replace("_", "-")] = r
    return out


def main():
    names = sys.argv[1:] or NETS
    os.environ.setdefault("T1_TOD", "const")
    panels = [E.Panel(p) for p in E.PANELS[:3]]
    # fills are the *net mean*; rebuild at weight 1.0 so the blend can be swept
    built = t1_fills.build(names, 1.0, panels, EVAL_LAST, W, E, NN, t1feat)

    per = {r: {w: [0.0, 0.0, 0] for w in WS} for r in ("R1", "R2", "R3")}
    for P in panels:
        vn, qn = built[P.panel]
        vt, qt, _ = t1_fills.trees(P, W, E)
        G = t1feat.load_grid(P.panel, "train", masked=True)
        reg = day_regimes(P.panel)
        nTb = P.obs_sp.shape[0]
        off = len(G["days"]) * 288 - nTb
        keep = P.ok & (P.ti >= nTb - 288 * EVAL_LAST)
        rr = np.array([reg.get(str(G["days"][d]), "?") for d in ((P.ti + off) // 288)])
        lanes = P.lanes[P.tj]
        for w in WS:
            v = w * vn + (1 - w) * vt
            q = w * qn + (1 - w) * qt
            dv = (v - P.v_t) ** 2
            dq = (q / lanes - P.q_t) ** 2
            for r in per:
                m = keep & (rr == r)
                if m.sum():
                    a = per[r][w]
                    a[0] += dv[m].sum(); a[1] += dq[m].sum(); a[2] += int(m.sum())

    print(f"\nlast {EVAL_LAST} days, {len(panels)} panels, net weight sweep")
    print(f"{'w':>5s} " + " ".join(f"{r:>9s}" for r in ("R1", "R2", "R3")) + f" {'S_state':>9s}")
    best_per = {}
    rows = {}
    for w in WS:
        S = {}
        for r in per:
            sv, sq, n = per[r][w]
            if not n:
                continue
            e = 0.54 * (sv / n) ** 0.5 / 25.0 + 0.46 * (sq / n) ** 0.5 / 600.0
            S[r] = float(np.exp(-e))
        rows[w] = S
        print(f"{w:5.2f} " + " ".join(f"{S[r]:9.5f}" for r in ("R1", "R2", "R3"))
              + f" {np.mean(list(S.values())):9.5f}")
    for r in ("R1", "R2", "R3"):
        best_per[r] = max(WS, key=lambda w: rows[w][r])
    mixed = np.mean([rows[best_per[r]][r] for r in ("R1", "R2", "R3")])
    flat = max(np.mean(list(rows[w].values())) for w in WS)
    print(f"\nbest single weight      : {flat:.5f}")
    print(f"best weight per regime  : {mixed:.5f}   {best_per}")
    print(f"gain from splitting     : {mixed - flat:+.5f}")


if __name__ == "__main__":
    main()
