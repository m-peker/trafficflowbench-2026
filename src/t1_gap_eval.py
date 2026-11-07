"""Score gap-fill rules on top of the shipped ensemble, S_state and S_LWR.

Reads out/gap_probe_<panel>.npz (t1_gap_probe.py with GAP_DUMP=1): the six-net
field predicted with organizer-style 90-minute blackouts injected into the last
7 train days. Gap target cells are then replaced by each rule and the whole
harness is scored. Reported both on the harness's own gap share and rescaled
to the board's 1.9%.

    python t1_gap_eval.py
"""
import nice  # noqa: F401
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import t1feat
from common import OUT
from t1_gap_probe import PANELS, GAP
from t1_gap_fill import fills
from t1_gap_analog import analog_fill
import os
from qfeat import HIST

BOARD_SHARE = 0.019


def dist_to_edge(nT, wins):
    d = np.full(nT, 99)
    for t in wins:
        for i in range(GAP):
            d[t + 1 + i] = min(i, GAP - 1 - i)
    return d


def main():
    orig = t1feat.load_grid
    rules = ["nets", "mix/profile", "analog", "analog_noadj", "analog+mix"] + [x for x in os.environ.get("GAP_EXTRA", "").split(",") if x]
    agg = {r: {"in": [], "out": [], "lwr": []} for r in rules}
    for panel in PANELS:
        z = np.load(OUT / f"gap_probe_{panel}.npz")
        nv, nq, wins, lo = z["v"], z["q"], [int(t) for t in z["wins"]], int(z["lo"])
        Gm = orig(panel, "train", masked=True); Gt = orig(panel, "train", masked=False)
        o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
        tv, tq, tp = Gt["speed"][:, o], Gt["flow"][:, o], Gt["pct"][:, o]
        nT, nL = tv.shape
        gap = np.zeros(nT, bool); hist = np.zeros(nT, bool)
        for t in wins:
            gap[t + 1:t + 1 + GAP] = True; hist[t - HIST:t] = True
        G = {}
        for c, full in (("speed", tv), ("flow", tq)):
            g = np.where(Gm["pct"] >= 75, Gm[c], np.nan)
            g[hist] = np.where(tp[hist] >= 75, full[hist], np.nan); g[gap] = np.nan
            G[c] = g
        days = pd.to_datetime(Gm["days"]); wk = np.repeat((days.dayofweek >= 5).astype(int), 288)
        tv_e = np.where(tp >= 75, tv, np.nan); tq_e = np.where(tp >= 75, tq, np.nan)
        F = fills(G, tv_e, tq_e, wins, wk, lo)
        vf_, cap_ = Gm["stat"]["vf"][None, :], Gm["stat"]["cap"][None, :]
        lib = np.arange(nT) < lo - 40 - 288 * int(os.environ.get("GAP_LIBGAP_DAYS", "0"))
        A = analog_fill(G["speed"], G["flow"], wins, tv_e, tq_e, lib, vf_, cap_)
        A0 = analog_fill(G["speed"], G["flow"], wins, tv_e, tq_e, lib, vf_, cap_, adjust=False,
                         k=int(os.environ.get("GAP_K", "10")), tod_win=int(os.environ.get("GAP_TODW", "6")))
        extra = {}
        for x in os.environ.get("GAP_EXTRA", "").split(","):
            if x and not x.endswith("+analog"):
                ez = np.load(OUT / f"gap_probe_{panel}_{x}.npz")
                extra[x] = (ez["v"], ez["q"])
        tgt = np.isnan(Gm["speed"]) & np.isfinite(tv) & (tp >= 75) & ~hist[:, None]
        tgt[:lo] = False
        ing = tgt & gap[:, None]
        d = dist_to_edge(nT, wins)[:, None] * np.ones((1, nL))
        lanes = np.maximum(Gm["stat"]["lanes"], 1.0)[None, :]
        L = Gm["stat"]["length"][None, :]
        # scorer-side grid: truth wherever the cell is not a target (it holds those)
        Nt = tq / np.maximum(tv, 1.0) * L
        dNt = Nt[1:] - Nt[:-1]
        rows = np.zeros(nT, bool); rows[lo:] = True
        for r in rules:
            v, q = nv.copy(), nq.copy()
            if r != "nets":
                mixv, mixq = F["mix"]["speed"], F["profile"]["flow"]
                if r == "mix/profile":
                    fs, fq = mixv, mixq
                elif r == "analog":
                    fs, fq = A
                elif r == "analog_noadj":
                    fs, fq = A0
                elif r == "analog+mix":
                    fs, fq = 0.5 * (A[0] + mixv), 0.5 * (A[1] + mixq)
                elif r.endswith("+analog"):
                    ev_, eq_ = extra[r[:-7]]
                    fs, fq = 0.5 * (ev_ + A0[0]), 0.5 * (eq_ + A0[1])
                else:
                    fs, fq = extra[r]
                fs = np.where(np.isfinite(fs), fs, mixv); fq = np.where(np.isfinite(fq), fq, mixq)
                ok = ing & np.isfinite(fs) & np.isfinite(fq)
                v[ok] = fs[ok]; q[ok] = fq[ok]
            for key, m in (("in", ing), ("out", tgt & ~gap[:, None])):
                ev = v[m] - tv[m]; eq = (q[m] - tq[m]) / np.broadcast_to(lanes, m.shape)[m]
                agg[r][key].append((ev, eq))
            sp = np.where(tgt, v, tv); fl = np.where(tgt, q, tq)
            N = fl / np.maximum(sp, 1.0) * L
            dN = N[1:] - N[:-1]
            m = np.isfinite(dN) & np.isfinite(dNt) & rows[1:, None]
            agg[r]["lwr"].append((np.abs(dN[m] - dNt[m]).sum(), np.abs(dNt[m]).sum(),
                                  np.abs(dN[m] - dNt[m])[ing[1:][m] | ing[:-1][m]].sum()))
        print(f"  {panel} done", flush=True)

    print(f"\n{'rule':16s} {'in-gap v':>8s} {'in-gap q':>8s} {'S_state':>8s} {'S_st@1.9%':>9s} "
          f"{'S_lwr':>7s} {'S_lwr@1.9%':>10s}")
    for r in rules:
        ci = lambda k, i: np.concatenate([x[i] for x in agg[r][k]])
        vi, qi, vo, qo = ci("in", 0), ci("in", 1), ci("out", 0), ci("out", 1)
        share = len(vi) / (len(vi) + len(vo))
        def s_state(sh):
            mv = sh * np.mean(vi ** 2) + (1 - sh) * np.mean(vo ** 2)
            mq = sh * np.mean(qi ** 2) + (1 - sh) * np.mean(qo ** 2)
            return 0.54 * (1 - np.sqrt(mv) / 25) + 0.46 * (1 - np.sqrt(mq) / 600)
        num = sum(x[0] for x in agg[r]["lwr"]); den = sum(x[1] for x in agg[r]["lwr"])
        gnum = sum(x[2] for x in agg[r]["lwr"])
        slwr = 1 - num / den
        slwr_b = 1 - ((num - gnum) + gnum * BOARD_SHARE / share) / den
        print(f"{r:16s} {np.sqrt(np.mean(vi ** 2)):8.2f} {np.sqrt(np.mean(qi ** 2)):8.1f} "
              f"{s_state(share):8.4f} {s_state(BOARD_SHARE):9.4f} {slwr:7.4f} {slwr_b:10.4f}")
    print(f"(harness gap share {share:.3f})")


if __name__ == "__main__":
    main()
