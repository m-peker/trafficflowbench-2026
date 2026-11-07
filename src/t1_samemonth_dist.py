"""Same-month analog library: label-free distance check (user's door 1, 2026-10-02).

For every validation/private Task 2 gap window, the analog matcher's score
(mean of the 10 best edge-signature distances, as in t1_gap_analog.analog_fill)
with two libraries:
  A  train truth (pct >= 75)  - what s83 uses
  B  the same month's own masked grid, starts more than one day away from the
     window and never overlapping any Task 2 gap span of that month
Also the trajectory coverage of the 10 best: share of the window's 18 x L cells
that at least one analog observes (B's library has hidden cells too).

    python t1_samemonth_dist.py
"""
import numpy as np
import pandas as pd
from common import ROOT, QUEUE_PANELS
from t1feat import load_grid
from t1_gap_analog import GAP, EDGE

MAXD = 0.002


def scores(Ev, Eq, wins, Lv, Lq, lib_ok, vf, cap, excl=None, k=10, tod_win=6):
    nv, nq, lv, lq = Ev / vf, Eq / cap, Lv / vf, Lq / cap
    tod = np.arange(Lv.shape[0]) % 288
    st = np.flatnonzero(lib_ok)
    st = st[(st >= EDGE) & (st + GAP + EDGE < Lv.shape[0])]
    out = []
    for t in wins:
        a = t + 1; b = a + GAP
        sig = np.r_[a - EDGE:a, b:b + EDGE]
        sv, sq = nv[sig], nq[sig]
        known = np.isfinite(sv) & np.isfinite(sq)
        if known.sum() < 10:
            out.append((np.nan, np.nan)); continue
        cand = st[np.abs((tod[st] - (a % 288) + 144) % 288 - 144) <= tod_win]
        if excl is not None:
            cand = cand[np.abs(cand - a) > 288]
        rows = cand[:, None] + np.r_[-EDGE:0, GAP:GAP + EDGE][None, :]
        cv, cq = lv[rows], lq[rows]
        ok = known[None] & np.isfinite(cv) & np.isfinite(cq)
        d = (np.where(ok, (cv - sv[None]) ** 2 + (cq - sq[None]) ** 2, 0.0).sum((1, 2))
             / np.maximum(ok.sum((1, 2)), 1))
        d[ok.sum((1, 2)) < 0.5 * known.sum()] = np.inf
        best = cand[np.argsort(d)[:k]]
        tr = lv[best[:, None] + np.arange(GAP)[None, :]]
        out.append((float(np.sort(d)[:k].mean()), float(np.isfinite(tr).any(0).mean())))
    return np.array(out)


def main():
    rows = []
    for panel in QUEUE_PANELS:
        Gt = load_grid(panel, "train", masked=False)
        for split in ("validation", "private"):
            G = load_grid(panel, split, masked=True)
            o = [Gt["link_ids"].index(l) for l in G["link_ids"]]
            vf, cap = G["stat"]["vf"][None, :], G["stat"]["cap"][None, :]
            Ev = np.where(G["pct"] >= 75, G["speed"], np.nan)
            Eq = np.where(G["pct"] >= 75, G["flow"], np.nan)
            day0 = pd.Timestamp(G["days"][0], tz="UTC")
            wi = pd.read_csv(ROOT / "task2" / panel / split / "window_index.csv")
            wins = [int((pd.Timestamp(r.forecast_start) - day0).total_seconds() // 300) - 1 for r in wi.itertuples()]
            wins = [t for t in wins if t + 1 - EDGE >= 0 and t + 1 + GAP + EDGE <= G["nT"]]
            # A: train truth
            Lv = np.where(Gt["pct"][:, o] >= 75, Gt["speed"][:, o], np.nan)
            Lq = np.where(Gt["pct"][:, o] >= 75, Gt["flow"][:, o], np.nan)
            A = scores(Ev, Eq, wins, Lv, Lq, np.ones(Lv.shape[0], bool), vf, cap)
            # B: same month, no start whose 30-step span touches any gap span
            gap = np.zeros(G["nT"], bool)
            for t in wins:
                gap[t + 1:t + 1 + GAP] = True
            span = np.convolve(gap.astype(int), np.ones(GAP + 2 * EDGE, int))[GAP + EDGE - 1:][:G["nT"]]
            lib_ok = np.zeros(G["nT"], bool)
            idx = np.arange(G["nT"])
            lib_ok[:] = True
            for t in wins:
                a = t + 1
                lib_ok[max(0, a - GAP - 2 * EDGE):a + GAP + EDGE] = False
            B = scores(Ev, Eq, wins, Ev, Eq, lib_ok, vf, cap, excl=True)
            for (da, ca), (db, cb) in zip(A, B):
                rows.append(dict(panel=panel, split=split, dA=da, dB=db, covA=ca, covB=cb))
        print(f"  {panel} done", flush=True)
    df = pd.DataFrame(rows).dropna()
    df["goodA"] = df.dA <= MAXD; df["goodB"] = df.dB <= MAXD
    df["bestB"] = df.dB < df.dA
    print(f"\n{len(df)} gap windows (validation + private)")
    print(df.groupby("split")[["goodA", "goodB", "bestB", "covA", "covB"]].mean().round(3))
    print("\nmedian distance A / B:", round(df.dA.median(), 5), round(df.dB.median(), 5))
    print("windows A fails (dA > 0.002) that B rescues:", int((~df.goodA & df.goodB).sum()), "of", int((~df.goodA).sum()))
    print(df.groupby("panel")[["goodA", "goodB", "covB"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
