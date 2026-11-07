"""s68 (analog) lost to s67 (edge+profile) on the board while winning locally.
Is it a level shift? For each gap: ratio of OUR edge level (6 steps either
side) to the chosen analogs' edge level, validation/private vs the harness.

    python t1_gap_level.py
"""
import nice  # noqa: F401
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from common import ROOT, OUT
from t1feat import load_grid
from t1_gap_analog import GAP, EDGE
from t1_gap_probe import PANELS as HPANELS
from qfeat import HIST

PANELS = ["D7_I10_E", "D7_I210_E", "D7_I405_N", "D12_I5_N"]


def neighbours(Ev, Eq, t, Lv, Lq, starts, vf, cap, k=10, tod_win=6):
    nv, nq, lv, lq = Ev / vf, Eq / cap, Lv / vf, Lq / cap
    a = t + 1; b = a + GAP
    sig = np.r_[a - EDGE:a, b:b + EDGE]
    sv, sq = nv[sig], nq[sig]; known = np.isfinite(sv) & np.isfinite(sq)
    tod = np.arange(Lv.shape[0]) % 288
    dt = (tod[starts] - (a % 288) + 144) % 288 - 144
    cand = starts[np.abs(dt) <= tod_win]
    rows = cand[:, None] + np.r_[-EDGE:0, GAP:GAP + EDGE][None, :]
    cv, cq = lv[rows], lq[rows]
    ok = known[None] & np.isfinite(cv) & np.isfinite(cq)
    d = (np.where(ok, (cv - sv[None]) ** 2 + (cq - sq[None]) ** 2, 0).sum((1, 2))
         / np.maximum(ok.sum((1, 2)), 1))
    d[ok.sum((1, 2)) < 0.5 * known.sum()] = np.inf
    best = cand[np.argsort(d)[:k]]
    er = best[:, None] + np.r_[-EDGE:0, GAP:GAP + EDGE][None, :]
    ours_q = np.nanmean(Eq[sig]); ours_v = np.nanmean(Ev[sig])
    m = known[None] & np.isfinite(Lq[er])
    lib_q = np.nanmean(np.where(m, Lq[er], np.nan)); lib_v = np.nanmean(np.where(m, Lv[er], np.nan))
    ours_q = np.nanmean(np.where(known, Eq[sig], np.nan)); ours_v = np.nanmean(np.where(known, Ev[sig], np.nan))
    return ours_q / lib_q, ours_v / lib_v, float(np.sort(d)[:k].mean()), best


def main():
    for panel in PANELS:
        Gt = load_grid(panel, "train", masked=False)
        for split in ("harness", "validation", "private"):
            if split == "harness":
                z = np.load(OUT / f"gap_probe_{panel}.npz")
                wins, lo = [int(t) for t in z["wins"]], int(z["lo"])
                G = load_grid(panel, "train", masked=True)
                o = [Gt["link_ids"].index(l) for l in G["link_ids"]]
                Ev = np.where(G["pct"] >= 75, G["speed"], np.nan); Eq = np.where(G["pct"] >= 75, G["flow"], np.nan)
                tv = np.where(Gt["pct"][:, o] >= 75, Gt["speed"][:, o], np.nan)
                tq = np.where(Gt["pct"][:, o] >= 75, Gt["flow"][:, o], np.nan)
                for t in wins:
                    Ev[t - HIST:t] = tv[t - HIST:t]; Eq[t - HIST:t] = tq[t - HIST:t]
                    Ev[t + 1:t + 1 + GAP] = np.nan; Eq[t + 1:t + 1 + GAP] = np.nan
                starts = np.arange(EDGE, lo - 40)
            else:
                G = load_grid(panel, split, masked=True)
                o = [Gt["link_ids"].index(l) for l in G["link_ids"]]
                tv = np.where(Gt["pct"][:, o] >= 75, Gt["speed"][:, o], np.nan)
                tq = np.where(Gt["pct"][:, o] >= 75, Gt["flow"][:, o], np.nan)
                Ev = np.where(G["pct"] >= 75, G["speed"], np.nan); Eq = np.where(G["pct"] >= 75, G["flow"], np.nan)
                day0 = pd.Timestamp(G["days"][0], tz="UTC")
                wi = pd.read_csv(ROOT / "task2" / panel / split / "window_index.csv")
                wins = [int((pd.Timestamp(r.forecast_start) - day0).total_seconds() // 300) - 1
                        for r in wi.itertuples()]
                starts = np.arange(EDGE, tv.shape[0] - GAP - EDGE)
            vf, cap = G["stat"]["vf"][None, :], G["stat"]["cap"][None, :]
            R = np.array([neighbours(Ev, Eq, t, tv, tq, starts, vf, cap)[:3] for t in wins
                          if t + 1 + GAP + EDGE <= Ev.shape[0]])
            print(f"{panel:10s} {split:10s} n={len(R):2d}  flow ours/analog {np.median(R[:, 0]):.3f} "
                  f"(mean {R[:, 0].mean():.3f})  speed {np.median(R[:, 1]):.3f}  match dist {np.median(R[:, 2]):.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
