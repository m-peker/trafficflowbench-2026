"""A fairer harness for gap fills: windows drawn the way the organizers draw them.

t1_gap_eval.py picked ongoing windows from every queued origin; the selector
keeps only ongoing windows with persistence IoU <= 0.9 and onset windows whose
queue first appears at step 6 - the moments where the state CHANGES inside the
gap. s68 (analogs) beat s67 (edge+profile) on the old harness and lost on the
board, and the analogs' match distance on the scored months is 1.5-9x the old
harness's. This draws contract windows over the last 60 train days of all eight
queue panels, more of them, and scores fills only (no nets needed).

    python t1_gap_fair.py
"""
import os
os.environ["QONSET_STEP6"] = "1"; os.environ["QONG_CONTRACT"] = "1"
import nice  # noqa: F401
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import t1feat
from common import QUEUE_PANELS
from qdata import load_panel, origins
from qfeat import HIST
from t1_gap_fill import fills
from t1_gap_analog import analog_fill, GAP
from t1_gap_level import neighbours

DAYS, NPER = int(os.environ.get("GAP_DAYS", "60")), int(os.environ.get("GAP_NPER", "15"))


def pick(panel, nT):
    D = load_panel(panel); lo = nT - DAYS * 288
    on, ong = origins(D)
    rng = np.random.default_rng(5); out = []
    for cond, pool in ((0, on), (1, ong)):
        pool = pool[(pool >= lo + HIST) & (pool < nT - GAP - 8)].copy()
        rng.shuffle(pool); k = 0
        for t in pool:
            if all(abs(t - c) >= 72 for c, _ in out):
                out.append((int(t), cond)); k += 1
            if k == NPER:
                break
    return sorted(out), lo


def main():
    nets = [x for x in os.environ.get("GAP_NETS", "").split(",") if x]
    rules = ["mix", "profile", "interp", "analog", "analog_adj", "analog+mix"]
    for n in nets:
        rules += [n, f"{n}+mix", f"gated:{n}"]
    acc = {c: {r: [] for r in rules} for c in (0, 1)}
    dists = []
    for panel in QUEUE_PANELS:
        Gm = t1feat.load_grid(panel, "train", masked=True); Gt = t1feat.load_grid(panel, "train", masked=False)
        o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
        tv, tq, tp = Gt["speed"][:, o], Gt["flow"][:, o], Gt["pct"][:, o]
        tv_e = np.where(tp >= 75, tv, np.nan); tq_e = np.where(tp >= 75, tq, np.nan)
        nT, nL = tv.shape
        W, lo = pick(panel, nT)
        wins = [t for t, _ in W]
        gap = np.zeros(nT, bool); hist = np.zeros(nT, bool)
        for t in wins:
            gap[t + 1:t + 1 + GAP] = True; hist[t - HIST:t] = True
        G = {}
        for c, full in (("speed", tv_e), ("flow", tq_e)):
            g = np.where(Gm["pct"] >= 75, Gm[c], np.nan)
            g[hist] = full[hist]; g[gap] = np.nan
            G[c] = g
        days = pd.to_datetime(Gm["days"]); wk = np.repeat((days.dayofweek >= 5).astype(int), 288)
        F = fills(G, tv_e, tq_e, wins, wk, lo)
        vf, cap = Gm["stat"]["vf"][None, :], Gm["stat"]["cap"][None, :]
        lib = np.arange(nT) < lo - 40
        A0 = analog_fill(G["speed"], G["flow"], wins, tv_e, tq_e, lib, vf, cap, adjust=False)
        A1 = analog_fill(G["speed"], G["flow"], wins, tv_e, tq_e, lib, vf, cap, adjust=True)
        starts = np.flatnonzero(lib)[6:]
        cand = {"mix": (F["mix"]["speed"], F["profile"]["flow"]),
                "profile": (F["profile"]["speed"], F["profile"]["flow"]),
                "interp": (F["interp"]["speed"], F["interp"]["flow"]),
                "analog": A0, "analog_adj": A1}
        cand["analog+mix"] = (0.5 * (A0[0] + cand["mix"][0]), 0.5 * (A0[1] + cand["mix"][1]))
        from common import OUT as _OUT
        for n in nets:
            z = np.load(_OUT / f"gap_fair_{panel}_{n}.npz")
            cand[n] = (z["v"], z["q"])
            cand[f"{n}+mix"] = (0.5 * (z["v"] + cand["mix"][0]), 0.5 * (z["q"] + cand["mix"][1]))
        tgt = np.isnan(Gm["speed"]) & np.isfinite(tv_e)
        lanes = np.maximum(Gm["stat"]["lanes"], 1.0)
        for t, cond in W:
            rows = slice(t + 1, t + 1 + GAP)
            m = tgt[rows]
            if not m.any():
                continue
            dcur = neighbours(G["speed"], G["flow"], t, tv_e, tq_e, starts, vf, cap)[2]
            dists.append(dcur)
            for n in nets:     # gated: analog where it matches well, the net elsewhere
                cand[f"gated:{n}"] = A0 if dcur <= 0.002 else cand[n]
            for r, (fs, fq) in cand.items():
                fs_ = np.where(np.isfinite(fs[rows]), fs[rows], cand["mix"][0][rows])
                fq_ = np.where(np.isfinite(fq[rows]), fq[rows], cand["mix"][1][rows])
                m2 = m & np.isfinite(cand["mix"][0][rows]) & np.isfinite(cand["mix"][1][rows])
                ev = (fs_ - tv[rows])[m2]; eq = ((fq_ - tq[rows]) / lanes[None, :])[m2]
                acc[cond][r].append((np.sum(ev ** 2), np.sum(eq ** 2), m2.sum(), dcur))
        print(f"  {panel}: {len(W)} windows", flush=True)
    print(f"\nmedian analog match distance: {np.median(dists):.4f} "
          f"(old harness ~0.001, scored months 0.0012-0.0094)")
    print(f"{'rule':12s} {'onset v':>8s} {'onset q':>8s} {'ong v':>7s} {'ong q':>7s} {'all v':>7s} {'all q':>7s}  in-gap S_state")
    for r in rules:
        row = []
        for cs in ((0,), (1,), (0, 1)):
            L = [x for c in cs for x in acc[c][r]]
            n = sum(x[2] for x in L)
            row += [np.sqrt(sum(x[0] for x in L) / n), np.sqrt(sum(x[1] for x in L) / n)]
        s = 0.54 * (1 - row[4] / 25) + 0.46 * (1 - row[5] / 600)
        print(f"{r:12s} " + " ".join(f"{x:7.2f}" for x in row) + f"   {s:.4f}")
    # window-level paired comparison, so noise is visible
    for r in ("analog", "mix"):
        v = np.array([x[0] / max(x[2], 1) for c in (0, 1) for x in acc[c][r]])
        tot = np.sort(np.array([x[0] for c in (0, 1) for x in acc[c][r]]))[::-1]
        print(f"{r:8s} per-window speed MSE p50 {np.median(v):7.1f} p90 {np.percentile(v, 90):7.1f} "
              f"max {v.max():8.1f}; top 5 windows carry {tot[:5].sum() / tot.sum():.0%} of total SE")
    d = np.array([x[3] for c in (0, 1) for x in acc[c]["analog"]])
    qs = [0, 0.002, 1.0]
    print("\nby match distance (<=0.002: analog used in s69; >0.002: the fallback's job):")
    for lo_, hi_ in zip(qs[:-1], qs[1:]):
        sel = (d > lo_) & (d <= hi_)
        f = lambda r, i: np.array([x[i] for c in (0, 1) for x in acc[c][r]])[sel].sum()
        n = f("analog", 2)
        print(f"  dist {lo_:.3f}-{hi_:.3f} ({sel.sum():3d} win): " + "  ".join(
            f"{r} v {np.sqrt(f(r, 0) / n):5.2f} q {np.sqrt(f(r, 1) / n):5.1f}" for r in ["mix", "analog"] + nets + [f"{x}+mix" for x in nets]))
    qs = np.quantile(d, [0, 0.5, 0.8, 0.95, 1.0])
    for lo_, hi_ in zip(qs[:-1], qs[1:]):
        sel = (d >= lo_) & (d <= hi_)
        f = lambda r, i: np.array([x[i] for c in (0, 1) for x in acc[c][r]])[sel].sum()
        n = f("analog", 2)
        rv = lambda r: np.sqrt(f(r, 0) / n); rq = lambda r: np.sqrt(f(r, 1) / n)
        print(f"match dist {lo_:.4f}-{hi_:.4f} ({sel.sum():3d} win): analog v {rv('analog'):5.2f} q {rq('analog'):6.1f} | "
              f"mix v {rv('mix'):5.2f} q {rq('mix'):6.1f}")
    for c, name in ((0, "onset"), (1, "ongoing")):
        a = np.array([x[0] / x[2] for x in acc[c]["analog"]]); b = np.array([x[0] / x[2] for x in acc[c]["mix"]])
        print(f"{name}: analog speed MSE lower than mix in {np.mean(a < b):.0%} of {len(a)} windows")


if __name__ == "__main__":
    main()
