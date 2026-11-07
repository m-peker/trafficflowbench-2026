"""Candidate fills for Task 1 target cells inside Task 2 gaps (no nets needed).

Same windows and gaps as t1_gap_probe.py (the shipped nets score speed RMSE
10.3 / 6.5 and flow/lane 316 / 260 in-gap on D7_I10_E / D7_I210_E). Each fill
uses only what the release shows: the raw history before the gap, the
observations that resume after it, and a per-link time-of-day profile from
earlier train days.

    interp    per link, linear in time between the last observed value before
              the gap and the first one after it
    profile   per-link time-of-day mean (weekday/weekend), scaled to the level
              seen at the two edges
    mix       average of the two

    python t1_gap_fill.py
"""
import nice  # noqa: F401
import os, warnings
import numpy as np
warnings.filterwarnings("ignore")
import t1feat
from t1_gap_probe import pick_windows, PANELS, DAYS, GAP
from qfeat import HIST

EDGE = 6


def edge_value(x, rows):
    """Mean of the finite values over `rows` per link (nan where none)."""
    blk = x[rows]
    with np.errstate(all="ignore"):
        return np.nanmean(blk, 0)


def profile(x, days, lo, wk):
    """Per-link tod mean over days before `lo`, split by weekend flag."""
    nT, nL = x.shape
    tod = np.arange(nT) % 288
    keep = np.arange(nT) < lo
    out = np.full((2, 288, nL), np.nan)
    for w in (0, 1):
        m = keep & (wk == w)
        s = np.zeros((288, nL)); c = np.zeros((288, nL))
        ok = np.isfinite(x) & m[:, None]
        np.add.at(s, tod[m], np.where(ok[m], x[m], 0.0)); np.add.at(c, tod[m], ok[m])
        out[w] = s / np.maximum(c, 1)
        out[w][c == 0] = np.nan
    return out


def fills(G, Gt_v, Gt_q, wins, wk, lo):
    nT, nL = G["speed"].shape
    tod = np.arange(nT) % 288
    res = {k: {c: np.full((nT, nL), np.nan) for c in ("speed", "flow")}
           for k in ("interp", "profile", "mix")}
    for ch, full in (("speed", Gt_v), ("flow", Gt_q)):
        x = G[ch]
        prof = profile(full, None, lo, wk)
        for t in wins:
            a, b = t + 1, t + GAP                     # gap rows a..b
            left = edge_value(x, slice(a - EDGE, a)); right = edge_value(x, slice(b + 1, b + 1 + EDGE))
            pl = np.nanmean(prof[wk[a - 1], tod[a - EDGE:a]], 0)
            pr = np.nanmean(prof[wk[b + 1], tod[(b + 1 + np.arange(EDGE)) % nT]], 0)
            rl = np.where(np.isfinite(left / pl), left / pl, np.nan)
            rr = np.where(np.isfinite(right / pr), right / pr, np.nan)
            rl = np.where(np.isfinite(rl), rl, rr); rr = np.where(np.isfinite(rr), rr, rl)
            L = np.where(np.isfinite(left), left, right); R = np.where(np.isfinite(right), right, left)
            for i, r in enumerate(range(a, b + 1)):
                w = (i + 1) / (GAP + 1)
                li = (1 - w) * L + w * R
                p = prof[wk[r], tod[r]] * ((1 - w) * rl + w * rr)
                p = np.where(np.isfinite(p), p, li); li = np.where(np.isfinite(li), li, p)
                res["interp"][ch][r] = li; res["profile"][ch][r] = p
                res["mix"][ch][r] = 0.5 * (li + p)
    return res


def main():
    orig = t1feat.load_grid
    pooled = {}
    for panel in PANELS:
        Gm = orig(panel, "train", masked=True); Gt = orig(panel, "train", masked=False)
        o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
        tv, tq = Gt["speed"][:, o], Gt["flow"][:, o]
        tp = Gt["pct"][:, o]
        tv_e = np.where(tp >= 75, tv, np.nan); tq_e = np.where(tp >= 75, tq, np.nan)
        nT = Gm["nT"]; lo = nT - DAYS * 288
        wins, _ = pick_windows(panel, nT)
        gap = np.zeros(nT, bool); hist = np.zeros(nT, bool)
        for t in wins:
            gap[t + 1:t + 1 + GAP] = True; hist[t - HIST:t] = True
        G = {c: Gm[c].copy() for c in ("speed", "flow")}
        mp = Gm["pct"]
        for c, full in (("speed", tv), ("flow", tq)):
            G[c] = np.where(mp >= 75, G[c], np.nan)
            G[c][hist] = np.where(tp[hist] >= 75, full[hist], np.nan)
            G[c][gap] = np.nan
        import pandas as pd
        days = pd.to_datetime(Gm["days"])
        wk = np.repeat((days.dayofweek >= 5).astype(int), 288)
        F = fills(G, tv_e, tq_e, wins, wk, lo)
        tgt = np.isnan(Gm["speed"]) & np.isfinite(tv) & (tp >= 75) & gap[:, None]
        tgt[:lo] = False
        lanes = np.maximum(Gm["stat"]["lanes"], 1.0)[None, :]
        print(f"\n{panel}: {int(tgt.sum())} gap targets")
        for k, f in F.items():
            ev = f["speed"][tgt] - tv[tgt]
            eq = (f["flow"][tgt] - tq[tgt]) / np.broadcast_to(lanes, tgt.shape)[tgt]
            ok = np.isfinite(ev) & np.isfinite(eq)
            sv, sq = np.sqrt(np.mean(ev[ok] ** 2)), np.sqrt(np.mean(eq[ok] ** 2))
            print(f"  {k:8s} speed RMSE {sv:6.2f}  flow/lane {sq:6.1f}  "
                  f"S_state {0.54 * (1 - sv / 25) + 0.46 * (1 - sq / 600):.4f}  coverage {ok.mean():.3f}")
            pooled.setdefault(k, []).append((ev[ok], eq[ok]))
    print("\npooled:")
    for k, v in pooled.items():
        ev = np.concatenate([a for a, _ in v]); eq = np.concatenate([b for _, b in v])
        print(f"  {k:8s} speed RMSE {np.sqrt(np.mean(ev ** 2)):6.2f}  flow/lane {np.sqrt(np.mean(eq ** 2)):6.1f}")


if __name__ == "__main__":
    main()
