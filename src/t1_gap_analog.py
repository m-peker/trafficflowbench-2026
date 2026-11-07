"""Analog fill for Task 2 gaps: borrow the in-between from similar moments.

For a gap on rows a..a+17, the signature is the whole corridor's v/vf and q/cap
over the EDGE steps before (the raw history) and after (where data resumes).
The library is every train position with the same time of day (+-TOD_WIN steps)
on library days; the k nearest signatures lend their true in-gap trajectories,
averaged. Level-adjusted per link so a quieter/busier day keeps its own level.
"""
import numpy as np

GAP, EDGE = 18, 6


def analog_fill(Ev, Eq, wins, Lv, Lq, lib_rows, vf, cap, k=10, tod_win=6, adjust=True, max_dist=None):
    """Ev/Eq: grid the release shows (nan where unknown), gaps blank.
    Lv/Lq: library truth grid (nan where unknown); lib_rows: bool over its rows,
    True where a library position may START (its gap and edges must lie inside).
    Returns speed/flow arrays shaped like Ev with only gap rows filled."""
    nT, nL = Ev.shape
    nv, nq = Ev / vf, Eq / cap
    lv, lq = Lv / vf, Lq / cap
    out_v = np.full_like(Ev, np.nan); out_q = np.full_like(Eq, np.nan)
    tod_lib = np.arange(Lv.shape[0]) % 288
    starts_all = np.flatnonzero(lib_rows)
    starts_all = starts_all[(starts_all >= EDGE) & (starts_all + GAP + EDGE < Lv.shape[0])]
    for t in wins:
        a = t + 1; b = a + GAP
        sig_rows = np.r_[a - EDGE:a, b:b + EDGE]
        sv, sq = nv[sig_rows], nq[sig_rows]
        known = np.isfinite(sv) & np.isfinite(sq)
        if known.sum() < 10:
            continue
        dt = (tod_lib[starts_all] - (a % 288) + 144) % 288 - 144
        cand = starts_all[np.abs(dt) <= tod_win]
        rows = cand[:, None] + np.r_[-EDGE:0, GAP:GAP + EDGE][None, :]
        cv, cq = lv[rows], lq[rows]                          # (n, 12, nL)
        ok = known[None] & np.isfinite(cv) & np.isfinite(cq)
        d = (np.where(ok, (cv - sv[None]) ** 2 + (cq - sq[None]) ** 2, 0.0).sum((1, 2))
             / np.maximum(ok.sum((1, 2)), 1))
        d[ok.sum((1, 2)) < 0.5 * known.sum()] = np.inf
        order = np.argsort(d)[:k]
        if max_dist is not None and d[order].mean() > max_dist:
            continue                      # poor match: leave it to the fallback fill
        best = cand[order]
        traj_rows = best[:, None] + np.arange(GAP)[None, :]
        tv, tq = lv[traj_rows], lq[traj_rows]                # (k, 18, nL)
        if adjust:
            # shift each analog so its edges match ours, blending the shift
            # linearly from the left edge to the right edge across the gap
            er = best[:, None] + np.r_[-EDGE:0][None, :]
            el = best[:, None] + np.r_[GAP:GAP + EDGE][None, :]
            w = ((np.arange(GAP) + 1) / (GAP + 1))[None, :, None]
            for arr, lib, ours in ((tv, lv, nv), (tq, lq, nq)):
                with np.errstate(all="ignore"):
                    dl = np.nanmean(ours[a - EDGE:a], 0)[None] - np.nanmean(lib[er], 1)
                    dr = np.nanmean(ours[b:b + EDGE], 0)[None] - np.nanmean(lib[el], 1)
                dl = np.where(np.isfinite(dl), dl, np.where(np.isfinite(dr), dr, 0.0))
                dr = np.where(np.isfinite(dr), dr, dl)
                arr += (1 - w) * dl[:, None, :] + w * dr[:, None, :]
        with np.errstate(all="ignore"):
            out_v[a:b] = np.nanmean(tv, 0) * vf
            out_q[a:b] = np.nanmean(tq, 0) * cap
    return out_v, out_q
