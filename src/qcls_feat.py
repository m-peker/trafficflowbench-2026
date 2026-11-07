"""Queue-specific features for the Task 2 classifier.

The classifier currently runs on the 106 Task 1 features, which were designed
for reconstruction: they carry the neighbouring cells' raw speeds, but not
whether those neighbours are below *their own* queue threshold. The threshold is
0.60 * free_speed_kmh and free speed varies per link, so the model has to
rediscover a per-link comparison it could simply be told.

Queues also propagate along the corridor and persist in time, so a blanked cell
flanked by queued cells is almost certainly queued. That is a strong, cheap
signal and none of the Task 1 features express it.

This block adds, all read from the masked layer that is available at test time:

  q_prev/q_next          was this link queued at its nearest published
                         observation before / after t
  q_up1..3, q_dn1..3     is the k-th link up / down the corridor queued at t
  n_q_neigh              how many of those six are queued
  margin_*               (speed - 0.60*v_f) / v_f at the same neighbours, so the
                         model sees how close to the threshold they sit
  corridor_q_frac        share of published links queued corridor-wide at t
  hist_rate_link/_tod    how often this link queues, overall and at this
                         time of day, from the train split
"""
import numpy as np
import pandas as pd

from common import WORK, links_table
from t1feat import load_grid, time_indices, link_indices

_CACHE = {}


def hist_rates(panel):
    """Queue rate per link and per (link, time-of-day), from the train split."""
    if panel in _CACHE:
        return _CACHE[panel]
    z = np.load(WORK / "cache" / f"{panel}_train.npz", allow_pickle=True)
    src = [str(x) for x in z["link_ids"]]
    sp, pct = z["speed"], z["pct"]
    lk = links_table(panel).drop_duplicates("link_id").set_index("link_id")
    vf = np.array([lk.free_speed_kmh.get(l, 105.0) for l in src], np.float32)
    elig = (pct >= 75) & np.isfinite(sp)
    q = elig & (sp <= 0.6 * vf[None, :])
    nT, nL = sp.shape
    tod = np.arange(nT) % 288
    rate_link = q.sum(0) / np.maximum(elig.sum(0), 1)
    num = np.zeros((288, nL)); den = np.zeros((288, nL))
    np.add.at(num, tod, q); np.add.at(den, tod, elig)
    k = np.ones(9) / 9                                    # smooth +/- 4 slots
    sm_n = np.apply_along_axis(
        lambda c: np.convolve(np.r_[c[-4:], c, c[:4]], k, "valid"), 0, num)
    sm_d = np.apply_along_axis(
        lambda c: np.convolve(np.r_[c[-4:], c, c[:4]], k, "valid"), 0, den)
    rate_tod = sm_n / np.maximum(sm_d, 1e-6)
    _CACHE[panel] = (src, rate_link.astype(np.float32), rate_tod.astype(np.float32))
    return _CACHE[panel]


def build_queue_block(panel, split, ti, tj, link_ids):
    """Queue-aware features at the given target cells, in corridor order."""
    G = load_grid(panel, split, masked=True)
    assert list(G["link_ids"]) == list(link_ids), "corridor order mismatch"
    sp, pct = G["speed"], G["pct"]
    vf = G["stat"]["vf"]
    tod_all = G["tod"].astype(np.int64)
    nT, nL = sp.shape
    thr = 0.6 * vf[None, :]
    pub = np.isfinite(sp) & (pct >= 75)
    q_pub = pub & (sp <= thr)
    margin = np.where(pub, (sp - thr) / vf[None, :], np.nan).astype(np.float32)

    cols, blocks = [], []

    def add(name, v):
        cols.append(name)
        blocks.append(np.asarray(v, np.float32))

    prev_i, next_i = time_indices(pub)
    pi = prev_i[ti, tj].astype(np.int64)
    ni = next_i[ti, tj].astype(np.int64)
    del prev_i, next_i
    okp, okn = pi >= 0, ni < nT
    for nm, idx, ok in (("prev", pi, okp), ("next", ni, okn)):
        qv = np.zeros(len(ti), np.float32)
        mv = np.full(len(ti), np.nan, np.float32)
        qv[ok] = q_pub[idx[ok], tj[ok]]
        mv[ok] = margin[idx[ok], tj[ok]]
        add(f"q_{nm}", qv)
        add(f"margin_{nm}", mv)
    add("gap_prev", np.where(okp, ti - pi, 999))
    add("gap_next", np.where(okn, ni - ti, 999))

    neigh = []
    for k in (1, 2, 3, -1, -2, -3):
        tag = f"up{k}" if k > 0 else f"dn{-k}"
        jj = tj - k
        ok = (jj >= 0) & (jj < nL)
        qv = np.zeros(len(ti), np.float32)
        mv = np.full(len(ti), np.nan, np.float32)
        sel = np.nonzero(ok)[0]
        if len(sel):
            a, b = ti[sel], jj[sel]
            qv[sel] = q_pub[a, b]
            mv[sel] = margin[a, b]
        add(f"q_{tag}", qv)
        add(f"margin_{tag}", mv)
        neigh.append(qv)
    add("n_q_neigh", np.sum(neigh, 0))

    frac = q_pub.sum(1) / np.maximum(pub.sum(1), 1)
    add("corridor_q_frac", frac[ti])
    with np.errstate(invalid="ignore"):
        add("corridor_margin", np.nanmedian(margin, 1)[ti])

    src, rate_link, rate_tod = hist_rates(panel)
    o = [src.index(l) for l in link_ids]
    add("hist_rate_link", rate_link[o][tj])
    add("hist_rate_tod", rate_tod[:, o][tod_all[ti], tj])

    del G, sp, pct, pub, q_pub, margin
    return np.stack(blocks, 1), cols
