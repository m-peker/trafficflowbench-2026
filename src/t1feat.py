"""Task 1 grid primitives.

Index arrays are computed once per panel and every feature is then read at the
target cells only, so a feature costs O(targets) instead of a full (time x link)
matrix.  Index grids are int32; on a 352-link corridor that halves 221 MB to
110 MB apiece.
"""
import numpy as np
import pandas as pd
from common import WORK, links_table
from qfeat import static_table

MISS_T = -1          # no earlier/later observation on this link
MISS_L = -1          # no observation up/down the corridor at this timestamp


def load_grid(panel, split, masked=True):
    tag = f"{panel}_{split}_masked" if masked else f"{panel}_{split}"
    z = np.load(WORK / "cache" / f"{tag}.npz", allow_pickle=True)
    link_ids = [str(x) for x in z["link_ids"]]
    stat = static_table(panel, link_ids)
    o = np.argsort(stat["order"])
    link_ids = [link_ids[i] for i in o]
    stat = {k: (v[o] if isinstance(v, np.ndarray) and v.shape == (len(o),) else v)
            for k, v in stat.items()}
    stat["rank"] = np.arange(len(o), dtype=np.float32)
    out = dict(panel=panel, split=split, link_ids=link_ids, stat=stat,
               speed=z["speed"][:, o], flow=z["flow"][:, o], occ=z["occ"][:, o],
               pct=z["pct"][:, o], days=[str(d) for d in z["days"]])
    if "regime" in z:
        out["regime"] = z["regime"]
    nd = len(out["days"])
    out["dow"] = np.repeat(pd.to_datetime(out["days"]).dayofweek.to_numpy(), 288
                           ).astype(np.int16)
    out["tod"] = np.tile(np.arange(288, dtype=np.int16), nd)
    out["nT"], out["nL"] = out["speed"].shape
    return out


def time_indices(avail):
    """Index of the previous / next available cell along time, per link."""
    nT, nL = avail.shape
    t = np.arange(nT, dtype=np.int32)[:, None]
    prev_i = np.maximum.accumulate(np.where(avail, t, np.int32(MISS_T)), 0)
    next_i = np.minimum.accumulate(
        np.where(avail, t, np.int32(nT))[::-1], 0)[::-1]
    return prev_i, next_i


def link_indices(avail):
    """Index of the nearest available link up / down the corridor, per timestamp."""
    nT, nL = avail.shape
    j = np.arange(nL, dtype=np.int32)[None, :]
    up_i = np.maximum.accumulate(np.where(avail, j, np.int32(MISS_L)), 1)
    dn_i = np.minimum.accumulate(
        np.where(avail, j, np.int32(nL))[:, ::-1], 1)[:, ::-1]
    return up_i, dn_i


def gather_t(mat, idx, tj, nT):
    """mat[idx, tj] with out-of-range positions returned as NaN."""
    ok = (idx >= 0) & (idx < nT)
    out = np.full(len(idx), np.nan, np.float32)
    out[ok] = mat[idx[ok], tj[ok]]
    return out


def gather_l(mat, ti, idx, nL):
    """mat[ti, idx] with out-of-range positions returned as NaN."""
    ok = (idx >= 0) & (idx < nL)
    out = np.full(len(idx), np.nan, np.float32)
    out[ok] = mat[ti[ok], idx[ok]]
    return out


def lin_interp(prev_v, prev_g, next_v, next_g):
    """Linear blend of the two flanking observations, falling back to whichever exists."""
    w = prev_g + next_g
    both = np.isfinite(prev_v) & np.isfinite(next_v) & (w > 0)
    out = np.where(both, (prev_v * next_g + next_v * prev_g) / np.where(w == 0, 1, w),
                   np.nan)
    out = np.where(np.isfinite(out), out, np.where(np.isfinite(prev_v), prev_v, next_v))
    return out.astype(np.float32)


def profile(mat, avail, key, nkey):
    """Mean of mat per (key, link); key is one value per timestamp."""
    nL = mat.shape[1]
    num = np.zeros((nkey, nL))
    den = np.zeros((nkey, nL))
    np.add.at(num, key, np.where(avail, np.nan_to_num(mat), 0.0))
    np.add.at(den, key, avail)
    out = num / np.maximum(den, 1e-9)
    out[den < 1] = np.nan
    return out.astype(np.float32)


def smooth_tod(prof, k=5):
    """Circular moving average across the time-of-day axis."""
    pad = np.concatenate([prof[-k:], prof, prof[:k]], 0)
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    ok = np.isfinite(pad).astype(np.float32)
    val = np.nan_to_num(pad)
    num = np.apply_along_axis(lambda c: np.convolve(c, ker, "valid"), 0, val)
    den = np.apply_along_axis(lambda c: np.convolve(c, ker, "valid"), 0, ok)
    return (num / np.maximum(den, 1e-9)).astype(np.float32)


def rolling_sums(mat, avail):
    """Prefix sums of the observed values and of the observation count."""
    z = np.zeros((1, mat.shape[1]), np.float32)
    cs_v = np.concatenate([z, np.cumsum(np.where(avail, np.nan_to_num(mat), 0.0)
                                        .astype(np.float32), 0)])
    cs_a = np.concatenate([z, np.cumsum(avail.astype(np.float32), 0)])
    return cs_v, cs_a


def rolling_at(cs_v, cs_a, ti, tj, w, nT):
    """Centred nan-aware moving average, read at the target cells."""
    lo = np.clip(ti - w, 0, nT)
    hi = np.clip(ti + w + 1, 0, nT)
    num = cs_v[hi, tj] - cs_v[lo, tj]
    den = cs_a[hi, tj] - cs_a[lo, tj]
    return (num / np.maximum(den, 1e-6)).astype(np.float32)
