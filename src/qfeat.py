"""Task 2 feature construction, identical for train windows and released windows."""
import numpy as np
import pandas as pd

import qramp
import qshock
from common import ROOT, WORK, links_table, net_dir

HIST = 12
HOR = 6


def link_order(panel, link_ids):
    """Corridor order index per link (upstream -> downstream)."""
    topo = pd.read_csv(net_dir(panel) / "lwr_mainline_topology.csv",
                       dtype={"mainline_link_id": str})
    key = "mainline_link_id" if "mainline_link_id" in topo.columns else "link_id"
    m = dict(zip(topo[key].astype(str), pd.to_numeric(topo.order_index, errors="coerce")))
    order = np.array([m.get(l, np.nan) for l in link_ids], float)
    if np.isnan(order).any():                       # fall back on link id order
        bad = np.isnan(order)
        order[bad] = np.arange(len(order))[bad] + np.nanmax(order[~bad] if (~bad).any() else [0]) + 1
    return order


def static_table(panel, link_ids):
    lk = links_table(panel).drop_duplicates("link_id").set_index("link_id")
    vf = np.array([lk.free_speed_kmh.get(l, 105.0) for l in link_ids], np.float32)
    lanes = np.array([lk.lanes.get(l, 4.0) for l in link_ids], np.float32)
    cap = np.array([lk.capacity_vph.get(l, 7000.0) for l in link_ids], np.float32)
    length = np.array([lk.length_km.get(l, 1.0) for l in link_ids], np.float32)
    order = link_order(panel, link_ids)
    rank = np.argsort(np.argsort(order))            # 0..nL-1 along the corridor
    return dict(vf=vf, lanes=lanes, cap=cap, length=length, order=order, rank=rank)


def _shift(arr, k, fill=np.nan):
    """Shift a per-link vector along the corridor: k>0 -> value of the k-th upstream link."""
    out = np.full_like(arr, fill, dtype=np.float32)
    if k > 0:
        out[k:] = arr[:-k]
    elif k < 0:
        out[:k] = arr[-k:]
    else:
        out[:] = arr
    return out


def history_summary(sp, fl, oc, pct, vf, cap, shock=False,
                    ramp=None, lanes=None):
    """sp/fl/oc/pct: (HIST, nL) in corridor order. Returns a dict of per-link vectors.

    `shock` adds the space-time front features from qshock. Off by default
    because out/queue_model.pkl is fitted on the narrower matrix.
    """
    nH, nL = sp.shape
    elig = (pct >= 75) & np.isfinite(sp)
    r = np.where(elig, sp / vf[None, :], np.nan)            # speed ratio
    fr = np.where(elig & np.isfinite(fl), fl / cap[None, :], np.nan)
    qq = elig & (r <= 0.6)

    has = elig.any(0)
    last_i = np.where(has, nH - 1 - elig[::-1].argmax(0), -1)
    idx = np.arange(nL)
    r_last = np.where(has, r[np.clip(last_i, 0, nH - 1), idx], np.nan)
    f_last = np.where(has, fr[np.clip(last_i, 0, nH - 1), idx], np.nan)
    o_last = np.where(has, oc[np.clip(last_i, 0, nH - 1), idx], np.nan)
    age = np.where(has, nH - 1 - last_i, 99).astype(np.float32)

    first_i = np.where(has, elig.argmax(0), -1)
    r_first = np.where(has, r[np.clip(first_i, 0, nH - 1), idx], np.nan)
    span = np.maximum(last_i - first_i, 1)
    slope = (r_last - r_first) / span

    with np.errstate(invalid="ignore"):
        r_mean = np.nanmean(r, 0)
        r_min = np.nanmin(np.where(np.isfinite(r), r, np.inf), 0)
        r_min = np.where(np.isfinite(r_min), r_min, np.nan)
        f_mean = np.nanmean(fr, 0)
        r_std = np.nanstd(r, 0)
    # last three available observations
    r_last3 = np.full(nL, np.nan, np.float32)
    q_last3 = np.zeros(nL, np.float32)
    for j in range(nL):
        ii = np.flatnonzero(elig[:, j])
        if len(ii):
            r_last3[j] = np.nanmean(r[ii[-3:], j])
            q_last3[j] = qq[ii[-3:], j].any()
    # short-window slope over the last 4 available points
    r_recent = np.full(nL, np.nan, np.float32)
    for j in range(nL):
        ii = np.flatnonzero(elig[:, j])
        if len(ii) >= 2:
            k = ii[-4:]
            r_recent[j] = (r[k[-1], j] - r[k[0], j]) / max(k[-1] - k[0], 1)
    if shock:
        _extra = qshock.features(r, qq, elig)
    else:
        _extra = {}
    if ramp is not None:
        _extra = dict(_extra, **qramp.features(
            ramp["on"], ramp["off"], fl, cap,
            np.ones_like(cap) if lanes is None else lanes, pct >= 75))
    return dict(**_extra,
                r_last=r_last, f_last=f_last, o_last=o_last, age=age, slope=slope,
                r_mean=r_mean, r_min=r_min, f_mean=f_mean, r_std=r_std,
                r_last3=r_last3, q_last3=q_last3, r_recent=r_recent,
                q_last=(r_last <= 0.6).astype(np.float32),
                q_any=qq.any(0).astype(np.float32),
                q_frac=qq.sum(0) / np.maximum(elig.sum(0), 1),
                n_obs=elig.sum(0).astype(np.float32), has=has.astype(np.float32))


FEATS = None


LONG = 36   # QLONG: masked-layer context T-36..T (3 h), longer than the 60-min window history


def long_feats(rows, vf):
    """Per-link summaries of the Task 1 masked layer over T-LONG..T (all stamped <= T).

    The window history is 60 minutes, so a queue's age is capped there; whether a queue
    has stood for 30 minutes or 2.5 hours is what decides when it dissipates."""
    r = rows / np.maximum(vf[None, :], 1.0)
    vis = np.isfinite(r); q = vis & (r <= 0.6); nq = vis & ~q
    n = rows.shape[0]; t = np.arange(n)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = q.sum(0) / np.maximum(vis.sum(0), 1)
        last_nq = np.where(nq, t, -1).max(0)             # last visible free-flow step
        last_q = np.where(q, t, -1).max(0)
        first_q = np.where(q, t, n).min(0)
        run = np.where(last_q > last_nq, n - 1 - last_nq, 0).astype(np.float32)   # standing-queue age
        since_first = np.where(first_q < n, n - 1 - first_q, -1).astype(np.float32)
        rmean = np.nanmean(np.where(vis, r, np.nan)[:-12], 0)
    anyq = q.any(1); cf = np.where(anyq)[0]
    corr_age = float(n - 1 - cf[0]) if len(cf) else -1.0
    return dict(lq_frac=frac.astype(np.float32), lq_run=run, lq_since=since_first,
                lr_mean=rmean.astype(np.float32), lvis=vis.mean(0).astype(np.float32),
                corr_lq=np.full(len(vf), np.float32(frac.mean())), corr_lage=np.full(len(vf), np.float32(corr_age)))


def build_rows(H, stat, prior, tod0, dow, cond, panel_id, shock=False,
               noprior=False, ramp=False, long=None):
    """One window -> (HOR*nL, F) feature matrix and the column names.

    H     : dict from history_summary, vectors already in corridor order
    stat  : static dict (vf, lanes, cap, length, rank)
    prior : dict of (HOR, nL) or (nL,) prior arrays
    tod0  : time-of-day slot index of the forecast origin T
    """
    nL = len(stat["vf"])
    cols, blocks = [], []

    def add(name, v):
        cols.append(name)
        blocks.append(np.asarray(v, np.float32))

    step = np.repeat(np.arange(1, HOR + 1, dtype=np.float32), nL)
    add("step", step)
    add("tod", ((tod0 + np.repeat(np.arange(1, HOR + 1), nL)) % 288).astype(np.float32))
    add("dow", np.full(HOR * nL, dow, np.float32))
    add("weekend", np.full(HOR * nL, float(dow >= 5), np.float32))
    add("cond", np.full(HOR * nL, float(cond), np.float32))
    add("panel_id", np.full(HOR * nL, float(panel_id), np.float32))

    for name in ("vf", "lanes", "cap", "length"):
        add(name, np.tile(stat[name], HOR))
    add("rank_norm", np.tile(stat["rank"] / max(nL - 1, 1), HOR).astype(np.float32))
    add("cap_per_lane", np.tile(stat["cap"] / np.maximum(stat["lanes"], 1), HOR))

    for name in ("r_last", "f_last", "o_last", "age", "slope", "r_mean", "r_min",
                 "f_mean", "r_std", "r_last3", "q_last3", "r_recent",
                 "q_last", "q_any", "q_frac", "n_obs", "has"):
        add(name, np.tile(H[name], HOR))
    add("dist_thr", np.tile(H["r_last"] - 0.6, HOR))
    add("dist_thr_min", np.tile(H["r_min"] - 0.6, HOR))

    for k in (1, 2, 3, -1, -2, -3):
        tag = f"up{k}" if k > 0 else f"dn{-k}"
        add(f"r_last_{tag}", np.tile(_shift(H["r_last"], k), HOR))
        add(f"q_last_{tag}", np.tile(_shift(H["q_last"], k), HOR))
        add(f"f_last_{tag}", np.tile(_shift(H["f_last"], k), HOR))
    add("r_last_up_min", np.tile(np.nanmin(np.stack(
        [_shift(H["r_last"], k) for k in (1, 2, 3, 4, 5)]), 0), HOR))
    add("r_last_dn_min", np.tile(np.nanmin(np.stack(
        [_shift(H["r_last"], -k) for k in (1, 2, 3, 4, 5)]), 0), HOR))
    add("q_neigh_sum", np.tile(np.nansum(np.stack(
        [_shift(H["q_last"], k) for k in (-3, -2, -1, 1, 2, 3)]), 0), HOR))

    with np.errstate(invalid="ignore"):
        corr_r = np.nanmean(H["r_last"])
        corr_q = np.nanmean(H["q_last"])
        corr_f = np.nanmean(H["f_last"])
        corr_slope = np.nanmean(H["slope"])
        corr_min = np.nanmean(H["r_min"])
    for name, v in (("corr_r", corr_r), ("corr_q", corr_q), ("corr_f", corr_f),
                    ("corr_slope", corr_slope), ("corr_min", corr_min)):
        add(name, np.full(HOR * nL, 0.0 if not np.isfinite(v) else v, np.float32))

    if long is not None:
        for name, v in long.items():
            add(name, np.tile(v, HOR))
        for k in (1, -1, 2, -2):
            tag = f"up{k}" if k > 0 else f"dn{-k}"
            add(f"lq_frac_{tag}", np.tile(_shift(long["lq_frac"], k), HOR))
            add(f"lq_run_{tag}", np.tile(_shift(long["lq_run"], k), HOR))

    if shock:
        for name in qshock.NAMES:
            add(name, np.tile(H[name], HOR))
    if ramp:
        for name in qramp.NAMES:
            add(name, np.tile(H[name], HOR))

    # Priors carry 77 percent of the onset model's gain, which makes it a
    # climatology: it predicts what usually happens at this link and hour and
    # barely reads the current state. QNOPRIOR=1 drops them, to see whether the
    # physical precursors - flow against capacity, falling speed - can be learned
    # once they are not crowded out, and whether such a model is complementary.
    for pname, parr in ({} if noprior else prior).items():
        v = parr if parr.ndim == 2 else np.tile(parr, (HOR, 1))
        add(pname, v.reshape(-1))

    X = np.stack(blocks, 1)
    return X, cols
