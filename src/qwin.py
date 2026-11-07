"""Window generation + IoU evaluation utilities for Task 2."""
import numpy as np
import pandas as pd
from common import WORK, links_table

HIST = 12          # observations in [T-60, T)
HOR = 6            # T+5 .. T+30


def load_panel(panel, split="train"):
    z = np.load(WORK / "cache" / f"{panel}_{split}.npz", allow_pickle=True)
    sp = z["speed"]; fl = z["flow"]; oc = z["occ"]; pct = z["pct"]
    link_ids = [str(x) for x in z["link_ids"]]
    lk = links_table(panel).drop_duplicates("link_id").set_index("link_id")
    vf = np.array([lk.free_speed_kmh.get(l, 105.0) for l in link_ids], np.float32)
    lanes = np.array([lk.lanes.get(l, 4.0) for l in link_ids], np.float32)
    length = np.array([lk.length_km.get(l, 1.0) for l in link_ids], np.float32)
    cap = np.array([lk.capacity_vph.get(l, 7000.0) for l in link_ids], np.float32)
    elig = (pct >= 75) & np.isfinite(sp)
    q = elig & (sp <= 0.6 * vf[None, :])
    return dict(panel=panel, speed=sp, flow=fl, occ=oc, pct=pct, elig=elig, q=q,
                link_ids=link_ids, vf=vf, lanes=lanes, length=length, cap=cap,
                t0=pd.Timestamp(int(z["t0"][0]), tz="UTC"), nT=sp.shape[0], nL=sp.shape[1])


def enumerate_windows(D, rng=None, per_cond=400, seed=0):
    """Return (onset_origins, ongoing_origins) as step indices."""
    q, nT = D["q"], D["nT"]
    lo, hi = HIST, nT - HOR - 1
    T = np.arange(lo, hi)
    # cumulative any-queue over rolling history / horizon via convolution
    anyq = q.any(1).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(anyq)])
    hist_cnt = cs[T] - cs[T - HIST]                # queued steps in [T-12, T)
    hor_cnt = cs[T + HOR + 1] - cs[T + 1]          # queued steps in [T+1, T+6]
    at_T = anyq[T]
    onset = T[(hist_cnt == 0) & (hor_cnt > 0) & (at_T == 0)]
    ongoing = T[at_T == 1]
    rng = rng or np.random.default_rng(seed)
    if per_cond and len(onset) > per_cond:
        onset = np.sort(rng.choice(onset, per_cond, replace=False))
    if per_cond and len(ongoing) > per_cond:
        ongoing = np.sort(rng.choice(ongoing, per_cond, replace=False))
    return onset, ongoing


def window_iou(D, T, pred):
    """pred: (HOR, nL) bool. Scored on eligible horizon cells only."""
    sl = slice(T + 1, T + 1 + HOR)
    e = D["elig"][sl]
    truth = D["q"][sl] & e
    p = pred & e
    inter = np.logical_and(p, truth).sum()
    union = np.logical_or(p, truth).sum()
    return 1.0 if union == 0 else inter / union
