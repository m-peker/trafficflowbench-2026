"""Build Task 2 training datasets from the train split."""
import numpy as np
import pandas as pd
from common import WORK, links_table
from qfeat import HIST, HOR, static_table, history_summary, build_rows

PANEL_ID = {}


def load_panel(panel, split="train"):
    z = np.load(WORK / "cache" / f"{panel}_{split}.npz", allow_pickle=True)
    link_ids = [str(x) for x in z["link_ids"]]
    stat = static_table(panel, link_ids)
    o = np.argsort(stat["order"])                     # corridor order permutation
    link_ids = [link_ids[i] for i in o]
    stat = {k: (v[o] if isinstance(v, np.ndarray) and v.shape == (len(o),) else v)
            for k, v in stat.items()}
    stat["rank"] = np.arange(len(o), dtype=np.float32)
    sp = z["speed"][:, o]; fl = z["flow"][:, o]; oc = z["occ"][:, o]; pct = z["pct"][:, o]
    elig = (pct >= 75) & np.isfinite(sp)
    q = elig & (sp <= 0.6 * stat["vf"][None, :])
    days = [str(d) for d in z["days"]]
    dow = np.repeat(pd.to_datetime(days).dayofweek.to_numpy(), 288)
    return dict(panel=panel, speed=sp, flow=fl, occ=oc, pct=pct, elig=elig, q=q,
                link_ids=link_ids, stat=stat, days=days, dow=dow,
                nT=sp.shape[0], nL=sp.shape[1])


import os

# An onset window requires the whole corridor to be queue-free for the full
# 60-minute history, which is why the pool is only 600-1400 windows per panel -
# and the data-volume sweep showed onset was the half that was starved, then hit
# that ceiling. QONSET_RELAX=k allows k queued timestamps in the history, which
# buys "nearly onset" windows: a corridor that is almost clear when a queue
# starts. It trades pool size against matching the evaluation distribution,
# which the calibrated harness will price.
ONSET_RELAX = int(os.environ.get("QONSET_RELAX", "0"))
# "flag" trains a classifier on the queue label; "speed" regresses the
# speed ratio and lets the threshold recover the label.
TARGET = os.environ.get("QTARGET", "flag")
# The organizers build every onset window with the queue first appearing at the
# last horizon step - 39 of 39 on train, every panel. Our own onset rule accepted
# a queue at any of the six steps, so 83.5 percent of the onset training pool was
# a different and easier problem (5-25 minutes ahead instead of 30). QONSET_STEP6=1
# keeps only the windows built the way the scored ones are.
ONSET_STEP6 = os.environ.get("QONSET_STEP6", "0") == "1"
# QONG_CONTRACT=1 filters ongoing origins by the selector's published safeguards
ONG_CONTRACT = os.environ.get("QONG_CONTRACT", "0") == "1"
# Task 2 has never read the ramp channel; QRAMP=1 turns it on.
RAMP = os.environ.get("QRAMP", "0") == "1"
_RAMP_CACHE = {}


def ramp_for(panel, split, link_ids):
    key = (panel, split)
    if key not in _RAMP_CACHE:
        from qramp import load_split
        _RAMP_CACHE[key] = load_split(panel, split, link_ids)
    return _RAMP_CACHE[key]


def origins(D, day_ok=None):
    """Onset and ongoing forecast-origin step indices."""
    q, nT = D["q"], D["nT"]
    T = np.arange(HIST, nT - HOR - 1)
    if day_ok is not None:
        T = T[day_ok[T // 288]]
    anyq = q.any(1).astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(anyq)])
    hist_cnt = cs[T] - cs[T - HIST]
    hor_cnt = cs[T + HOR + 1] - cs[T + 1]
    at_T = anyq[T]
    if ONSET_STEP6:
        early = cs[T + HOR] - cs[T + 1]          # steps 1..HOR-1
        last = anyq[T + HOR]                     # step HOR
        onset = T[(hist_cnt <= ONSET_RELAX) & (at_T == 0) & (early == 0) & (last == 1)]
    else:
        onset = T[(hist_cnt <= ONSET_RELAX) & (hor_cnt > 0) & (at_T == 0)]
    ongoing = T[at_T == 1]
    if ONG_CONTRACT:
        ongoing = contract_ongoing(D, ongoing)
    return onset, ongoing


def persistence_iou(D, T):
    """IoU of carrying the last history frame across the horizon, per origin."""
    q, e = D["q"], D["elig"]
    out = np.empty(len(T))
    for i, t in enumerate(T):
        tr = q[t + 1:t + 1 + HOR] & e[t + 1:t + 1 + HOR]
        pr = q[t][None, :] & e[t + 1:t + 1 + HOR]
        u = (pr | tr).sum()
        out[i] = 1.0 if u == 0 else (pr & tr).sum() / u
    return out


def contract_ongoing(D, T):
    """Keep only ongoing origins the organizers' selector could have drawn.

    synthetic_release_v1.json: every scored window has a queue in the horizon
    and maximum_persistence_iou 0.9. On the 40 organizer train ongoing windows
    both hold 40/40 (max persistence IoU 0.862), while 8.8 percent of the raw
    at_T pool is above 0.9 and 1.5 percent has an empty horizon - easy windows
    the board never asks about.
    """
    anyq = D["q"].any(1)
    T = T[np.array([anyq[t + 1:t + 1 + HOR].any() for t in T], bool)]
    return T[persistence_iou(D, T) <= 0.9]


def make_priors(D, day_ok=None):
    """Historical queue-rate priors, optionally restricted to a set of days."""
    q, elig, nT, nL = D["q"], D["elig"], D["nT"], D["nL"]
    if day_ok is None:
        day_ok = np.ones(nT // 288, bool)
    step_ok = np.repeat(day_ok, 288)
    qq, ee = q[step_ok], elig[step_ok]
    tod = (np.arange(nT) % 288)[step_ok]
    dw = D["dow"][step_ok]

    rate_link = qq.sum(0) / np.maximum(ee.sum(0), 1)
    num = np.zeros((288, nL)); den = np.zeros((288, nL))
    np.add.at(num, tod, qq); np.add.at(den, tod, ee)
    # circular smoothing over +/- 3 slots
    k = np.ones(7) / 7
    sm_n = np.apply_along_axis(lambda c: np.convolve(np.r_[c[-3:], c, c[:3]], k, "valid"), 0, num)
    sm_d = np.apply_along_axis(lambda c: np.convolve(np.r_[c[-3:], c, c[:3]], k, "valid"), 0, den)
    rate_tod = sm_n / np.maximum(sm_d, 1e-6)

    wk = (dw >= 5).astype(int)
    num2 = np.zeros((2, 288, nL)); den2 = np.zeros((2, 288, nL))
    np.add.at(num2, (wk, tod), qq); np.add.at(den2, (wk, tod), ee)
    sm2n = np.stack([np.apply_along_axis(
        lambda c: np.convolve(np.r_[c[-5:], c, c[:5]], np.ones(11) / 11, "valid"), 0, num2[i])
        for i in range(2)])
    sm2d = np.stack([np.apply_along_axis(
        lambda c: np.convolve(np.r_[c[-5:], c, c[:5]], np.ones(11) / 11, "valid"), 0, den2[i])
        for i in range(2)])
    rate_tod_wk = sm2n / np.maximum(sm2d, 1e-6)

    on, ong = origins(D, day_ok)
    P_on = np.zeros((HOR, nL)); P_ong = np.zeros((HOR, nL))
    for T in on:
        P_on += (q[T + 1:T + 1 + HOR] & elig[T + 1:T + 1 + HOR])
    P_on /= max(len(on), 1)
    sub = ong if len(ong) <= 4000 else np.random.default_rng(0).choice(ong, 4000, replace=False)
    for T in sub:
        P_ong += (q[T + 1:T + 1 + HOR] & elig[T + 1:T + 1 + HOR])
    P_ong /= max(len(sub), 1)
    return dict(rate_link=rate_link.astype(np.float32),
                rate_tod=rate_tod.astype(np.float32),
                rate_tod_wk=rate_tod_wk.astype(np.float32),
                P_on=P_on.astype(np.float32), P_ong=P_ong.astype(np.float32))


# ------------------------------------------------------------ rolling priors
# QROLL=N: queue-rate priors from the N days before the window's day, computed on
# the Task 1 MASKED layer of the window's own split (for private, the validation
# month precedes it; all of it is stamped before T, so the online rule allows it).
# Train-derived priors misled on the separately generated scored months (s104);
# these carry the scored month's own queue pattern.
QROLL = int(os.environ.get("QROLL", "0"))
_MP, _RP = {}, {}


def masked_panel(panel, split):
    if (panel, split) in _MP:
        return _MP[(panel, split)]
    from t1feat import load_grid
    parts = [split] if split != "private" else ["validation", "private"]
    Gs = [load_grid(panel, sp, masked=True) for sp in parts]
    vf = Gs[0]["stat"]["vf"][None, :]
    sp_ = np.concatenate([g["speed"] for g in Gs]); pc = np.concatenate([g["pct"] for g in Gs])
    elig = (pc >= 75) & np.isfinite(sp_)
    D = dict(q=elig & (sp_ <= 0.6 * vf), elig=elig, nT=sp_.shape[0], nL=sp_.shape[1],
             dow=np.concatenate([g["dow"] for g in Gs]),
             offset_days=0 if split != "private" else Gs[0]["nT"] // 288)
    _MP[(panel, split)] = D
    return D


def rolling_priors(panel, split, day):
    """Priors from the QROLL days before `day` (day index within `split`)."""
    key = (panel, split, day)
    if key not in _RP:
        D = masked_panel(panel, split)
        d = day + D["offset_days"]; nd = D["nT"] // 288
        ok = np.zeros(nd, bool); ok[max(0, d - QROLL):d] = True
        if not ok.any() and split != "train":
            # first day of the validation month: no earlier day of this split, so use the
            # last QROLL days of the train split's masked layer (they precede it). Never the
            # window's own day - that would read rows stamped after T.
            Dt = masked_panel(panel, "train"); ndt = Dt["nT"] // 288
            okt = np.zeros(ndt, bool); okt[ndt - QROLL:] = True
            _RP[key] = make_priors(Dt, okt)
            return _RP[key]
        if not ok.any():
            ok[:max(d, 1)] = True      # train split only (training windows, not scored)
        _RP[key] = make_priors(D, ok)
    return _RP[key]


def window_prior(pri, T, cond):
    tods = (np.arange(T + 1, T + 1 + HOR) % 288)
    return dict(pri_link=np.tile(pri["rate_link"], (HOR, 1)),
                pri_tod=pri["rate_tod"][tods],
                pri_on=pri["P_on"],
                pri_ong=pri["P_ong"],
                pri_cond=pri["P_on"] if cond == 0 else pri["P_ong"])


QT_T = os.environ.get("QT_T", "0") == "1"
QLONG = os.environ.get("QLONG", "0") == "1"
_MS = {}


def masked_speed(panel, split):
    """Task 1 masked-layer speeds of a split, corridor order (QLONG context)."""
    if (panel, split) not in _MS:
        from t1feat import load_grid
        _MS[(panel, split)] = load_grid(panel, split, masked=True)["speed"]
    return _MS[(panel, split)]
QT_HIDE = float(os.environ.get("QT_HIDE", "0.35"))


def window_xy(D, pri, T, cond, panel_id, with_y=True, shock=False,
              noprior=False, ramp=False):
    h = slice(T - HIST, T)
    sp_, fl_, oc_, pc_ = D["speed"], D["flow"], D["occ"], D["pct"]
    if QT_T:
        # the organizers allow the row AT the origin T (released in the Task 1 masked
        # layer, so ~35% of its cells are blank): history = T-55..T, T row thinned
        h = slice(T - HIST + 1, T + 1)
        hide = np.random.default_rng(int(T) * 7919 + int(panel_id)).random(D["nL"]) < QT_HIDE
        if D.get("split", "train") != "train":
            hide[:] = False          # masked-layer panels: the T row is already the released one
        sp_ = D["speed"][h].copy(); fl_ = D["flow"][h].copy(); oc_ = D["occ"][h].copy(); pc_ = D["pct"][h].copy()
        sp_[-1, hide] = np.nan; fl_[-1, hide] = np.nan; oc_[-1, hide] = np.nan; pc_[-1, hide] = 0
        hh = slice(None)
    else:
        hh = h
    R = None
    if ramp:
        M = ramp_for(D["panel"], "train", D["link_ids"])
        R = {k: M[k][h] for k in ("on", "off")}
    H = history_summary(sp_[hh], fl_[hh], oc_[hh], pc_[hh],
                        D["stat"]["vf"], D["stat"]["cap"], shock=shock,
                        ramp=R, lanes=D["stat"]["lanes"])
    if QROLL:
        pri = rolling_priors(D["panel"], D.get("split", "train"), T // 288)
    LF = None
    if QLONG:
        from qfeat import long_feats, LONG
        Ms = masked_speed(D["panel"], D.get("split", "train"))
        assert Ms.shape == D["speed"].shape
        LF = long_feats(Ms[T - LONG:T + 1], D["stat"]["vf"])
    X, cols = build_rows(H, D["stat"], window_prior(pri, T, cond),
                         T % 288, int(D["dow"][T]), cond, panel_id, shock=shock,
                         noprior=noprior, ramp=ramp, long=LF)
    if not with_y:
        return X, cols, None, None
    sl = slice(T + 1, T + 1 + HOR)
    m = D["elig"][sl].reshape(-1)
    if TARGET == "speed":
        # The queue flag is `speed <= 0.6 * vf`, so the flag throws away
        # everything about how far a cell is from that line: 0.61 and 0.95 of
        # free-flow are both "not queued" and one of them is about to be. The
        # ratio keeps it, and thresholding the prediction recovers the flag.
        r = D["speed"][sl] / np.maximum(D["stat"]["vf"][None, :], 1.0)
        y = np.where(D["elig"][sl], r, np.nan).reshape(-1).astype(np.float32)
        m = m & np.isfinite(y)
        y = np.nan_to_num(y, nan=1.0)
    else:
        y = (D["q"][sl] & D["elig"][sl]).reshape(-1).astype(np.int8)
    return X, cols, y, m


def load_masked_split(panel, split):
    """A scored month's Task 1 masked layer as a training panel (QADD_SPLIT): labels
    exist only on visible cells, which is what elig marks."""
    from t1feat import load_grid
    G = load_grid(panel, split, masked=True)
    sp = G["speed"]; pct = np.nan_to_num(G["pct"])
    elig = (pct >= 75) & np.isfinite(sp)
    q = elig & (sp <= 0.6 * G["stat"]["vf"][None, :])
    return dict(panel=panel, split=split, speed=sp, flow=G["flow"], occ=G["occ"], pct=pct.astype(np.int16),
                elig=elig, q=q, link_ids=G["link_ids"], stat=G["stat"], days=G["days"],
                dow=G["dow"].astype(int), nT=sp.shape[0], nL=sp.shape[1])
