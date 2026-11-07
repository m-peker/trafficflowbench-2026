"""Ramp features for Task 2: the demand that causes a queue, which it never sees.

Four ways of varying the model have now been measured - more data, a shallower
learner, a narrower feature set, and a different target formulation - and all
four produce models whose scores correlate about 0.91 with the original and whose
averages lose to it. They all read the same 62 features, so they all know the
same things. The ceiling is the information, not the estimator.

And there is a whole released channel Task 2 does not read. `qfeat` uses mainline
speed, flow and occupancy; `grep -c ramp qdata.py qfeat.py qtrain.py` returns
zero. Ramp flows are released for every split, and Task 1 already uses them for
its conservation terms.

That matters most exactly where Task 2 is weakest. Onset scores 0.79 against
ongoing's 0.90, and onset is the *formation* of a queue - which on a freeway is
usually a merge: on-ramp demand arriving faster than the mainline can absorb.
A model with no ramp channel cannot see the cause, only the symptom once the
mainline has already slowed.

The features are built around that mechanism rather than dumped in wholesale:

    on_last, off_last     ramp flow at the last observed step, per lane
    on_slope              how the on-ramp flow has been trending over the history
    merge_load            (mainline flow + on-ramp) / capacity - the loading that
                          actually decides whether a merge breaks down
    merge_load_max        its worst value across the history
    on_cum_up3            on-ramp inflow entering the three links upstream, which
                          is the demand about to arrive here
    ramp_miss             share of the history where the ramp was unobserved, so
                          the model can discount the rest when it is blind

Off by default; QRAMP=1 opts in, and the models that exist keep loading.
"""
import numpy as np

EPS = 1e-6


def load_split(panel, split, link_ids):
    """on/off/miss matrices for a split, in corridor order.

    Imported here rather than at module scope: t1build reaches back to qfeat for
    static_table, so a top-level import closes a cycle qfeat -> qramp -> t1build
    -> t1feat -> qfeat.
    """
    from t1build import ramp_matrices
    return ramp_matrices(panel, split, list(link_ids))


_HIST_CACHE = {}


def history_for(panel, split, link_ids, T, hist):
    """{on, off} over the `hist` steps ending just before timestamp T.

    The evaluation scripts read the mainline history out of
    `window_history.parquet`, which carries no ramp columns, so the ramp channel
    has to be indexed out of the split's own cache instead. That cache stores its
    day list, which is all that is needed to turn a timestamp into a row.
    """
    key = (panel, split)
    if key not in _HIST_CACHE:
        import numpy as _np
        from common import WORK
        z = _np.load(WORK / "cache" / f"{panel}_{split}_ramp.npz", allow_pickle=True)
        src = [str(x) for x in z["link_ids"]]
        order = [src.index(l) for l in link_ids]
        _HIST_CACHE[key] = (
            z["on"][:, order].astype(_np.float32),
            z["off"][:, order].astype(_np.float32),
            {str(d): i for i, d in enumerate(z["days"])})
    on, off, dix = _HIST_CACHE[key]
    i = dix.get(T.strftime("%Y-%m-%d"))
    if i is None:
        return None
    t0 = i * 288 + T.hour * 12 + T.minute // 5
    if t0 - hist < 0 or t0 > on.shape[0]:
        return None
    return {"on": on[t0 - hist:t0], "off": off[t0 - hist:t0]}


def _shift_sum(a, k):
    """Sum of a over the k links immediately upstream (lower index)."""
    out = np.zeros_like(a)
    for j in range(1, k + 1):
        out[j:] += a[:-j]
    return out


def features(on, off, fl, cap, lanes, pct_ok):
    """(HIST, nL) ramp and mainline history -> per-link vectors.

    `on`/`off` are ramp flows attached to each mainline link, `fl` the mainline
    flow, all over the same history window.
    """
    nH, nL = on.shape
    lanes = np.maximum(lanes, 1.0)
    obs = np.isfinite(on) & pct_ok
    n_obs = obs.sum(0)

    def last_of(a):
        out = np.full(nL, np.nan, np.float32)
        for j in range(nL):
            ii = np.flatnonzero(obs[:, j])
            if len(ii):
                out[j] = a[ii[-1], j]
        return out

    on_last, off_last = last_of(on), last_of(off)
    with np.errstate(invalid="ignore"):
        on_mean = np.where(n_obs > 0, np.nansum(np.where(obs, on, 0.0), 0)
                           / np.maximum(n_obs, 1), np.nan)

    # trend of the on-ramp flow: last third minus first third
    k = max(nH // 3, 1)
    with np.errstate(invalid="ignore"):
        early = np.nanmean(np.where(obs[:k], on[:k], np.nan), 0)
        late = np.nanmean(np.where(obs[-k:], on[-k:], np.nan), 0)
    on_slope = (late - early) / k

    # the merge loading: everything arriving divided by what the link can take
    total = fl + np.nan_to_num(on)
    load = total / np.maximum(cap[None, :], EPS)
    with np.errstate(invalid="ignore"):
        merge_load = np.nanmean(load, 0)
        merge_load_max = np.nanmax(np.where(np.isfinite(load), load, -np.inf), 0)
    merge_load_max = np.where(np.isfinite(merge_load_max), merge_load_max, np.nan)

    on_cum_up3 = _shift_sum(np.nan_to_num(on_last), 3)

    return {
        "on_last": (on_last / lanes).astype(np.float32),
        "off_last": (off_last / lanes).astype(np.float32),
        "on_mean": (on_mean / lanes).astype(np.float32),
        "on_slope": (on_slope / lanes).astype(np.float32),
        "merge_load": merge_load.astype(np.float32),
        "merge_load_max": merge_load_max.astype(np.float32),
        "on_cum_up3": (on_cum_up3 / lanes).astype(np.float32),
        "ramp_miss": (1.0 - n_obs / nH).astype(np.float32),
    }


NAMES = ["on_last", "off_last", "on_mean", "on_slope", "merge_load",
         "merge_load_max", "on_cum_up3", "ramp_miss"]
