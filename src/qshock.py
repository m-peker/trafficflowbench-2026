"""Space-time features for Task 2: where the queue front is, and how fast it moves.

The existing 62 features are all either one time slice across space - every
spatial term is built from `H["r_last"]` - or one link across time - `slope`,
`r_recent`, `r_mean`. Nothing combines the two. So the model can see that a link
is slow now and that its neighbour is slow now, but it cannot see the *diagonal*
in the space-time history that a propagating front draws, and a propagating
front is the entire mechanism of queue formation: the back of a queue moves
upstream at a characteristic speed, and where it will be in five to thirty
minutes is a question about that speed and the distance still to travel.

Three families, all computed from the (HIST, nL) speed-ratio grid the summary
already builds. Corridor order runs upstream to downstream, so a queue growing
upstream moves toward *lower* indices and reaches a link from its downstream
side.

  d_dn_q, d_up_q      links to the nearest queued cell downstream / upstream at
                      the last observed step. `r_last_dn_min` already says how
                      slow the downstream neighbourhood is; this says how far
                      away the slow part starts, which is the part that decides
                      whether it arrives inside the horizon.

  d_dn_q_slope        how that distance has been changing, links per step, from
                      a least-squares line over the history. Negative means the
                      front is approaching. `eta_dn` turns it into steps-to-
                      arrival, which is directly comparable to the horizon index.

  r_diag_up{k}        the speed ratio k steps back in time and k links
                      downstream - a probe along the characteristic of a front
                      moving upstream at one link per step. `r_diag_dn{k}` is
                      the mirror, for a recovery wave moving downstream. If a
                      front is really propagating at that speed these read like
                      the link's own future.

Everything is index-based rather than in kilometres. Link lengths vary, but
within a corridor they vary little, and an index distance needs no extra input
and cannot go wrong when a length is missing.

Off by default everywhere: `queue_model.pkl` was fitted on the 62 and would
break on a wider matrix. `build_rows(..., shock=True)` opts in.
"""
import numpy as np

MAXD = 12.0          # cap on "no queue found in this direction", in links
NDIAG = 3            # how many steps along the characteristic to probe


def _dist_to_queue(qq, downstream=True):
    """(HIST, nL) queued mask -> (HIST, nL) distance in links to the nearest
    queued cell in that direction, MAXD where there is none."""
    nH, nL = qq.shape
    out = np.full((nH, nL), MAXD, np.float32)
    rng = range(nL - 2, -1, -1) if downstream else range(1, nL)
    step = 1 if downstream else -1
    # one sweep: the distance at j is 0 if j is queued, else 1 + distance at j+step
    for t in range(nH):
        row = out[t]
        row[qq[t]] = 0.0
        for j in rng:
            nb = row[j + step]
            if nb + 1 < row[j]:
                row[j] = nb + 1
    return out


def _slope_over_time(a):
    """Least-squares slope per column of a (HIST, nL) array, ignoring NaNs."""
    nH, nL = a.shape
    t = np.arange(nH, dtype=np.float64)[:, None]
    m = np.isfinite(a)
    n = m.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(m, t, 0.0)
        y = np.where(m, np.nan_to_num(a), 0.0)
        sx, sy = x.sum(0), y.sum(0)
        sxx, sxy = (x * x).sum(0), (x * y).sum(0)
        den = n * sxx - sx * sx
        slope = np.where(den > 0, (n * sxy - sx * sy) / np.where(den == 0, 1, den),
                         np.nan)
    return np.where(n >= 3, slope, np.nan).astype(np.float32)


def _diag_probe(r, elig, k, downstream=True):
    """r at k steps back in time and k links along the corridor.

    Following the characteristic of a front travelling one link per step: for a
    front moving upstream, the cell that is now at link j was, k steps ago, k
    links downstream of it.
    """
    nH, nL = r.shape
    out = np.full(nL, np.nan, np.float32)
    t = nH - 1 - k
    if t < 0:
        return out
    src = np.arange(nL) + (k if downstream else -k)
    ok = (src >= 0) & (src < nL)
    v = np.where(elig[t], r[t], np.nan)
    out[ok] = v[src[ok]]
    return out


def features(r, qq, elig):
    """(HIST, nL) grids -> dict of per-link vectors. Names are stable."""
    nH, nL = r.shape
    d_dn = _dist_to_queue(qq, downstream=True)
    d_up = _dist_to_queue(qq, downstream=False)

    out = {
        "d_dn_q": d_dn[-1].copy(),
        "d_up_q": d_up[-1].copy(),
        "d_dn_q_slope": _slope_over_time(np.where(d_dn < MAXD, d_dn, np.nan)),
        "d_up_q_slope": _slope_over_time(np.where(d_up < MAXD, d_up, np.nan)),
        "d_dn_q_min": d_dn.min(0).astype(np.float32),
    }

    # steps until the downstream front reaches this link, if it is approaching
    with np.errstate(invalid="ignore", divide="ignore"):
        closing = -out["d_dn_q_slope"]
        eta = np.where(closing > 1e-3, out["d_dn_q"] / closing, np.inf)
    out["eta_dn"] = np.clip(np.where(np.isfinite(eta), eta, 99.0), 0.0, 99.0
                            ).astype(np.float32)

    # how much of the corridor is queued, and whether that is growing
    frac = qq.mean(1).astype(np.float32)
    out["corr_q_frac_last"] = np.full(nL, frac[-1], np.float32)
    out["corr_q_frac_slope"] = np.full(
        nL, float(_slope_over_time(frac[:, None])[0]), np.float32)

    # the most upstream queued link, and how fast that boundary is moving
    idx = np.arange(nL, dtype=np.float32)
    front = np.array([idx[qq[t]].min() if qq[t].any() else np.nan
                      for t in range(nH)], np.float32)
    out["corr_front_pos"] = np.full(nL, front[-1] if np.isfinite(front[-1])
                                    else np.nan, np.float32)
    out["corr_front_speed"] = np.full(
        nL, float(_slope_over_time(front[:, None])[0]), np.float32)
    # distance from this link to that boundary, signed: positive = downstream of me
    out["d_front"] = (out["corr_front_pos"] - idx).astype(np.float32)

    for k in range(1, NDIAG + 1):
        out[f"r_diag_up{k}"] = _diag_probe(r, elig, k, downstream=True)
        out[f"r_diag_dn{k}"] = _diag_probe(r, elig, k, downstream=False)

    return out


NAMES = (["d_dn_q", "d_up_q", "d_dn_q_slope", "d_up_q_slope", "d_dn_q_min",
          "eta_dn", "corr_q_frac_last", "corr_q_frac_slope", "corr_front_pos",
          "corr_front_speed", "d_front"]
         + [f"r_diag_up{k}" for k in range(1, NDIAG + 1)]
         + [f"r_diag_dn{k}" for k in range(1, NDIAG + 1)])
