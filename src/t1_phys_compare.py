"""Blend weight for net vs trees, scored on 0.35*S_state + 0.15*S_physics.

s36 (+0.00529 on the board for a 50/50 blend) showed the physics half carries
most of the gain: S_state moved ~+0.0006 of composite, so ~+0.0047 came through
S_LWR. That makes the blend weight worth choosing on the physics-aware metric
rather than on RMSE, which is what ens_search.Panel already computes - this just
injects the conv net as one more member, reading its fills at the same cells.

    python t1_phys_compare.py NAME [NAME2 ...]
"""
import nice  # noqa: F401
import os, sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
import ens_search as E
import t1_nn as NN
from common import OUT

# T1_EVAL_LAST=7 scores only the last 7 days of the eval block. a1 held out the
# last 30 train days and c1 only the last 7, so that window is the one both of
# them are honest on; anything wider flatters c1.
EVAL_LAST = int(os.environ.get("T1_EVAL_LAST", "0"))

W = {"speed": {"own": 0.51, "pooled": 0.12, "family": 0.21, "alt": 0.16},
     "flow": {"own": 0.47, "pooled": 0.25, "family": 0.17, "alt": 0.12},
     "density": {"own": 0.44, "pooled": 0.16, "family": 0.17, "alt": 0.23}}


def score_fields(P, v, q):
    """Panel.score()'s formulas, taking the filled speed/flow directly."""
    tj = P.tj
    o = P.ok
    if EVAL_LAST:
        o = o & (P.ti >= P.obs_sp.shape[0] - 288 * EVAL_LAST)
    S_v = 1 - float(np.sqrt(np.mean((v[o] - P.v_t[o]) ** 2))) / 25.0
    S_q = 1 - float(np.sqrt(np.mean((q[o] / P.lanes[tj][o] - P.q_t[o]) ** 2))) / 600.0
    S_state = 0.54 * S_v + 0.46 * S_q
    sp = P.obs_sp.astype(np.float64).copy()
    fl = P.obs_fl.astype(np.float64).copy()
    sp[P.ti, tj] = v
    fl[P.ti, tj] = q
    N = (fl / np.maximum(sp, 1.0)) * P.length[None, :]
    dN = N[1:] - N[:-1]
    m = np.isfinite(dN) & P.dNt_ok
    if EVAL_LAST:
        keep = np.zeros(m.shape[0], bool)
        keep[-288 * EVAL_LAST:] = True
        m = m & keep[:, None]
    e = np.abs(dN[m] - P.dNt[m]).sum() / (np.abs(P.dNt[m]).sum() + 1e-9)
    S_lwr = max(0.0, 1.0 - min(1.0, float(e)))
    # S_FD reads the whole grid, so with EVAL_LAST it must be cut to the same
    # rows: a partially filled grid otherwise scores its unfilled remainder,
    # and blending filler with real values invents (20 km/h, 588 vph) points
    # that no fundamental diagram allows.
    if EVAL_LAST:
        S_fd = P._fd(sp[-288 * EVAL_LAST:], fl[-288 * EVAL_LAST:])
    else:
        S_fd = P._fd(sp, fl)
    S_phys = (S_fd + 2 * S_lwr) / 3.0
    return dict(S_state=S_state, S_lwr=S_lwr, S_physics=S_phys,
                partial=0.35 * S_state + 0.15 * S_phys)


def main():
    names = sys.argv[1:] or ["a1"]
    nets = {n: NN.load_net(n) for n in names}
    panels = [E.Panel(p) for p in E.PANELS[:4]]
    fields = {}
    for P in panels:
        tj, ti = P.tj, P.ti
        v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[tj] * 1.05)
        k = P.mix(W, "density")
        q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v_t)
                      * P.lanes[tj], 5.0, P.cap[tj] * 1.15)
        nT_block = P.obs_sp.shape[0]
        per_net = {}
        for n, net in nets.items():
            G, sp_n, fl_n, _ = NN.predict_grid(net, P.panel, "train")
            lo = G["nT"] - nT_block
            per_net[n] = (sp_n[ti + lo, tj], fl_n[ti + lo, tj])
        fields[P.panel] = (v_t, q_t, per_net)
        print(f"  {P.panel} ready", flush=True)

    def agg(make):
        out = {}
        for P in panels:
            v_t, q_t, per_net = fields[P.panel]
            v, q = make(v_t, q_t, per_net)
            for k2, v2 in score_fields(P, v, q).items():
                out.setdefault(k2, []).append(v2)
        return {k2: float(np.mean(v2)) for k2, v2 in out.items()}

    rows = [("trees", lambda v_t, q_t, pn: (v_t, q_t))]
    for n in names:
        rows.append((f"net {n}", lambda v_t, q_t, pn, n=n: pn[n]))
        for w in (0.3, 0.5, 0.7):
            rows.append((f"{w:.1f} net {n} + {1 - w:.1f} trees",
                         lambda v_t, q_t, pn, w=w, n=n:
                         (w * pn[n][0] + (1 - w) * v_t, w * pn[n][1] + (1 - w) * q_t)))
    if len(names) > 1:
        for w in (0.5, 0.6, 0.7):
            rows.append((f"{w:.1f} mean(nets) + {1 - w:.1f} trees",
                         lambda v_t, q_t, pn, w=w:
                         (w * np.mean([pn[n][0] for n in names], 0) + (1 - w) * v_t,
                          w * np.mean([pn[n][1] for n in names], 0) + (1 - w) * q_t)))
    print(f"\n{'candidate':30s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}")
    for label, make in rows:
        a = agg(make)
        print(f"{label:30s} {a['S_state']:8.4f} {a['S_lwr']:7.4f} {a['partial']:8.5f}")


if __name__ == "__main__":
    main()
