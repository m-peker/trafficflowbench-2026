"""The shipped Task 1 answer on the eval block, built once and cached.

t1_sharpen.py, t1_density.py and t1_interp.py all post-process the same fields -
0.85 * mean(nets) + 0.15 * trees on the last T1_EVAL_LAST days - and rebuilding
them costs a predict_grid pass per net per panel. This caches them under
out/fills_<key>.npz so only the first sweep pays.

The cache key covers everything that changes the fields: the net list, the blend
weight, the panels, the eval window and the inference convention.
"""
import os
import numpy as np

from common import OUT


def _key(names, weight, panels, eval_last):
    tod = os.environ.get("T1_TOD", "abs")
    tt = os.environ.get("T1_TILE_T", "256")
    tl = os.environ.get("T1_TILE_L", "0")
    return (f"{'-'.join(names)}_w{weight}_{len(panels)}p_e{eval_last}"
            f"_{tod}{tt}x{tl}")


def build(names, weight, panels, eval_last, W, E, NN, t1feat):
    """{panel: (v, q)} for the filled cells, from cache when it is there."""
    f = OUT / f"fills_{_key(names, weight, panels, eval_last)}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        print(f"  fills from cache: {f.name}", flush=True)
        return {p.panel: (z[f"{p.panel}_v"], z[f"{p.panel}_q"]) for p in panels}
    nets = {n: NN.load_net(n) for n in names}
    out, flat = {}, {}
    for P in panels:
        tj, ti = P.tj, P.ti
        v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[tj] * 1.05)
        k = P.mix(W, "density")
        q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v_t)
                      * P.lanes[tj], 5.0, P.cap[tj] * 1.15)
        nTb = P.obs_sp.shape[0]
        nT_full = t1feat.load_grid(P.panel, "train", masked=True)["nT"]
        lo = max(0, nT_full - 288 * eval_last - 256)
        vs, qs = [], []
        for n in names:
            G, sp, fl, _ = NN.predict_grid(nets[n], P.panel, "train", t_lo=lo, t_hi=nT_full)
            off = G["nT"] - nTb
            vs.append(sp[ti + off, tj]); qs.append(fl[ti + off, tj])
        v = weight * np.mean(vs, 0) + (1 - weight) * v_t
        q = weight * np.mean(qs, 0) + (1 - weight) * q_t
        out[P.panel] = (v, q)
        flat[f"{P.panel}_v"] = v; flat[f"{P.panel}_q"] = q
        print(f"  {P.panel} ready", flush=True)
    np.savez_compressed(f, **flat)
    return out


def trees(P, W, E):
    """The tree ensemble's own speed, flow and density on the same cells."""
    tj = P.tj
    v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[tj] * 1.05)
    k = P.mix(W, "density")
    q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v_t)
                  * P.lanes[tj], 5.0, P.cap[tj] * 1.15)
    return v_t, q_t, k
