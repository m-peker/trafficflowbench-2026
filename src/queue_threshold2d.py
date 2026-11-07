"""One threshold per window condition, instead of one for both.

task2_run applies a single P_STAR = 0.20 to every blanked horizon cell. But the
two scored conditions are different regimes: in queue_onset nothing is queued in
the history and a queue forms during the horizon, in queue_ongoing a queue is
already there and mostly persists. Prevalence and error structure differ, and
the single-threshold sweeps hinted the optima do too - ongoing kept preferring
higher values (0.25-0.50) than onset (0.20-0.25).

The window's condition is in window_index.csv, so it is known at prediction
time; using it costs one extra parameter, not a per-panel fit. This computes the
IoU surface over (p_onset, p_ongoing) on held-out train windows for every panel,
aggregates it the way the leaderboard does, and reports what a condition-aware
pair buys over the best single value.
"""
import pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from common import OUT, WORK, QUEUE_PANELS, FAMILY
from t1build import build
from qdata import load_panel, origins
from qfeat import HOR

EVAL_DAYS = 60
N_WIN = 300
GRID = np.arange(0.05, 0.71, 0.025)


def panel_curves(panel):
    """IoU per (condition, threshold) on held-out train windows."""
    D = load_panel(panel)
    z = np.load(WORK / "cache" / f"{panel}_train_masked.npz", allow_pickle=True)
    src = [str(x) for x in z["link_ids"]]
    o = [src.index(l) for l in D["link_ids"]]
    msp, mfl = z["speed"][:, o], z["flow"][:, o]
    mavail = np.isfinite(msp) & np.isfinite(mfl)
    thr = 0.6 * D["stat"]["vf"][None, :]
    nT, nL = msp.shape
    nd = nT // 288

    rng = np.random.default_rng(3)
    on, ong = origins(D)
    on = on[on // 288 >= nd - EVAL_DAYS]
    ong = ong[ong // 288 >= nd - EVAL_DAYS]
    on = rng.choice(on, min(N_WIN, len(on)), replace=False)
    ong = rng.choice(ong, min(N_WIN, len(ong)), replace=False)

    need = np.zeros((nT, nL), bool)
    for T in np.concatenate([on, ong]):
        need[T + 1: T + 1 + HOR] = True
    r = build(panel, "train", targets="blanked", target_mask=need, verbose=False)
    ti, tj = r["ti"], r["tj"]
    with open(OUT / "queue_cls" / f"{panel}.pkl", "rb") as fh:
        model = pickle.load(fh)["model"]
    prob = model.predict(r["X"], num_iteration=model.best_iteration)
    del r

    base = mavail & (msp <= thr)
    out = {}
    for cond, origs in (("onset", on), ("ongoing", ong)):
        vals = []
        for p_star in GRID:
            g = base.copy()
            g[ti, tj] = prob >= p_star
            v = []
            for T in origs:
                sl = slice(T + 1, T + 1 + HOR)
                e = D["elig"][sl]
                t = D["q"][sl] & e
                pr = g[sl] & e
                u = np.logical_or(pr, t).sum()
                v.append(1.0 if u == 0 else np.logical_and(pr, t).sum() / u)
            vals.append(float(np.mean(v)))
        out[cond] = np.array(vals)
    return out


def main():
    panels = sys.argv[1:] or QUEUE_PANELS
    t0 = time.time()
    curves = {}
    for p in panels:
        curves[p] = panel_curves(p)
        best_on = GRID[int(np.argmax(curves[p]["onset"]))]
        best_ong = GRID[int(np.argmax(curves[p]["ongoing"]))]
        print(f"{p:11s} onset best {curves[p]['onset'].max():.4f} @{best_on:.3f}   "
              f"ongoing best {curves[p]['ongoing'].max():.4f} @{best_ong:.3f}"
              f"   [{time.time()-t0:.0f}s]", flush=True)
    pickle.dump({"grid": GRID, "curves": curves}, open(OUT / "qthr2d.pkl", "wb"))

    fams = sorted({FAMILY[p] for p in curves})

    def agg(i_on, i_ong):
        """Leaderboard aggregation: panel-condition -> panel -> family -> mean."""
        per_panel = {p: 0.5 * (c["onset"][i_on] + c["ongoing"][i_ong])
                     for p, c in curves.items()}
        return float(np.mean([np.mean([v for p, v in per_panel.items()
                                       if FAMILY[p] == f]) for f in fams]))

    single = [(agg(i, i), GRID[i]) for i in range(len(GRID))]
    b_single = max(single)
    best2d, arg2d = -1, None
    for i in range(len(GRID)):
        for j in range(len(GRID)):
            s = agg(i, j)
            if s > best2d:
                best2d, arg2d = s, (GRID[i], GRID[j])
    cur = agg(int(np.argmin(np.abs(GRID - 0.20))), int(np.argmin(np.abs(GRID - 0.20))))

    print(f"\nS_queue with the shipped single p*=0.20 : {cur:.4f}")
    print(f"best single threshold  p*={b_single[1]:.3f}       : {b_single[0]:.4f}")
    print(f"best per-condition pair onset={arg2d[0]:.3f} ongoing={arg2d[1]:.3f}: {best2d:.4f}")
    print(f"\ncondition-aware gain over what we ship: {best2d-cur:+.4f} S_queue"
          f"  ->  {0.30*(best2d-cur):+.4f} composite")
    print(f"of which just retuning the single threshold: {b_single[0]-cur:+.4f}")


if __name__ == "__main__":
    main()
