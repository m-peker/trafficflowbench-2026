"""Does the queue classifier actually move window IoU?

Cell-level F1 is not the score. Task 2 is scored by space-time IoU per window,
averaged per panel and condition, so an improvement only counts if it lands on
cells that are inside a scored window's horizon. This rebuilds both grids on
held-out train windows and compares them the way the leaderboard would.

  read     published cells -> the observation itself
  baseline blanked cells   -> Task 1 speed model, thresholded at 0.60*v_f
  new      blanked cells   -> queue classifier probability, thresholded at p*
"""
import pickle, sys, time
from pathlib import Path
import numpy as np

from common import WORK, links_table
from t1build import build
from qdata import load_panel, origins
from qfeat import HOR

FROZEN = Path(r"D:/BigData Competition/work/out")
EVAL_DAYS = 60
N_WIN = 300


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
    t0 = time.time()
    D = load_panel(panel)                       # unmasked truth, corridor order
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
    r = build(panel, "train", targets="blanked", target_mask=need)
    ti, tj = r["ti"], r["tj"]
    print(f"{panel}: {len(on)}+{len(ong)} windows, {len(ti):,} blanked horizon cells "
          f"[{time.time()-t0:.0f}s]", flush=True)

    with open(FROZEN / f"t1_model_{panel}.pkl", "rb") as fh:
        sp_model = pickle.load(fh)["models"]["speed"]
    sp_pred = sp_model.predict(r["X"], num_iteration=sp_model.best_iteration)
    base_q = sp_pred <= thr[0, tj]

    with open(WORK / "out" / "queue_cls" / f"{panel}.pkl", "rb") as fh:
        cls = pickle.load(fh)
    prob = cls["model"].predict(r["X"], num_iteration=cls["model"].best_iteration)
    del r

    grids = {}
    for name, flag in (("threshold the speed", base_q), ("classifier", None)):
        g = mavail & (msp <= thr)               # published cells, read directly
        if flag is not None:
            g[ti, tj] = flag
        grids[name] = g
    prob_grid = np.zeros((nT, nL), np.float32)
    prob_grid[ti, tj] = prob

    def iou(g, origins_):
        v = []
        for T in origins_:
            sl = slice(T + 1, T + 1 + HOR)
            e = D["elig"][sl]
            t = D["q"][sl] & e
            p = g[sl] & e
            u = np.logical_or(p, t).sum()
            v.append(1.0 if u == 0 else np.logical_and(p, t).sum() / u)
        return float(np.mean(v))

    print(f"\n{'variant':26s} {'onset':>8s} {'ongoing':>8s} {'panel':>8s}")
    a, b = iou(grids["threshold the speed"], on), iou(grids["threshold the speed"], ong)
    print(f"{'threshold the speed':26s} {a:8.4f} {b:8.4f} {(a+b)/2:8.4f}")

    best = None
    for p_star in (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.35, 0.50):
        g = mavail & (msp <= thr)
        g[ti, tj] = prob >= p_star
        c, d = iou(g, on), iou(g, ong)
        tag = f"classifier p>={p_star:.2f}"
        print(f"{tag:26s} {c:8.4f} {d:8.4f} {(c+d)/2:8.4f}")
        if best is None or (c + d) / 2 > best[0]:
            best = ((c + d) / 2, p_star)
    print(f"\nbest {best[0]:.4f} at p>={best[1]:.2f}  "
          f"(baseline {(a+b)/2:.4f}, {best[0]-(a+b)/2:+.4f})")
    print(f"-> S_queue would move that much; on the composite "
          f"{0.30*(best[0]-(a+b)/2):+.4f}")


if __name__ == "__main__":
    main()
