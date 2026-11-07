"""Compare decision rules for the Task 2 queue model on held-out train windows."""
import sys, pickle, time
import numpy as np
import pandas as pd
from common import OUT, QUEUE_PANELS, FAMILY
from qfeat import HOR
from qdata import load_panel, origins, make_priors, window_xy
from qtrain import collect, VAL_DAYS

CACHE = OUT / "qtune_probs.pkl"


def gather(store, models):
    """Probabilities, truth and eligibility for every held-out window."""
    rows = []
    for panel, S in store.items():
        D, pri, pid = S["D"], S["pri"], S["pid"]
        for T, cond in S["val"]:
            X, _, y, m = window_xy(D, pri, T, cond, pid)
            p = models[cond].predict(X)
            rows.append(dict(panel=panel, T=int(T), cond=cond, p=p.astype(np.float32),
                             y=y.astype(bool), elig=m.astype(bool)))
        print(f"  gathered {panel}", flush=True)
    return rows


def iou(pred, y, elig):
    p = pred & elig
    t = y & elig
    u = np.logical_or(p, t).sum()
    return 1.0 if u == 0 else float(np.logical_and(p, t).sum() / u)


def expected_iou_set(p, floor=0.0):
    """Plug-in maximiser of E[IoU] under independence: take the top-k by probability."""
    o = np.argsort(-p)
    ps = p[o]
    cs = np.cumsum(ps)
    tot = cs[-1]
    k = np.arange(1, len(ps) + 1)
    score = cs / (k + tot - cs)
    best = int(np.argmax(score))
    sel = np.zeros(len(p), bool)
    if score[best] > 0:
        sel[o[: best + 1]] = True
    if floor > 0:
        sel &= p >= floor
    return sel


def aggregate(rows, key="iou"):
    df = pd.DataFrame([{"panel": r["panel"], "cond": r["cond"], "iou": r[key]} for r in rows])
    pc = df.groupby(["panel", "cond"]).iou.mean().reset_index()
    pm = pc.groupby("panel").iou.mean().reset_index()
    pm["family"] = pm.panel.map(FAMILY)
    return float(pm.groupby("family").iou.mean().mean()), pc


def main():
    with open(OUT / "queue_model.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    models, thr = bundle["models"], bundle["thr"]
    if CACHE.exists():
        rows = pickle.load(open(CACHE, "rb"))
        print(f"loaded {len(rows)} cached windows")
    else:
        store = collect(QUEUE_PANELS)
        rows = gather(store, models)
        pickle.dump(rows, open(CACHE, "wb"))

    print("saved thresholds:", thr)
    # 1) fixed threshold, tuned per condition
    for cond, name in ((0, "onset"), (1, "ongoing")):
        sub = [r for r in rows if r["cond"] == cond]
        grid = np.arange(0.05, 0.95, 0.025)
        best = (None, -1)
        for th in grid:
            for r in sub:
                r["iou"] = iou(r["p"] >= th, r["y"], r["elig"])
            s, _ = aggregate(sub)
            if s > best[1]:
                best = (th, s)
        print(f"{name:8s} fixed-threshold best thr={best[0]:.3f} IoU={best[1]:.4f}")
        thr[name] = float(best[0])

    # 2) per-window expected-IoU set
    for cond, name in ((0, "onset"), (1, "ongoing")):
        sub = [r for r in rows if r["cond"] == cond]
        for floor in (0.0, 0.05, 0.1, 0.2, 0.3):
            for r in sub:
                r["iou"] = iou(expected_iou_set(r["p"], floor), r["y"], r["elig"])
            s, _ = aggregate(sub)
            print(f"{name:8s} expected-IoU floor={floor:.2f} IoU={s:.4f}")

    # 3) combination: expected-IoU set intersected with the tuned threshold
    for cond, name in ((0, "onset"), (1, "ongoing")):
        sub = [r for r in rows if r["cond"] == cond]
        for r in sub:
            r["iou"] = iou(expected_iou_set(r["p"], thr[name] * 0.5), r["y"], r["elig"])
        s, _ = aggregate(sub)
        print(f"{name:8s} expected-IoU floor=thr/2 IoU={s:.4f}")

    # final: report the fixed-threshold total, which is what qpredict applies
    for r in rows:
        r["iou"] = iou(r["p"] >= (thr["onset"] if r["cond"] == 0 else thr["ongoing"]),
                       r["y"], r["elig"])
    total, pc = aggregate(rows)
    print("\npanel-condition:\n", pc.to_string())
    print(f"\nS_queue fixed-threshold = {total:.4f}")
    bundle["thr"] = thr
    pickle.dump(bundle, open(OUT / "queue_model.pkl", "wb"))


if __name__ == "__main__":
    main()
