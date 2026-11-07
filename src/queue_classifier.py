"""Predict the queue flag directly at blanked cells, instead of thresholding a
reconstructed speed.

Task 2 reads the published speed where the horizon cell was not blanked; the
error is entirely in the blanked ones, where we currently take the Task 1 speed
model's output and compare it to 0.60*free_speed. That is a regression's
conditional mean being used as a classifier, and it is weakest exactly where the
decision is hard - onset windows, where the speed is crossing the threshold.

A classifier fitted on the same 106 features models P(queued) directly, so cells
near the boundary get a calibrated probability rather than a point estimate that
happens to land on one side.

    python queue_classifier.py train           # fit one model per panel
    python queue_classifier.py eval D7_I10_E   # IoU against thresholding
"""
import gc, pickle, sys, time
from pathlib import Path
import numpy as np
import lightgbm as lgb

from common import OUT, QUEUE_PANELS, links_table
from t1build import build
from t1feat import load_grid

MODELS = OUT / "queue_cls"
NUM_THREADS = 12
SAMPLE = 1_200_000
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbosity=-1,
              num_threads=NUM_THREADS)


def thresholds(panel, link_ids):
    lk = links_table(panel).drop_duplicates("link_id").set_index("link_id")
    return np.array([0.6 * lk.free_speed_kmh.get(l, 105.0) for l in link_ids],
                    np.float32)


def train_panel(panel):
    """Target: was this blanked cell queued, per the unmasked layer."""
    r = build(panel, "train", truth="train", sample=SAMPLE, seed=1)
    G = load_grid(panel, "train", masked=True)
    thr = thresholds(panel, G["link_ids"])
    del G
    y = (r["y_speed"] <= thr[r["tj"]]).astype(np.int8)
    ok = np.isfinite(r["y_speed"])
    day = r["ti"] // 288
    nd = day.max() + 1
    tr, va = (day < nd - 45) & ok, (day >= nd - 45) & ok
    print(f"[{panel}] {tr.sum():,} train / {va.sum():,} valid cells, "
          f"queued {y[tr].mean():.4f}", flush=True)

    ds = lgb.Dataset(r["X"][tr], label=y[tr], feature_name=r["cols"])
    dv = lgb.Dataset(r["X"][va], label=y[va], feature_name=r["cols"], reference=ds)
    b = lgb.train(PARAMS, ds, 1500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(60, verbose=False)])

    p = b.predict(r["X"][va], num_iteration=b.best_iteration)
    # what thresholding the speed model would have said, for comparison
    from t1build import build as _b  # noqa: F401  (kept explicit for clarity)
    with open(Path(r"D:/BigData Competition/work/out") /
              f"t1_model_{panel}.pkl", "rb") as fh:
        sp_model = pickle.load(fh)["models"]["speed"]
    sp = sp_model.predict(r["X"][va], num_iteration=sp_model.best_iteration)
    base = (sp <= thr[r["tj"][va]]).astype(np.int8)
    truth = y[va].astype(bool)

    def f1(pred):
        tp = int((pred & truth).sum())
        return 2 * tp / max(int(pred.sum()) + int(truth.sum()), 1)

    best = max(((f1(p >= t), t) for t in np.arange(0.05, 0.95, 0.025)))
    print(f"[{panel}] cell-level F1  threshold-the-speed {f1(base.astype(bool)):.4f}"
          f"  ->  classifier {best[0]:.4f} at p>={best[1]:.3f}", flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / f"{panel}.pkl", "wb") as fh:
        pickle.dump({"model": b, "cols": r["cols"], "p_star": float(best[1])}, fh)
    del r, ds, dv
    gc.collect()
    return best


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        panels = sys.argv[2:] or QUEUE_PANELS
        t0 = time.time()
        for panel in panels:
            train_panel(panel)
            print(f"    [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
