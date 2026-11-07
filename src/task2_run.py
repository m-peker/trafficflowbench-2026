"""Task 2 queue flags: read where published, classify where blanked.

Every window's horizon timestamp is also in the released masked mainline layer,
so a cell that was not blanked is labelled from its published speed directly.
The blanked ones used to be labelled by thresholding the Task 1 speed
reconstruction, which is a regression's conditional mean standing in for a
classifier — weakest exactly in onset windows, where the speed is crossing the
threshold. A classifier fitted on the same features and thresholded at P_STAR
replaces that step.

Held-out train windows, panel IoU:

    D7_I10_E   0.9757 -> 0.9921
    D7_I10_W   0.9393 -> 0.9748
    D7_I210_E  0.9941 -> 0.9990

P_STAR is one global constant rather than a per-panel fit: tuning it per panel
bought 0.0003 more, which is not worth the extra degrees of freedom on a private
month with different incidents.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from common import ROOT, OUT, WORK, QUEUE_PANELS, links_table
from t1build import build

GRIDS = WORK / "grids"
CLS = OUT / "queue_cls"
HOR = 6
P_STAR = 0.20
SPLITS = ["validation", "private"]


def horizon_rows(base, di):
    """Window index -> the grid rows and timestamps of every horizon cell."""
    wi = pd.read_csv(base / "window_index.csv")
    out = []
    for w in wi.itertuples(index=False):
        T = pd.Timestamp(w.forecast_origin)
        rows, stamps = [], []
        for s in range(1, HOR + 1):
            t = T + pd.Timedelta(minutes=5 * s)
            d = t.strftime("%Y-%m-%d")
            if d in di:
                rows.append(di[d] * 288 + t.hour * 12 + t.minute // 5)
                stamps.append(t)
        if rows:
            out.append((w.window_id, rows, stamps))
    return out


def main():
    frames, diag = [], []
    for panel in QUEUE_PANELS:
        lk = links_table(panel).drop_duplicates("link_id").set_index("link_id")
        cls_path = CLS / f"{panel}.pkl"
        model = None
        if cls_path.exists():
            with open(cls_path, "rb") as fh:
                model = pickle.load(fh)["model"]
        for split in SPLITS:
            base = ROOT / "task2" / panel / split
            if not (base / "window_index.csv").exists():
                continue
            z = np.load(GRIDS / f"{panel}_{split}.npz", allow_pickle=True)
            speed, filled = z["speed"], z["filled"]
            link_ids = [str(x) for x in z["link_ids"]]
            days = [str(d) for d in z["days"]]
            di = {d: i for i, d in enumerate(days)}
            thr = np.array([0.6 * lk.free_speed_kmh.get(l, 105.0) for l in link_ids])

            windows = horizon_rows(base, di)
            wanted = sorted({r for _, rows, _ in windows for r in rows})
            flag = speed <= thr[None, :]          # published cells read directly

            n_cls = 0
            if model is not None:
                mask = np.zeros(speed.shape, bool)
                mask[wanted] = True
                mask &= filled                    # only the blanked ones
                if mask.any():
                    r = build(panel, split, targets="blanked", target_mask=mask,
                              verbose=False)
                    p = model.predict(r["X"], num_iteration=model.best_iteration)
                    flag[r["ti"], r["tj"]] = p >= P_STAR
                    n_cls = len(p)
                    del r

            for wid, rows, stamps in windows:
                frames.append(pd.DataFrame({
                    "window_id": wid,
                    "timestamp": np.repeat([s.strftime("%Y-%m-%dT%H:%M:%SZ")
                                            for s in stamps], len(link_ids)),
                    "link_id": np.tile(link_ids, len(rows)),
                    "queue_pred": flag[rows].astype(int).reshape(-1)}))
            share = float(filled[wanted].mean())
            diag.append(dict(panel=panel, split=split, windows=len(windows),
                             blanked_share=share, classified=n_cls))
            print(f"[{panel}/{split}] {len(windows)} windows, "
                  f"{share:.1%} of horizon cells blanked, {n_cls:,} classified"
                  f"{'' if model is not None else '  (NO CLASSIFIER - thresholding)'}",
                  flush=True)

    sub = pd.concat(frames, ignore_index=True)
    sub = sub.drop_duplicates(["window_id", "timestamp", "link_id"])
    sub.to_csv(OUT / "queue_submission.csv", index=False)
    pd.DataFrame(diag).to_csv(OUT / "queue_diag.csv", index=False)
    print(f"\nwrote {OUT / 'queue_submission.csv'}  {len(sub):,} rows, "
          f"positive rate {sub.queue_pred.mean():.4f}")


if __name__ == "__main__":
    main()
