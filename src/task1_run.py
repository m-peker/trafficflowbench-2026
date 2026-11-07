"""Task 1: per-panel gradient-boosted reconstruction of speed and flow.

Writes the Task 1 submission rows and, for every scored split, a full
(time x link) reconstructed grid that Tasks 2 and 3 reuse.

Restart-safe: panels already present in out/state_submission.csv for both
splits are skipped and new rows are appended.
"""
import sys, gc, time, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import nice
from common import ROOT, OUT, WORK, PANELS
from t1build import build
from t1feat import load_grid

GRIDS = WORK / "grids"
GRIDS.mkdir(exist_ok=True)
STATE = OUT / "state_submission.csv"
# leave a couple of cores free: GBDT tree building is memory-bandwidth bound,
# so 20 hyperthreads on 14 physical cores is slower than 12 and locks up the UI
NUM_THREADS = nice.THREADS
# 2.5M rather than 900k: on D7_I10_E it takes speed RMSE 1.397 -> 1.356 and flow
# 30.02 -> 29.70. Small for S_state, but S_LWR is driven by the error in density
# q/v, so it compounds there.
SAMPLE = 2_500_000
# The submitted flow is a half-and-half blend of the direct flow model and
# density*speed. Fitting density directly cuts its error 10 percent (0.0598 ->
# 0.0538) because occupancy at the neighbouring cells is a density proxy, but
# taking flow entirely from it costs 19 percent on flow RMSE. The blend keeps
# 97 percent of the density gain for 4 percent of the flow cost - and density,
# not flow, is what the conservation residual sees.
FLOW_BLEND = 0.5
PRED_CHUNK = 400_000
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=100, feature_fraction=0.75, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1, num_threads=NUM_THREADS)


def peak_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")


def completed_panels():
    if not STATE.exists():
        return set()
    expect = {}
    for panel in PANELS:
        for split in ("validation", "private"):
            f = ROOT / "task1" / panel / split / "sample_submission_state.csv"
            expect[(panel, split)] = sum(1 for _ in open(f)) - 1
    have = {}
    for ch in pd.read_csv(STATE, usecols=["panel", "timestamp"],
                          chunksize=2_000_000, dtype=str):
        split = ch.timestamp.str[:7].map({"2031-03": "validation",
                                          "2031-04": "private"})
        for k, n in ch.groupby([ch.panel, split]).size().items():
            have[k] = have.get(k, 0) + int(n)
    return {p for p in PANELS
            if all(have.get((p, s), 0) == expect[(p, s)]
                   for s in ("validation", "private"))}


def train_panel(panel):
    r = build(panel, "train", truth="train", sample=SAMPLE, seed=1)
    X, cols = r["X"], r["cols"]
    day = r["ti"] // 288
    nd = day.max() + 1
    tr, va = day < nd - 45, day >= nd - 45
    y_dens = r["y_flowpl"] / np.maximum(r["y_speed"], 1.0)
    models, scores = {}, {}
    for nm, y, sc in (("speed", r["y_speed"], 25.0), ("flow", r["y_flowpl"], 600.0),
                      ("density", y_dens, None)):
        m = np.isfinite(y)
        ds = lgb.Dataset(X[tr & m], label=y[tr & m], feature_name=cols)
        dv = lgb.Dataset(X[va & m], label=y[va & m], feature_name=cols, reference=ds)
        b = lgb.train(PARAMS, ds, 2500, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        p = b.predict(X[va & m], num_iteration=b.best_iteration)
        e = float(np.sqrt(np.mean((p - y[va & m]) ** 2)))
        scores[nm] = max(0.0, 1 - e / sc) if sc else e
        # keep only the trees that are actually used
        models[nm] = lgb.Booster(model_str=b.model_to_string(
            num_iteration=b.best_iteration))
        models[nm].best_iteration = b.best_iteration
        del ds, dv, p, b
        gc.collect()
        tag = f"S={scores[nm]:.4f}" if sc else f"RMSE={e:.4f}"
        print(f"  [{panel}] {nm}: RMSE={e:.4f} {tag}  rss={peak_mb():.0f}MB",
              flush=True)
    print(f"  [{panel}] S_state(holdout) = "
          f"{0.54 * scores['speed'] + 0.46 * scores['flow']:.4f}", flush=True)
    del r, X, tr, va, day, y_dens
    gc.collect()
    return models, cols, scores


def predict_chunked(model, X):
    """Chunked prediction with an explicit thread cap.

    The cap is the point. A booster restored from a pickle carries no
    num_threads, so without this every predict() call opens one thread per
    logical core no matter what PARAMS said at training time.
    """
    out = np.empty(len(X), np.float64)
    it = model.best_iteration
    kw = nice.predict_kwargs()
    for a in range(0, len(X), PRED_CHUNK):
        b = min(a + PRED_CHUNK, len(X))
        out[a:b] = model.predict(X[a:b], num_iteration=it, **kw)
    return out


def reconstruct(panel, split, models):
    """Full grid: published observations where present, model fit at blanked cells."""
    G = load_grid(panel, split, masked=True)
    stat = G["stat"]
    sp_grid = G["speed"].copy()
    fl_grid = G["flow"].copy()
    pct = G["pct"]
    filled = ~np.isfinite(G["speed"])
    link_ids = list(G["link_ids"])
    days = np.array(G["days"])
    del G
    gc.collect()

    r = build(panel, split, targets="blanked")
    ti, tj = r["ti"], r["tj"]
    sp = predict_chunked(models["speed"], r["X"])
    fl_pl = predict_chunked(models["flow"], r["X"])
    k_pl = predict_chunked(models["density"], r["X"])
    del r["X"], r
    gc.collect()

    sp = np.clip(sp, 5.0, stat["vf"][tj] * 1.05)
    fl_pl = (1 - FLOW_BLEND) * fl_pl + FLOW_BLEND * k_pl * sp
    flow = np.clip(fl_pl * stat["lanes"][tj], 5.0, stat["cap"][tj] * 1.15)
    del k_pl
    sp_grid[ti, tj] = sp
    fl_grid[ti, tj] = flow
    np.savez_compressed(GRIDS / f"{panel}_{split}.npz", speed=sp_grid, flow=fl_grid,
                        pct=pct, filled=filled,
                        link_ids=np.array(link_ids, dtype=object),
                        days=np.array(days, dtype=object))
    del sp_grid, fl_grid, pct, filled
    gc.collect()

    ts = pd.to_datetime(days[ti // 288]) + pd.to_timedelta((ti % 288) * 5, unit="m")
    out = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "link_id": np.array(link_ids)[tj],
                        "speed_kmh": sp, "flow_vph": flow})
    del sp, flow, ti, tj
    gc.collect()
    return out


def main():
    t0 = time.time()
    done = completed_panels()
    todo = [p for p in (sys.argv[1:] or PANELS) if p not in done]
    print("already complete:", sorted(done))
    print("to do:", todo, flush=True)
    header = not STATE.exists() or STATE.stat().st_size == 0
    diag = []
    for panel in todo:
        models, cols, scores = train_panel(panel)
        diag.append({"panel": panel, "speed": scores["speed"], "flow": scores["flow"], "density_rmse": scores["density"]})
        tmpl = pd.read_csv(ROOT / "task1" / panel / "validation" /
                           "sample_submission_state.csv",
                           dtype={"link_id": str, "station_id": str})
        st = tmpl.drop_duplicates("link_id").set_index("link_id").station_id
        del tmpl
        for split in ("validation", "private"):
            d = reconstruct(panel, split, models)
            key = pd.read_csv(ROOT / "task1" / panel / split /
                              "sample_submission_state.csv",
                              dtype={"link_id": str, "station_id": str})
            merged = key[["panel", "timestamp", "station_id", "link_id",
                          "mask_regime"]].merge(d, on=["timestamp", "link_id"],
                                                how="left")
            miss = int(merged.speed_kmh.isna().sum())
            if miss:
                print(f"  [{panel}/{split}] WARNING {miss}/{len(merged)} unmatched "
                      f"target rows, filling with link medians", flush=True)
                for c in ("speed_kmh", "flow_vph"):
                    merged[c] = merged[c].fillna(
                        merged.groupby("link_id")[c].transform("median"))
                    merged[c] = merged[c].fillna(merged[c].median())
            merged.to_csv(STATE, mode="w" if header else "a", header=header,
                          index=False)
            header = False
            print(f"  [{panel}/{split}] {len(merged):,} rows, unmatched {miss} "
                  f"[{time.time()-t0:.0f}s] rss={peak_mb():.0f}MB", flush=True)
            del d, key, merged
            gc.collect()
        with open(OUT / f"t1_model_{panel}.pkl", "wb") as fh:
            pickle.dump({"models": models, "cols": cols}, fh)
        del models
        gc.collect()
    if diag:
        dd = pd.DataFrame(diag)
        path = OUT / "t1_diag.csv"
        dd.to_csv(path, mode="a" if path.exists() else "w",
                  header=not path.exists(), index=False)
        print(dd.to_string())
    print("mean S_state = done for", len(todo), "panels")


if __name__ == "__main__":
    main()
