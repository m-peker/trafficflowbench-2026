"""Task 1 trained across all ten corridors at once, against the per-panel models.

Ten independent models each see one corridor. The features are already
panel-agnostic - gaps, ratios, profile deviations - with the link's static
attributes (v_f, lanes, capacity, length, corridor position) carried alongside,
so pooling is well posed. A pooled fit sees ten times the data over the same
physics, and ten times as much of the rare regime: congestion is ~7 percent of
cells, which is 175k per panel but 1.75M pooled.

Three things get compared on the same per-panel holdout (last 45 train days):
the per-panel model, the pooled model, and their average. The average is the
interesting one - two fits with different bias usually beat either alone, and
that costs nothing but disk.

    python pooled_t1.py train      # fit the pooled models
    python pooled_t1.py eval       # per-panel / pooled / blend on the holdout
"""
import gc, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from common import OUT, PANELS, FAMILY
from t1build import build
from t1feat import load_grid
from task1_run import PARAMS, FLOW_BLEND

PER_PANEL = 500_000          # per corridor, so ~5M rows pooled
POOLED = OUT / "t1_pooled.pkl"
TARGETS = (("speed", 25.0), ("flow", 600.0), ("density", None))


def collect(sample=PER_PANEL):
    """Pooled design matrix, with a per-row panel index so holdouts stay separable."""
    Xs, ys, ds, pid, cols = [], {n: [] for n, _ in TARGETS}, [], [], None
    for i, panel in enumerate(PANELS):
        r = build(panel, "train", truth="train", sample=sample, seed=1, verbose=False)
        cols = r["cols"]
        ok = np.isfinite(r["y_speed"]) & np.isfinite(r["y_flowpl"])
        Xs.append(r["X"][ok].astype(np.float32))
        ys["speed"].append(r["y_speed"][ok])
        ys["flow"].append(r["y_flowpl"][ok])
        ys["density"].append(r["y_flowpl"][ok] / np.maximum(r["y_speed"][ok], 1.0))
        ds.append((r["ti"][ok] // 288).astype(np.int16))
        pid.append(np.full(int(ok.sum()), i, np.int8))
        print(f"  pooled {panel}: {int(ok.sum()):,} rows "
              f"(total {sum(len(a) for a in pid):,})", flush=True)
        del r
        gc.collect()
    X = np.vstack(Xs); del Xs; gc.collect()
    return (X, {n: np.concatenate(v) for n, v in ys.items()},
            np.concatenate(ds), np.concatenate(pid), cols)


def train():
    t0 = time.time()
    X, y, day, pid, cols = collect()
    print(f"pooled matrix {X.shape} ({time.time()-t0:.0f}s)", flush=True)
    nd = day.max() + 1
    tr, va = day < nd - 45, day >= nd - 45
    models = {}
    for name, scale in TARGETS:
        ds = lgb.Dataset(X[tr], label=y[name][tr], feature_name=cols)
        dv = lgb.Dataset(X[va], label=y[name][va], feature_name=cols, reference=ds)
        b = lgb.train(PARAMS, ds, 3000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        p = b.predict(X[va], num_iteration=b.best_iteration)
        e = float(np.sqrt(np.mean((p - y[name][va]) ** 2)))
        models[name] = lgb.Booster(model_str=b.model_to_string(
            num_iteration=b.best_iteration))
        models[name].best_iteration = b.best_iteration
        tag = f"S={max(0,1-e/scale):.4f}" if scale else ""
        print(f"  pooled {name}: it={b.best_iteration} RMSE={e:.4f} {tag} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del ds, dv, b, p
        gc.collect()
    with open(POOLED, "wb") as fh:
        pickle.dump({"models": models, "cols": cols}, fh)
    print("saved", POOLED)


def evaluate(panels=None):
    with open(POOLED, "rb") as fh:
        pooled = pickle.load(fh)["models"]
    rows = []
    for panel in (panels or PANELS):
        r = build(panel, "train", truth="train", sample=900_000, seed=1, verbose=False)
        G = load_grid(panel, "train", masked=True)
        vf, lanes, cap = G["stat"]["vf"], G["stat"]["lanes"], G["stat"]["cap"]
        del G
        day = r["ti"] // 288
        m = (day >= day.max() + 1 - 45) & np.isfinite(r["y_speed"]) \
            & np.isfinite(r["y_flowpl"])
        X, tj = r["X"][m], r["tj"][m]
        v_t, q_t = r["y_speed"][m], r["y_flowpl"][m]
        k_t = q_t / np.maximum(v_t, 1.0)
        del r
        with open(OUT / f"t1_model_{panel}.pkl", "rb") as fh:
            own = pickle.load(fh)["models"]

        def emit(mods, w_pooled):
            out = {}
            for name in ("speed", "flow", "density"):
                a = mods[name].predict(X, num_iteration=mods[name].best_iteration)
                if w_pooled > 0:
                    p = pooled[name].predict(X, num_iteration=pooled[name].best_iteration)
                    a = (1 - w_pooled) * a + w_pooled * p
                out[name] = a
            v = np.clip(out["speed"], 5.0, vf[tj] * 1.05)
            q = (1 - FLOW_BLEND) * out["flow"] + FLOW_BLEND * out["density"] * v
            q = np.clip(q * lanes[tj], 5.0, cap[tj] * 1.15) / lanes[tj]
            rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
            return rm(v, v_t), rm(q, q_t), rm(q / v, k_t) / k_t.mean()

        for label, mods, w in (("per-panel", own, 0.0),
                               ("pooled", pooled, 1.0),
                               ("blend 50/50", own, 0.5),
                               ("blend 70/30", own, 0.3)):
            ev, eq, ek = emit(mods, w)
            rows.append(dict(panel=panel, variant=label, family=FAMILY[panel],
                             S_v=1 - ev / 25, S_q=1 - eq / 600,
                             S_state=0.54 * (1 - ev / 25) + 0.46 * (1 - eq / 600),
                             dens=ek))
        cur = [x for x in rows if x["panel"] == panel]
        best = max(cur, key=lambda x: x["S_state"])
        print(f"{panel:11s} " + "  ".join(
            f"{x['variant']}: {x['S_state']:.4f}/{x['dens']:.4f}" for x in cur)
            + f"   best={best['variant']}", flush=True)
        del X
        gc.collect()

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "pooled_eval.csv", index=False)
    print("\n--- leaderboard aggregation ---")
    for v in d.variant.unique():
        s = d[d.variant == v]
        agg = s.groupby("family").S_state.mean().mean()
        print(f"{v:12s} S_state {agg:.4f}   density {s.dens.mean():.5f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    else:
        evaluate(sys.argv[2:] or None)
