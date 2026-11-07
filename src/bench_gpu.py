"""Benchmark the Task 1 regressors: LightGBM on CPU vs XGBoost / CatBoost on GPU.

Same features, same day-wise holdout, so the RMSE numbers are comparable and we
only switch backends if accuracy holds.

Run with the interpreter that has a CUDA build, e.g.
  C:/Users/musa.peker/AppData/Local/anaconda3/envs/last-env/python.exe bench_gpu.py
"""
import sys, time
import numpy as np
from t1build import build

PANEL = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
SAMPLE = 900_000


def report(name, secs, rmse, scale):
    print(f"{name:22s} {secs:7.1f}s  RMSE={rmse:8.4f}  S={max(0.0, 1 - rmse / scale):.4f}",
          flush=True)


def main():
    t0 = time.time()
    r = build(PANEL, "train", truth="train", sample=SAMPLE, seed=1)
    print(f"features built in {time.time()-t0:.0f}s  X={r['X'].shape}", flush=True)
    X, cols = r["X"], r["cols"]
    day = r["ti"] // 288
    nd = day.max() + 1
    tr, va = day < nd - 45, day >= nd - 45

    for nm, y, scale in (("speed", r["y_speed"], 25.0), ("flow", r["y_flowpl"], 600.0)):
        m = np.isfinite(y)
        Xtr, ytr = X[tr & m], y[tr & m]
        Xva, yva = X[va & m], y[va & m]
        print(f"\n--- {nm}: train {Xtr.shape[0]:,} valid {Xva.shape[0]:,} ---", flush=True)

        try:
            import lightgbm as lgb
            t = time.time()
            ds = lgb.Dataset(Xtr, label=ytr, feature_name=cols)
            dv = lgb.Dataset(Xva, label=yva, feature_name=cols, reference=ds)
            b = lgb.train(dict(objective="l2", learning_rate=0.05, num_leaves=255,
                               min_data_in_leaf=100, feature_fraction=0.75,
                               bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                               num_threads=0), ds, 2500, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(60, verbose=False)])
            p = b.predict(Xva, num_iteration=b.best_iteration)
            report(f"lightgbm cpu ({b.best_iteration})", time.time() - t,
                   float(np.sqrt(np.mean((p - yva) ** 2))), scale)
        except Exception as e:
            print("lightgbm failed:", e)

        try:
            import xgboost as xgb
            t = time.time()
            dtr = xgb.DMatrix(Xtr, label=ytr)
            dva = xgb.DMatrix(Xva, label=yva)
            params = dict(objective="reg:squarederror", eta=0.05, max_depth=0,
                          grow_policy="lossguide", max_leaves=255, min_child_weight=100,
                          colsample_bytree=0.75, subsample=0.8, tree_method="hist",
                          device="cuda", eval_metric="rmse")
            ev = {}
            b = xgb.train(params, dtr, 2500, evals=[(dva, "va")],
                          early_stopping_rounds=60, verbose_eval=False,
                          evals_result=ev)
            p = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
            report(f"xgboost gpu ({b.best_iteration})", time.time() - t,
                   float(np.sqrt(np.mean((p - yva) ** 2))), scale)
        except Exception as e:
            print("xgboost gpu failed:", type(e).__name__, e)

        try:
            from catboost import CatBoostRegressor
            t = time.time()
            b = CatBoostRegressor(iterations=2500, learning_rate=0.05, depth=8,
                                  task_type="GPU", devices="0", verbose=False,
                                  early_stopping_rounds=60, loss_function="RMSE")
            b.fit(Xtr, ytr, eval_set=(Xva, yva))
            p = b.predict(Xva)
            report(f"catboost gpu ({b.best_iteration_})", time.time() - t,
                   float(np.sqrt(np.mean((p - yva) ** 2))), scale)
        except Exception as e:
            print("catboost gpu failed:", type(e).__name__, e)


if __name__ == "__main__":
    main()
