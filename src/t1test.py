import sys, time
import numpy as np, lightgbm as lgb
from t1build import build

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
t = time.time()
r = build(panel, "train", truth="train", sample=700000, seed=1)
print("built %.0fs" % (time.time() - t), "X", r["X"].shape)
X, cols = r["X"], r["cols"]
ys, yf = r["y_speed"], r["y_flowpl"]
day = r["ti"] // 288
tr = day < 213
va = ~tr
res = {}
for nm, y, sc in [("speed", ys, 25.0), ("flow", yf, 600.0)]:
    m = np.isfinite(y)
    ds = lgb.Dataset(X[tr & m], label=y[tr & m], feature_name=cols)
    dv = lgb.Dataset(X[va & m], label=y[va & m], feature_name=cols, reference=ds)
    p = dict(objective="l2", learning_rate=0.06, num_leaves=255, min_data_in_leaf=100,
             feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
             verbosity=-1, num_threads=0)
    b = lgb.train(p, ds, 2000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    pr = b.predict(X[va & m], num_iteration=b.best_iteration)
    e = float(np.sqrt(np.mean((pr - y[va & m]) ** 2)))
    res[nm] = max(0.0, 1 - e / sc)
    print(f"{nm}: it={b.best_iteration} RMSE={e:.3f} S={res[nm]:.4f}")
    top = sorted(zip(cols, b.feature_importance("gain")), key=lambda x: -x[1])[:10]
    print("  top:", [c for c, _ in top])
print("S_state =", 0.54 * res["speed"] + 0.46 * res["flow"])
