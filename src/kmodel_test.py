"""Does modelling density directly beat dividing two independent predictions?

S_LWR is driven by the error in k = q/v, not by the error in q and v separately.
Our current answer takes the ratio of two independently fitted models, so their
errors add.  Fitting k directly (occupancy at the neighbouring cells is a direct
density proxy) and then reporting q = k*v should shrink that error.
"""
import sys, time
import numpy as np
import lightgbm as lgb
from t1build import build

PANEL = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 900_000
PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=100, feature_fraction=0.75, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1, num_threads=12)


def fit(X, y, tr, va, cols, rounds=2500):
    m = np.isfinite(y)
    ds = lgb.Dataset(X[tr & m], label=y[tr & m], feature_name=cols)
    dv = lgb.Dataset(X[va & m], label=y[va & m], feature_name=cols, reference=ds)
    b = lgb.train(PARAMS, ds, rounds, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    return b


def main():
    t0 = time.time()
    r = build(PANEL, "train", truth="train", sample=SAMPLE, seed=1)
    X, cols = r["X"], r["cols"]
    day = r["ti"] // 288
    nd = day.max() + 1
    tr, va = day < nd - 45, day >= nd - 45
    y_v = r["y_speed"]
    y_q = r["y_flowpl"]
    y_k = y_q / np.maximum(y_v, 1.0)          # per-lane density, veh/km/lane
    ok = np.isfinite(y_v) & np.isfinite(y_q)

    mv = fit(X, y_v, tr, va, cols)
    mq = fit(X, y_q, tr, va, cols)
    mk = fit(X, y_k, tr, va, cols)
    print(f"models trained in {time.time()-t0:.0f}s", flush=True)

    s = va & ok
    v_hat = np.clip(mv.predict(X[s], num_iteration=mv.best_iteration), 5.0, None)
    q_hat = mq.predict(X[s], num_iteration=mq.best_iteration)
    k_hat = mk.predict(X[s], num_iteration=mk.best_iteration)
    v_t, q_t, k_t = y_v[s], y_q[s], y_k[s]

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print(f"\nspeed  RMSE = {rmse(v_hat, v_t):.4f}   S_v = {1 - rmse(v_hat, v_t)/25:.4f}")
    print(f"\n{'density k (veh/km/lane)':32s} {'RMSE':>8s} {'rel':>8s}")
    k_ratio = q_hat / np.maximum(v_hat, 1.0)
    for name, k in (("q_hat / v_hat  (current)", k_ratio), ("direct k model", k_hat)):
        print(f"{name:32s} {rmse(k, k_t):8.4f} {rmse(k, k_t)/k_t.mean():8.4f}")

    print(f"\n{'flow (vph/lane)':32s} {'RMSE':>8s}    {'S_q':>7s}")
    q_from_k = k_hat * v_hat
    for name, q in (("direct q model (current)", q_hat), ("k_hat * v_hat", q_from_k),
                    ("mean of the two", 0.5 * (q_hat + q_from_k))):
        e = rmse(q, q_t)
        print(f"{name:32s} {e:8.3f}    {1 - e/600:7.4f}")

    # what the physics residual actually cares about: the error in k
    print("\nrelative density error (drives S_LWR):")
    for name, k in (("current", k_ratio), ("direct k", k_hat),
                    ("blend", 0.5 * (k_ratio + k_hat))):
        rel = rmse(k, k_t) / k_t.mean()
        print(f"  {name:10s} {rel:.5f}")


if __name__ == "__main__":
    main()
