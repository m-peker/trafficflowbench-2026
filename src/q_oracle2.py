"""Task 2 via the Task 1 reconstruction: published cells read, blanked cells modelled."""
import sys, time, pickle
import numpy as np
import lightgbm as lgb
from common import WORK, OUT
from t1build import build
from qdata import load_panel, origins
from qfeat import HOR

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
D = load_panel(panel)
z = np.load(WORK / "cache" / f"{panel}_train_masked.npz", allow_pickle=True)
src = [str(x) for x in z["link_ids"]]
o = [src.index(l) for l in D["link_ids"]]
msp, mpct = z["speed"][:, o], z["pct"][:, o]
mflow = z["flow"][:, o]
mavail = np.isfinite(msp) & np.isfinite(mflow)
thr = 0.6 * D["stat"]["vf"][None, :]
nT, nL = msp.shape

rng = np.random.default_rng(3)
on, ong = origins(D)
nd = nT // 288
on = on[on // 288 >= nd - 60]
ong = ong[ong // 288 >= nd - 60]
on = rng.choice(on, min(300, len(on)), replace=False)
ong = rng.choice(ong, min(300, len(ong)), replace=False)

# cells we actually need
need = np.zeros((nT, nL), bool)
for T in np.concatenate([on, ong]):
    need[T + 1: T + 1 + HOR] = True
print(f"{panel}: windows {len(on)}+{len(ong)}, horizon cells {need.sum():,}", flush=True)

# train the Task 1 speed model on the earlier days
t0 = time.time()
r = build(panel, "train", truth="train", sample=700000, seed=1)
day = r["ti"] // 288
tr = day < nd - 60
m = tr & np.isfinite(r["y_speed"])
ds = lgb.Dataset(r["X"][m], label=r["y_speed"][m], feature_name=r["cols"])
mdl = lgb.train(dict(objective="l2", learning_rate=0.05, num_leaves=255,
                     min_data_in_leaf=100, feature_fraction=0.75,
                     bagging_fraction=0.8, bagging_freq=1, verbosity=-1,
                     num_threads=12), ds, 900)
print(f"speed model trained ({time.time()-t0:.0f}s)", flush=True)
del r, ds

rr = build(panel, "train", targets="blanked", target_mask=need)
pred = mdl.predict(rr["X"])
grid = msp.copy()
grid[rr["ti"], rr["tj"]] = pred
print(f"reconstructed {len(pred):,} cells ({time.time()-t0:.0f}s)", flush=True)

# interpolation-only reference for the same cells
idx = np.where(mavail, np.arange(nT)[:, None], -1)
pi = np.maximum.accumulate(idx, 0)
idx2 = np.where(mavail, np.arange(nT)[:, None], nT)
ni = np.minimum.accumulate(idx2[::-1], 0)[::-1]
cc = np.arange(nL)[None, :]
pv = np.where(pi >= 0, msp[np.clip(pi, 0, nT - 1), cc], np.nan)
nv = np.where(ni < nT, msp[np.clip(ni, 0, nT - 1), cc], np.nan)
tt = np.arange(nT)[:, None]
pg = np.where(pi >= 0, tt - pi, 9999); ng = np.where(ni < nT, ni - tt, 9999)
interp = np.where(np.isfinite(pv) & np.isfinite(nv),
                  (pv * ng + nv * pg) / np.maximum(pg + ng, 1),
                  np.where(np.isfinite(pv), pv, nv))
grid_i = np.where(mavail, msp, interp)


def ev(origins_, g, name):
    v = []
    for T in origins_:
        sl = slice(T + 1, T + 1 + HOR)
        e = D["elig"][sl]
        t = D["q"][sl] & e
        p = (g[sl] <= thr) & e
        u = np.logical_or(p, t).sum()
        v.append(1.0 if u == 0 else np.logical_and(p, t).sum() / u)
    print(f"  {name}: IoU={np.mean(v):.4f}")
    return float(np.mean(v))


print("interp grid:"); a = ev(on, grid_i, "onset  "); b = ev(ong, grid_i, "ongoing")
print("gbm grid:");    c = ev(on, grid, "onset  ");   d = ev(ong, grid, "ongoing")
print(f"panel IoU  interp={(a+b)/2:.4f}  gbm={(c+d)/2:.4f}")
