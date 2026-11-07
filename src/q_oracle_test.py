"""How far does reading the released masked grid get us on Task 2?

For a window horizon we may read the published masked observation wherever the
cell was not blanked, and must reconstruct the rest.  Truth here is the
unmasked observation layer, which the released train split ships.
"""
import sys
import numpy as np
import pandas as pd
from common import WORK, links_table
from qfeat import static_table, HIST, HOR
from qdata import load_panel, origins

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"

D = load_panel(panel)                                    # unmasked truth grid
z = np.load(WORK / "cache" / f"{panel}_train_masked.npz", allow_pickle=True)
src = [str(x) for x in z["link_ids"]]
o = [src.index(l) for l in D["link_ids"]]                # match corridor order
msp, mpct = z["speed"][:, o], z["pct"][:, o]
mavail = np.isfinite(msp) & (mpct >= 75)
thr = 0.6 * D["stat"]["vf"][None, :]

print(f"{panel}: masked-grid observed fraction = {mavail.mean():.3f}")

# temporal linear interpolation of the masked speed grid, per link
nT, nL = msp.shape
idx = np.where(mavail, np.arange(nT)[:, None], -1)
prev_i = np.maximum.accumulate(idx, 0)
idx2 = np.where(mavail, np.arange(nT)[:, None], nT)
next_i = np.minimum.accumulate(idx2[::-1], 0)[::-1]
cols = np.arange(nL)[None, :]
pv = np.where(prev_i >= 0, msp[np.clip(prev_i, 0, nT - 1), cols], np.nan)
nv = np.where(next_i < nT, msp[np.clip(next_i, 0, nT - 1), cols], np.nan)
tt = np.arange(nT)[:, None]
pg = np.where(prev_i >= 0, tt - prev_i, 9999)
ng = np.where(next_i < nT, next_i - tt, 9999)
w = pg + ng
interp = np.where(np.isfinite(pv) & np.isfinite(nv),
                  (pv * ng + nv * pg) / np.maximum(w, 1),
                  np.where(np.isfinite(pv), pv, nv))

filled = np.where(mavail, msp, interp)
q_pred_grid = filled <= thr
q_true_grid = D["q"]
elig = D["elig"]

rng = np.random.default_rng(3)
on, ong = origins(D)
on = rng.choice(on, min(400, len(on)), replace=False)
ong = rng.choice(ong, 400, replace=False)


def score(origins_, name):
    obs_only, full, persist = [], [], []
    for T in origins_:
        sl = slice(T + 1, T + 1 + HOR)
        e = elig[sl]
        t = q_true_grid[sl] & e
        # (a) read where published, reconstruct the rest by interpolation
        p = q_pred_grid[sl] & e
        u = np.logical_or(p, t).sum()
        full.append(1.0 if u == 0 else np.logical_and(p, t).sum() / u)
        # (b) read where published, predict 0 elsewhere
        p2 = (msp[sl] <= thr) & mavail[sl] & e
        u2 = np.logical_or(p2, t).sum()
        obs_only.append(1.0 if u2 == 0 else np.logical_and(p2, t).sum() / u2)
    print(f"  {name}: published+interp IoU={np.mean(full):.4f}   "
          f"published-only IoU={np.mean(obs_only):.4f}")


score(on, "onset  ")
score(ong, "ongoing")

# how noisy is the threshold crossing itself?
near = elig & (np.abs(D["speed"] - thr) <= 2.0)
print(f"cells within 2 km/h of the queue threshold: {near.sum() / elig.sum():.5f} of eligible")
