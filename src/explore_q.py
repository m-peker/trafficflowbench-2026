import sys, numpy as np, pandas as pd
from common import WORK, links_table

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
z = np.load(WORK / "cache" / f"{panel}_train.npz", allow_pickle=True)
sp, pct = z["speed"], z["pct"]
link_ids = list(z["link_ids"])
lk = links_table(panel).set_index("link_id")
vf = np.array([lk.free_speed_kmh.get(l, 105.0) for l in link_ids], np.float32)
thr = 0.6 * vf

elig = (pct >= 75) & np.isfinite(sp)
q = np.where(elig, sp <= thr, False)
nT, nL = sp.shape
print(f"{panel}: {nT}x{nL}  eligible={elig.mean():.3f}  queued|elig={q.sum()/elig.sum():.4f}")

# per link queue rate
rate = q.sum(0) / np.maximum(elig.sum(0), 1)
o = np.argsort(-rate)
print("top-8 queued links:", [(link_ids[i], round(float(rate[i]), 3)) for i in o[:8]])
print("links with rate>0.02:", int((rate > 0.02).sum()), "of", nL)

# time of day profile
tod = np.arange(nT) % 288
qr = np.array([q[tod == h].sum() / max(elig[tod == h].sum(), 1) for h in range(288)])
peak = np.argsort(-qr)[:6]
print("peak tod slots (hh:mm):", [f"{p//12:02d}:{(p%12)*5:02d}={qr[p]:.3f}" for p in sorted(peak)])

# persistence: from t (last obs = T-5) to T+5..T+30 -> lags 2..7
for lag in [1, 2, 4, 7]:
    a = q[:-lag]; b = q[lag:]
    m = elig[:-lag] & elig[lag:]
    p11 = (a & b & m).sum() / max((a & m).sum(), 1)
    p01 = (~a & b & m).sum() / max((~a & m).sum(), 1)
    print(f"lag {lag} ({lag*5}min): P(q_t+k|q_t)={p11:.3f}  P(q_t+k|~q_t)={p01:.4f}")
