import sys, numpy as np
from qwin import load_panel, enumerate_windows, window_iou, HIST, HOR

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
D = load_panel(panel)
onset, ongoing = enumerate_windows(D, per_cond=300, seed=1)
print(f"{panel}: onset windows available/sampled={len(onset)}, ongoing={len(ongoing)}")

nL = D["nL"]


def strat_last_step(D, T):
    """Baseline: queue state at the final history step only."""
    last = T - 1
    row = D["q"][last]
    return np.repeat(row[None, :], HOR, 0)


def strat_last_avail(D, T, k=HIST):
    """Most recent eligible observation per link inside the history."""
    h = slice(T - k, T)
    e = D["elig"][h]; qq = D["q"][h]
    idx = np.where(e.any(0), e.shape[0] - 1 - e[::-1].argmax(0), -1)
    row = np.zeros(nL, bool)
    ok = idx >= 0
    row[ok] = qq[idx[ok], np.arange(nL)[ok]]
    return np.repeat(row[None, :], HOR, 0)


def strat_last_avail_any3(D, T):
    """Queued if any of the last 3 available observations was queued."""
    h = slice(T - HIST, T)
    e = D["elig"][h]; qq = D["q"][h]
    row = np.zeros(nL, bool)
    for j in range(nL):
        idx = np.where(e[:, j])[0]
        if len(idx):
            row[j] = qq[idx[-3:], j].any()
    return np.repeat(row[None, :], HOR, 0)


for name, fn in [("last_step(baseline)", strat_last_step),
                 ("last_available", strat_last_avail),
                 ("last_avail_any3", strat_last_avail_any3)]:
    for cond, origins in [("onset", onset), ("ongoing", ongoing)]:
        v = [window_iou(D, T, fn(D, T)) for T in origins]
        print(f"  {name:22s} {cond:8s} IoU={np.mean(v):.4f}")
