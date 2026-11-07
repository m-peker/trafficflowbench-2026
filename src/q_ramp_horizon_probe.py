"""Does the ramp count released DURING a Task 2 horizon carry the mainline queue?

Validation/private blank the mainline for the horizon (+60 min) but ship ramp
counts straight through it. This measures, on train, how much the ramp counts
at T+s add to a history-only forecast of the queue at T+s.

Three LightGBM models on the same rows, same split by day:
    hist      mainline state at T around link j, step s, time of day
    hist+rT   + ramp counts at T (information we already legitimately have)
    hist+rTs  + ramp counts at T+s (the horizon's own ramp counts)

    python q_ramp_horizon_probe.py
"""
import nice  # noqa: F401
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import QUEUE_PANELS
from qdata import load_panel
from qramp import load_split

HOR, NB, VAL = 6, 2, 60


def auc(y, p):
    o = np.argsort(p); y = y[o].astype(np.int64)
    r = np.arange(1, len(y) + 1, dtype=np.int64)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) // 2) / (n1 * n0))


def rows(panel, pid, rng):
    D = load_panel(panel)
    R = load_split(panel, "train", D["link_ids"])
    nT, nL = D["nT"], D["nL"]
    tod = np.arange(nT) % 288
    ratio = D["speed"] / np.maximum(D["stat"]["vf"][None, :], 1.0)
    feats = {}
    for k in ("on", "off"):
        x = R[k].astype(np.float64)
        prof = np.zeros((288, nL)); cnt = np.zeros((288, nL))
        ok = np.isfinite(x)
        np.add.at(prof, tod, np.where(ok, x, 0.0)); np.add.at(cnt, tod, ok)
        feats[k] = x / np.maximum(prof / np.maximum(cnt, 1), 1.0)[tod]
    qa = D["q"].any(1)
    T = np.arange(12, nT - HOR - 1)
    T = T[qa[T] | qa[T + HOR]]                       # windows near congestion
    T = rng.choice(T, size=min(1500, len(T)), replace=False)
    out = []
    pad = lambda a: np.pad(a, ((0, 0), (NB, NB)), constant_values=np.nan)
    rp, op, fp = pad(ratio), pad(feats["on"]), pad(feats["off"])
    for t in T:
        for s in range(1, HOR + 1):
            y = D["q"][t + s]; e = D["elig"][t + s]
            d = {"pid": pid, "day": t // 288, "s": s, "tod": (t + s) % 288,
                 "link": np.arange(nL), "y": y.astype(int), "e": e}
            for o in range(-NB, NB + 1):
                sl = slice(NB + o, NB + o + nL)
                d[f"r{o}"] = rp[t, sl]
                d[f"onT{o}"] = op[t, sl]; d[f"offT{o}"] = fp[t, sl]
                d[f"onS{o}"] = op[t + s, sl]; d[f"offS{o}"] = fp[t + s, sl]
            out.append(pd.DataFrame(d))
    df = pd.concat(out, ignore_index=True)
    df["val"] = df.day >= (nT // 288 - VAL)
    return df[df.e]


def main():
    rng = np.random.default_rng(0)
    df = pd.concat([rows(p, i, rng) for i, p in enumerate(QUEUE_PANELS)], ignore_index=True)
    print(f"rows {len(df):,}  queued {df.y.mean():.3f}", flush=True)
    base = ["pid", "s", "tod", "link"] + [f"r{o}" for o in range(-NB, NB + 1)]
    rT = [f"{k}T{o}" for k in ("on", "off") for o in range(-NB, NB + 1)]
    rS = [f"{k}S{o}" for k in ("on", "off") for o in range(-NB, NB + 1)]
    tr, te = df[~df.val], df[df.val]
    params = dict(objective="binary", num_leaves=63, learning_rate=0.05,
                  min_child_samples=200, feature_fraction=0.8, verbose=-1)
    for name, cols in (("hist", base), ("hist+rT", base + rT), ("hist+rT+rTs", base + rT + rS)):
        m = lgb.train(params, lgb.Dataset(tr[cols], tr.y), 300)
        p = m.predict(te[cols])
        by_s = " ".join(f"{auc(te.y.values[te.s.values == s], p[te.s.values == s]):.3f}"
                        for s in range(1, HOR + 1))
        print(f"{name:12s} AUC {auc(te.y.values, p):.4f}   by step {by_s}", flush=True)


if __name__ == "__main__":
    main()
