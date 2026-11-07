"""Do the board's Task 1 targets sit in structures the local holdout never has?

The Task 2 gaps were exactly that (s67-s71). This compares, for Task 1 target
cells outside those gaps, validation/private against the train masked grid's
last 60 days (the local holdout's population):
    run     length of the blank run in time that contains the target (per link)
    cov     share of the corridor observed at the target's timestamp
    near    steps to the nearest observed cell on the same link
    edge    steps to the nearest Task 2 gap (how many targets sit beside one)

    python t1_structure_diff.py
"""
import nice  # noqa: F401
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from common import ROOT, QUEUE_PANELS
from t1feat import load_grid

PANELS = QUEUE_PANELS + ["D12_I405_N", "D12_I405_S"]


def gaps(panel, split, G):
    g = np.zeros(G["nT"], bool)
    f = ROOT / "task2" / panel / split / "window_index.csv"
    if split == "train" or not f.exists():
        return g
    day0 = pd.Timestamp(G["days"][0], tz="UTC")
    for r in pd.read_csv(f).itertuples():
        a = int((pd.Timestamp(r.forecast_start) - day0).total_seconds() // 300)
        g[max(a, 0):a + 18] = True
    return g


def stats(G, tgt, gap):
    obs = np.isfinite(G["speed"]) & (G["pct"] >= 75)
    nT, nL = obs.shape
    blank = ~obs
    # run length of blanks per link, assigned to every cell of the run
    run = np.zeros((nT, nL), np.int32)
    for j in range(nL):
        b = blank[:, j].astype(np.int8)
        d = np.diff(np.r_[0, b, 0]); s = np.flatnonzero(d == 1); e = np.flatnonzero(d == -1)
        for a, z in zip(s, e):
            run[a:z, j] = z - a
    t = np.arange(nT)[:, None]
    last = np.maximum.accumulate(np.where(obs, t, -10 ** 6), 0)
    nxt = np.minimum.accumulate(np.where(obs, t, 10 ** 6)[::-1], 0)[::-1]
    near = np.minimum(t - last, nxt - t)
    cov = obs.mean(1)[:, None] * np.ones((1, nL))
    gi = np.flatnonzero(gap)
    dg = np.full(nT, 999)
    if len(gi):
        idx = np.searchsorted(gi, np.arange(nT)); idx = np.clip(idx, 1, len(gi) - 1)
        dg = np.minimum(np.abs(np.arange(nT) - gi[idx - 1]), np.abs(np.arange(nT) - gi[idx]))
    m = tgt & ~gap[:, None]
    return run[m], cov[m], near[m], np.broadcast_to(dg[:, None], (nT, nL))[m]


def main():
    rows = []
    for panel in PANELS:
        for split in ("train", "validation", "private"):
            G = load_grid(panel, split, masked=True)
            nT = G["nT"]
            if split == "train":
                Gt = load_grid(panel, "train", masked=False)
                o = [Gt["link_ids"].index(l) for l in G["link_ids"]]
                tgt = np.isnan(G["speed"]) & np.isfinite(Gt["speed"][:, o]) & (Gt["pct"][:, o] >= 75)
                tgt[: nT - 60 * 288] = False
            else:
                # targets = blanked cells whose quality flags say they were observed
                tgt = np.isnan(G["speed"]) & (G["pct"] >= 75)
            gp = gaps(panel, split, G)
            r, c, n, d = stats(G, tgt, gp)
            rows.append(dict(panel=panel, split=split, n=len(r),
                             run1=np.mean(r == 1), run_ge4=np.mean(r >= 4), run_mean=r.mean(),
                             run_ge12=np.mean(r >= 12), cov=c.mean(), near_ge3=np.mean(n >= 3),
                             beside_gap=np.mean(d <= 6)))
        print(f"  {panel}", flush=True)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.groupby("split")[["n", "run1", "run_ge4", "run_ge12", "run_mean", "cov", "near_ge3", "beside_gap"]]
          .agg({"n": "sum", "run1": "mean", "run_ge4": "mean", "run_ge12": "mean", "run_mean": "mean",
                "cov": "mean", "near_ge3": "mean", "beside_gap": "mean"}).round(4).to_string())
    print()
    print(df.pivot(index="panel", columns="split", values="run_ge4").round(4).to_string())


if __name__ == "__main__":
    main()
