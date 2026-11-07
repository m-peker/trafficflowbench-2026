"""Cache per-panel dense (time x link) matrices, filling incrementally."""
import sys
import numpy as np
import pandas as pd
from common import ROOT, WORK, PANELS, mainline_files, links_table

CACHE = WORK / "cache"
CACHE.mkdir(exist_ok=True)
COLS = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed"]


def build(panel, split="train", masked=False):
    files = mainline_files(panel, split, masked=masked)
    link_ids = sorted(links_table(panel).link_id.astype(str).unique())
    li = {l: i for i, l in enumerate(link_ids)}
    nL = len(link_ids)

    # day grid from the file set
    days = sorted({pd.read_parquet(f, columns=["date"]).date.iloc[0] for f in files})
    t0 = pd.Timestamp(days[0] + "T00:00:00Z")
    day_idx = {d: i for i, d in enumerate(days)}
    nT = len(days) * 288

    sp = np.full((nT, nL), np.nan, np.float32)
    fl = np.full((nT, nL), np.nan, np.float32)
    oc = np.full((nT, nL), np.nan, np.float32)
    pc = np.zeros((nT, nL), np.int8)
    for f in files:
        d = pd.read_parquet(f, columns=COLS)
        ts = pd.to_datetime(d.timestamp, utc=True)
        day = ts.dt.strftime("%Y-%m-%d")
        row = day.map(day_idx).to_numpy(np.int64) * 288 + \
            (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(np.int64)
        col = d.link_id.astype(str).map(li).to_numpy(np.int64)
        ok = np.isfinite(row) & (col >= 0)
        sp[row[ok], col[ok]] = d.speed_kmh.to_numpy(np.float32)[ok]
        fl[row[ok], col[ok]] = d.flow_vph.to_numpy(np.float32)[ok]
        oc[row[ok], col[ok]] = d.occupancy.to_numpy(np.float32)[ok]
        pc[row[ok], col[ok]] = d.pct_observed.to_numpy(np.int8)[ok]

    np.savez_compressed(
        CACHE / f"{panel}_{split}.npz", speed=sp, flow=fl, occ=oc, pct=pc,
        link_ids=np.array(link_ids, dtype=object),
        days=np.array(days, dtype=object),
        t0=np.array([t0.value]))
    print(f"[{panel}/{split}] {nT}x{nL} days={len(days)} nan={np.isnan(sp).mean():.3f}",
          flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    split = "train"
    masked = False
    if args and args[0] in ("train", "validation", "private"):
        split = args.pop(0)
        masked = split != "train"
    for p in (args or PANELS):
        build(p, split, masked=masked)


def build_masked(panel, split):
    """Masked observation grid + per-day mask regime, for any split."""
    files = sorted((ROOT / "corridors" / panel / split /
                    "mainline_states_masked").glob("**/*.parquet"))
    link_ids = sorted(links_table(panel).link_id.astype(str).unique())
    li = {l: i for i, l in enumerate(link_ids)}
    nL = len(link_ids)
    metas = []
    for f in files:
        d = pd.read_parquet(f, columns=["date", "mask_regime"])
        metas.append((d.date.iloc[0], d.mask_regime.iloc[0]))
    days = sorted({m[0] for m in metas})
    day_idx = {d: i for i, d in enumerate(days)}
    regime = np.zeros(len(days), np.int8)
    for d, r in metas:
        regime[day_idx[d]] = {"R1": 1, "R2": 2, "R3": 3}.get(str(r), 0)
    nT = len(days) * 288
    sp = np.full((nT, nL), np.nan, np.float32)
    fl = np.full((nT, nL), np.nan, np.float32)
    oc = np.full((nT, nL), np.nan, np.float32)
    pc = np.zeros((nT, nL), np.int8)
    for f in files:
        d = pd.read_parquet(f, columns=["date", "timestamp", "link_id", "speed_kmh",
                                        "flow_vph", "occupancy", "pct_observed"])
        ts = pd.to_datetime(d.timestamp, utc=True)
        row = ts.dt.strftime("%Y-%m-%d").map(day_idx).to_numpy(np.int64) * 288 + \
            (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(np.int64)
        col = d.link_id.astype(str).map(li).to_numpy(np.int64)
        sp[row, col] = d.speed_kmh.to_numpy(np.float32)
        fl[row, col] = d.flow_vph.to_numpy(np.float32)
        oc[row, col] = d.occupancy.to_numpy(np.float32)
        pc[row, col] = d.pct_observed.to_numpy(np.int8)
    np.savez_compressed(CACHE / f"{panel}_{split}_masked.npz", speed=sp, flow=fl,
                        occ=oc, pct=pc, regime=regime,
                        link_ids=np.array(link_ids, dtype=object),
                        days=np.array(days, dtype=object))
    print(f"[{panel}/{split}/masked] {nT}x{nL} days={len(days)} "
          f"nan={np.isnan(sp).mean():.3f}", flush=True)
