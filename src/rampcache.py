"""Aggregate ramp flows onto mainline links as (time x link) matrices."""
import sys
import numpy as np
import pandas as pd
from common import ROOT, WORK, PANELS, ramp_files, links_table, net_dir

CACHE = WORK / "cache"


def build(panel, split):
    amap = pd.read_csv(net_dir(panel) / "ramp_attachment_map.csv",
                       dtype={"ramp_link_id": str, "nearest_mainline_link_id": str})
    link_ids = sorted(links_table(panel).link_id.astype(str).unique())
    li = {l: i for i, l in enumerate(link_ids)}
    amap["col"] = amap.nearest_mainline_link_id.map(li)
    amap = amap.dropna(subset=["col"])
    m_col = dict(zip(amap.ramp_link_id, amap.col.astype(int)))
    m_typ = dict(zip(amap.ramp_link_id, amap.ramp_type.astype(str)))

    files = ramp_files(panel, split)
    days = sorted({pd.read_parquet(f, columns=["date"]).date.iloc[0] for f in files})
    di = {d: i for i, d in enumerate(days)}
    nT, nL = len(days) * 288, len(link_ids)
    on = np.zeros((nT, nL), np.float32); on_n = np.zeros((nT, nL), np.int16)
    off = np.zeros((nT, nL), np.float32); off_n = np.zeros((nT, nL), np.int16)
    miss = np.zeros((nT, nL), np.int16)
    for f in files:
        d = pd.read_parquet(f, columns=["timestamp", "ramp_link_id", "flow_vph",
                                        "pct_observed"])
        d["ramp_link_id"] = d.ramp_link_id.astype(str)
        col = d.ramp_link_id.map(m_col)
        keep = col.notna()
        d = d[keep]; col = col[keep].to_numpy(int)
        ts = pd.to_datetime(d.timestamp, utc=True)
        row = ts.dt.strftime("%Y-%m-%d").map(di).to_numpy(np.int64) * 288 + \
            (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(np.int64)
        typ = d.ramp_link_id.map(m_typ).to_numpy()
        val = pd.to_numeric(d.flow_vph, errors="coerce").to_numpy(np.float32)
        ok = (pd.to_numeric(d.pct_observed, errors="coerce").to_numpy() >= 75) & np.isfinite(val)
        is_on = typ == "OR"
        np.add.at(on, (row[ok & is_on], col[ok & is_on]), val[ok & is_on])
        np.add.at(on_n, (row[ok & is_on], col[ok & is_on]), 1)
        np.add.at(off, (row[ok & ~is_on], col[ok & ~is_on]), val[ok & ~is_on])
        np.add.at(off_n, (row[ok & ~is_on], col[ok & ~is_on]), 1)
        np.add.at(miss, (row[~ok], col[~ok]), 1)
    np.savez_compressed(CACHE / f"{panel}_{split}_ramp.npz", on=on, off=off,
                        on_n=on_n, off_n=off_n, miss=miss,
                        link_ids=np.array(link_ids, dtype=object),
                        days=np.array(days, dtype=object))
    print(f"[{panel}/{split}/ramp] {nT}x{nL} on_mean={on.mean():.1f} "
          f"off_mean={off.mean():.1f} miss={miss.sum():,}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    splits = ["train", "validation", "private"]
    for p in (args or PANELS):
        for s in splits:
            build(p, s)
