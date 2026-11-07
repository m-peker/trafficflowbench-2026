"""Drop any panel from out/state_submission.csv that is not complete for both splits.

An interrupted run leaves a partial block; the resume step re-does those panels
from scratch so a panel's two splits always come from the same model.
"""
import sys
import numpy as np
import pandas as pd
from common import ROOT, OUT, WORK, PANELS

path = OUT / "state_submission.csv"
expect = {}
for panel in PANELS:
    for split in ("validation", "private"):
        f = ROOT / "task1" / panel / split / "sample_submission_state.csv"
        expect[(panel, split)] = sum(1 for _ in open(f)) - 1

d = pd.read_csv(path, dtype={"panel": str, "timestamp": str})
d["split"] = d.timestamp.str[:7].map(
    {"2031-03": "validation", "2031-04": "private"})
have = d.groupby(["panel", "split"]).size()
print("rows found per block:")
keep_panels = []
for panel in PANELS:
    ok = True
    for split in ("validation", "private"):
        n = int(have.get((panel, split), 0))
        e = expect[(panel, split)]
        if n:
            print(f"  {panel:12s} {split:11s} {n:>9,} / {e:>9,}"
                  f"{'' if n == e else '   INCOMPLETE'}")
        ok &= n == e
    if ok:
        keep_panels.append(panel)
print("complete panels:", keep_panels)

out = d[d.panel.isin(keep_panels)].drop(columns=["split"])
dup = int(out.duplicated(["panel", "timestamp", "station_id", "link_id",
                          "mask_regime"]).sum())
print(f"kept {len(out):,} rows, duplicates {dup}")
if dup:
    out = out.drop_duplicates(["panel", "timestamp", "station_id", "link_id",
                               "mask_regime"])
out.to_csv(path, index=False)
print("rewrote", path)

# grids belonging to dropped panels are stale
for p in (WORK / "grids").glob("*.npz"):
    panel = p.stem.rsplit("_", 1)[0]
    if panel not in keep_panels:
        p.unlink()
        print("removed stale grid", p.name)
