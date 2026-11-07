"""Build a probe submission that zeroes one task, to read its component score.

The leaderboard reports one number. Zeroing a single task's value column and
resubmitting makes the drop equal to that task's weight times its score, which
is the only way to separate S_physics from S_ODME - the two we cannot measure
locally. The probe scores badly on purpose; Kaggle ranks you on your best
submission, so it costs a slot and nothing else.
"""
import sys, time, zipfile
from pathlib import Path
import pandas as pd

TASK = sys.argv[1] if len(sys.argv) > 1 else "odme"
COL = {"odme": "path_flow", "queue": "queue_pred", "state": "speed_kmh"}[TASK]
SRC = Path(r"D:/BigData Competition/work/out/submission.csv")
OUT = Path(r"D:/BigData Competition") / f"probe_{TASK}_zeroed.zip"

t0 = time.time()
tmp = Path(f"out/probe_{TASK}.csv")
first, n, zeroed = True, 0, 0
for ch in pd.read_csv(SRC, chunksize=1_000_000):
    m = ch.task == TASK
    zeroed += int(m.sum())
    ch.loc[m, COL] = 0
    if TASK == "state":
        ch.loc[m, "flow_vph"] = 0
    ch["speed_kmh"] = ch.speed_kmh.round(2)
    ch["flow_vph"] = ch.flow_vph.round(1)
    ch["path_flow"] = ch.path_flow.round(2)
    ch["queue_pred"] = ch.queue_pred.astype("int8")
    ch.to_csv(tmp, mode="w" if first else "a", header=first, index=False,
              lineterminator="\n")
    first = False
    n += len(ch)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(tmp, arcname="submission.csv")
tmp.unlink()
print(f"{n:,} rows, zeroed {COL} on {zeroed:,} {TASK} rows")
print(f"-> {OUT}  {OUT.stat().st_size/1e6:.0f} MB  ({time.time()-t0:.0f}s)")
