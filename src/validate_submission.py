"""Check the assembled upload file against the rules the evaluator enforces."""
import sys
import numpy as np
import pandas as pd
from common import ROOT, OUT

CHUNK = 2_000_000


def main():
    sub_path = OUT / "submission.csv"
    tmpl_path = ROOT / "sample_submission.csv"
    cols = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]

    n_sub = n_tmpl = 0
    ids_ok = True
    problems = []
    stats = {t: {"n": 0, "nonzero": 0} for t in ("state", "queue", "odme")}
    it_s = pd.read_csv(sub_path, chunksize=CHUNK)
    it_t = pd.read_csv(tmpl_path, chunksize=CHUNK, usecols=["submission_id", "task"])
    for s, t in zip(it_s, it_t):
        if list(s.columns) != cols:
            problems.append(f"column mismatch: {list(s.columns)}")
            break
        if len(s) != len(t) or not (s.submission_id.to_numpy() == t.submission_id.to_numpy()).all():
            ids_ok = False
        if not (s.task.to_numpy() == t.task.to_numpy()).all():
            problems.append("task column does not line up with the template")
        if s.isna().any().any():
            problems.append(f"{int(s.isna().any(1).sum())} rows contain empty cells")
        q = s.queue_pred
        if not q.isin([0, 1]).all():
            problems.append("queue_pred is not strictly 0/1")
        if (s.path_flow < 0).any() or not np.isfinite(s.path_flow).all():
            problems.append("path_flow has negative or non-finite values")
        if (s.speed_kmh < 0).any() or (s.flow_vph < 0).any():
            problems.append("negative speed or flow")
        for task, vcol in (("state", "speed_kmh"), ("queue", "queue_pred"), ("odme", "path_flow")):
            m = s.task == task
            stats[task]["n"] += int(m.sum())
            stats[task]["nonzero"] += int((s.loc[m, vcol] != 0).sum())
        n_sub += len(s)
        n_tmpl += len(t)
    print(f"rows: submission={n_sub:,} template={n_tmpl:,} ids_aligned={ids_ok}")
    for task, d in stats.items():
        frac = d["nonzero"] / max(d["n"], 1)
        print(f"  {task:6s}: {d['n']:>9,} rows, {frac:6.2%} non-zero in its value column")
    if problems:
        print("PROBLEMS:")
        for p in dict.fromkeys(problems):
            print("  -", p)
    else:
        print("no problems found")


if __name__ == "__main__":
    main()
