"""Join the three per-task prediction files onto the upload template.

Keys are folded into a single int64 per row through small string vocabularies,
so 6.7 million state rows cost ~110 MB instead of several GB of Python strings.
Both the prediction files and submission_key.csv are streamed in chunks.
"""
import sys
import numpy as np
import pandas as pd
from common import ROOT, OUT

CHUNK = 400_000
OUT_COLS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
KEYS = {"state": ["panel", "timestamp", "link_id"],
        "queue": ["window_id", "timestamp", "link_id"],
        "odme": ["panel", "departure_time", "path_id"]}
VALS = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
FILES = {"state": OUT / "state_submission.csv",
         "queue": OUT / "queue_submission.csv",
         "odme": OUT / "odme_submission.csv"}


class Vocab:
    """Growing string -> small int map, one per key column."""

    def __init__(self):
        self.m = {}

    def add(self, s):
        for v in s.unique():
            if v not in self.m:
                self.m[v] = len(self.m)
        return s.map(self.m).to_numpy(np.int64)

    def lookup(self, s):
        return s.map(self.m).fillna(-1).to_numpy(np.int64)

    def __len__(self):
        return len(self.m)


def fold(codes, widths):
    key = np.zeros(len(codes[0]), np.int64)
    bad = np.zeros(len(codes[0]), bool)
    for c, w in zip(codes, widths):
        bad |= c < 0
        key = key * w + np.maximum(c, 0)
    key[bad] = -1
    return key


def load_task(task):
    """Stream our predictions into sorted int64 keys plus value arrays."""
    vocabs = [Vocab() for _ in KEYS[task]]
    keys, vals = [], []
    dtypes = {c: str for c in KEYS[task]}
    for ch in pd.read_csv(FILES[task], chunksize=CHUNK, dtype=dtypes):
        codes = [v.add(ch[c]) for v, c in zip(vocabs, KEYS[task])]
        keys.append(codes)
        vals.append(np.column_stack([pd.to_numeric(ch[c], errors="coerce")
                                     .to_numpy(np.float32) for c in VALS[task]]))
    widths = [max(len(v), 1) for v in vocabs]
    key = np.concatenate([fold(c, widths) for c in keys])
    val = np.concatenate(vals)
    del keys, vals
    order = np.argsort(key, kind="stable")
    key, val = key[order], val[order]
    keep = np.concatenate([[True], key[1:] != key[:-1]])
    if not keep.all():
        print(f"  {task}: dropped {int((~keep).sum())} duplicate keys")
        key, val = key[keep], val[keep]
    print(f"loaded {task}: {len(key):,} rows, vocab sizes {widths}", flush=True)
    return vocabs, widths, key, val


def main():
    tables = {t: load_task(t) for t in ("state", "queue", "odme")}
    usecols = ["submission_id", "task", "panel", "timestamp", "link_id",
               "window_id", "departure_time", "path_id"]
    dest = OUT / "submission.csv"
    first, total = True, 0
    stats = {t: [0, 0] for t in tables}
    for chunk in pd.read_csv(ROOT / "submission_key.csv", chunksize=CHUNK,
                             usecols=usecols, dtype=str):
        task_col = chunk.task.to_numpy()
        out = pd.DataFrame({"submission_id": chunk.submission_id.astype("int64"),
                            "task": task_col})
        for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
            out[c] = 0.0
        for task, (vocabs, widths, key, val) in tables.items():
            rows = task_col == task
            n = int(rows.sum())
            if not n:
                continue
            sub = chunk.loc[rows, KEYS[task]]
            codes = [v.lookup(sub[c]) for v, c in zip(vocabs, KEYS[task])]
            want = fold(codes, widths)
            pos = np.searchsorted(key, want)
            hit = (pos < len(key)) & (want >= 0)
            pos_c = np.clip(pos, 0, len(key) - 1)
            hit &= key[pos_c] == want
            stats[task][0] += int(hit.sum())
            stats[task][1] += n
            for i, c in enumerate(VALS[task]):
                v = np.where(hit, val[pos_c, i], 0.0)
                out.loc[rows, c] = v
        out["queue_pred"] = out.queue_pred.round().clip(0, 1).astype(np.int8)
        out[OUT_COLS].to_csv(dest, mode="w" if first else "a", header=first,
                             index=False, float_format="%.4f")
        first = False
        total += len(out)
        print(f"\r  {total:,} rows", end="", flush=True)
        del chunk, out
    print()
    for task, (m, n) in stats.items():
        print(f"{task}: matched {m:,} / {n:,} ({100 * m / max(n, 1):.2f}%)")
    print("wrote", dest, f"{total:,} rows")


if __name__ == "__main__":
    main()
