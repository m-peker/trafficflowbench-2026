import sys, numpy as np, pandas as pd
from qwin import load_panel, enumerate_windows, HIST, HOR

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
D = load_panel(panel)
onset, ongoing = enumerate_windows(D, per_cond=0)
print(f"{panel}: all onset origins={len(onset)}, ongoing={len(ongoing)}, nL={D['nL']}")

q, elig = D["q"], D["elig"]
rows = []
for T in onset:
    sl = slice(T + 1, T + 1 + HOR)
    t = q[sl] & elig[sl]
    n_cells = int(t.sum())
    n_links = int(t.any(0).sum())
    first = int(np.argmax(t.any(1)))          # first horizon step with any queue
    rows.append((T, n_cells, n_links, first, int(elig[sl].sum())))
r = pd.DataFrame(rows, columns=["T", "cells", "links", "first_step", "elig"])
print(r[["cells", "links", "first_step"]].describe().to_string())
print("\nfirst_step distribution:"); print(r.first_step.value_counts().sort_index())
print("\ntruth density (queued/eligible):", r.cells.sum() / r.elig.sum())

# which links appear
cnt = np.zeros(D["nL"])
for T in onset:
    sl = slice(T + 1, T + 1 + HOR)
    cnt += (q[sl] & elig[sl]).any(0)
o = np.argsort(-cnt)
print("\ntop onset links:", [(D['link_ids'][i], int(cnt[i])) for i in o[:12]])
print("share of onset windows covered by top-2 links:",
      float(cnt[o[:2]].sum()) / max(cnt.sum(), 1))
print("cumulative link coverage of onset cells: top5=%.3f top10=%.3f top20=%.3f" % (
    cnt[o[:5]].sum()/cnt.sum(), cnt[o[:10]].sum()/cnt.sum(), cnt[o[:20]].sum()/cnt.sum()))

# time of day of onset origins
tod = (onset % 288)
print("\nonset origin hour distribution:")
print(pd.Series(tod // 12).value_counts().sort_index().to_string())
