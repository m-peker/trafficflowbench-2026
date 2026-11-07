"""Cache one net's raw speed/flow[/density] on every submission row, once.

Five different Task 1 strategies (swap, fractional add, density-head swap,
robust trimmed mean, ...) all recombine the SAME underlying net predictions in
different ways. Predicting each net once and caching it, instead of once per
strategy, is what makes building five submissions overnight tractable.

    python t1_field_cache.py f1 h1 i1 j1 m1 n1 v1 dn1 dn2

writes `_field_<net>.csv` (panel,timestamp,link_id,speed_kmh,flow_vph[,density])
aligned exactly to the submission's own rows, for every panel and both splits.
"""
import nice  # noqa: F401
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
sys.path.insert(0, str(HERE))
import t1_nn as NN                                     # noqa: E402
from common import ROOT                                # noqa: E402


def field_for(name):
    out = RUNS / f"_field_{name}.csv"
    net = NN.load_net(name)
    n_out = net.out[-1].out_channels
    parts = []
    for panel in NN.PANELS:
        for split in ("validation", "private"):
            key = pd.read_csv(ROOT / "task1" / panel / split / "sample_submission_state.csv",
                              dtype={"link_id": str, "station_id": str})
            key = key.drop(columns=["speed_kmh", "flow_vph"])
            ts = pd.to_datetime(key.timestamp)
            if n_out >= 3:
                G, sp, fl, _, dens = NN.predict_grid(net, panel, split)
            else:
                G, sp, fl, _ = NN.predict_grid(net, panel, split)
                dens = None
            di = {d: i for i, d in enumerate(G["days"])}
            li = {l: i for i, l in enumerate(G["link_ids"])}
            row = (ts.dt.strftime("%Y-%m-%d").map(di).to_numpy() * 288 +
                  (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy())
            col = key.link_id.map(li).to_numpy()
            assert not np.isnan(row.astype(float)).any() and not pd.isna(col).any(), \
                f"{name}/{panel}/{split}: unmapped row/col"
            row = row.astype(int); col = col.astype(int)
            k = key.copy()
            k["speed_kmh"] = sp[row, col]
            k["flow_vph"] = fl[row, col]
            if dens is not None:
                k["density"] = dens[row, col]
            parts.append(k[["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"] +
                          (["density"] if dens is not None else [])])
        print(f"  [{name}/{panel}] done", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out.name}: {len(df):,} rows{' (with density)' if dens is not None else ''}",
         flush=True)


def main():
    for name in sys.argv[1:]:
        field_for(name)


if __name__ == "__main__":
    main()
