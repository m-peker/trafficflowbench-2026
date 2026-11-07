"""Set queue_pred = 1 on every horizon cell the release marks ineligible.

Used by runs/s54_elig-probe. The released `is_score_eligible` flag lives in
mainline_states_masked and, on train, equals true eligibility on every horizon
cell checked (57k/57k); on validation and private it carries the same marginal
rate (0.758-0.765 against the 0.764 baseline), so it is not degenerate there.
"""
import glob
import numpy as np
import pandas as pd

from common import ROOT, QUEUE_PANELS

SPLITS = ("validation", "private")


def eligibility():
    """(window_id, timestamp, link_id) -> is_score_eligible for scored horizons."""
    frames = []
    for panel in QUEUE_PANELS:
        for split in SPLITS:
            wi = pd.read_csv(ROOT / "task2" / panel / split / "window_index.csv")
            wi["forecast_origin"] = pd.to_datetime(wi.forecast_origin, utc=True)
            for day, grp in wi.groupby(wi.forecast_origin.dt.strftime("%Y_%m_%d")):
                f = glob.glob(str(ROOT / "corridors" / panel / split /
                                  "mainline_states_masked" / "*" / f"*{day}.parquet"))
                assert len(f) == 1, f"{panel}/{split}/{day}: {len(f)} files"
                d = pd.read_parquet(f[0], columns=["timestamp", "link_id",
                                                   "is_score_eligible"])
                d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
                d["link_id"] = d.link_id.astype(str)
                for w in grp.itertuples(index=False):
                    T = w.forecast_origin
                    hz = d[(d.timestamp > T) &
                           (d.timestamp <= T + pd.Timedelta(minutes=30))].copy()
                    hz["window_id"] = w.window_id
                    frames.append(hz[["window_id", "timestamp", "link_id",
                                      "is_score_eligible"]])
            print(f"  [{panel}/{split}] {len(wi)} windows", flush=True)
    return pd.concat(frames, ignore_index=True)


def flip(src_csv, out_csv):
    sub = pd.read_csv(src_csv, dtype={"link_id": str})
    sub["timestamp"] = pd.to_datetime(sub.timestamp, utc=True)
    el = eligibility()
    m = sub.merge(el, on=["window_id", "timestamp", "link_id"], how="left")
    assert len(m) == len(sub), f"{len(m)} vs {len(sub)}"
    miss = m.is_score_eligible.isna().sum()
    assert miss == 0, f"{miss} rows without an eligibility flag"
    before = int(m.queue_pred.sum())
    m.loc[m.is_score_eligible == 0, "queue_pred"] = 1
    after = int(m.queue_pred.sum())
    out = m[["window_id", "timestamp", "link_id", "queue_pred"]].copy()
    out["timestamp"] = out.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["queue_pred"] = out.queue_pred.astype(int)
    out.to_csv(out_csv, index=False)
    print(f"positives {before:,} -> {after:,} of {len(out):,} cells "
          f"({after / len(out):.3f}); ineligible share "
          f"{(m.is_score_eligible == 0).mean():.3f}")
