"""Shared helpers for TrafficFlowBench."""
import nice  # noqa: F401  - first, so its thread caps land before numpy
from pathlib import Path
import json, os
import numpy as np
import pandas as pd

# TFB_DATA: the unzipped competition release; TFB_WORK: caches, models (out/) and outputs
ROOT = Path(os.environ.get("TFB_DATA", r"d:/BigData Competition/2026-ieee-big-data-traffic-flow-bench"))
WORK = Path(os.environ.get("TFB_WORK", Path(__file__).resolve().parent))
OUT = WORK / "out"
OUT.mkdir(parents=True, exist_ok=True)

PANELS = [p["corridor_id"] for p in
          json.loads((ROOT / "config" / "corridors.json").read_text())["panels"]]
FAMILY = {p["corridor_id"]: p["family_id"] for p in
          json.loads((ROOT / "config" / "corridors.json").read_text())["panels"]}
QUEUE_PANELS = [p for p in PANELS if not p.startswith("D12_I405")]


def net_dir(panel):
    return ROOT / "corridors" / panel / "network"


def links_table(panel):
    """link_id -> length_km, lanes, free_speed_kmh, capacity_vph (+ milepost order)."""
    df = pd.read_csv(net_dir(panel) / "links.csv", dtype={"link_id": str})
    return df


def station_lanes(panel):
    """Mirror of the organizer lane lookup: station_id and link_id -> lanes."""
    lanes = {}
    df = pd.read_csv(net_dir(panel) / "links.csv", dtype={"link_id": str})
    ln = pd.to_numeric(df["lanes"], errors="coerce")
    ok = ln.notna() & (ln >= 1) & (ln <= 16)
    for lid, v in zip(df.link_id[ok], ln[ok]):
        lanes[str(lid)] = float(v)
    fd = pd.read_csv(net_dir(panel) / "fd_parameters.csv", dtype={"link_id": str})
    ln = pd.to_numeric(fd["lanes"], errors="coerce")
    ok = ln.notna() & (ln >= 1) & (ln <= 16)
    for lid, v in zip(fd.link_id[ok], ln[ok]):
        lanes[str(lid)] = float(v)
    return lanes


def lane_vector(stations, links, lanes_map):
    out = np.ones(len(links), dtype=float)
    st = np.asarray(stations, dtype=object)
    lk = np.asarray(links, dtype=object)
    for i in range(len(out)):
        s, l = str(st[i]), str(lk[i])
        if s in lanes_map:
            out[i] = lanes_map[s]
        elif l in lanes_map:
            out[i] = lanes_map[l]
    return out


def mainline_files(panel, split, masked=True):
    sub = "mainline_states_masked" if masked else "mainline_states"
    return sorted((ROOT / "corridors" / panel / split / sub).glob("**/*.parquet"))


def ramp_files(panel, split):
    return sorted((ROOT / "corridors" / panel / split / "ramp_states").glob("**/*.parquet"))
