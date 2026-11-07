"""LightGBM residual reconstruction with temporal and spatial context."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import RELEASE, OUT, THREADS
from base_panel import Panel as BasePanel

TRAIN_MONTHS = ("2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01")
VALID_MONTH = "2031-02"
SEED = 20261001
RATE = np.array([.20, .30, .50], dtype=np.float32)


class Panel(BasePanel):
    """Add milepost order from static, public training metadata."""

    def __init__(self, name, panel_no):
        super().__init__(name, panel_no)
        sample = sorted((RELEASE / "corridors" / name / "train" / "mainline_states" /
                         "year_month=2030-06").glob("*.parquet"))[0]
        location = pd.read_parquet(sample, columns=["link_id", "milepost"]).drop_duplicates("link_id")
        location["link_id"] = location.link_id.astype(str)
        milepost = location.set_index("link_id").reindex(self.links).milepost.to_numpy(dtype=np.float32)
        if not np.isfinite(milepost).all():
            raise ValueError(f"Missing mileposts for {name}")
        self.milepost = milepost
        self.sorted_links = np.argsort(milepost)
        self.link_rank = np.empty(len(self.links), dtype=np.int32)
        self.link_rank[self.sorted_links] = np.arange(len(self.links))


def all_panels():
    config = json.loads((RELEASE / "config" / "corridors.json").read_text())
    return [x["corridor_id"] for x in config["panels"]]


def month_frame(panel, month, split="train"):
    directory = RELEASE / "corridors" / panel / split / "mainline_states"
    if split != "train":
        directory = RELEASE / "corridors" / panel / split / "mainline_states_masked"
        paths = sorted(directory.glob("**/*.parquet"))
    else:
        paths = sorted(directory.glob(f"year_month={month}/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No state data: {panel} {split} {month}")
    cols = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
    frame = pd.concat((pd.read_parquet(path, columns=cols) for path in paths), ignore_index=True)
    frame["link_id"] = frame.link_id.astype(str)
    frame["station_id"] = frame.station_id.astype(str)
    if frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError(f"Duplicate link/time for {panel} {month}")
    return frame


def matrices(frame, panel):
    index = pd.Index(sorted(frame.timestamp.unique()))
    cols = panel.links
    speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=cols).to_numpy(dtype=np.float32)
    flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=cols).to_numpy(dtype=np.float32)
    eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=cols).fillna(0).to_numpy(dtype=bool)
    return index, speed, flow, eligible


def simulated_mask(panel_name, times, speed, flow, eligible):
    dates = pd.Index([text[:10] for text in times])
    result = np.zeros(speed.shape, dtype=bool)
    regimes = np.empty(speed.shape[0], dtype=np.uint8)
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|{panel_name}|{date}".encode(), digest_size=8).digest(), "big")
        regime = seed % 3
        rng = np.random.default_rng(seed)
        regimes[rows] = regime
        result[rows] = (rng.random((len(rows), speed.shape[1])) < RATE[regime]) & eligible[rows] \
                       & np.isfinite(speed[rows]) & np.isfinite(flow[rows])
    return result, regimes


def sampled_targets(mask, times, per_day):
    dates = pd.Index([text[:10] for text in times])
    row_out, col_out = [], []
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        coords = np.argwhere(mask[rows])
        if not len(coords):
            continue
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|sample|{date}|{mask.shape[1]}".encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        chosen = coords[rng.choice(len(coords), size=min(per_day, len(coords)), replace=False)]
        row_out.append(rows[chosen[:, 0]])
        col_out.append(chosen[:, 1])
    return np.concatenate(row_out), np.concatenate(col_out)


def graph_mean(graph, values):
    observed = np.isfinite(values).astype(np.float32)
    count = (graph @ observed.T).T
    mean = (graph @ np.nan_to_num(values, nan=0.0).T).T / np.maximum(count, 1)
    mean = np.where(count > 0, mean, np.nan)
    return mean, count


def compute_features(panel, panel_no, times, observed_speed, observed_flow, rows, cols, regime):
    n_times, n_links = observed_speed.shape
    temporal_speed = pd.DataFrame(observed_speed).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_flow = pd.DataFrame(observed_flow).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_speed = np.where(np.isfinite(temporal_speed), temporal_speed, panel.vf[None, :])
    temporal_flow = np.where(np.isfinite(temporal_flow), temporal_flow, .6 * panel.cap[None, :])

    graph = (panel.incoming + panel.outgoing).tocsr()
    graph.data[:] = 1.0
    spatial_flow, spatial_count = graph_mean(graph, observed_flow)
    flow_base = np.where(spatial_count > 0, .5 * temporal_flow + .5 * spatial_flow, temporal_flow)
    up_speed, up_count = graph_mean(panel.incoming, observed_speed)
    down_speed, down_count = graph_mean(panel.outgoing, observed_speed)
    up_flow, _ = graph_mean(panel.incoming, observed_flow)
    down_flow, _ = graph_mean(panel.outgoing, observed_flow)

    valid = np.isfinite(observed_speed)
    positions = np.arange(n_times, dtype=np.int32)[:, None]
    prev_index = np.maximum.accumulate(np.where(valid, positions, -1), axis=0)
    next_index = np.minimum.accumulate(np.where(valid, positions, n_times)[::-1], axis=0)[::-1]
    p = prev_index[rows, cols]
    n = next_index[rows, cols]
    prev_v = observed_speed[np.maximum(p, 0), cols].copy()
    next_v = observed_speed[np.minimum(n, n_times - 1), cols].copy()
    prev_q = observed_flow[np.maximum(p, 0), cols].copy()
    next_q = observed_flow[np.minimum(n, n_times - 1), cols].copy()
    prev_v[p < 0] = np.nan
    prev_q[p < 0] = np.nan
    next_v[n >= n_times] = np.nan
    next_q[n >= n_times] = np.nan
    prev_gap = np.where(p >= 0, rows - p, 300).astype(np.float32)
    next_gap = np.where(n < n_times, n - rows, 300).astype(np.float32)

    stamp = pd.to_datetime(times, utc=True)
    tod = stamp.hour.to_numpy(dtype=np.float32) + stamp.minute.to_numpy(dtype=np.float32) / 60
    dow = stamp.dayofweek.to_numpy(dtype=np.float32)
    lanes = np.maximum(panel.lanes[cols], 1)
    base_speed = temporal_speed[rows, cols]
    base_flow = flow_base[rows, cols] / lanes
    temporal_flow_lane = temporal_flow[rows, cols] / lanes
    spatial_flow_lane = spatial_flow[rows, cols] / lanes
    up_q = up_flow[rows, cols] / lanes
    down_q = down_flow[rows, cols] / lanes
    regime_vector = regime if len(regime) == len(rows) else regime[rows]
    features = [
        np.full(len(rows), panel_no), cols, regime_vector,
        tod[rows], np.sin(2*np.pi*tod[rows]/24), np.cos(2*np.pi*tod[rows]/24), dow[rows],
        panel.vf[cols], panel.cap[cols] / lanes, lanes, panel.length[cols],
        base_speed, base_flow, temporal_flow_lane, spatial_flow_lane,
        base_speed / panel.vf[cols], base_flow * lanes / panel.cap[cols],
        prev_v, next_v, prev_q / lanes, next_q / lanes, prev_gap, next_gap,
        next_v - prev_v, (next_q - prev_q) / lanes,
        up_speed[rows, cols], down_speed[rows, cols], up_q, down_q,
        up_count[rows, cols], down_count[rows, cols], spatial_count[rows, cols],
        up_speed[rows, cols] - down_speed[rows, cols], up_q - down_q,
    ]
    for offset in (-2, -1, 1, 2):
        ranks = panel.link_rank[cols] + offset
        valid_neighbor = (ranks >= 0) & (ranks < n_links)
        neighbor = panel.sorted_links[np.clip(ranks, 0, n_links-1)]
        neighbor_lanes = np.maximum(panel.lanes[neighbor], 1)
        same_time_v = observed_speed[rows, neighbor]
        same_time_q = observed_flow[rows, neighbor] / neighbor_lanes
        filled_v = temporal_speed[rows, neighbor]
        filled_q = temporal_flow[rows, neighbor] / neighbor_lanes
        features.extend((
            np.where(valid_neighbor, np.abs(panel.milepost[neighbor]-panel.milepost[cols]), np.nan),
            np.where(valid_neighbor, same_time_v, np.nan),
            np.where(valid_neighbor, same_time_q, np.nan),
            np.where(valid_neighbor, filled_v, np.nan),
            np.where(valid_neighbor, filled_q, np.nan),
        ))
    X = np.column_stack(features).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, base_speed, base_flow
