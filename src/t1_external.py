"""Score the public LightGBM Task 1 pipeline against ours, on held-out days.

`Public-Code/` (MIT, Mayank Sharma) scores 0.86144 against our 0.86168, so its
Task 1 is worth testing as an ensemble partner: it is LightGBM over a 2-D grid
with spatial-interpolation and diurnal-profile features, a different family from
our conv nets. The worry is that it is the *same* family as our tree side, which
the board already down-weights to 0.15 - adding it would then just raise the
tree weight, which s47/s48 measured as harmful.

This settles it instead of guessing. Their pipeline is trained with the last
HOLD days of train excluded, then asked for those days' blanked cells; ours is
predicted on the same cells; both are scored against `mainline_states` truth
with the official per-regime metric, and the blend is swept.

    T1_HOLD=7 python t1_external.py [panel ...]
"""
import nice  # noqa: F401
import glob
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from common import ROOT
import t1_nn as NN

PUB = Path(r"D:\BigData Competition\Public-Code")
HOLD = int(os.environ.get("T1_HOLD", "7"))
NETS = os.environ.get("T1_NETS", "f1,h1,i1,j1,m1,n1").split(",")
NETW = float(os.environ.get("T1_NETW", "0.85"))
PANELS = sys.argv[1:] or ["D7_I10_E", "D7_I10_W", "D7_I210_E"]
BLENDS = [float(x) for x in os.environ.get("T1_BLENDS", "0,0.1,0.2,0.3,0.5,0.7,1.0").split(",")]


def load_theirs():
    """Import their module with DATA_DIR pointed at our release."""
    sys.path.insert(0, str(PUB / "scripts"))
    import importlib
    mod = importlib.import_module("build_task1_lightgbm")
    mod.DATA_DIR = ROOT
    return mod


def held_out_days(panel):
    """The last HOLD train days, as 'YYYY_MM_DD' strings in date order."""
    days = set()
    for r in ("R1", "R2", "R3"):
        for f in glob.glob(str(ROOT / "corridors" / panel / "train" /
                               "mainline_states_masked" / f"mask_regime={r}" / "*.parquet")):
            days.add(re.search(r"(\d{4}_\d{2}_\d{2})", f).group(1))
    return sorted(days)[-HOLD:]


def their_predictions(mod, panel, hold):
    """Train excluding the held-out days, then predict their blanked cells."""
    orig_glob = mod.glob.glob

    def filtered(pattern, **kw):
        files = orig_glob(pattern, **kw)
        if "mainline_states_masked" in pattern and "train" in pattern:
            files = [f for f in files if not any(d in f for d in hold)]
        return files

    mod.glob.glob = filtered
    try:
        model_s, model_f, feats, meta, profiles = mod.train_corridor_models(panel, n_train_days=36)
    finally:
        mod.glob.glob = orig_glob

    rows = []
    for day in hold:
        f = glob.glob(str(ROOT / "corridors" / panel / "train" /
                          "mainline_states_masked" / "*" / f"*{day}.parquet"))
        assert len(f) == 1, f"{panel}/{day}: {len(f)} files"
        df_v = pd.read_parquet(f[0])
        st = df_v[["station_id", "milepost"]].drop_duplicates().sort_values("milepost")
        fdf = mod.extract_features_from_day(df_v, st.station_id.tolist(), meta, profiles)
        need = (df_v["is_missing"] == 1) | df_v.speed_kmh.isna() | df_v.flow_vph.isna()
        sub = fdf[need.to_numpy()].copy()
        if sub.empty:
            continue
        X = sub[feats]
        sub["ext_speed"] = np.clip(model_s.predict(X), 5.0, sub.free_speed.values * 1.05)
        sub["ext_flow"] = np.clip(model_f.predict(X) * sub.lanes.values, 50.0,
                                  sub.capacity.values * 1.02)
        rows.append(sub[["timestamp", "link_id", "mask_regime", "ext_speed", "ext_flow"]])
    return pd.concat(rows, ignore_index=True)


def our_predictions(panel, hold):
    """Our shipped chain on the same cells: 0.85 * mean(nets) + 0.15 * trees."""
    import ens_search as E
    from t1_phys_compare import W
    import t1feat
    os.environ.setdefault("T1_TOD", "const")
    G = t1feat.load_grid(panel, "train", masked=True)
    day_idx = {d: i for i, d in enumerate(G["days"])}
    keep = np.zeros(G["nT"], bool)
    for d in hold:
        keep[day_idx[d.replace("_", "-")] * 288:(day_idx[d.replace("_", "-")] + 1) * 288] = True

    vs, qs = [], []
    for n in NETS:
        _, sp, fl, _ = NN.predict_grid(NN.load_net(n), panel, "train")
        vs.append(sp); qs.append(fl)
    v_net = np.mean(vs, 0); q_net = np.mean(qs, 0)

    P = E.Panel(panel)
    nTb = P.obs_sp.shape[0]
    off = G["nT"] - nTb
    tj, ti = P.tj, P.ti
    v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[tj] * 1.05)
    k = P.mix(W, "density")
    q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v_t)
                  * P.lanes[tj], 5.0, P.cap[tj] * 1.15)
    v = NETW * v_net[ti + off, tj] + (1 - NETW) * v_t
    q = NETW * q_net[ti + off, tj] + (1 - NETW) * q_t

    t0 = pd.Timestamp(G["days"][0] + " 00:00:00", tz="UTC")
    rows = pd.DataFrame({
        "timestamp": t0 + pd.to_timedelta((ti + off) * 5, unit="m"),
        "link_id": np.array(G["link_ids"], dtype=object)[tj],
        "our_speed": v, "our_flow": q,
        "v_true": P.v_t, "q_true_lane": P.q_t, "lanes": P.lanes[tj],
    })
    return rows[keep[ti + off]]


def official(sub, vcol, qcol):
    """S_state as TASK1_GUIDE defines it: macro-mean of exp(-e) over regimes."""
    S = {}
    for r, g in sub.groupby("mask_regime"):
        rv = float(np.sqrt(np.mean((g[vcol] - g.v_true) ** 2)))
        rq = float(np.sqrt(np.mean((g[qcol] / g.lanes - g.q_true_lane) ** 2)))
        S[r] = float(np.exp(-(0.54 * rv / 25.0 + 0.46 * rq / 600.0)))
    return float(np.mean(list(S.values()))), S


def main():
    mod = load_theirs()
    parts = []
    for panel in PANELS:
        hold = held_out_days(panel)
        print(f"[{panel}] held out {hold[0]}..{hold[-1]}", flush=True)
        ext = their_predictions(mod, panel, hold)
        ours = our_predictions(panel, hold)
        ours["timestamp"] = pd.to_datetime(ours.timestamp, utc=True)
        ext["timestamp"] = pd.to_datetime(ext.timestamp, utc=True)
        ext["link_id"] = ext.link_id.astype(str)
        ours["link_id"] = ours.link_id.astype(str)
        m = ours.merge(ext, on=["timestamp", "link_id"], how="inner")
        print(f"  ours {len(ours):,} cells, theirs {len(ext):,}, matched {len(m):,}", flush=True)
        parts.append(m)
    sub = pd.concat(parts, ignore_index=True)

    print(f"\n{len(sub):,} common cells over {len(PANELS)} panels, last {HOLD} train days")
    print(f"{'blend':>6s} {'S_state':>9s}   (0 = ours, 1 = theirs)")
    for b in BLENDS:
        v = (1 - b) * sub.our_speed + b * sub.ext_speed
        q = (1 - b) * sub.our_flow + b * sub.ext_flow
        sub["_v"], sub["_q"] = v, q
        s, per = official(sub, "_v", "_q")
        tag = "  <- shipped" if b == 0 else ""
        print(f"{b:6.2f} {s:9.5f}   " + " ".join(f"{k}={x:.5f}" for k, x in sorted(per.items())) + tag)


if __name__ == "__main__":
    main()
