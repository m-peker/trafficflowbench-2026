"""Second-stage residual model for Task 1: correct the ensemble's own output.

Every Task 1 member solves the same problem independently - inpaint the masked
cells - and four different architectures plateau at the same S_state
(0.9508-0.9516). A residual model conditioned on the ensemble's OWN
reconstruction (its mean, how much the members disagree, distance to the
nearest real observation, mask regime) is solving a different problem: not
"what is the state", but "where is the ensemble wrong, and by how much". More
members of the same kind cannot buy that; s58/s59/s60 just measured members
seven through nine as negative.

Honesty about validation: all six nets share one 7-day held-out week
(T1_VAL_DAYS=7), so genuinely unseen (net, truth) pairs exist only there. This
does leave-one-family-out CV across that single week (5 folds, both panels of a
family held out together each time) - a test of generalising across corridors,
not across time, since there is no second held-out period to test against.
Treat the CV number as a directional signal, not proof: this is a *model-class*
change, and four of four previous model-class changes were mis-ranked by this
same kind of local harness (m1, p1, v1, s49's fifth seed) while every
inference-class change was predicted almost exactly (the tod fix to 0.00006).

    python t1_residual.py cv                      # leave-one-family-out
    python t1_residual.py build submission_tag     # fit on everything, correct
                                                    # validation/private, write
                                                    # runs/<tag>/out/state_submission.csv
"""
import nice  # noqa: F401
import glob
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import ens_search as E
from t1_phys_compare import score_fields, EVAL_LAST, W
from t1_interp import bracket
from common import ROOT, FAMILY
import t1_nn as NN
import t1feat

NETS = os.environ.get("T1_NETS", "f1,h1,i1,j1,m1,n1").split(",")
NETW = float(os.environ.get("T1_NETW", "0.85"))
PANELS = E.PANELS
PANEL_TO_CODE = {p: i for i, p in enumerate(sorted(PANELS))}
REG_CODE = {"R1": 1, "R2": 2, "R3": 3}
FEATURES = ["net_mean_v", "net_mean_q", "net_std_v", "net_std_q", "tree_v", "tree_q",
            "blend_v", "blend_q", "vf", "lanes", "cap", "length", "rank",
            "da", "db", "up_v", "dn_v", "local_std_v", "sin_tod", "cos_tod",
            "dow", "is_weekend", "regime", "panel_id"]
CAT_FEATURES = ["panel_id", "regime", "dow"]
os.environ.setdefault("T1_TOD", "const")


def rolling_std(x, half=3):
    """O(nT) rolling std along axis 0, via cumulative sums."""
    k = 2 * half + 1
    pad = np.pad(x, ((half, half), (0, 0)), mode="edge")
    c1 = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), pad]), 0)
    c2 = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), pad ** 2]), 0)
    n = k
    s1 = c1[k:] - c1[:-k]; s2 = c2[k:] - c2[:-k]
    return np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))


def day_regime(panel, split, day_dash):
    """mask_regime for one calendar day, read straight off its own column -
    works uniformly for train/validation/private, no folder-name parsing."""
    day_us = day_dash.replace("-", "_")
    f = glob.glob(str(ROOT / "corridors" / panel / split / "mainline_states_masked" /
                      "*" / f"*{day_us}.parquet"))
    assert len(f) == 1, f"{panel}/{split}/{day_dash}: {len(f)} files"
    return pd.read_parquet(f[0], columns=["mask_regime"]).mask_regime.iloc[0]


def held_out_rows(panel):
    """Feature/target rows for panel's 7-day train holdout, one row per masked
    cell, with the residual against truth as the target."""
    P = E.Panel(panel)
    nL = len(P.vf)
    nTb = P.obs_sp.shape[0]

    nets = {n: NN.load_net(n) for n in NETS}
    G0 = t1feat.load_grid(panel, "train", masked=True)
    nT_full = G0["nT"]
    lo = max(0, nT_full - nTb - 256)
    per_v, per_q = [], []
    for n in NETS:
        G, sp, fl, _ = NN.predict_grid(nets[n], panel, "train", t_lo=lo, t_hi=nT_full)
        off = G["nT"] - nTb
        per_v.append(sp[off:off + nTb]); per_q.append(fl[off:off + nTb])
    per_v = np.stack(per_v); per_q = np.stack(per_q)          # (nnet, nTb, nL)
    net_mean_v = per_v.mean(0); net_mean_q = per_q.mean(0)
    net_std_v = per_v.std(0); net_std_q = per_q.std(0)

    # mirrors t1_phys_compare.score_fields's construction exactly, including
    # the lane multiplication and clip - a bug here (missing *lanes) once made
    # the tree side ~4x too small and the "shipped" baseline flow RMSE looked
    # like 110 instead of ~28.
    tree_v_full = np.full((nTb, nL), np.nan)
    tree_q_full = np.full((nTb, nL), np.nan)
    v_tree = np.clip(P.mix(W, "speed"), 5.0, P.vf[P.tj] * 1.05)
    k = P.mix(W, "density")
    q_from_tree = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v_tree)
                          * P.lanes[P.tj], 5.0, P.cap[P.tj] * 1.15)
    tree_v_full[P.ti, P.tj] = v_tree
    tree_q_full[P.ti, P.tj] = q_from_tree

    blend_v = NETW * net_mean_v + (1 - NETW) * np.nan_to_num(tree_v_full)
    blend_q = NETW * net_mean_q + (1 - NETW) * np.nan_to_num(tree_q_full)
    lstd_v = rolling_std(blend_v)

    obs = np.isfinite(P.obs_sp)
    up_v, up_d, dn_v, dn_d = bracket(np.nan_to_num(P.obs_sp), obs)
    vf2 = np.repeat(P.vf[None, :], nTb, 0)

    day_of = np.arange(nTb) // 288
    ndays = nTb // 288
    assert nTb % 288 == 0, f"{panel}: nTb={nTb} not a whole number of days"
    days = G0["days"][-ndays:]
    reg_by_day = {d: day_regime(panel, "train", d) for d in days}
    row_regime = np.array([REG_CODE.get(reg_by_day.get(days[d], ""), 0) for d in day_of])
    dow = np.array([pd.Timestamp(days[d]).dayofweek for d in day_of])
    tsod = np.arange(nTb) % 288

    i, j = P.ti, P.tj
    lanes_j = P.lanes[j]
    df = pd.DataFrame({
        "net_mean_v": net_mean_v[i, j], "net_mean_q": net_mean_q[i, j],
        "net_std_v": net_std_v[i, j], "net_std_q": net_std_q[i, j],
        "tree_v": tree_v_full[i, j], "tree_q": tree_q_full[i, j],
        "blend_v": blend_v[i, j], "blend_q": blend_q[i, j],
        "vf": P.vf[j], "lanes": lanes_j, "cap": P.cap[j], "length": P.length[j],
        "rank": j / max(nL, 1),
        "da": np.minimum(up_d[i, j], 50), "db": np.minimum(dn_d[i, j], 50),
        "up_v": up_v[i, j] / np.maximum(vf2[i, j], 1.0),
        "dn_v": dn_v[i, j] / np.maximum(vf2[i, j], 1.0),
        "local_std_v": lstd_v[i, j], "sin_tod": np.sin(2 * np.pi * tsod[i] / 288),
        "cos_tod": np.cos(2 * np.pi * tsod[i] / 288), "dow": dow[i],
        "is_weekend": (dow[i] >= 5).astype(int), "regime": row_regime[i],
        "panel_id": PANEL_TO_CODE[panel],
        "true_v": P.v_t, "true_q": P.q_t, "ok": P.ok,
    })
    return df[df.ok].drop(columns="ok")


def cache_path(panel):
    return Path("out") / f"resid_holdout_{panel}.parquet"


def get_holdout(panel):
    f = cache_path(panel)
    if f.exists():
        return pd.read_parquet(f)
    df = held_out_rows(panel)
    df.to_parquet(f)
    return df


def official_S(df, v, q):
    S = {}
    for r, g in df.groupby("regime"):
        if not len(g):
            continue
        rv = float(np.sqrt(np.mean((g[v] - g.true_v) ** 2)))
        rq = float(np.sqrt(np.mean((g[q] / g.lanes - g.true_q) ** 2)))
        S[r] = float(np.exp(-(0.54 * rv / 25.0 + 0.46 * rq / 600.0)))
    return float(np.mean(list(S.values()))), S


def fit(train_df, target):
    # panel_id/regime/dow are plain small-integer codes, kept consistent between
    # training and inference via PANEL_TO_CODE/REG_CODE rather than pandas
    # 'category' dtype (whose category list could otherwise silently differ
    # between the CV frame and the validation/private frame built later).
    ds = lgb.Dataset(train_df[FEATURES], label=train_df[target],
                     categorical_feature=CAT_FEATURES, free_raw_data=False)
    params = dict(objective="regression", num_leaves=31, min_child_samples=200,
                  learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1)
    return lgb.train(params, ds, num_boost_round=300)


def cv():
    panels = sys.argv[2:] or PANELS
    print(f"loading {len(panels)} panels' held-out week...", flush=True)
    frames = []
    for p in panels:
        df = get_holdout(p)
        df["family"] = FAMILY[p]
        frames.append(df)
        print(f"  {p}: {len(df):,} cells", flush=True)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["resid_v"] = all_df.true_v - all_df.blend_v
    all_df["resid_q"] = all_df.true_q - all_df.blend_q / all_df.lanes

    fams = sorted(set(all_df.family))
    corrected_v = np.full(len(all_df), np.nan)
    corrected_q = np.full(len(all_df), np.nan)
    for fam in fams:
        te = all_df.family == fam
        tr = ~te
        mv = fit(all_df[tr], "resid_v")
        mq = fit(all_df[tr], "resid_q")
        corrected_v[te] = all_df.loc[te, "blend_v"] + mv.predict(all_df.loc[te, FEATURES])
        corrected_q[te] = (all_df.loc[te, "blend_q"] +
                           mq.predict(all_df.loc[te, FEATURES]) * all_df.loc[te, "lanes"])
        print(f"  fold held out {fam}: {te.sum():,} cells", flush=True)

    all_df["corr_v"] = np.clip(corrected_v, 5.0, all_df.vf * 1.05)
    all_df["corr_q"] = np.clip(corrected_q, 50.0, all_df.cap * 1.15)

    s_base, per_base = official_S(all_df, "blend_v", "blend_q")
    s_corr, per_corr = official_S(all_df, "corr_v", "corr_q")
    print(f"\nleave-one-family-out CV, {len(panels)} panels, {len(all_df):,} cells")
    print(f"{'':10s} " + " ".join(f"R{r}={per_base.get(r, float('nan')):.5f}" for r in (1, 2, 3)) + f"  S_state={s_base:.5f}")
    print(f"{'shipped':10s}")
    print(f"{'corrected':10s} " + " ".join(f"R{r}={per_corr.get(r, float('nan')):.5f}" for r in (1, 2, 3)) + f"  S_state={s_corr:.5f}")
    print(f"\ndelta S_state: {s_corr - s_base:+.5f}")

    rv0 = np.sqrt(np.mean((all_df.blend_v - all_df.true_v) ** 2))
    rv1 = np.sqrt(np.mean((all_df.corr_v - all_df.true_v) ** 2))
    rq0 = np.sqrt(np.mean((all_df.blend_q / all_df.lanes - all_df.true_q) ** 2))
    rq1 = np.sqrt(np.mean((all_df.corr_q / all_df.lanes - all_df.true_q) ** 2))
    print(f"speed RMSE {rv0:.4f} -> {rv1:.4f}   flow/lane RMSE {rq0:.3f} -> {rq1:.3f}")

    # density level, as a proxy for S_LWR direction (not the full dN metric,
    # which needs adjacent-cell structure this row-wise dataframe does not keep)
    N_true = all_df.true_q * all_df.lanes / np.maximum(all_df.true_v, 1.0) * all_df.length
    N_base = all_df.blend_q / np.maximum(all_df.blend_v, 1.0) * all_df.length
    N_corr = all_df.corr_q / np.maximum(all_df.corr_v, 1.0) * all_df.length
    print(f"density |err| mean: base {np.abs(N_base-N_true).mean():.4f} -> "
          f"corrected {np.abs(N_corr-N_true).mean():.4f}")
    per_fam = all_df.groupby("family").apply(
        lambda g: official_S(g, "corr_v", "corr_q")[0] - official_S(g, "blend_v", "blend_q")[0])
    print("\nper-family delta:"); print(per_fam.to_string())

    print(f"\nestimated composite gain (S_state only, S_LWR flat): "
          f"{(s_corr - s_base) * 0.35:+.5f}")
    imp = pd.Series(mv.feature_importance(importance_type="gain"), index=FEATURES)
    print("\ntop speed-residual features (gain):")
    print(imp.sort_values(ascending=False).head(8).to_string())


def features_for_split(panel, split, key):
    """key: DataFrame with timestamp (tz-aware), link_id, blend_v, blend_q,
    tree_v, tree_q for this panel/split's submission rows. Returns key with
    every FEATURES column added, built the same way as held_out_rows()."""
    G = t1feat.load_grid(panel, split, masked=True)
    nT, nL = G["nT"], G["nL"]
    li = {l: i for i, l in enumerate(G["link_ids"])}
    di = {d: i for i, d in enumerate(G["days"])}
    ts = key.timestamp
    row = (ts.dt.strftime("%Y-%m-%d").map(di).to_numpy() * 288 +
          (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy())
    col = key.link_id.map(li).to_numpy()
    assert not np.isnan(row.astype(float)).any() and not pd.isna(col).any(), \
        f"{panel}/{split}: unmapped row/col"
    row = row.astype(int); col = col.astype(int)

    per_v, per_q = [], []
    for n in NETS:
        _, sp, fl, _ = NN.predict_grid(NN.load_net(n), panel, split)
        per_v.append(sp); per_q.append(fl)
    per_v = np.stack(per_v); per_q = np.stack(per_q)
    net_mean_v = per_v.mean(0); net_mean_q = per_q.mean(0)
    net_std_v = per_v.std(0); net_std_q = per_q.std(0)
    # local_std_v uses the net mean alone (no full tree grid is available for
    # validation/private outside the submission's own masked cells); this is a
    # low-importance feature in the CV (not in the top 8), so the small
    # definition drift against held_out_rows()'s blend-based version is
    # accepted rather than building a full tree-inference grid under time
    # pressure.
    lstd = rolling_std(net_mean_v)

    obs = np.isfinite(G["speed"]) & (G["pct"] >= 75)
    up_v, up_d, dn_v, dn_d = bracket(np.nan_to_num(G["speed"]), obs)
    st = G["stat"]

    days_needed = sorted({G["days"][d] for d in set(row // 288)})
    reg_by_day = {d: day_regime(panel, split, d) for d in days_needed}
    row_regime = np.array([REG_CODE.get(reg_by_day[G["days"][d]], 0) for d in row // 288])
    tsod = row % 288
    dow = ts.dt.dayofweek.to_numpy()

    out = key.copy()
    out["net_mean_v"] = net_mean_v[row, col]; out["net_mean_q"] = net_mean_q[row, col]
    out["net_std_v"] = net_std_v[row, col]; out["net_std_q"] = net_std_q[row, col]
    out["vf"] = st["vf"][col]; out["lanes"] = np.maximum(st["lanes"][col], 1.0)
    out["cap"] = st["cap"][col]; out["length"] = st["length"][col]
    out["rank"] = col / max(nL, 1)
    out["da"] = np.minimum(up_d[row, col], 50); out["db"] = np.minimum(dn_d[row, col], 50)
    out["up_v"] = up_v[row, col] / np.maximum(st["vf"][col], 1.0)
    out["dn_v"] = dn_v[row, col] / np.maximum(st["vf"][col], 1.0)
    out["local_std_v"] = lstd[row, col]
    out["sin_tod"] = np.sin(2 * np.pi * tsod / 288); out["cos_tod"] = np.cos(2 * np.pi * tsod / 288)
    out["dow"] = dow; out["is_weekend"] = (dow >= 5).astype(int)
    out["regime"] = row_regime
    out["panel_id"] = PANEL_TO_CODE[panel]
    return out


def build():
    tag = sys.argv[2] if len(sys.argv) > 2 else "s61_t1_residual"
    print(f"loading {len(PANELS)} panels' held-out week to fit the final models...",
         flush=True)
    frames = [get_holdout(p) for p in PANELS]
    all_df = pd.concat(frames, ignore_index=True)
    all_df["resid_v"] = all_df.true_v - all_df.blend_v
    all_df["resid_q"] = all_df.true_q - all_df.blend_q / all_df.lanes
    mv = fit(all_df, "resid_v")
    mq = fit(all_df, "resid_q")
    print(f"fitted on {len(all_df):,} cells from all {len(PANELS)} panels", flush=True)

    HERE = Path(__file__).resolve().parent.parent
    RUNS = HERE / "runs"
    ship = pd.read_csv(RUNS / "_t1_tod_state6.csv", dtype={"link_id": str, "station_id": str})
    tree = pd.read_csv(RUNS / "s15_onset-step6" / "out" / "state_submission.csv",
                       dtype={"link_id": str, "station_id": str})
    ship["timestamp"] = pd.to_datetime(ship.timestamp, utc=True)
    tree["timestamp"] = pd.to_datetime(tree.timestamp, utc=True)

    out_parts = []
    for panel in PANELS:
        for split in ("validation", "private"):
            key = pd.read_csv(ROOT / "task1" / panel / split / "sample_submission_state.csv",
                              dtype={"link_id": str, "station_id": str})
            # the template ships its own placeholder speed_kmh/flow_vph columns,
            # which would otherwise silently _x/_y-suffix the merges below
            key = key.drop(columns=["speed_kmh", "flow_vph"])
            key["timestamp"] = pd.to_datetime(key.timestamp, utc=True)
            m = key.merge(ship[["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"]],
                         on=["panel", "timestamp", "link_id"], how="left")
            assert len(m) == len(key) and not m.speed_kmh.isna().any(), \
                f"{panel}/{split}: {m.speed_kmh.isna().sum()} rows missing from the shipped file"
            m = m.merge(tree[["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"]],
                       on=["panel", "timestamp", "link_id"], suffixes=("", "_tree"), how="left")
            assert not m.speed_kmh_tree.isna().any()
            m = m.rename(columns={"speed_kmh": "blend_v", "flow_vph": "blend_q",
                                  "speed_kmh_tree": "tree_v", "flow_vph_tree": "tree_q"})
            feats = features_for_split(panel, split, m)
            corr_v = feats.blend_v + mv.predict(feats[FEATURES])
            corr_q = feats.blend_q + mq.predict(feats[FEATURES]) * feats.lanes
            feats["speed_kmh"] = np.clip(corr_v, 5.0, feats.vf * 1.05)
            feats["flow_vph"] = np.clip(corr_q, 50.0, feats.cap * 1.15)
            out_parts.append(feats[["panel", "timestamp", "station_id", "link_id",
                                    "mask_regime", "speed_kmh", "flow_vph"]])
            print(f"  [{panel}/{split}] {len(feats):,} rows corrected", flush=True)

    out = pd.concat(out_parts, ignore_index=True)
    out["timestamp"] = out.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir = RUNS / tag / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "state_submission.csv", index=False)
    d = np.abs(out.speed_kmh.to_numpy() -
              pd.read_csv(RUNS / "_t1_tod_state6.csv").speed_kmh.to_numpy())
    print(f"\nwrote {out_dir/'state_submission.csv'}: {len(out):,} rows, "
         f"mean |correction| {d.mean():.4f} km/h, max {d.max():.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if cmd == "cv":
        cv()
    elif cmd == "build":
        build()
    else:
        print(f"unknown command {cmd!r}: use 'cv' or 'build'", file=sys.stderr)
