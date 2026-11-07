"""A real multi-feature classifier for Task 2 horizon eligibility - stronger
due diligence than the single-feature bucketing that closed this question.

The earlier analysis bucketed P(eligible) by ONE feature (history observed
rate) and found it flat (0.73-0.78 across the whole range). That leaves two
real gaps: (a) per-LINK structure (some sensors may be chronically more or
less reliable in a way a single continuous feature misses), and (b)
interactions among features that single-feature bucketing cannot see at all.
A LightGBM classifier with link identity, family, time-of-day, day-of-week,
and the history-observed-rate together closes both gaps in one shot.

If its held-out AUC is near 0.5, eligibility really is unpredictable and the
0.764 ceiling is confirmed by a much stronger method than before. If it is
meaningfully above 0.5 (say > 0.60), there is a real, previously-missed
signal worth exploiting.

    python q_elig_classifier.py
"""
import nice  # noqa: F401
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

from common import QUEUE_PANELS, FAMILY
from qdata import load_panel
from qfeat import HIST, HOR

VAL = 60


def rows_for(panel, pid):
    D = load_panel(panel)
    el = D["elig"]; nT = el.shape[0]; nd = nT // 288
    T = np.arange(HIST, nT - HOR - 1)
    T = T[(T // 288 >= nd - VAL)]           # the same held-out window as elsewhere
    rng = np.random.default_rng(7)
    T = rng.choice(T, size=min(4000, len(T)), replace=False)
    recs = []
    for t0 in T:
        t0 = int(t0)
        hist_rate = el[t0 - HIST:t0].mean(0)     # per-link, this window
        last = el[t0 - 1].astype(float)
        dow = int(D["dow"][t0]); tod0 = t0 % 288
        fut = el[t0 + 1:t0 + 1 + HOR]
        for s in range(HOR):
            recs.append(pd.DataFrame({
                "panel_id": pid, "link": np.arange(el.shape[1]),
                "hist_rate": hist_rate, "last": last,
                "step": s, "dow": dow, "weekend": int(dow >= 5),
                "tod_sin": np.sin(2 * np.pi * ((tod0 + s + 1) % 288) / 288),
                "tod_cos": np.cos(2 * np.pi * ((tod0 + s + 1) % 288) / 288),
                "eligible": fut[s].astype(int),
            }))
    return pd.concat(recs, ignore_index=True)


def main():
    frames = []
    for pid, panel in enumerate(QUEUE_PANELS):
        df = rows_for(panel, pid)
        df["family"] = FAMILY[panel]
        frames.append(df)
        print(f"  {panel}: {len(df):,} rows, base rate {df.eligible.mean():.4f}", flush=True)
    all_df = pd.concat(frames, ignore_index=True)
    fam_code = {f: i for i, f in enumerate(sorted(set(all_df.family)))}
    all_df["family_id"] = all_df.family.map(fam_code)
    # a per-(panel,link) integer id, so the classifier can learn "this exact
    # sensor is chronically flaky" if that pattern exists
    all_df["link_uid"] = all_df.panel_id.astype(str) + "_" + all_df.link.astype(str)
    all_df["link_uid"] = all_df.link_uid.astype("category").cat.codes

    features = ["panel_id", "family_id", "link_uid", "hist_rate", "last",
                "step", "dow", "weekend", "tod_sin", "tod_cos"]
    cat = ["panel_id", "family_id", "link_uid", "step", "dow"]

    rng = np.random.default_rng(0)
    n = len(all_df)
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, te = all_df.iloc[idx[:cut]], all_df.iloc[idx[cut:]]

    ds = lgb.Dataset(tr[features], label=tr.eligible, categorical_feature=cat,
                     free_raw_data=False)
    params = dict(objective="binary", num_leaves=63, min_child_samples=100,
                  learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1)
    model = lgb.train(params, ds, num_boost_round=300)

    def auc(y, p, cap=300_000, seed=1):
        # numpy's default int dtype on Windows is 32-bit, and the rank sum
        # below overflows it well before y reaches a million rows (silently
        # wrapping to nonsense, which is what produced train AUC -0.60 / test
        # 5.55 the first time this ran) - force int64, and subsample, since
        # 300k rows estimates AUC to within noise anyway.
        y = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        p = np.asarray(p)
        if len(y) > cap:
            idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
            y, p = y[idx], p[idx]
        o = np.argsort(p); y = y[o].astype(np.int64)
        r = np.arange(1, len(y) + 1, dtype=np.int64)
        n1 = np.int64(y.sum()); n0 = np.int64(len(y)) - n1
        if n1 == 0 or n0 == 0:
            return float("nan")
        return float((r[y == 1].sum() - n1 * (n1 + 1) // 2) / (n1 * n0))

    p_tr = model.predict(tr[features]); p_te = model.predict(te[features])
    print(f"\nrows: train {len(tr):,}  test {len(te):,}  base rate {all_df.eligible.mean():.4f}")
    print(f"AUC  train {auc(tr.eligible, p_tr):.4f}   test {auc(te.eligible, p_te):.4f}")
    print(f"test accuracy at 0.5 threshold: {((p_te >= 0.5).astype(int) == te.eligible).mean():.4f}"
         f"  (predicting the base rate everywhere gets "
         f"{max(all_df.eligible.mean(), 1 - all_df.eligible.mean()):.4f})")

    imp = pd.Series(model.feature_importance(importance_type="gain"), index=features)
    print("\nfeature importance (gain):")
    print(imp.sort_values(ascending=False).to_string())

    # calibration: does sorting by predicted probability actually separate
    # eligible-heavy cells from ineligible-heavy ones in the test set?
    te = te.copy(); te["p"] = p_te
    te["bucket"] = pd.qcut(te.p, 10, duplicates="drop")
    print("\neligibility rate by predicted-probability decile (test set):")
    print(te.groupby("bucket", observed=True).eligible.mean().to_string())


if __name__ == "__main__":
    main()
