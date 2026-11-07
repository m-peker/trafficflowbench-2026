"""Does the forecaster get worse later in the year, or is the gap somewhere else?

Two numbers refuse to meet. On the organizers' own train windows - all of which
sit in the first ten days of June 2030 - the shipped forecaster scores 0.7596.
The leaderboard, on March and April 2031, implies 0.5025. Neither in-sample
fitting nor the eligibility rule nor the 0.60 queue threshold explains it: the
eligibility rule reproduces `is_score_eligible` exactly, the threshold is pinned
by all 120 released `queue_ongoing` windows having a queue in history at 0.60
and not below, and windows sampled from the *held-out* tail of train score
higher still, at 0.8189.

That leaves drift. Train runs June 2030 to February 2031 and the scored splits
are March and April 2031, so if the corridor's behaviour moves through the year
a model fitted on the first 213 days would decay. This measures that directly:
the same window rule, the same model, month by month across train.

A flat line says the gap is not drift and sends the search elsewhere. A falling
line says the model needs to be refitted on recent data, or given features that
do not assume last summer's regime.

    python qtrend.py
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors, origins, window_xy
from qfeat import HOR

N_PER_COND = 60          # per panel per month
PSTAR = {0: 0.50, 1: 0.45}


def main():
    with open(OUT / "queue_model.pkl", "rb") as fh:
        models = pickle.load(fh)["models"]

    rows = []
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel)
        pri = make_priors(D)
        days = np.array([str(d) for d in D["days"]])
        month = np.array([d[:7] for d in days])
        on, ong = origins(D)
        rng = np.random.default_rng(7)
        for m in sorted(set(month)):
            day_ix = np.flatnonzero(month == m)
            lo, hi = day_ix[0] * 288, (day_ix[-1] + 1) * 288
            for cond, pool in ((0, on), (1, ong)):
                sel = pool[(pool >= lo + 288) & (pool < hi - 288)]
                if len(sel) < 10:
                    continue
                sel = rng.choice(sel, min(N_PER_COND, len(sel)), replace=False)
                vals = []
                for T in sel:
                    sl = slice(T + 1, T + 1 + HOR)
                    truth = (D["q"][sl] & D["elig"][sl]).reshape(-1)
                    em = D["elig"][sl].reshape(-1)
                    X, _, _, _ = window_xy(D, pri, T, cond, pid, with_y=False)
                    pred = (models[cond].predict(X) >= PSTAR[cond]) & em
                    u = np.logical_or(pred, truth).sum()
                    vals.append(1.0 if u == 0 else
                                float(np.logical_and(pred, truth).sum() / u))
                rows.append(dict(panel=panel, month=m,
                                 cond="onset" if cond == 0 else "ongoing",
                                 iou=float(np.mean(vals)), n=len(sel)))
        print(f"  {panel} done", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "qtrend.csv", index=False)
    d["family"] = d.panel.map(FAMILY)

    # leaderboard-style: condition mean -> panel -> family -> mean, per month
    per = (d.groupby(["month", "panel", "family"]).iou.mean().reset_index()
             .groupby(["month", "family"]).iou.mean().reset_index()
             .groupby("month").iou.mean())
    print(f"\n{'month':>9s} {'S_queue':>9s}   windows")
    n_by_month = d.groupby("month").n.sum()
    for m, v in per.items():
        print(f"{m:>9s} {v:9.4f}   {n_by_month[m]:,}")
    first, last = per.iloc[0], per.iloc[-1]
    print(f"\nfirst month {per.index[0]}: {first:.4f}   "
          f"last month {per.index[-1]}: {last:.4f}   change {last-first:+.4f}")
    print(f"leaderboard (Mar+Apr 2031): 0.5025")


if __name__ == "__main__":
    main()
