"""Where does the Task 1 error live by traffic regime, not by mask distance?

error_anatomy.py stratified the holdout error by distance to the nearest
observation and found it flat, which I read as "we are at the noise floor". But
93 percent of cells are in free flow, where speed sits at v_f and prediction is
nearly trivial. If the error mass concentrates in the congested minority, that
average hides a regime where a specialised model could still win - and
congestion is exactly where S_LWR lives too, because density swings there.

This splits the same holdout by v/v_f and reports, per bucket, the share of
cells, the RMSE, and the share of total squared error.
"""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

from common import OUT, PANELS
from t1build import build
from t1feat import load_grid
from task1_run import FLOW_BLEND

MODELS = OUT                                   # v3 models live here


def main():
    panels = sys.argv[1:] or ["D7_I10_E", "D12_I5_N", "D7_I405_N"]
    rows = []
    for panel in panels:
        r = build(panel, "train", truth="train", sample=900_000, seed=1, verbose=False)
        G = load_grid(panel, "train", masked=True)
        vf, lanes, cap = G["stat"]["vf"], G["stat"]["lanes"], G["stat"]["cap"]
        del G
        day = r["ti"] // 288
        m = (day >= day.max() + 1 - 45) & np.isfinite(r["y_speed"]) \
            & np.isfinite(r["y_flowpl"])
        X, tj = r["X"][m], r["tj"][m]
        v_t, q_t = r["y_speed"][m], r["y_flowpl"][m]

        with open(MODELS / f"t1_model_{panel}.pkl", "rb") as fh:
            mods = pickle.load(fh)["models"]
        v_p = np.clip(mods["speed"].predict(X, num_iteration=mods["speed"].best_iteration),
                      5.0, vf[tj] * 1.05)
        q_d = mods["flow"].predict(X, num_iteration=mods["flow"].best_iteration)
        k_p = mods["density"].predict(X, num_iteration=mods["density"].best_iteration)
        q_p = np.clip(((1 - FLOW_BLEND) * q_d + FLOW_BLEND * k_p * v_p) * lanes[tj],
                      5.0, cap[tj] * 1.15) / lanes[tj]
        del r, X

        ratio = v_t / vf[tj]                    # true regime, from the truth
        ev, eq = (v_p - v_t) ** 2, (q_p - q_t) ** 2
        buckets = (("congested  v<0.6vf", ratio < 0.6),
                   ("transition 0.6-0.8", (ratio >= 0.6) & (ratio < 0.8)),
                   ("near-free  0.8-0.95", (ratio >= 0.8) & (ratio < 0.95)),
                   ("free       v>=0.95vf", ratio >= 0.95))
        print(f"\n=== {panel} — holdout {len(v_t):,} cells ===")
        print(f"{'regime':22s} {'share':>7s} {'speed RMSE':>11s} {'of sq err':>10s}"
              f" {'flow RMSE':>10s} {'of sq err':>10s}")
        for name, sel in buckets:
            if sel.sum() == 0:
                continue
            print(f"{name:22s} {sel.mean():7.1%} "
                  f"{np.sqrt(ev[sel].mean()):11.3f} {ev[sel].sum()/ev.sum():10.1%} "
                  f"{np.sqrt(eq[sel].mean()):10.2f} {eq[sel].sum()/eq.sum():10.1%}")
            rows.append(dict(panel=panel, regime=name, share=sel.mean(),
                             v_rmse=np.sqrt(ev[sel].mean()),
                             v_err_share=ev[sel].sum()/ev.sum(),
                             q_rmse=np.sqrt(eq[sel].mean()),
                             q_err_share=eq[sel].sum()/eq.sum()))
        print(f"{'overall':22s} {1.0:7.1%} {np.sqrt(ev.mean()):11.3f} "
              f"{1.0:10.1%} {np.sqrt(eq.mean()):10.2f} {1.0:10.1%}")

    d = pd.DataFrame(rows)
    if d.empty:
        return
    print("\n--- averaged over panels ---")
    a = d.groupby("regime")[["share", "v_err_share", "q_err_share"]].mean()
    print(a.to_string(float_format=lambda x: f"{x:.1%}"))
    print("\nIf a regime holds a much larger share of the squared error than of the")
    print("cells, that is where a specialised model would pay - and the flat")
    print("mask-distance profile would not have shown it.")
    d.to_csv(OUT / "error_by_regime.csv", index=False)


if __name__ == "__main__":
    main()
