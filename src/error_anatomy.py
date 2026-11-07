"""Where does the Task 1 error actually live?

Physics is 48 percent of the room left to 0.95, and S_LWR is purely a function
of Task 1 accuracy, so the question that decides the target is whether our
reconstruction error is model error or measurement noise.

The test: stratify the holdout error by how far the nearest observation is. A
cell whose neighbours sit 5 minutes either side is nearly interpolation; one in
the middle of a long blanked run is nearly extrapolation. If the easy cells are
already at a floor and the hard ones are far above it, the headroom is real and
sits in the hard cells. If every bucket is flat, we are at the noise floor and
physics cannot be bought.
"""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from t1build import build

FROZEN = Path(r"D:/BigData Competition/work/out")
PANEL = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"


def main():
    r = build(PANEL, "train", truth="train", sample=900_000, seed=1)
    X, cols = r["X"], r["cols"]
    day = r["ti"] // 288
    va = day >= day.max() + 1 - 45

    with open(FROZEN / f"t1_model_{PANEL}.pkl", "rb") as fh:
        models = pickle.load(fh)["models"]

    i_pg, i_ng = cols.index("sp_prevg"), cols.index("sp_nextg")
    pg, ng = X[:, i_pg], X[:, i_ng]
    span = np.minimum(pg, ng)          # distance to the nearest observation

    out = []
    for nm, y, scale, unit in (("speed", r["y_speed"], 25.0, "km/h"),
                               ("flow", r["y_flowpl"], 600.0, "vph/lane")):
        m = va & np.isfinite(y)
        pred = models[nm].predict(X[m], num_iteration=models[nm].best_iteration)
        err = pred - y[m]
        s = span[m]
        print(f"\n=== {nm} ({unit}) — holdout {m.sum():,} cells ===")
        print(f"{'nearest obs':>12s} {'share':>7s} {'RMSE':>9s} {'bias':>8s} "
              f"{'S if all':>9s}")
        rows = []
        for lo, hi, lab in ((1, 1, "5 min"), (2, 2, "10 min"), (3, 3, "15 min"),
                            (4, 6, "20-30 min"), (7, 1000, ">30 min")):
            k = (s >= lo) & (s <= hi)
            if k.sum() == 0:
                continue
            rmse = float(np.sqrt(np.mean(err[k] ** 2)))
            rows.append((lab, k.mean(), rmse))
            print(f"{lab:>12s} {k.mean():7.1%} {rmse:9.3f} {err[k].mean():+8.3f} "
                  f"{max(0, 1 - rmse / scale):9.4f}")
        overall = float(np.sqrt(np.mean(err ** 2)))
        print(f"{'overall':>12s} {1.0:7.1%} {overall:9.3f} {err.mean():+8.3f} "
              f"{max(0, 1 - overall / scale):9.4f}")

        # what the score would be if every bucket hit the easiest bucket's error
        floor = rows[0][2]
        lifted = float(np.sqrt(sum(sh * max(rmse, floor) ** 2 * 0 + sh * floor ** 2
                                   for _, sh, rmse in rows)))
        print(f"  if every cell reached the 5-min bucket's error: RMSE {lifted:.3f} "
              f"-> S {max(0, 1 - lifted / scale):.4f}")
        out.append((nm, overall, floor, rows))

    print("\n--- what that buys ---")
    (_, sv, fv, _), (_, sq, fq, _) = out
    now = 0.54 * (1 - sv / 25) + 0.46 * (1 - sq / 600)
    best = 0.54 * (1 - fv / 25) + 0.46 * (1 - fq / 600)
    print(f"S_state now {now:.4f}, at the easy-bucket floor {best:.4f} "
          f"({0.35 * (best - now):+.4f} on the composite)")
    dens_now = np.hypot(sq * 4 / 3700, sv / 110)
    dens_best = np.hypot(fq * 4 / 3700, fv / 110)
    print(f"relative density error {dens_now:.4f} -> {dens_best:.4f} "
          f"(x{dens_best/dens_now:.2f}); S_LWR scales roughly with it")
    e_now = 0.44
    e_best = 0.06 + (e_now - 0.06) * dens_best / dens_now
    sp_now = (0.98 + 2 * max(0, 1 - e_now)) / 3
    sp_best = (0.98 + 2 * max(0, 1 - e_best)) / 3
    print(f"S_physics {sp_now:.3f} -> {sp_best:.3f} ({0.15 * (sp_best - sp_now):+.4f})")
    print(f"combined ceiling for this line of attack: "
          f"{0.35 * (best - now) + 0.15 * (sp_best - sp_now):+.4f}")


if __name__ == "__main__":
    main()
