"""What the retrained Task 1 actually buys, measured on what gets submitted.

train_panel prints the direct flow model's score, but the submitted flow is
`0.5*q_hat + 0.5*k_hat*v_hat`, which is slightly worse on flow RMSE and
noticeably better on density. Density is what the conservation residual sees, so
the headline number in the training log understates the physics gain and
overstates the state gain. This measures both on the blend that is actually
written.
"""
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

from common import OUT, PANELS, FAMILY
from t1build import build
from t1feat import load_grid
from task1_run import FLOW_BLEND

FROZEN = Path(r"D:/BigData Competition/work/out")     # the v1/v2 models


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate(panel):
    r = build(panel, "train", truth="train", sample=900_000, seed=1, verbose=False)
    G = load_grid(panel, "train", masked=True)
    vf, lanes, cap = G["stat"]["vf"], G["stat"]["lanes"], G["stat"]["cap"]
    del G
    day = r["ti"] // 288
    va = (day >= day.max() + 1 - 45) & np.isfinite(r["y_speed"]) & np.isfinite(r["y_flowpl"])
    X = r["X"][va]
    tj = r["tj"][va]
    v_t, q_t = r["y_speed"][va], r["y_flowpl"][va]
    k_t = q_t / np.maximum(v_t, 1.0)
    out = {}

    def as_submitted(v, q_pl):
        """Exactly the clipping reconstruct() applies before writing a row."""
        v = np.clip(v, 5.0, vf[tj] * 1.05)
        q = np.clip(q_pl * lanes[tj], 5.0, cap[tj] * 1.15) / lanes[tj]
        return v, q

    with open(FROZEN / f"t1_model_{panel}.pkl", "rb") as fh:
        old = pickle.load(fh)["models"]
    v_o = old["speed"].predict(X, num_iteration=old["speed"].best_iteration)
    q_o = old["flow"].predict(X, num_iteration=old["flow"].best_iteration)
    v_o, q_o = as_submitted(v_o, q_o)
    out["old"] = (rmse(v_o, v_t), rmse(q_o, q_t), rmse(q_o / v_o, k_t) / k_t.mean())

    with open(OUT / f"t1_model_{panel}.pkl", "rb") as fh:
        new = pickle.load(fh)["models"]
    v_n = new["speed"].predict(X, num_iteration=new["speed"].best_iteration)
    q_d = new["flow"].predict(X, num_iteration=new["flow"].best_iteration)
    k_n = new["density"].predict(X, num_iteration=new["density"].best_iteration)
    v_n = np.clip(v_n, 5.0, vf[tj] * 1.05)          # reconstruct() clips first,
    q_b = (1 - FLOW_BLEND) * q_d + FLOW_BLEND * k_n * v_n   # then blends
    v_n, q_b = as_submitted(v_n, q_b)
    out["new"] = (rmse(v_n, v_t), rmse(q_b, q_t), rmse(q_b / v_n, k_t) / k_t.mean())
    del r, X
    return out


def main():
    panels = sys.argv[1:] or PANELS
    rows = []
    for p in panels:
        try:
            e = evaluate(p)
        except FileNotFoundError:
            print(f"{p}: not retrained yet, skipping", flush=True)
            continue
        for tag in ("old", "new"):
            v, q, k = e[tag]
            rows.append(dict(panel=p, version=tag, speed_rmse=v, flow_rmse=q,
                             dens_rel=k, S_v=1 - v / 25, S_q=1 - q / 600,
                             S_state=0.54 * (1 - v / 25) + 0.46 * (1 - q / 600),
                             family=FAMILY[p]))
        a, b = rows[-2], rows[-1]
        print(f"{p:11s} S_state {a['S_state']:.4f} -> {b['S_state']:+.4f}"
              f"   density {a['dens_rel']:.5f} -> {b['dens_rel']:.5f} "
              f"(x{b['dens_rel']/a['dens_rel']:.3f})", flush=True)

    d = pd.DataFrame(rows)
    if d.empty:
        return
    agg = d.groupby(["version", "family"]).S_state.mean().groupby("version").mean()
    dens = d.groupby("version").dens_rel.mean()
    print(f"\nS_state (leaderboard aggregation)  {agg['old']:.4f} -> {agg['new']:.4f}"
          f"   composite {0.35 * (agg['new'] - agg['old']):+.4f}")
    ratio = dens["new"] / dens["old"]
    print(f"relative density error              {dens['old']:.5f} -> {dens['new']:.5f}"
          f"   (x{ratio:.3f})")
    print("\nS_LWR scales with the density error; e_lwr = 0.06 + k*density_error,")
    print("anchored on the perfect answer's 0.06. Under the two readings of the")
    print("leaderboard we could not separate:")
    for name, e_now in (("S_physics ~0.70", 0.44), ("S_physics ~0.84", 0.226)):
        e_new = 0.06 + (e_now - 0.06) * ratio
        sp_now = (0.98 + 2 * max(0.0, 1 - e_now)) / 3
        sp_new = (0.98 + 2 * max(0.0, 1 - e_new)) / 3
        print(f"  {name}: {sp_now:.3f} -> {sp_new:.3f}   composite "
              f"{0.15 * (sp_new - sp_now):+.4f}")
    d.to_csv(OUT / "v3_measure.csv", index=False)


if __name__ == "__main__":
    main()
