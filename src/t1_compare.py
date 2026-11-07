"""Net against trees on the same cells: the train split's real masked targets.

The trees were fitted on every train day, so they see these days; the net holds
out the last 30. That handicap is deliberate - if the net matches the trees here
it is ahead on validation and private, where neither has seen the month.

    python t1_compare.py NAME [--days 30] [--panels 4]
"""
import nice  # noqa: F401
import argparse, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from common import PANELS, OUT
from t1feat import load_grid
from t1build import build
from ens_search import MEMBERS
from task1_run import FLOW_BLEND, predict_chunked
import t1_nn as NN

W = {"speed": {"own": 0.51, "pooled": 0.12, "family": 0.21, "alt": 0.16},
     "flow": {"own": 0.47, "pooled": 0.25, "family": 0.17, "alt": 0.12},
     "density": {"own": 0.44, "pooled": 0.16, "family": 0.17, "alt": 0.23}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--panels", type=int, default=4)
    a = ap.parse_args()
    net = NN.Net().to(NN.DEV)
    net.load_state_dict(torch.load(OUT / f"t1_nn_{a.name}.pt", map_location=NN.DEV))
    tot = {k: [0.0, 0.0, 0] for k in ("net", "tree", "blend")}
    for panel in PANELS[:a.panels]:
        U = load_grid(panel, "train", masked=False)
        nT, nL = U["nT"], U["nL"]
        recent = np.zeros((nT, nL), bool); recent[nT - 288 * a.days:] = True
        G, sp_n, fl_n, _ = NN.predict_grid(net, panel, "train")
        r = build(panel, "train", targets="eligible", target_mask=recent, verbose=False)
        ti, tj = r["ti"], r["tj"]
        members = {n: MEMBERS[n](panel) for n in ("own", "pooled", "family", "alt")}
        mix = {}
        for t in ("speed", "flow", "density"):
            acc, wt = 0.0, W[t]
            s = sum(wt.values())
            for n, M in members.items():
                acc = acc + predict_chunked(M[t], r["X"]) * (wt[n] / s)
            mix[t] = acc
        stat = G["stat"]
        v_t = np.clip(mix["speed"], 5.0, stat["vf"][tj] * 1.05)
        q_pl = (1 - FLOW_BLEND) * mix["flow"] + FLOW_BLEND * mix["density"] * v_t
        q_t = np.clip(q_pl * stat["lanes"][tj], 5.0, stat["cap"][tj] * 1.15)
        v_n, q_n = sp_n[ti, tj], fl_n[ti, tj]
        truth_v = U["speed"][ti, tj]
        truth_q = U["flow"][ti, tj] / stat["lanes"][tj]
        ok = np.isfinite(truth_v) & np.isfinite(truth_q)
        for key, (v, q) in (("net", (v_n, q_n)), ("tree", (v_t, q_t)),
                            ("blend", (0.5 * (v_n + v_t), 0.5 * (q_n + q_t)))):
            dv = (v[ok] - truth_v[ok]); dq = (q[ok] / stat["lanes"][tj][ok] - truth_q[ok])
            tot[key][0] += float((dv ** 2).sum()); tot[key][1] += float((dq ** 2).sum())
            tot[key][2] += int(ok.sum())
        print(f"  {panel}: {int(ok.sum()):,} cells", flush=True)
    print(f"\nlast {a.days} train days, real masked targets, {a.panels} panels")
    for key, (sv, sq, n) in tot.items():
        rv, rq = (sv / n) ** 0.5, (sq / n) ** 0.5
        print(f"  {key:6s} speed RMSE {rv:.3f}  flow/lane {rq:.2f}  "
              f"S_state {0.54*(1-rv/25)+0.46*(1-rq/600):.4f}")


if __name__ == "__main__":
    main()
