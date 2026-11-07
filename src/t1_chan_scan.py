"""Per-channel blend weights: the net's speed and its flow need not mix alike.

S_LWR reads k = q/v, so the two channels do not enter the score symmetrically,
and the board says the net is better than the local harness admits (s37 beat
s32 by +0.0038 while the local partial ranked the trees first). This scans the
two weights separately on 0.35*S_state + 0.15*S_physics.
"""
import nice  # noqa: F401
import sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
import ens_search as E
import t1_nn as NN
from common import OUT
from t1_phys_compare import score_fields, W

name = sys.argv[1] if len(sys.argv) > 1 else "a1"
net = NN.Net().to(NN.DEV)
net.load_state_dict(torch.load(OUT / f"t1_nn_{name}.pt", map_location=NN.DEV))
net.eval()
panels = [E.Panel(p) for p in E.PANELS[:4]]
data = []
for P in panels:
    tj = P.tj
    v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[tj] * 1.05)
    k = P.mix(W, "density")
    q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v_t)
                  * P.lanes[tj], 5.0, P.cap[tj] * 1.15)
    G, sp_n, fl_n, _ = NN.predict_grid(net, P.panel, "train")
    lo = G["nT"] - P.obs_sp.shape[0]
    data.append((P, v_t, q_t, sp_n[P.ti + lo, tj], fl_n[P.ti + lo, tj]))
    print(f"  {P.panel} ready", flush=True)

ws = (0.0, 0.3, 0.5, 0.7, 1.0)
print(f"\nrows: weight on the net's speed, columns: on its flow ({name})")
print("        " + "".join(f"{c:>9.1f}" for c in ws))
for a in ws:
    line = []
    for b in ws:
        acc = []
        for P, v_t, q_t, v_n, q_n in data:
            acc.append(score_fields(P, a * v_n + (1 - a) * v_t,
                                    b * q_n + (1 - b) * q_t)["partial"])
        line.append(float(np.mean(acc)))
    print(f"  {a:4.1f}  " + "".join(f"{x:9.5f}" for x in line))
