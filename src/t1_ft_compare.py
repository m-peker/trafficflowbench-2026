"""Per-panel finetunes against the shared f1, on the week both held out.

Each panel uses its own ft_<panel> checkpoint; f1 is the same net everywhere.
Scored on 0.35*S_state + 0.15*S_physics over the last 7 days (T1_EVAL_LAST=7).
"""
import os
os.environ.setdefault("T1_EVAL_LAST", "7")
import nice  # noqa: F401
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import ens_search as E
import t1_nn as NN
from t1_phys_compare import score_fields, W

panels = [E.Panel(p) for p in E.PANELS[:4]]
f1 = NN.load_net("f1")
rows = {}
for P in panels:
    tj = P.tj
    v_t = np.clip(P.mix(W, "speed"), 5.0, P.vf[tj] * 1.05)
    k = P.mix(W, "density")
    q_t = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v_t)
                  * P.lanes[tj], 5.0, P.cap[tj] * 1.15)
    fields = {}
    for tag, net in (("f1", f1), ("ft", NN.load_net(f"ft_{P.panel}"))):
        G, sp, fl, _ = NN.predict_grid(net, P.panel, "train")
        lo = G["nT"] - P.obs_sp.shape[0]
        fields[tag] = (sp[P.ti + lo, tj], fl[P.ti + lo, tj])
    cands = {
        "trees": (v_t, q_t),
        "f1 alone": fields["f1"], "ft alone": fields["ft"],
        "0.6 f1 + 0.4 trees": (0.6 * fields["f1"][0] + 0.4 * v_t, 0.6 * fields["f1"][1] + 0.4 * q_t),
        "0.6 ft + 0.4 trees": (0.6 * fields["ft"][0] + 0.4 * v_t, 0.6 * fields["ft"][1] + 0.4 * q_t),
        "0.6 mean(f1,ft) + 0.4 trees": (0.3 * (fields["f1"][0] + fields["ft"][0]) + 0.4 * v_t,
                                        0.3 * (fields["f1"][1] + fields["ft"][1]) + 0.4 * q_t),
    }
    for name, (v, q) in cands.items():
        s = score_fields(P, v, q)
        rows.setdefault(name, []).append((P.panel, s["partial"], s["S_state"], s["S_lwr"]))
    print(f"  {P.panel} ready", flush=True)

print(f"\n{'candidate':30s} {'S_state':>8s} {'S_lwr':>7s} {'partial':>8s}   per panel partial")
for name, r in rows.items():
    print(f"{name:30s} {np.mean([x[2] for x in r]):8.4f} {np.mean([x[3] for x in r]):7.4f} "
          f"{np.mean([x[1] for x in r]):8.5f}   " + " ".join(f"{x[1]:.5f}" for x in r))
