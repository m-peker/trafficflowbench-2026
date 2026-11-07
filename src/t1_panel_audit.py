"""Per-panel Task 1 accuracy on all ten corridors, aggregated the way the board does.

The board averages S_state panel-by-regime, then panel, then corridor family,
then the five families - so the 44-link D12_I405_S weighs as much as the
352-link D7_I405_N, and the D12_I405 family is a fifth of S_state. The local
harnesses only ever scored four panels, pooled by cell, and none of them from
D12_I405. If the net is much worse on the small corridors, that alone could be
part of why the board's error reads ~1.55x ours.

Last 7 train days (held out by every T1_VAL_DAYS=7 net), the release's real
masked targets, f1 alone and the tree file for comparison is not needed - this
is about where the net is weak.
"""
import nice  # noqa: F401
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from common import PANELS, FAMILY
from t1feat import load_grid
import t1_nn as NN

DAYS = 7
net = NN.load_net("f1")
rows = []
for panel in PANELS:
    U = load_grid(panel, "train", masked=False)
    G, sp, fl, _ = NN.predict_grid(net, panel, "train")
    nT = G["nT"]
    lanes = np.maximum(G["stat"]["lanes"], 1.0)[None, :]
    tgt = ~np.isfinite(G["speed"]) & np.isfinite(U["speed"]) & (U["pct"] >= 75)
    tgt[: nT - 288 * DAYS] = False
    if "regime" in G:
        reg = np.asarray(G["regime"])
    dv = (sp - U["speed"])[tgt]
    dq = ((fl - U["flow"]) / lanes)[tgt]
    rv, rq = float(np.sqrt(np.mean(dv ** 2))), float(np.sqrt(np.mean(dq ** 2)))
    S = 0.54 * (1 - rv / 25) + 0.46 * (1 - rq / 600)
    rows.append((panel, FAMILY.get(panel, panel[:-2]), G["nL"], int(tgt.sum()), rv, rq, S))
    print(f"  {panel:11s} links {G['nL']:3d}  targets {int(tgt.sum()):7d}  speed RMSE {rv:5.2f}  "
          f"flow/lane {rq:6.2f}  S_state {S:.4f}", flush=True)

pooled_v = np.sqrt(sum(r[4] ** 2 * r[3] for r in rows) / sum(r[3] for r in rows))
pooled_q = np.sqrt(sum(r[5] ** 2 * r[3] for r in rows) / sum(r[3] for r in rows))
pooled = 0.54 * (1 - pooled_v / 25) + 0.46 * (1 - pooled_q / 600)
fam = {}
for r in rows:
    fam.setdefault(r[1], []).append(r[6])
board_like = np.mean([np.mean(v) for v in fam.values()])
print(f"\npooled over cells: S_state {pooled:.4f}")
print("by family: " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in fam.items()))
print(f"board-style (panel -> family -> mean of 5): S_state {board_like:.4f}")
