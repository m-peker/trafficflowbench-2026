"""Run nets on t1_gap_fair.py's organizer-style gap windows and cache the fields.

    python t1_gap_fair_net.py gf1 [more nets, averaged]  -> out/gap_fair_<panel>_<tag>.npz
"""
import os, sys
os.environ["QONSET_STEP6"] = "1"; os.environ["QONG_CONTRACT"] = "1"
os.environ.setdefault("T1_TOD", "const"); os.environ.setdefault("T1_TILE_L", "64")
import nice  # noqa: F401
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import t1_nn as NN
import t1feat
from common import QUEUE_PANELS
from qfeat import HIST
from t1_gap_fair import pick
from t1_gap_analog import GAP


def main():
    names = sys.argv[1:]
    tag = os.environ.get("GAP_TAG", "+".join(names))
    nets = [NN.load_net(n) for n in names]
    orig = t1feat.load_grid
    for panel in QUEUE_PANELS:
        Gm = orig(panel, "train", masked=True); Gt = orig(panel, "train", masked=False)
        o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
        nT = Gm["nT"]
        W, lo = pick(panel, nT)
        Gg = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Gm.items()}
        for t, _ in W:
            for ch in ("speed", "flow", "occ", "pct"):
                Gg[ch][t - HIST:t] = Gt[ch][:, o][t - HIST:t]
            for ch in ("speed", "flow", "occ"):
                Gg[ch][t + 1:t + 1 + GAP] = np.nan
        NN.load_grid = lambda p, s, masked=True, G=Gg: G
        V, Q = [], []
        for net in nets:
            _, sp, fl, _ = NN.predict_grid(net, panel, "train", t_lo=lo - 256)[:4]
            V.append(sp); Q.append(fl)
        NN.load_grid = orig
        np.savez_compressed(NN.OUT / f"gap_fair_{panel}_{tag}.npz",
                            v=np.mean(V, 0).astype(np.float32), q=np.mean(Q, 0).astype(np.float32))
        print(f"  {panel}: {len(W)} windows", flush=True)


if __name__ == "__main__":
    main()
