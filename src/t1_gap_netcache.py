"""Cache one net's fields on the validation/private masked grids, for the gap fill.

The release already blanks the gaps and publishes the history raw, so the net
is run on the grid exactly as released.
    python t1_gap_netcache.py gf1   -> out/gapnet_<net>_<panel>_<split>.npz
"""
import os, sys
os.environ.setdefault("T1_TOD", "const"); os.environ.setdefault("T1_TILE_L", "64")
import nice  # noqa: F401
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import t1_nn as NN

PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N",
          "D7_I405_S", "D12_I5_N", "D12_I5_S"]

if __name__ == "__main__":
    name = sys.argv[1]
    net = NN.load_net(name)
    # GAPNET_TAG (e.g. "_i") names a variant cache - s89 runs the gap nets with
    # T1_PCTMIN=0 T1_PCTCLIP=75 into gapnet_<net>_i_*, leaving the originals intact
    out_name = name + os.environ.get("GAPNET_TAG", "")
    for panel in PANELS:
        for split in ("validation", "private"):
            _, sp, fl, _ = NN.predict_grid(net, panel, split)[:4]
            np.savez_compressed(NN.OUT / f"gapnet_{out_name}_{panel}_{split}.npz",
                                v=sp.astype(np.float32), q=fl.astype(np.float32))
            print(f"  {panel}/{split}", flush=True)
