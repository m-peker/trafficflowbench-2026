"""Helper stages for reproduce.py (run from src/, paths from TFB_DATA / TFB_WORK).

    python repro_steps.py nn_static
    python repro_steps.py nets TREES_STATE.csv OUT_STATE.csv
    python repro_steps.py package FINAL_DIR OUT.zip
    python repro_steps.py verify NEW.zip REFERENCE.zip
"""
import hashlib, os, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, WORK, PANELS

SIX = ["f1", "h1", "i1", "j1", "m1", "n1"]
# s88 swaps in the six members fine-tuned with ineligible readings as inputs
if os.environ.get("T1_SIX"):
    SIX = os.environ["T1_SIX"].split(",")
W_TREE = 0.15
COLS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def md5(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def nn_static():
    """work/nn/<panel>_static.npz - the per-panel constants predict_grid reads.
    Identical to what `t1_nn.py prep` writes, without the training memmaps."""
    from t1feat import load_grid
    nn_dir = WORK / "nn"; nn_dir.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        G = load_grid(panel, "train", masked=False)
        st = G["stat"]
        occ_scale = max(float(np.nanpercentile(G["occ"][::7], 99)), 1e-3)
        np.savez(nn_dir / f"{panel}_static.npz", vf=st["vf"], lanes=st["lanes"],
                 cap=st["cap"], length=st["length"], occ_scale=occ_scale,
                 nT=G["nT"], nL=G["nL"], days=np.array(G["days"]))
        print(f"  static {panel}", flush=True)


def nets(trees_csv, out_csv):
    """Mean of the six Task 1 nets (T1_TOD=const, T1_TILE_L=64 in the env),
    blended 0.85 / 0.15 with the tree ensemble - the s55 state file."""
    import t1_nn as NN
    models = {n: NN.load_net(n) for n in SIX}
    parts = []
    for panel in NN.PANELS:
        for split in ("validation", "private"):
            key = pd.read_csv(ROOT / "task1" / panel / split / "sample_submission_state.csv",
                              dtype={"link_id": str, "station_id": str})
            ts = pd.to_datetime(key.timestamp)
            row = col = None
            acc_v = acc_q = None
            for n in SIX:
                G, speed, flow, _ = NN.predict_grid(models[n], panel, split)[:4]
                if row is None:
                    di = {d: i for i, d in enumerate(G["days"])}
                    li = {l: i for i, l in enumerate(G["link_ids"])}
                    row = (ts.dt.strftime("%Y-%m-%d").map(di).to_numpy() * 288 +
                           (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy())
                    col = key.link_id.map(li).to_numpy()
                    assert not np.isnan(row).any() and not pd.isna(col).any()
                    row = row.astype(int); col = col.astype(int)
                v, q = speed[row, col], flow[row, col]
                acc_v = v if acc_v is None else acc_v + v
                acc_q = q if acc_q is None else acc_q + q
            d = key.copy()
            d["speed_kmh"] = acc_v / len(SIX); d["flow_vph"] = acc_q / len(SIX)
            parts.append(d)
            print(f"  [{panel}/{split}] {len(key):,} rows", flush=True)
    sub = pd.concat(parts, ignore_index=True)
    tree = pd.read_csv(trees_csv, dtype={"link_id": str, "station_id": str})
    kcols = ["panel", "timestamp", "link_id"]
    m = sub.merge(tree[kcols + ["speed_kmh", "flow_vph"]], on=kcols, suffixes=("", "_t"))
    assert len(m) == len(sub), f"{len(m)} vs {len(sub)}"
    for c in ("speed_kmh", "flow_vph"):
        m[c] = (1 - W_TREE) * m[c] + W_TREE * m[f"{c}_t"]
    m[COLS].to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(m):,} rows", flush=True)


def package(final_dir, zip_path):
    """Merge the three task files onto the upload template, validate, round, zip -
    the same steps as runs/build_variant.py."""
    final_dir = Path(final_dir)
    import merge_submission as M
    import validate_submission as V
    M.OUT = final_dir
    M.FILES = {t: final_dir / f"{t}_submission.csv"
               for t in ("state", "queue", "odme")}
    M.main()
    V.OUT = final_dir
    V.main()
    out_csv = final_dir / "submission_compact.csv"
    first = True
    for ch in pd.read_csv(final_dir / "submission.csv", chunksize=1_000_000):
        ch["speed_kmh"] = ch.speed_kmh.round(2)
        ch["flow_vph"] = ch.flow_vph.round(1)
        ch["path_flow"] = ch.path_flow.round(2)
        ch["queue_pred"] = ch.queue_pred.astype("int8")
        ch.to_csv(out_csv, mode="w" if first else "a", header=first, index=False,
                  lineterminator="\n")
        first = False
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(out_csv, arcname="submission.csv")
    (final_dir / "submission.csv").unlink(); out_csv.unlink()
    print(f"wrote {zip_path}  md5 {md5(zip_path)}", flush=True)


def verify(new_zip, ref_zip):
    """Compare a rebuilt upload with the submitted one, value by value."""
    rd = lambda p: pd.read_csv(zipfile.ZipFile(p).open("submission.csv"))
    a, b = rd(new_zip), rd(ref_zip)
    assert len(a) == len(b) and (a.submission_id.values == b.submission_id.values).all()
    print(f"md5 new {md5(new_zip)}  reference {md5(ref_zip)}  identical zip: {md5(new_zip) == md5(ref_zip)}")
    for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        d = (a[c] - b[c]).abs()
        print(f"  {c:10s} rows differing {int((d > 0).sum()):>9,}  max |diff| {d.max():.4g}")
    return a, b


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"nn_static": nn_static, "nets": nets, "package": package, "verify": verify}[cmd](*args)
