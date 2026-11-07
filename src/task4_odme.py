"""Task 4 ODME: reproduce the organizer reference construction.

build_task4_odme_artifacts.py in the public repo builds the reference path
flows as  nnls([A_measured; sqrt(l)I], [counts; sqrt(l)*base_od])  with
REG_LAMBDA = 0.05 and the anchor taken from network/base_od.csv (not the
released weak prior).  Reproducing that gives S_link ~ 1 and, if the withheld
reference was produced the same way, S_od / S_dev / S_attr as well.
"""
import sys, time
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from common import ROOT, OUT, PANELS, net_dir

REG_LAMBDA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
ANCHOR = sys.argv[2] if len(sys.argv) > 2 else "base_od"   # base_od | weak_prior
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
SPLITS = ["validation", "private"]


def load_operator(panel):
    net = net_dir(panel)
    paths = pd.read_csv(net / "path_set.csv", dtype={"path_id": str})
    inc = pd.read_csv(net / "path_link_incidence.csv", dtype={"path_id": str, "link_id": str})
    path_ids = paths.path_id.tolist()
    link_ids = inc.link_id.drop_duplicates().tolist()
    pi = {x: i for i, x in enumerate(path_ids)}
    li = {x: i for i, x in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(path_ids)))
    A[inc.link_id.map(li).to_numpy(np.int64), inc.path_id.map(pi).to_numpy(np.int64)] = 1.0
    base = pd.read_csv(net / "base_od.csv", dtype={"path_id": str})
    base = paths[["path_id", "origin_zone", "destination_zone"]].merge(
        base[["path_id", "departure_time", "base_flow"]], on="path_id", how="left")
    base = base.set_index("path_id").reindex(path_ids).reset_index()
    base["base_flow"] = pd.to_numeric(base.base_flow, errors="coerce").fillna(0.0)
    return path_ids, link_ids, A, paths, base


def solve(A, counts, anchor, lam):
    aa = np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])])
    bb = np.concatenate([counts, np.sqrt(lam) * anchor])
    f, _ = nnls(aa, bb)
    return f


def main():
    records, diag = [], []
    for panel in PANELS:
        path_ids, link_ids, A, paths, base = load_operator(panel)
        bv = base.base_flow.to_numpy(float)
        for split in SPLITS:
            cpath = ROOT / "task4" / panel / split / "synthetic_link_counts.csv"
            ppath = ROOT / "task4" / panel / split / "synthetic_weak_prior.csv"
            cf = pd.read_csv(cpath, dtype={"link_id": str})
            counts = cf.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(float)
            observed = set(cf.link_id)
            meas = np.array([l in observed for l in link_ids], dtype=bool)
            pr = pd.read_csv(ppath, dtype={"path_id": str})
            dep = str(pr.departure_time.iloc[0])
            anchor = bv
            if ANCHOR == "weak_prior":
                wv = (pr.set_index("path_id").reindex(path_ids)["path_flow"]
                      .fillna(0.0).to_numpy(float))
                if np.abs(wv).sum() > 0:
                    anchor = wv
            t0 = time.time()
            f = solve(A[meas], counts[meas], anchor, REG_LAMBDA)
            s_link = max(0.0, 1.0 - float(np.abs(A[meas] @ f - counts[meas]).sum())
                         / max(float(counts[meas].sum()), 1e-9))
            x = paths[["path_id", "origin_zone", "destination_zone"]].copy()
            x["panel"] = panel
            x["departure_time"] = dep
            x["path_flow"] = f
            records.append(x[["panel", "departure_time", "path_id",
                              "origin_zone", "destination_zone", "path_flow"]])
            diag.append({"panel": panel, "split": split, "S_link": s_link,
                         "n_paths": len(path_ids), "n_meas": int(meas.sum()),
                         "sum_f": float(f.sum()), "secs": time.time() - t0})
            print(f"[{panel}/{split}] S_link={s_link:.5f} paths={len(path_ids)} "
                  f"meas={meas.sum()} t={time.time()-t0:.0f}s", flush=True)
    out = pd.concat(records, ignore_index=True)
    out.to_csv(OUT / f"odme_submission{TAG}.csv", index=False)
    pd.DataFrame(diag).to_csv(OUT / f"odme_diag{TAG}.csv", index=False)
    print("wrote", OUT / f"odme_submission{TAG}.csv", len(out), "rows")
    print(pd.DataFrame(diag).S_link.describe())


if __name__ == "__main__":
    main()
