"""S_FD the way the board most likely computes it: per submitted state row.

Compares, per panel (regime-averaged):
    sub      our s55 submission rows (validation + private) - exactly what the board sees
    truth_t  the truth at train target cells (last 60 days) - what a perfect answer scores
    truth_a  the truth on every train cell (what the whole-grid local proxy is dominated by)

    python fd_rows_probe.py [path/to/state_submission.csv]
"""
import sys
import numpy as np
import pandas as pd
from common import ROOT, net_dir
from t1feat import load_grid

PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N",
          "D7_I405_S", "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]


def fd_table(panel):
    fd = pd.read_csv(net_dir(panel) / "fd_parameters.csv", dtype={"link_id": str})
    return fd.drop_duplicates("link_id").set_index("link_id")


def s_fd(v, q, lanes, vf, cap, kc, kj):
    lanes = np.maximum(lanes, 1.0)
    ql = q / lanes; kl = q / np.maximum(v, 1.0) / lanes
    kc, kj, cap = kc / lanes, kj / lanes, cap / lanes
    qfd = np.where(kl <= kc, vf * kl, cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - kl, 0))
    m = np.isfinite(ql) & np.isfinite(qfd)
    return float(max(0.0, 1 - np.abs(ql[m] - qfd[m]).sum() / (np.abs(ql[m]).sum() + 1e-9)))


def params(fd, links):
    f = fd.reindex(links)
    return (f.lanes.to_numpy(float), f.free_speed_kmh.to_numpy(float), f.capacity_vph.to_numpy(float),
            f.critical_density.to_numpy(float), f.k_jam.to_numpy(float))


def main():
    sub_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "runs/s55_t1-tod-fix/out/state_submission.csv")
    sub = pd.read_csv(sub_path, dtype={"link_id": str})
    print(f"{'panel':12s} {'sub':>7s} {'truth_t':>8s} {'truth_a':>8s}")
    res = []
    for panel in PANELS:
        fd = fd_table(panel)
        s = sub[sub.panel == panel]
        vals = [np.mean([s_fd(g.speed_kmh.to_numpy(), g.flow_vph.to_numpy(), *params(fd, g.link_id))
                         for _, g in s.groupby("mask_regime")])]
        Gm = load_grid(panel, "train", masked=True); Gt = load_grid(panel, "train", masked=False)
        o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
        v_t, q_t = Gt["speed"][:, o], Gt["flow"][:, o]
        nT = v_t.shape[0]; rows = np.arange(nT) >= nT - 60 * 288
        tgt = np.isnan(Gm["speed"]) & np.isfinite(v_t) & rows[:, None]
        allc = np.isfinite(v_t) & rows[:, None]
        P = params(fd, Gm["link_ids"])
        for m in (tgt, allc):
            ti, tj = np.nonzero(m)
            vals.append(s_fd(v_t[ti, tj], q_t[ti, tj], *(p[tj] for p in P)))
        res.append(vals)
        print(f"{panel:12s} " + " ".join(f"{x:8.4f}" for x in vals), flush=True)
    print(f"{'mean':12s} " + " ".join(f"{x:8.4f}" for x in np.mean(res, 0)))


if __name__ == "__main__":
    main()
