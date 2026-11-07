"""Does the released topology explain the conservation residual we cannot reproduce?

physics_local.py builds the flux balance from positional neighbours: the link to
the left in corridor order is the inflow, the link itself is the outflow. On the
*ground truth* that gives e_lwr = 0.9865 - the residual is as large as the term
it is normalised by, i.e. no conservation at all. Meanwhile the organizers score
a perfect answer at S_physics 0.9544, so their residual is nearly zero on the
same data. One of the two constructions is wrong, and it is not theirs.

The release ships `network/lwr_mainline_topology.csv`, which names the actual
`incoming_link_ids`, `outgoing_link_ids`, `on_ramp_link_ids` and
`off_ramp_link_ids` per mainline link - the thing our proxy guessed at. It also
records how many of those boundaries carry a sensor, and the counts are grim:
every link has at least one unsensored incoming connector. So this asks two
separate questions.

  1. Scale. Positional neighbours give mean|rhs| = 10.0 veh against mean|dN| =
     0.57 veh, an 18x mismatch that alone explains e_lwr ~ 1. Does summing the
     topology's real incoming and outgoing sets bring rhs down to the size of
     dN? If yes, the convention was the bug.

  2. Coverage. On how many (link, timestamp) cells is every boundary flow in the
     topology actually observable from the release? If that number is zero, no
     local reconstruction of S_LWR is possible at any convention, and the
     leaderboard stays the only instrument for the physics component.

Read-only; writes nothing.
"""
import sys
import numpy as np
import pandas as pd

from common import ROOT, net_dir
from t1feat import load_grid
from t1build import ramp_matrices

DT_H = 5.0 / 60.0


def parse_ids(cell):
    if not isinstance(cell, str) or not cell.strip() or cell.strip().lower() == "nan":
        return []
    return [x.strip() for x in cell.replace(";", ",").split(",") if x.strip()]


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
    topo = pd.read_csv(net_dir(panel) / "lwr_mainline_topology.csv", dtype=str)
    Gm = load_grid(panel, "train", masked=True)
    Gt = load_grid(panel, "train", masked=False)
    o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
    sp, fl = Gt["speed"][:, o], Gt["flow"][:, o]
    del Gt
    link_ids = list(Gm["link_ids"])
    pos = {l: i for i, l in enumerate(link_ids)}
    R = ramp_matrices(panel, "train", link_ids)
    L = Gm["stat"]["length"][None, :]
    N = (fl / np.maximum(sp, 1.0)) * L
    dN = N[1:] - N[:-1]
    nT, nL = fl.shape

    key = "mainline_link_id" if "mainline_link_id" in topo.columns else "link_id"
    topo = topo.drop_duplicates(key).set_index(key)

    # ---- coverage: how much of the declared boundary is actually observable
    rows = []
    for l in link_ids:
        if l not in topo.index:
            rows.append(dict(link=l, n_in=0, n_out=0, in_seen=0, out_seen=0))
            continue
        r = topo.loc[l]
        inc = parse_ids(r.get("incoming_link_ids", ""))
        out = parse_ids(r.get("outgoing_link_ids", ""))
        rows.append(dict(link=l, n_in=len(inc), n_out=len(out),
                         in_seen=sum(x in pos for x in inc),
                         out_seen=sum(x in pos for x in out)))
    cov = pd.DataFrame(rows)
    full = (cov.n_in > 0) & (cov.n_out > 0) & (cov.in_seen == cov.n_in) \
        & (cov.out_seen == cov.n_out)
    print(f"{panel}: {len(cov)} mainline links")
    print(f"  declared incoming per link : {cov.n_in.value_counts().to_dict()}")
    print(f"  of which observable        : {(cov.in_seen == cov.n_in).sum()} links fully covered")
    print(f"  declared outgoing per link : {cov.n_out.value_counts().to_dict()}")
    print(f"  of which observable        : {(cov.out_seen == cov.n_out).sum()} links fully covered")
    print(f"  links with BOTH sides fully observable: {int(full.sum())}\n")

    # ---- scale: rhs from the topology, on the links where it can be formed
    q_in = np.zeros((nT, nL))
    q_out = np.zeros((nT, nL))
    usable = np.zeros(nL, bool)
    for j, l in enumerate(link_ids):
        if l not in topo.index:
            continue
        r = topo.loc[l]
        inc = [pos[x] for x in parse_ids(r.get("incoming_link_ids", "")) if x in pos]
        out = [pos[x] for x in parse_ids(r.get("outgoing_link_ids", "")) if x in pos]
        if not inc or not out:
            continue
        q_in[:, j] = fl[:, inc].sum(1)
        q_out[:, j] = fl[:, out].sum(1)
        usable[j] = True

    print(f"links where a two-sided balance can be formed from observed "
          f"mainline flows alone: {int(usable.sum())}")
    if not usable.any():
        print("nothing to measure; the boundary flux is not in the release.")
        return

    variants = {
        "topology in/out": (q_in, q_out),
        "topology + ramps": (q_in + R["on"], q_out + R["off"]),
        "positional (current proxy)": (
            np.concatenate([np.full((nT, 1), np.nan), fl[:, :-1]], 1) + R["on"],
            fl + R["off"]),
    }
    print(f"\nmean|dN| over usable links = "
          f"{np.nanmean(np.abs(dN[:, usable])):.3f} veh\n")
    print(f"{'variant':30s} {'mean|rhs|':>10s} {'e_lwr':>8s} {'S_LWR':>7s} "
          f"{'corr':>7s} {'cells':>10s}")
    for name, (qi, qo) in variants.items():
        rhs = (DT_H * (qi - qo))[:-1]
        m = (np.isfinite(dN) & np.isfinite(rhs) & (R["miss"][:-1] == 0)
             & usable[None, :])
        if not m.any():
            print(f"{name:30s} {'-':>10s}")
            continue
        e = np.abs(dN[m] - rhs[m]).sum() / (np.abs(rhs[m]).sum() + 1e-9)
        c = np.corrcoef(dN[m], rhs[m])[0, 1]
        print(f"{name:30s} {np.abs(rhs[m]).mean():10.3f} {e:8.4f} "
              f"{max(0, 1 - min(1, e)):7.4f} {c:+7.4f} {int(m.sum()):10,}")


if __name__ == "__main__":
    main()
