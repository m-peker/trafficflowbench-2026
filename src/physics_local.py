"""Local Task 3 proxy on the train split.

score_task3.py cannot be run on the public package because the organizer
boundary flows are withheld.  On train we have the unmasked layer, so the true
flow into and out of every link is available and the check can be reproduced:

    rhs   = dt * (inflow + on_ramp - outflow - off_ramp)     from the truth
    dN    = N(t+dt) - N(t),   N = (q/v) * L                  from our state
    e_lwr = sum|dN - rhs| / sum|rhs|
    S_LWR = max(0, 1 - min(1, e_lwr))

The point is the denominator: it is the net flux, roughly a percent of the
vehicles on a link, so S_LWR is far more sensitive to noise in dN than S_state
is to noise in a cell.  That is what this script is for - measuring how much a
reconstruction choice moves S_LWR, which the leaderboard only reports folded
into one composite number.
"""
import sys, gc, pickle
import numpy as np
import pandas as pd
from common import ROOT, OUT, WORK, net_dir, links_table
from t1feat import load_grid
from t1build import build, ramp_matrices

DT_H = 5.0 / 60.0
EVAL_DAYS = 45          # the tail of train the saved models early-stopped on


def corridor_arrays(panel):
    """Truth grid, masked grid and ramp aggregates, all in corridor order."""
    Gm = load_grid(panel, "train", masked=True)
    Gt = load_grid(panel, "train", masked=False)
    o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
    truth_sp = Gt["speed"][:, o]
    truth_fl = Gt["flow"][:, o]
    del Gt
    R = ramp_matrices(panel, "train", Gm["link_ids"])
    return Gm, truth_sp, truth_fl, R


def lwr_score(N, truth_fl, R, stat, mask_rows, ramp_ok):
    """S_LWR over the given rows, with the truth supplying the boundary flows."""
    nT, nL = N.shape
    dN = N[1:] - N[:-1]
    q_in = np.full_like(truth_fl, np.nan)
    q_in[:, 1:] = truth_fl[:, :-1]                 # flow of the upstream link
    q_out = truth_fl                               # flow leaving the link
    rhs = DT_H * (q_in + R["on"] - q_out - R["off"])
    rhs = rhs[:-1]
    ok = (np.isfinite(dN) & np.isfinite(rhs) & ramp_ok[:-1]
          & mask_rows[:-1, None])
    ok[:, 0] = False                               # first link has no upstream
    if not ok.any():
        return 0.0, 0.0, 0
    e = np.abs(dN[ok] - rhs[ok]).sum() / (np.abs(rhs[ok]).sum() + 1e-9)
    return max(0.0, 1.0 - min(1.0, float(e))), float(e), int(ok.sum())


def fd_score(speed, flow, stat, rows):
    lanes = np.maximum(stat["lanes"], 1.0)[None, :]
    q_lane = flow / lanes
    k_lane = flow / np.maximum(speed, 1.0) / lanes
    fd = pd.read_csv(net_dir(stat["panel"]) / "fd_parameters.csv",
                     dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    kcrit = np.array([fd.critical_density.get(l, 55.0) for l in stat["link_ids"]])[None, :] / lanes
    kjam = np.array([fd.k_jam.get(l, 300.0) for l in stat["link_ids"]])[None, :] / lanes
    cap = stat["cap"][None, :] / lanes
    vf = stat["vf"][None, :]
    q_fd = np.where(k_lane <= kcrit, vf * k_lane,
                    (cap / np.maximum(kjam - kcrit, 1e-9)) * np.maximum(kjam - k_lane, 0))
    m = rows[:, None] & np.isfinite(q_lane) & np.isfinite(q_fd)
    return float(max(0.0, 1.0 - np.abs(q_lane[m] - q_fd[m]).sum()
                     / (np.abs(q_lane[m]).sum() + 1e-9)))


def smooth_density(N, filled, w):
    """Blend the predicted vehicle count toward a centred moving average."""
    if w <= 0:
        return N
    k = 2 * w + 1
    pad = np.pad(N, ((w, w), (0, 0)), mode="edge")
    cs = np.cumsum(np.nan_to_num(pad), 0)
    cs = np.concatenate([np.zeros((1, N.shape[1])), cs])
    sm = (cs[k:] - cs[:-k]) / k
    return np.where(filled, sm, N)


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
    Gm, truth_sp, truth_fl, R = corridor_arrays(panel)
    stat = dict(Gm["stat"])
    stat["panel"] = panel
    stat["link_ids"] = Gm["link_ids"]
    nT, nL = Gm["speed"].shape
    nd = nT // 288
    day = np.repeat(np.arange(nd), 288)
    rows = day >= nd - EVAL_DAYS
    L = stat["length"][None, :]

    ramp_ok = R["miss"] == 0

    with open(OUT / f"t1_model_{panel}.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    models = bundle["models"]

    target_mask = np.zeros((nT, nL), bool)
    target_mask[rows] = True
    r = build(panel, "train", targets="blanked", target_mask=target_mask)
    ti, tj = r["ti"], r["tj"]
    pred_sp = np.clip(models["speed"].predict(r["X"], num_iteration=models["speed"].best_iteration),
                      5.0, stat["vf"][tj] * 1.05)
    pred_fl = np.clip(models["flow"].predict(r["X"], num_iteration=models["flow"].best_iteration)
                      * stat["lanes"][tj], 5.0, stat["cap"][tj] * 1.15)
    del r
    gc.collect()

    filled = np.zeros((nT, nL), bool)
    filled[ti, tj] = True

    variants = {}
    # (a) perfect answer: the released observations themselves
    variants["perfect"] = (truth_sp.copy(), truth_fl.copy())
    # (b) our reconstruction
    sp = Gm["speed"].copy(); fl = Gm["flow"].copy()
    sp[ti, tj] = pred_sp; fl[ti, tj] = pred_fl
    variants["ours"] = (sp, fl)
    # (c) historical-mean style naive fill, for the low anchor
    prof = np.zeros((288, nL))
    cnt = np.zeros((288, nL))
    tod = np.arange(nT) % 288
    av = np.isfinite(Gm["speed"])
    np.add.at(prof, tod, np.where(av, np.nan_to_num(Gm["speed"]), 0.0))
    np.add.at(cnt, tod, av)
    prof_sp = prof / np.maximum(cnt, 1)
    prof2 = np.zeros((288, nL)); cnt2 = np.zeros((288, nL))
    np.add.at(prof2, tod, np.where(av, np.nan_to_num(Gm["flow"]), 0.0))
    np.add.at(cnt2, tod, av)
    prof_fl = prof2 / np.maximum(cnt2, 1)
    sp_n = Gm["speed"].copy(); fl_n = Gm["flow"].copy()
    sp_n[ti, tj] = prof_sp[ti % 288, tj]
    fl_n[ti, tj] = prof_fl[ti % 288, tj]
    variants["naive_profile"] = (sp_n, fl_n)

    print(f"{panel}: evaluating last {EVAL_DAYS} train days, "
          f"{int(filled[rows].sum()):,} filled cells "
          f"({filled[rows].mean():.1%} of the grid)\n", flush=True)
    print(f"{'variant':22s} {'S_LWR':>8s} {'e_lwr':>8s} {'S_FD':>7s} "
          f"{'S_physics':>10s} {'cells':>10s}")
    for name, (s, f) in variants.items():
        N = (f / np.maximum(s, 1.0)) * L
        slwr, e, n = lwr_score(N, truth_fl, R, stat, rows, ramp_ok)
        sfd = fd_score(s, f, stat, rows)
        print(f"{name:22s} {slwr:8.4f} {e:8.4f} {sfd:7.4f} "
              f"{(sfd + 2 * slwr) / 3:10.4f} {n:10,}")

    # (d) how much does smoothing the filled cells help?
    s, f = variants["ours"]
    N0 = (f / np.maximum(s, 1.0)) * L
    for w in (1, 2, 3, 5):
        N = smooth_density(N0, filled, w)
        slwr, e, n = lwr_score(N, truth_fl, R, stat, rows, ramp_ok)
        print(f"{'ours smooth w=' + str(w):22s} {slwr:8.4f} {e:8.4f} "
              f"{'':7s} {'':10s} {n:10,}")


if __name__ == "__main__":
    main()
