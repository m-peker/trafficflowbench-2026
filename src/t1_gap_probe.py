"""How badly does Task 1 do inside the Task 2 gaps?

Validation/private blank the WHOLE corridor for the Task 2 horizon plus the 60
minutes after it (18 steps), at queue onsets and ongoing queues, and ~1.9% of
the scored Task 1 rows sit inside those spans. The nets were trained on
scattered single-cell blanks only and have never met a full-corridor blackout.

On the last 7 train days of each panel: place organizer-style windows (5 onset
at step 6, 5 ongoing, >= 6 h apart), blank rows T+1..T+18 on every link, restore
the raw history T-12..T-1 (the release publishes it unmasked), run the shipped
ensemble, and score the SAME target cells with and without the gaps.

    python t1_gap_probe.py
"""
import nice  # noqa: F401
import os, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("T1_TOD", "const"); os.environ.setdefault("T1_TILE_L", "64")
import t1_nn as NN
import t1feat
from qdata import load_panel, origins
from qfeat import HIST, HOR

NETS = os.environ.get("GAP_NETS", "f1,h1,i1,j1,m1,n1").split(",")
PANELS = ["D7_I10_E", "D7_I210_E", "D7_I405_N", "D12_I5_N"]
DAYS, GAP = 7, 18


def pick_windows(panel, nT):
    D = load_panel(panel)
    lo = nT - DAYS * 288
    on, ong = origins(D)
    rng = np.random.default_rng(3)
    chosen = []
    for pool in (on, ong):
        pool = pool[(pool >= lo + HIST) & (pool < nT - GAP - 1)]
        rng.shuffle(pool); k = 0
        for t in pool:
            if all(abs(t - c) >= 72 for c in chosen):
                chosen.append(int(t)); k += 1
            if k == 5:
                break
    return sorted(chosen), D["link_ids"]


def main():
    nets = {n: NN.load_net(n) for n in NETS}
    orig = t1feat.load_grid
    tot = {"in_gap": [], "out_gap": []}
    for panel in PANELS:
        Gm = orig(panel, "train", masked=True); Gt = orig(panel, "train", masked=False)
        o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
        tv, tq = Gt["speed"][:, o], Gt["flow"][:, o]
        nT = Gm["nT"]
        wins, qlinks = pick_windows(panel, nT)
        gap = np.zeros(nT, bool); hist = np.zeros(nT, bool)
        for t in wins:
            gap[t + 1:t + 1 + GAP] = True; hist[t - HIST:t] = True
        Gg = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in Gm.items()}
        for ch in ("speed", "flow", "occ"):
            Gg[ch][hist] = Gt[ch][:, o][hist]           # raw history, as released
            Gg[ch][gap] = np.nan                         # the blackout
        lo = nT - DAYS * 288
        tgt = np.isnan(Gm["speed"]) & np.isfinite(tv) & (Gt["pct"][:, o] >= 75)
        tgt[:lo] = False
        tgt_hist = tgt & hist[:, None]
        tgt &= ~hist[:, None]                            # history cells are not targets
        res = {}
        variants = (("gapped", Gg),) if os.environ.get("GAP_DUMP") else (("base", Gm), ("gapped", Gg))
        for tag, G in variants:
            NN.load_grid = lambda p, s, masked=True, G=G: G
            V, Q = [], []
            for n, net in nets.items():
                _, sp, fl, _ = NN.predict_grid(net, panel, "train", t_lo=lo - 256)[:4]
                V.append(sp); Q.append(fl)
            res[tag] = (np.mean(V, 0), np.mean(Q, 0))
        NN.load_grid = orig
        if os.environ.get("GAP_DUMP"):
            v, q = res["gapped"]
            tag = os.environ.get("GAP_TAG", "")
            np.savez_compressed(NN.OUT / (f"gap_probe_{panel}_{tag}.npz" if tag else f"gap_probe_{panel}.npz"), v=v.astype(np.float32),
                                q=q.astype(np.float32), wins=np.array(wins), lo=lo)
            continue
        lanes = np.maximum(Gm["stat"]["lanes"], 1.0)[None, :]
        print(f"\n{panel}: {len(wins)} windows, {int((tgt & gap[:, None]).sum())} targets in gaps, "
              f"{int((tgt & ~gap[:, None]).sum())} outside", flush=True)
        for where, m in (("in_gap", tgt & gap[:, None]), ("out_gap", tgt & ~gap[:, None])):
            for tag in ("base", "gapped"):
                v, q = res[tag]
                ev = v[m] - tv[m]; eq = (q[m] - tq[m]) / np.broadcast_to(lanes, m.shape)[m]
                sv, sq = np.sqrt(np.mean(ev ** 2)), np.sqrt(np.mean(eq ** 2))
                print(f"  {where:8s} {tag:7s} speed RMSE {sv:6.2f}  flow/lane RMSE {sq:6.1f}  "
                      f"S_state {0.54 * (1 - sv / 25) + 0.46 * (1 - sq / 600):.4f}", flush=True)
                if tag == "gapped":
                    tot[where].append((ev, eq))
    print("\npooled (gapped):")
    ev_in = np.concatenate([a for a, _ in tot["in_gap"]]); eq_in = np.concatenate([b for _, b in tot["in_gap"]])
    ev_out = np.concatenate([a for a, _ in tot["out_gap"]]); eq_out = np.concatenate([b for _, b in tot["out_gap"]])
    share = 0.019
    for name, a, b in (("in_gap", ev_in, eq_in), ("out_gap", ev_out, eq_out)):
        print(f"  {name:8s} speed MSE {np.mean(a ** 2):8.2f}  flow MSE {np.mean(b ** 2):9.1f}")
    mv = share * np.mean(ev_in ** 2) + (1 - share) * np.mean(ev_out ** 2)
    mq = share * np.mean(eq_in ** 2) + (1 - share) * np.mean(eq_out ** 2)
    print(f"  at the board's 1.9% gap share: speed RMSE x{np.sqrt(mv / np.mean(ev_out ** 2)):.3f}, "
          f"flow RMSE x{np.sqrt(mq / np.mean(eq_out ** 2)):.3f} over the no-gap error")


if __name__ == "__main__":
    main()
