"""Rescore cached OOS probabilities with predictions masked to eligible cells.

window_index.csv carries future_coverage = 1.0 on every window of every split,
while only ~77% of horizon cells are eligible: the queue truth is the underlying
state on all cells. Dropout is MCAR, so IoU over eligible cells (predictions and
truth both masked) estimates the all-cell IoU; the old harness counted every
prediction on an unobserved cell as a false positive.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
from common import OUT, QUEUE_PANELS
from qdata import load_panel, contract_ongoing
from qfeat import HIST, HOR
from oos_onset import agg

VAL = 60


def elig_lists():
    f = OUT / "oos_elig_masks.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True); return z["on"], z["ong"]
    on, ong = [], []
    for panel in QUEUE_PANELS:
        D = load_panel(panel); q = D["q"]; nT = q.shape[0]; nd = nT // 288
        anyq = q.any(1).astype(np.int32); cs = np.concatenate([[0], np.cumsum(anyq)])
        T = np.arange(HIST, nT - HOR - 1)
        s6 = T[((cs[T] - cs[T - HIST]) == 0) & (anyq[T] == 0)
               & ((cs[T + HOR] - cs[T + 1]) == 0) & (anyq[T + HOR] == 1)]
        pool = s6[s6 // 288 >= nd - VAL]
        for t in np.random.default_rng(11).choice(pool, min(150, len(pool)), replace=False):
            on.append(D["elig"][t + HOR])
        To = contract_ongoing(D, T[(T // 288 >= nd - VAL) & q[T].any(1)])
        for t in np.sort(np.random.default_rng(21).choice(To, min(120, len(To)), replace=False)):
            ong.append(D["elig"][t + 1:t + 1 + HOR].reshape(-1))
    on, ong = np.array(on, dtype=object), np.array(ong, dtype=object)
    np.savez_compressed(f, on=on, ong=ong)
    return on, ong


def score(pv, truth, panels, masks, th):
    v = []
    for p, t, m in zip(pv, truth, masks):
        pr = (p >= th) & m; u = (pr | t).sum()
        v.append(1.0 if u == 0 else (pr & t).sum() / u)
    return agg(v, panels)


THRS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def main():
    m_on, m_ong = elig_lists()
    names = sys.argv[1:]
    Z = np.load(OUT / "oos_onset_probs.npz", allow_pickle=True)
    G = np.load(OUT / "oos_ong_both.npz", allow_pickle=True)
    on = {"both": Z["p_both"]}; ong = {"both": G["p"]}
    for n in names:
        a = np.load(OUT / f"nn_oos_onset_{n}.npz", allow_pickle=True)["p"]
        b = np.load(OUT / f"nn_oos_ong_{n}.npz", allow_pickle=True)["p"]
        on[f"nn_{n}"] = a; ong[f"nn_{n}"] = b
        on[f"avg(both,{n})"] = [(x + y) / 2 for x, y in zip(Z["p_both"], a)]
        ong[f"avg(both,{n})"] = [(x + y) / 2 for x, y in zip(G["p"], b)]
    for title, c, T, P, M in (("ONSET step 6", on, Z["truth"], Z["panel"], m_on),
                              ("ONGOING", ong, G["truth"], G["panel"], m_ong)):
        print(f"\n{title}, predictions masked to eligible cells ({len(T)} windows)")
        print(f"{'candidate':20s} " + " ".join(f"{t:>6.2f}" for t in THRS) + "   best")
        for k, pv in c.items():
            s = [score(pv, T, P, M, th) for th in THRS]
            print(f"{k:20s} " + " ".join(f"{x:6.4f}" for x in s) + f"  {max(s):.4f}")


if __name__ == "__main__":
    main()
