"""Neural queue model vs LightGBM `both`, and their average, on the OOS harnesses.

Reads the cached probabilities that oos_onset.py / oos_ongoing.py / nn_queue.py
eval wrote for the same windows in the same order, so nothing is recomputed.
Reports each candidate at every threshold and at its own best, since a gain that
only exists at one threshold is threshold tuning, not a better model.

    python nn_compare.py c1 [c2 ...]
"""
import sys
import numpy as np
from common import OUT
from oos_onset import agg, iou

THRS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def table(title, truth, panels, cands):
    print(f"\n{title} ({len(truth)} windows)")
    print(f"{'candidate':22s} " + " ".join(f"{t:>6.2f}" for t in THRS) + "   best")
    for name, pv in cands.items():
        s = [agg([iou(p >= th, t) for p, t in zip(pv, truth)], panels) for th in THRS]
        print(f"{name:22s} " + " ".join(f"{v:6.4f}" for v in s) + f"  {max(s):.4f}")


def main():
    names = sys.argv[1:]
    Z = np.load(OUT / "oos_onset_probs.npz", allow_pickle=True)
    c = {"both": Z["p_both"], "both+both6+negall":
         [np.mean([Z["p_both"][i], Z["p_both6"][i], Z["p_negall"][i]], 0)
          for i in range(len(Z["truth"]))]}
    for n in names:
        nn = np.load(OUT / f"nn_oos_onset_{n}.npz", allow_pickle=True)["p"]
        c[f"nn_{n}"] = nn
        c[f"avg(both,nn_{n})"] = [(a + b) / 2 for a, b in zip(Z["p_both"], nn)]
    if len(names) > 1:
        c["nn_all"] = [np.mean([c[f"nn_{n}"][i] for n in names], 0) for i in range(len(Z["truth"]))]
        c["avg(both,nn_all)"] = [(a + b) / 2 for a, b in zip(Z["p_both"], c["nn_all"])]
    table("ONSET step 6, out of sample", Z["truth"], Z["panel"], c)

    G = np.load(OUT / "oos_ong_both.npz", allow_pickle=True)
    c = {"both": G["p"]}
    for n in names:
        nn = np.load(OUT / f"nn_oos_ong_{n}.npz", allow_pickle=True)["p"]
        c[f"nn_{n}"] = nn
        c[f"avg(both,nn_{n})"] = [(a + b) / 2 for a, b in zip(G["p"], nn)]
    if len(names) > 1:
        c["nn_all"] = [np.mean([c[f"nn_{n}"][i] for n in names], 0) for i in range(len(G["truth"]))]
        c["avg(both,nn_all)"] = [(a + b) / 2 for a, b in zip(G["p"], c["nn_all"])]
    table("ONGOING contract windows, out of sample", G["truth"], G["panel"], c)


if __name__ == "__main__":
    main()
