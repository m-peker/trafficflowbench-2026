"""Score arbitrary Task 2 model mixes on the board-matching OOS harnesses.

nn_compare.py only builds a fixed handful of combinations. This takes mixes on
the command line, reading whatever probabilities oos_onset.py / oos_ongoing.py /
nn_queue.py eval / nn_queue2.py eval already cached for the same windows in the
same order, so nothing is recomputed.

Members: `both`, `both6`, `negall` (LightGBM), `nn:NAME` (1-D conv),
`nn2:NAME` (2-D conv). A mix is members joined by `+`, averaged before the
threshold, exactly as qpredict2.py averages them.

    python q_mix.py ongoing both+nn:c1+nn:c2+nn:q1 both+nn:c1+nn:c2+nn:q1+nn:q4
    python q_mix.py onset both+both6+negall both+both6+negall+nn:q4
"""
import sys
import numpy as np
from common import OUT
from oos_onset import agg, iou

THRS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def reference(cond):
    """Truth and panels for the window set every member shares."""
    f = "oos_onset_probs.npz" if cond == "onset" else "oos_ong_both.npz"
    Z = np.load(OUT / f, allow_pickle=True)
    return Z["truth"], Z["panel"]


def load(cond, name):
    """Probabilities for one member. The conv-net caches hold only `p`; their
    windows are the same ones, in the same order, as the LightGBM caches."""
    tag = "onset" if cond == "onset" else "ong"
    if name.startswith("nn2:"):
        return np.load(OUT / f"nn2_oos_{tag}_{name[4:]}.npz", allow_pickle=True)["p"]
    if name.startswith("nn:"):
        return np.load(OUT / f"nn_oos_{tag}_{name[3:]}.npz", allow_pickle=True)["p"]
    if cond == "onset":
        return np.load(OUT / "oos_onset_probs.npz", allow_pickle=True)[f"p_{name}"]
    return np.load(OUT / f"oos_ong_{name}.npz", allow_pickle=True)["p"]


def main():
    cond = sys.argv[1]
    mixes = sys.argv[2:]
    truth, panels = reference(cond)
    print(f"{cond} - board-matching harness ({len(truth)} windows, "
          f"predictions counted everywhere)")
    print(f"{'mix':44s} " + " ".join(f"{t:>6.2f}" for t in THRS) + "    best")
    for mix in mixes:
        ps = []
        for m in mix.split("+"):
            p = load(cond, m)
            assert len(p) == len(truth), f"{m}: {len(p)} windows vs {len(truth)}"
            ps.append(p)
        avg = [np.mean([p[i] for p in ps], 0) for i in range(len(truth))]
        s = [agg([iou(p >= th, t) for p, t in zip(avg, truth)], panels) for th in THRS]
        print(f"{mix:44s} " + " ".join(f"{v:6.4f}" for v in s) + f"   {max(s):.4f}")


if __name__ == "__main__":
    main()
