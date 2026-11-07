"""Which Task 2 truth convention reproduces the board?

Board (s32 split by s33): onset 0.7614, ongoing 0.7586.

H3  truth = queued AND eligible, predictions counted everywhere (the harness
    every Task 2 decision since 2026-09-16 was made on)
H2  truth = queued on every cell the scorer's underlying state covers; locally
    approximated by every cell with a released speed, eligible or not, and
    cells with no speed at all dropped from both sides

Same windows and cached probabilities as oos_onset.py / oos_ongoing.py.
    python q_truth_convention.py
"""
import nice  # noqa: F401
import numpy as np
from common import OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, contract_ongoing
from qfeat import HIST, HOR

VAL = 60


def agg(vals, panels):
    per = {}
    for v, p in zip(vals, panels): per.setdefault(p, []).append(v)
    fams = sorted({FAMILY[p] for p in per})
    return float(np.mean([np.mean([np.mean(per[p]) for p in per if FAMILY[p] == f])
                          for f in fams]))


def iou(pr, t, known=None):
    if known is not None:
        pr, t = pr & known, t & known
    u = (pr | t).sum()
    return 1.0 if u == 0 else (pr & t).sum() / u


def truths(D, rows):
    sp = D["speed"][rows]
    slow = np.isfinite(sp) & (sp <= 0.6 * D["stat"]["vf"][None, :])
    return D["q"][rows] & D["elig"][rows], slow, np.isfinite(sp)


def onset():
    Z = np.load(OUT / "oos_onset_probs.npz", allow_pickle=True)
    P = np.mean([Z[f"p_{n}"].tolist() for n in ("both", "both6", "negall")], 0) \
        if False else None
    pv = [np.mean([Z[f"p_{n}"][i] for n in ("both", "both6", "negall")], 0)
          for i in range(len(Z["truth"]))]
    i = 0; res = {"H3": [], "H2": [], "H2_known": []}; pan = []
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel); q = D["q"]; nT = q.shape[0]; nd = nT // 288
        anyq = q.any(1).astype(np.int32); cs = np.concatenate([[0], np.cumsum(anyq)])
        T = np.arange(HIST, nT - HOR - 1)
        s6 = T[((cs[T] - cs[T - HIST]) == 0) & (anyq[T] == 0)
               & ((cs[T + HOR] - cs[T + 1]) == 0) & (anyq[T + HOR] == 1)]
        pool = s6[s6 // 288 >= nd - VAL]
        pick = np.random.default_rng(11).choice(pool, min(150, len(pool)), replace=False)
        for t0 in pick:
            p = pv[i]; i += 1
            pr = p >= 0.60
            if not pr.any():
                pr = np.zeros_like(pr); pr[np.argmax(p)] = True       # never-empty rule
            h3, h2, kn = truths(D, slice(t0 + HOR, t0 + HOR + 1))
            assert (h3[0] == Z["truth"][i - 1]).all()
            res["H3"].append(iou(pr, h3[0])); res["H2"].append(iou(pr, h2[0]))
            res["H2_known"].append(iou(pr, h2[0], kn[0])); pan.append(panel)
    return {k: agg(v, pan) for k, v in res.items()}


def ongoing(name="both", th=0.50):
    Z = np.load(OUT / f"oos_ong_{name}.npz", allow_pickle=True)
    P = Z["p"]; i = 0; res = {"H3": [], "H2": [], "H2_known": []}; pan = []
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel); q = D["q"]; nT = q.shape[0]; nd = nT // 288
        T = np.arange(HIST, nT - HOR - 1)
        T = T[(T // 288 >= nd - VAL) & q[T].any(1)]
        T = contract_ongoing(D, T)
        pick = np.sort(np.random.default_rng(21).choice(T, min(120, len(T)), replace=False))
        for t0 in pick:
            pr = (P[i] >= th).reshape(HOR, -1)
            h3, h2, kn = truths(D, slice(t0 + 1, t0 + 1 + HOR))
            assert (h3.reshape(-1) == Z["truth"][i]).all()
            i += 1
            res["H3"].append(iou(pr, h3)); res["H2"].append(iou(pr, h2))
            res["H2_known"].append(iou(pr, h2, kn)); pan.append(panel)
    return {k: agg(v, pan) for k, v in res.items()}


if __name__ == "__main__":
    print("board        onset 0.7614   ongoing 0.7586")
    on = onset(); og = ongoing()
    for k in ("H3", "H2", "H2_known"):
        print(f"{k:12s} onset {on[k]:.4f}   ongoing {og[k]:.4f}")


def masks_ongoing(name="both", th=0.50):
    """Board-check: s24 zeroed pct<75 horizon cells and lost 0.154 of S_queue."""
    Z = np.load(OUT / f"oos_ong_{name}.npz", allow_pickle=True)
    P = Z["p"]; i = 0; pan = []
    res = {k: [] for k in ("none", "zero_inelig(s24)", "zero_missing", "zero_partial")}
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel); q = D["q"]; nT = q.shape[0]; nd = nT // 288
        T = np.arange(HIST, nT - HOR - 1)
        T = T[(T // 288 >= nd - VAL) & q[T].any(1)]
        T = contract_ongoing(D, T)
        pick = np.sort(np.random.default_rng(21).choice(T, min(120, len(T)), replace=False))
        for t0 in pick:
            rows = slice(t0 + 1, t0 + 1 + HOR)
            pr = (P[i] >= th).reshape(HOR, -1); i += 1
            _, h2, kn = truths(D, rows)
            el = D["elig"][rows]
            res["none"].append(iou(pr, h2))
            res["zero_inelig(s24)"].append(iou(pr & el, h2))
            res["zero_missing"].append(iou(pr & kn, h2))
            res["zero_partial"].append(iou(pr & (el | ~kn), h2))
            pan.append(panel)
    return {k: agg(v, pan) for k, v in res.items()}


if __name__ == "__main__":
    import sys
    if "masks" in sys.argv:
        for k, v in masks_ongoing().items():
            print(f"  ongoing under H2, {k:18s} {v:.4f}")
