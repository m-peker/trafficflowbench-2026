"""Out-of-sample ongoing harness: contract ongoing windows from the last 60 train days.

Windows are drawn the way synthetic_release_v1.json says the selector draws them
(queue at the origin, queue in the horizon, persistence IoU <= 0.9), so this is
the scored population, and the days are unseen by every QVAL_DAYS=60 model.
Every horizon cell is scored. Per-model probabilities cache to
out/oos_ong_<model>.npz, so adding a model costs only that model.

    python oos_ongoing.py both bothC
"""
import nice  # noqa: F401
import sys, pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")
from common import OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors, window_xy, contract_ongoing
from qfeat import HIST, HOR
import qkind

VAL, N = 60, 120


def windows():
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel); q = D["q"]; nT = q.shape[0]; nd = nT // 288
        fit = np.zeros(nd, bool); fit[:nd - VAL] = True
        T = np.arange(HIST, nT - HOR - 1)
        T = T[(T // 288 >= nd - VAL) & q[T].any(1)]
        T = contract_ongoing(D, T)
        pick = np.sort(np.random.default_rng(21).choice(T, min(N, len(T)), replace=False))
        yield pid, panel, D, make_priors(D, fit), pick


def probs(name):
    f = OUT / f"oos_ong_{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z["p"], z["truth"], z["panel"]
    b = pickle.load(open(OUT / f"queue_model_{name}.pkl", "rb"))
    M, kind = b["models"][1], qkind.kind_of(b)
    names = set(M.feature_name())
    assert "pri_on" in names and "d_dn_q" not in names and "merge_load" not in names
    P, Tr, Pa = [], [], []
    for pid, panel, D, pri, pick in windows():
        for t0 in pick:
            X, _, _, _ = window_xy(D, pri, int(t0), 1, pid, with_y=False)
            P.append(qkind.to_score(kind, M.predict(X)))
            Tr.append((D["q"][t0 + 1:t0 + 1 + HOR] & D["elig"][t0 + 1:t0 + 1 + HOR]).reshape(-1))
            Pa.append(panel)
        print(f"  [{name}] {panel}: {len(pick)}", flush=True)
    P, Tr, Pa = np.array(P, dtype=object), np.array(Tr, dtype=object), np.array(Pa)
    np.savez_compressed(f, p=P, truth=Tr, panel=Pa)
    return P, Tr, Pa


def agg(vals, panels):
    per = {}
    for v, p in zip(vals, panels): per.setdefault(p, []).append(v)
    fams = sorted({FAMILY[p] for p in per})
    return float(np.mean([np.mean([np.mean(per[p]) for p in per if FAMILY[p] == f])
                          for f in fams]))


def main():
    names = sys.argv[1:] or ["both"]
    got = {n: probs(n) for n in names}
    thrs = (0.40, 0.45, 0.50, 0.525, 0.55, 0.60)
    print(f"\n{len(got[names[0]][1])} contract ongoing windows, last {VAL} train days")
    print(f"{'model':12s} " + " ".join(f"{t:>6.3f}" for t in thrs))
    for n, (P, Tr, Pa) in got.items():
        s = []
        for th in thrs:
            v = []
            for p, t in zip(P, Tr):
                pr = p >= th; u = (pr | t).sum()
                v.append(1.0 if u == 0 else (pr & t).sum() / u)
            s.append(agg(v, Pa))
        print(f"{n:12s} " + " ".join(f"{x:6.4f}" for x in s))


if __name__ == "__main__":
    main()
