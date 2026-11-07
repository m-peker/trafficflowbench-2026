"""Out-of-sample onset harness: step-6 onset windows from the last 60 train days.

The organizers' train windows are in-sample for every model, so they cannot rank
onset models honestly. Every model here was fit with QVAL_DAYS=60, so windows
opening in those days are unseen (bar early stopping). Only horizon step 6 is
scored, matching the s15 rule (onset queue first appears exactly at step 6).

Probabilities are cached to out/oos_onset_probs.npz; rerun is free.
    python oos_onset.py
"""
import nice  # noqa: F401
import itertools, pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")
from common import OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors, window_xy
from qfeat import HIST, HOR
import qkind

VAL, N = 60, 150
MODELS = {"both": "queue_model_both.pkl", "both6": "queue_model_both6.pkl",
          "relaxfit": "queue_model_relaxfit.pkl", "win4x": "queue_model_win4x.pkl",
          "base": "queue_model.pkl", "negall": "queue_model_negall.pkl"}
CACHE = OUT / "oos_onset_probs.npz"


def collect():
    B = {}
    for n, f in MODELS.items():
        b = pickle.load(open(OUT / f, "rb"))
        names = set(b["models"][0].feature_name())
        assert "pri_on" in names and "d_dn_q" not in names and "merge_load" not in names, n
        B[n] = (b["models"][0], qkind.kind_of(b))
    P = {n: [] for n in B}; truth, panel_of = [], []
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel); q = D["q"]; nT = q.shape[0]; nd = nT // 288
        fit = np.zeros(nd, bool); fit[:nd - VAL] = True; pri = make_priors(D, fit)
        anyq = q.any(1).astype(np.int32); cs = np.concatenate([[0], np.cumsum(anyq)])
        T = np.arange(HIST, nT - HOR - 1)
        s6 = T[((cs[T] - cs[T - HIST]) == 0) & (anyq[T] == 0)
               & ((cs[T + HOR] - cs[T + 1]) == 0) & (anyq[T + HOR] == 1)]
        pool = s6[s6 // 288 >= nd - VAL]
        pick = np.random.default_rng(11).choice(pool, min(N, len(pool)), replace=False)
        for t0 in pick:
            X, _, _, _ = window_xy(D, pri, int(t0), 0, pid, with_y=False)
            for n, (M, kind) in B.items():
                P[n].append(qkind.to_score(kind, M.predict(X)).reshape(HOR, D["nL"])[-1])
            truth.append(q[t0 + HOR] & D["elig"][t0 + HOR]); panel_of.append(panel)
        print(f"  {panel}: {len(pick)} windows", flush=True)
    np.savez_compressed(CACHE, panel=np.array(panel_of),
                        truth=np.array(truth, dtype=object),
                        **{f"p_{n}": np.array(v, dtype=object) for n, v in P.items()})


def agg(vals, panels):
    per = {}
    for v, p in zip(vals, panels): per.setdefault(p, []).append(v)
    fams = sorted({FAMILY[p] for p in per})
    return float(np.mean([np.mean([np.mean(per[p]) for p in per if FAMILY[p] == f])
                          for f in fams]))


def iou(pr, t):
    u = (pr | t).sum()
    return 1.0 if u == 0 else (pr & t).sum() / u


def main():
    if not CACHE.exists():
        collect()
    Z = np.load(CACHE, allow_pickle=True)
    panels, truth = Z["panel"], Z["truth"]
    P = {k[2:]: Z[k] for k in Z.files if k.startswith("p_")}
    names = sorted(P)
    cands = dict(P)
    for k in (2, 3):
        for c in itertools.combinations(names, k):
            cands["+".join(c)] = [np.mean([P[n][i] for n in c], 0) for i in range(len(truth))]
    thrs = (0.45, 0.50, 0.55, 0.60, 0.65)
    rows = []
    for name, pv in cands.items():
        s = [agg([iou(p >= th, t) for p, t in zip(pv, truth)], panels) for th in thrs]
        rows.append((name, s))
    base = dict(rows)["both"][thrs.index(0.60)]
    rows.sort(key=lambda r: -np.mean(r[1]))
    print(f"{len(truth)} windows; both@0.60 = {base:.4f}")
    print(f"{'candidate':26s} " + " ".join(f"{t:>6.2f}" for t in thrs) + "   mean")
    for name, s in rows[:25]:
        print(f"{name:26s} " + " ".join(f"{v:6.4f}" for v in s) + f"  {np.mean(s):.4f}")


if __name__ == "__main__":
    main()
