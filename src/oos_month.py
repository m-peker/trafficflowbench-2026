"""Per-family queue score by month of the train split, out of sample in time.

The board's per-family S_queue (s29-s31 isolation) is uneven - D7_I210 0.866,
D7_I405 0.781, D7_I10 0.588, D12_I5 0.554 - while our last-60-day harness says
0.96 / 0.93 / 0.87 / 0.90. Validation is March 2031 and private April, past the
end of train (Feb 2031), so a family whose queues drift would show it as a
downward trend across train months. Priors come from the fit days only
(everything before the last 60), so months inside the holdout are honest.

    python oos_month.py
"""
import nice  # noqa: F401
import pickle, warnings
import numpy as np
warnings.filterwarnings("ignore")
from common import OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors, window_xy, contract_ongoing
from qfeat import HIST, HOR
import qkind

VAL = 60
N_PER_MONTH = 40


def main():
    b = pickle.load(open(OUT / "queue_model_both.pkl", "rb"))
    M, kind = b["models"], qkind.kind_of(b)
    rows = {}
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel); q = D["q"]; e = D["elig"]; nT = q.shape[0]; nd = nT // 288
        fit = np.zeros(nd, bool); fit[:nd - VAL] = True
        pri = make_priors(D, fit)
        months = np.array([d[:7] for d in D["days"]])
        anyq = q.any(1).astype(np.int32); cs = np.concatenate([[0], np.cumsum(anyq)])
        T = np.arange(HIST, nT - HOR - 1)
        s6 = T[((cs[T] - cs[T - HIST]) == 0) & (anyq[T] == 0)
               & ((cs[T + HOR] - cs[T + 1]) == 0) & (anyq[T + HOR] == 1)]
        ong = contract_ongoing(D, T[anyq[T] == 1])
        for m in sorted(set(months)):
            days = np.where(months == m)[0]
            sel = lambda a: a[np.isin(a // 288, days)]
            rng = np.random.default_rng(3)
            for cond, pool, thr in ((0, sel(s6), 0.60), (1, sel(ong), 0.50)):
                if len(pool) == 0:
                    continue
                pick = rng.choice(pool, min(N_PER_MONTH, len(pool)), replace=False)
                v = []
                for t0 in pick:
                    X, _, _, _ = window_xy(D, pri, int(t0), cond, pid, with_y=False)
                    p = qkind.to_score(kind, M[cond].predict(X)).reshape(HOR, D["nL"])
                    if cond == 0:
                        pr = (p[-1] >= thr) & e[t0 + HOR]; tr = q[t0 + HOR] & e[t0 + HOR]
                    else:
                        sl = slice(t0 + 1, t0 + 1 + HOR)
                        pr = (p >= thr) & e[sl]; tr = q[sl] & e[sl]
                    u = (pr | tr).sum(); v.append(1.0 if u == 0 else (pr & tr).sum() / u)
                rows.setdefault((FAMILY[panel], m, cond), []).append(float(np.mean(v)))
        print(f"  {panel} done", flush=True)
    fams = sorted({k[0] for k in rows}); mons = sorted({k[1] for k in rows})
    print("\nall-cell IoU by family and month (mean of onset and ongoing)")
    print(f"{'family':10s} " + " ".join(f"{m[2:]:>8s}" for m in mons))
    for f in fams:
        line = []
        for m in mons:
            vals = [np.mean(rows[(f, m, c)]) for c in (0, 1) if (f, m, c) in rows]
            line.append(np.mean(vals) if vals else np.nan)
        print(f"{f:10s} " + " ".join(f"{v:8.3f}" for v in line))
    print("\nonset only")
    for f in fams:
        print(f"{f:10s} " + " ".join(f"{np.mean(rows[(f,m,0)]):8.3f}" if (f,m,0) in rows else "     nan" for m in mons))
    print("\nongoing only")
    for f in fams:
        print(f"{f:10s} " + " ".join(f"{np.mean(rows[(f,m,1)]):8.3f}" if (f,m,1) in rows else "     nan" for m in mons))


if __name__ == "__main__":
    main()
