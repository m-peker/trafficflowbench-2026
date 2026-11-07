"""Score the organizers' own 80 train windows with a model that never saw them.

Every scored window on the board is drawn by the organizers' selector; our
harnesses use pools we built ourselves. The only selector-drawn windows whose
truth we have are the 80 train ones, all in 2030-06-01..20 - and every model so
far was fitted on those days. `queue_model_nojune.pkl` (QHOLD_FIRST=30) is not,
so this reads our true performance on selector-drawn, unseen windows.

Reports both conventions: predictions everywhere against truth on eligible
cells (what the old harness did), and predictions masked to eligible cells (the
all-cell estimate under MCAR).

    python oos_selector.py nojune both
"""
import nice  # noqa: F401
import sys, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from common import ROOT, OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors, window_prior
from qfeat import HIST, HOR, history_summary, build_rows
from qpredict import history_arrays
import qkind

HOLD_FIRST = 30
THR = {0: 0.60, 1: 0.50}


def main():
    names = sys.argv[1:] or ["nojune"]
    bundles = {}
    for n in names:
        fn = "queue_model.pkl" if n == "base" else f"queue_model_{n}.pkl"
        b = pickle.load(open(OUT / fn, "rb"))
        bundles[n] = (b["models"], qkind.kind_of(b))
    out = {}
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel); lids = D["link_ids"]; stat = D["stat"]
        nd = len(D["days"])
        fit = np.ones(nd, bool); fit[:HOLD_FIRST] = False        # priors as the model saw them
        pri = make_priors(D, fit)
        dix = {d: i for i, d in enumerate(D["days"])}
        base = ROOT / "task2" / panel / "train"
        wi = pd.read_csv(base / "window_index.csv")
        hist = pd.read_parquet(base / "window_history.parquet")
        hist["link_id"] = hist.link_id.astype(str)
        hist["timestamp"] = pd.to_datetime(hist.timestamp, utc=True)
        for w in wi.itertuples(index=False):
            T = pd.Timestamp(w.forecast_origin)
            t0 = dix[T.strftime("%Y-%m-%d")] * 288 + T.hour * 12 + T.minute // 5
            cond = 0 if "onset" in str(w.condition) else 1
            times = [T - pd.Timedelta(minutes=5 * (HIST - i)) for i in range(HIST)]
            sp, fl, oc, pc = history_arrays(hist[hist.window_id == w.window_id], lids, times)
            H = history_summary(sp, fl, oc, pc, stat["vf"], stat["cap"])
            tod0 = T.hour * 12 + T.minute // 5
            X, _ = build_rows(H, stat, window_prior(pri, t0, cond), tod0,
                              int(T.dayofweek), cond, pid)
            sl = slice(t0 + 1, t0 + 1 + HOR)
            truth = (D["q"][sl] & D["elig"][sl]); elig = D["elig"][sl]
            for n, (M, kind) in bundles.items():
                p = qkind.to_score(kind, M[cond].predict(X)).reshape(HOR, D["nL"])
                f = p >= THR[cond]
                if cond == 0:
                    f[:HOR - 1] = False
                for tag, pred in (("all-cells-counted", f), ("eligible-only", f & elig)):
                    u = (pred | truth).sum()
                    iou = 1.0 if u == 0 else (pred & truth).sum() / u
                    out.setdefault((n, tag, cond, FAMILY[panel]), []).append(iou)
        print(f"  {panel} done", flush=True)
    for n in names:
        print(f"\n{n}:")
        for tag in ("all-cells-counted", "eligible-only"):
            for cond, cn in ((0, "onset"), (1, "ongoing")):
                per = {f: float(np.mean(v)) for (nn, tg, c, f), v in out.items()
                       if nn == n and tg == tag and c == cond}
                print(f"   {tag:18s} {cn:8s} " + "  ".join(f"{f} {v:.3f}" for f, v in sorted(per.items()))
                      + f"   | mean {np.mean(list(per.values())):.3f}")


if __name__ == "__main__":
    main()
