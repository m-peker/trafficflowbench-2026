"""Choose the predicted set to maximise expected IoU, not by a fixed threshold.

The model is fitted to log loss and the score is space-time IoU, which is a set
metric. A fixed probability cutoff is not the rule that maximises it. For a
candidate set S with independent per-cell probabilities,

    E|S n T| = sum_{i in S} p_i
    E|S u T| = |S| + sum_{i not in S} p_i

so the ratio of expectations is

    E[IoU] ~ ( sum_{i in S} p_i ) / ( |S| + sum_{i not in S} p_i )

Adding the cells in descending order of p makes this unimodal in |S|, so the
best set is found by one sort and one scan. The cutoff it implies moves per
window: in a window where the model is confident about many cells it pays to be
generous, and in a quiet one it pays to predict almost nothing. A single global
threshold cannot do both, and 0.50 was chosen on the average.

This is decision rule only - the model, the features and the training are
untouched, so it can be tested on the probabilities the current model already
produces.

    python qdecide.py            # fixed threshold vs expected-IoU set, same windows
"""
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import ROOT, OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors
from qfeat import HIST, HOR, history_summary, build_rows
from qpredict import history_arrays

SPLIT = "train"
FIXED = {0: 0.50, 1: 0.45}
FLOORS = (0.0, 0.05, 0.10, 0.20)     # never include a cell below this


def best_set(p, floor=0.0):
    """Size of the expected-IoU-optimal prefix of p sorted descending."""
    o = np.argsort(-p)
    ps = p[o]
    csum = np.cumsum(ps)
    total = ps.sum()
    k = np.arange(1, len(ps) + 1)
    # numerator: sum of p inside; denominator: |S| + sum of p outside
    val = csum / (k + (total - csum))
    if floor > 0:
        allowed = ps >= floor
        val = np.where(allowed, val, -1.0)
    if not np.isfinite(val).any() or val.max() <= 0:
        return np.zeros(len(p), bool)
    kbest = int(np.argmax(val)) + 1
    out = np.zeros(len(p), bool)
    out[o[:kbest]] = True
    return out


def main():
    with open(OUT / "queue_model.pkl", "rb") as fh:
        models = pickle.load(fh)["models"]
    names = ["fixed"] + [f"expIoU floor={f:.2f}" for f in FLOORS]
    acc = {n: {} for n in names}
    sizes = {n: [] for n in names}

    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel)
        pri = make_priors(D)
        lids = D["link_ids"]
        dix = {str(d): i for i, d in enumerate(D["days"])}
        base = ROOT / "task2" / panel / SPLIT
        wi = pd.read_csv(base / "window_index.csv")
        h = pd.read_parquet(base / "window_history.parquet")
        h["link_id"] = h.link_id.astype(str)
        h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
        tmp = {n: [] for n in names}

        for w in wi.itertuples(index=False):
            T = pd.Timestamp(w.forecast_origin)
            d0 = T.strftime("%Y-%m-%d")
            if d0 not in dix:
                continue
            t0 = dix[d0] * 288 + T.hour * 12 + T.minute // 5
            rows = np.arange(t0 + 1, t0 + 1 + HOR)
            truth = (D["q"][rows] & D["elig"][rows]).reshape(-1)
            times = [T - pd.Timedelta(minutes=5 * (HIST - i)) for i in range(HIST)]
            sp, fl, oc, pc = history_arrays(h[h.window_id == w.window_id], lids, times)
            H = history_summary(sp, fl, oc, pc, D["stat"]["vf"], D["stat"]["cap"])
            cond = 0 if "onset" in str(w.condition) else 1
            tod0 = T.hour * 12 + T.minute // 5
            pw = dict(pri_link=np.tile(pri["rate_link"], (HOR, 1)),
                      pri_tod=pri["rate_tod"][[(tod0 + s) % 288 for s in range(1, HOR + 1)]],
                      pri_on=pri["P_on"], pri_ong=pri["P_ong"],
                      pri_cond=pri["P_on"] if cond == 0 else pri["P_ong"])
            X, _ = build_rows(H, D["stat"], pw, tod0, int(T.dayofweek), cond, pid)
            prob = models[cond].predict(X)

            preds = {"fixed": prob >= FIXED[cond]}
            for f in FLOORS:
                preds[f"expIoU floor={f:.2f}"] = best_set(prob, f)
            for n, pred in preds.items():
                u = np.logical_or(pred, truth).sum()
                tmp[n].append(1.0 if u == 0 else
                              float(np.logical_and(pred, truth).sum() / u))
                sizes[n].append(int(pred.sum()))
        for n in names:
            acc[n][panel] = float(np.mean(tmp[n]))
        print(f"  {panel} done", flush=True)

    fams = sorted({FAMILY[p] for p in QUEUE_PANELS})
    print(f"\n{'rule':22s} {'S_queue':>9s} {'mean cells predicted':>21s}")
    base = None
    for n in names:
        a = float(np.mean([np.mean([v for p, v in acc[n].items() if FAMILY[p] == f])
                           for f in fams]))
        if base is None:
            base = a
        print(f"{n:22s} {a:9.4f} {np.mean(sizes[n]):21.1f}"
              f"   {a - base:+.4f}")
    print("\n(the harness reads about 1.2x high; a gain here should carry at ~0.83 of"
          "\nits size to the board)")


if __name__ == "__main__":
    main()
