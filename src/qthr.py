"""Cache one model's window probabilities, then make threshold choices instant.

Every threshold question so far has cost a ten-minute rebuild of the same
predictions. They do not change, so they are computed once into
`out/qprob_<model>.npz` and every sweep after that is arithmetic.

The question that prompted this: the sweeps had all been constrained to a single
threshold shared by both conditions, and `qpredict --thr X` sets both to X, which
is what s12 shipped. But onset and ongoing are different problems - roughly 1.5
percent of cells positive against 14 percent - and the model's own tuning had
already picked 0.450 and 0.525 for them separately. A shared value cannot be
right for both.

    python qthr.py both            # cache, then sweep shared vs per-condition
"""
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from common import ROOT, OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors
from qfeat import HIST, HOR, history_summary, build_rows
import qkind
import qramp
from qpredict import history_arrays

SPLIT = "train"
GRID = np.round(np.arange(0.20, 0.81, 0.025), 3)


def build_cache(model_name):
    path = OUT / f"qprob_{model_name}.npz"
    if path.exists():
        return path
    fn = "queue_model.pkl" if model_name == "base" else f"queue_model_{model_name}.pkl"
    with open(OUT / fn, "rb") as fh:
        models = pickle.load(fh)["models"]
    names = set(models[0].feature_name())
    shock, noprior = "d_dn_q" in names, "pri_on" not in names
    wants_ramp = "merge_load" in names
    with open(OUT / fn, "rb") as fh:
        kind = qkind.kind_of(pickle.load(fh))

    probs, truths, panels, conds = [], [], [], []
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
        for w in wi.itertuples(index=False):
            T = pd.Timestamp(w.forecast_origin)
            d0 = T.strftime("%Y-%m-%d")
            if d0 not in dix:
                continue
            t0 = dix[d0] * 288 + T.hour * 12 + T.minute // 5
            rows = np.arange(t0 + 1, t0 + 1 + HOR)
            if rows[-1] >= D["q"].shape[0]:
                continue
            truths.append((D["q"][rows] & D["elig"][rows]).reshape(-1))
            times = [T - pd.Timedelta(minutes=5 * (HIST - i)) for i in range(HIST)]
            sp, fl, oc, pc = history_arrays(h[h.window_id == w.window_id], lids, times)
            R = (qramp.history_for(panel, SPLIT, lids, T, HIST)
                 if wants_ramp else None)
            H = history_summary(sp, fl, oc, pc, D["stat"]["vf"], D["stat"]["cap"],
                                shock=shock, ramp=R, lanes=D["stat"]["lanes"])
            cond = 0 if "onset" in str(w.condition) else 1
            tod0 = T.hour * 12 + T.minute // 5
            pw = dict(pri_link=np.tile(pri["rate_link"], (HOR, 1)),
                      pri_tod=pri["rate_tod"][[(tod0 + s) % 288 for s in range(1, HOR + 1)]],
                      pri_on=pri["P_on"], pri_ong=pri["P_ong"],
                      pri_cond=pri["P_on"] if cond == 0 else pri["P_ong"])
            X, _ = build_rows(H, D["stat"], pw, tod0, int(T.dayofweek), cond, pid,
                              shock=shock, noprior=noprior,
                              ramp=wants_ramp)
            probs.append(qkind.to_score(kind, models[cond].predict(X)).astype(np.float32))
            panels.append(panel)
            conds.append(cond)
        print(f"  {panel} cached", flush=True)
    np.savez_compressed(
        path,
        probs=np.array(probs, dtype=object),
        truths=np.array(truths, dtype=object),
        panels=np.array(panels, dtype=object),
        conds=np.array(conds, np.int8), allow_pickle=True)
    return path


def load(model_name):
    z = np.load(OUT / f"qprob_{model_name}.npz", allow_pickle=True)
    return z["probs"], z["truths"], z["panels"], z["conds"]


def agg(probs, truths, panels, conds, t_on, t_ong):
    per = {}
    for p, t, pan, c in zip(probs, truths, panels, conds):
        pred = p >= (t_on if c == 0 else t_ong)
        u = np.logical_or(pred, t).sum()
        per.setdefault((pan, c), []).append(
            1.0 if u == 0 else float(np.logical_and(pred, t).sum() / u))
    pv = {}
    for (pan, c), v in per.items():
        pv.setdefault(pan, []).append(float(np.mean(v)))
    pv = {p: float(np.mean(v)) for p, v in pv.items()}
    fams = sorted({FAMILY[p] for p in pv})
    return float(np.mean([np.mean([v for p, v in pv.items() if FAMILY[p] == f])
                          for f in fams]))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "both"
    build_cache(name)
    P, T, PAN, C = load(name)
    print(f"\n{len(P)} windows cached for `{name}`\n")

    shared = [(agg(P, T, PAN, C, t, t), t) for t in GRID]
    bs, bt = max(shared)
    print(f"best shared threshold: {bt:.3f} -> {bs:.4f}")

    best, arg = -1, None
    for a in GRID:
        for b in GRID:
            v = agg(P, T, PAN, C, a, b)
            if v > best:
                best, arg = v, (a, b)
    print(f"best pair: onset={arg[0]:.3f} ongoing={arg[1]:.3f} -> {best:.4f}"
          f"   ({best-bs:+.4f} over shared)\n")

    print("ongoing profile at the best onset:")
    for b in GRID[::2]:
        v = agg(P, T, PAN, C, arg[0], b)
        print(f"  ongoing {b:.3f}  {v:.4f}{'   <-- best' if b == arg[1] else ''}")


if __name__ == "__main__":
    main()
