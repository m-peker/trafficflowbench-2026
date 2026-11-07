"""S_queue on the organizers' own windows, scored over every horizon cell.

Two corrections, both found the hard way on 2026-09-10.

`qeval2.py` samples windows out of the train grid with our `origins()` rule and
scored the shipped forecaster at 0.8189. The first post-fix submission implies
about 0.49. The gap is not the model and not its inputs - the prediction-time
history is fully populated, 12 of 12 timestamps and every link - it is the
window selection. The release ships exactly ten windows per panel per split,
five onset and five ongoing, chosen by the organizers, and it ships them for
**train** as well. Those are the ones to score against, because they are drawn
the same way as the validation and private ones.

This runs the submitted code path - `window_history.parquet` through
`qpredict.history_arrays` - against the truth in the unmasked train grid, and
aggregates the way the board does: per window, per condition, per panel, per
family, then the mean over families.

Second: score every cell in the horizon, not only the `is_score_eligible` ones.
The release is synthetic, so the organizers hold the true state everywhere and
`is_score_eligible` marks what *we* can see rather than what they grade. The
published persistence baseline settles which convention they use - it scores
0.252, and this harness reads 0.2792 over all cells against 0.3339 over eligible
ones only. Under the all-cells convention the shipped forecaster reads 0.60
where the board says 0.5025, so the harness is about 1.2x optimistic; under the
eligible-only convention it read 0.76, which is 1.5x out.

Treat the level as indicative and the ordering as usable.

    python qeval_official.py                 # shipped narrow model
    python qeval_official.py --both          # narrow and space-time side by side
"""
import os, pickle, sys, time
import numpy as np
import pandas as pd

from common import ROOT, OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors, window_prior
from qfeat import HIST, HOR, history_summary, build_rows
from qpredict import history_arrays

SPLIT = "train"
SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)


_DAYIX = {}


def grid_index(D, ts):
    """Timestamp -> row in the train grid, or None if that day is not in it."""
    key = id(D)
    if key not in _DAYIX:
        _DAYIX[key] = {str(d): i for i, d in enumerate(D["days"])}
    i = _DAYIX[key].get(ts.strftime("%Y-%m-%d"))
    if i is None:
        return None
    return i * 288 + ts.hour * 12 + ts.minute // 5


def evaluate(bundles):
    per_panel = {}
    t0 = time.time()
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel)
        pri = make_priors(D)
        link_ids = D["link_ids"]
        base = ROOT / "task2" / panel / SPLIT
        wi = pd.read_csv(base / "window_index.csv")
        hist = pd.read_parquet(base / "window_history.parquet")
        hist["link_id"] = hist.link_id.astype(str)
        hist["timestamp"] = pd.to_datetime(hist.timestamp, utc=True)

        acc = {n: {"onset": {p: [] for p in SWEEP},
                   "ongoing": {p: [] for p in SWEEP}} for n in bundles}
        used = 0
        for w in wi.itertuples(index=False):
            T = pd.Timestamp(w.forecast_origin)
            rows = [grid_index(D, T + pd.Timedelta(minutes=5 * s))
                    for s in range(1, HOR + 1)]
            if any(r is None for r in rows):
                continue
            rows = np.array(rows)
            truth = (D["q"][rows] & D["elig"][rows]).reshape(-1)
            times = [T - pd.Timedelta(minutes=5 * (HIST - i)) for i in range(HIST)]
            h = hist[hist.window_id == w.window_id]
            sp, fl, oc, pc = history_arrays(h, link_ids, times)
            cond = 0 if "onset" in str(w.condition) else 1
            cname = "onset" if cond == 0 else "ongoing"
            tod0 = T.hour * 12 + T.minute // 5
            used += 1
            for name, b in bundles.items():
                H = history_summary(sp, fl, oc, pc, D["stat"]["vf"],
                                    D["stat"]["cap"], shock=b["shock"])
                pw = dict(pri_link=np.tile(pri["rate_link"], (HOR, 1)),
                          pri_tod=pri["rate_tod"][[(tod0 + s) % 288
                                                   for s in range(1, HOR + 1)]],
                          pri_on=pri["P_on"], pri_ong=pri["P_ong"],
                          pri_cond=pri["P_on"] if cond == 0 else pri["P_ong"])
                X, _ = build_rows(H, D["stat"], pw, tod0, int(T.dayofweek), cond,
                                  pid, shock=b["shock"])
                p = b["models"][cond].predict(X)
                for pstar in SWEEP:
                    # score every horizon cell, not just the eligible ones. The
                    # data is synthetic, so the organizers hold the true state
                    # everywhere; is_score_eligible says what *we* can see, not
                    # what they grade. The persistence baseline settles it: they
                    # publish 0.252 for it, this harness reads 0.2792 scoring
                    # all cells and 0.3339 scoring only eligible ones.
                    pred = p >= pstar
                    u = np.logical_or(pred, truth).sum()
                    acc[name][cname][pstar].append(
                        1.0 if u == 0 else
                        float(np.logical_and(pred, truth).sum() / u))
        per_panel[panel] = {n: {c: {p: (float(np.mean(v)) if v else np.nan)
                                    for p, v in d.items()}
                                for c, d in cd.items()} for n, cd in acc.items()}
        msg = "  ".join(
            f"{n}: " + "/".join(
                f"{per_panel[panel][n][c][0.50]:.3f}" for c in ("onset", "ongoing"))
            for n in bundles)
        print(f"{panel:11s} {used:2d}/{len(wi)} windows   {msg} (at p*=0.50)"
              f"   [{time.time()-t0:.0f}s]", flush=True)
    return per_panel


def aggregate(per_panel, name, pstar):
    fams = sorted({FAMILY[p] for p in per_panel})
    pv = {}
    for p in per_panel:
        a = per_panel[p][name]["onset"][pstar]
        b = per_panel[p][name]["ongoing"][pstar]
        vals = [v for v in (a, b) if np.isfinite(v)]
        if vals:
            pv[p] = float(np.mean(vals))
    return float(np.mean([np.mean([v for p, v in pv.items() if FAMILY[p] == f])
                          for f in fams if any(FAMILY[p] == f for p in pv)]))


def main():
    which = ("narrow", "shock") if "--both" in sys.argv[1:] else (
        ("shock",) if os.environ.get("QSHOCK") == "1" else ("narrow",))
    bundles = {}
    for name, fn, shock in (("narrow", "queue_model.pkl", False),
                            ("shock", "queue_model_shock.pkl", True)):
        if name not in which or not (OUT / fn).exists():
            continue
        with open(OUT / fn, "rb") as fh:
            b = pickle.load(fh)
        bundles[name] = {"models": b["models"], "shock": shock}
    assert bundles, "no model available"
    print(f"scoring {list(bundles)} on the organizers' own {SPLIT} windows\n")

    per_panel = evaluate(bundles)
    print(f"\n{'threshold':>10s} " + "".join(f"{n:>12s}" for n in bundles))
    rows = {p: {n: aggregate(per_panel, n, p) for n in bundles} for p in SWEEP}
    for p in SWEEP:
        print(f"{p:10.2f} " + "".join(f"{rows[p][n]:12.4f}" for n in bundles))
    print()
    for n in bundles:
        b = max(SWEEP, key=lambda p: rows[p][n])
        print(f"{n:8s} best S_queue {rows[b][n]:.4f} at p*={b:.2f}"
              f"   -> composite contribution {0.30*rows[b][n]:.4f}")


if __name__ == "__main__":
    main()
