"""Rank every trained queue model, and their averages, on the calibrated harness.

Scores each `out/queue_model*.pkl` on the organizers' own train windows, over
every horizon cell, with the leaderboard's panel -> family aggregation - the
convention `qeval_official.py` settled on, which reads about 1.2x high against
the board rather than 1.5x.

Averaging is included because it is the one thing that has reliably paid on this
competition: on Task 1 the pooled/per-panel blend was the only gain in a batch of
five. Members here differ in what they were fed - more windows, all the
negatives, a shorter holdout - rather than in architecture, so they are less
diverse than Task 1's were, but the test is free once the probabilities exist.

Probabilities are computed once per (window, model) and cached in memory, so
adding a combination costs nothing.

    python qcompare.py
"""
import itertools
import pickle
import warnings
from pathlib import Path

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
SWEEP = (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)
MAX_COMBO = 3


def _describe(models):
    """What feature set a bundle wants, read from the booster rather than its name.

    Detecting this from the filename broke the moment a model was called
    `bothnp` instead of `bothnp...noprior`: a 57-feature model was handed 62
    columns and LightGBM refused. The booster carries its own feature names, so
    ask it.
    """
    names = set(models[0].feature_name())
    return {"shock": "d_dn_q" in names, "noprior": "pri_on" not in names,
            "ramp": "merge_load" in names}


def discover():
    """Every queue model on disk, with whether it needs the qshock features."""
    out = {}
    for f in sorted(OUT.glob("queue_model*.pkl")):
        name = f.stem.replace("queue_model", "").lstrip("_") or "base"
        try:
            with open(f, "rb") as fh:
                b = pickle.load(fh)
        except Exception as e:
            print(f"  ({f.name} unreadable: {e})")
            continue
        out[name] = dict(models=b["models"], kind=qkind.kind_of(b),
                         **_describe(b["models"]))
    return out


def collect_probs(bundles):
    """(model, panel, window) -> probability vector, plus the truth per window."""
    probs = {n: [] for n in bundles}
    truths, keys = [], []
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
            truths.append((D["q"][rows] & D["elig"][rows]).reshape(-1))
            keys.append((panel, "onset" if "onset" in str(w.condition) else "ongoing"))
            times = [T - pd.Timedelta(minutes=5 * (HIST - i)) for i in range(HIST)]
            sp, fl, oc, pc = history_arrays(h[h.window_id == w.window_id], lids, times)
            cond = 0 if "onset" in str(w.condition) else 1
            tod0 = T.hour * 12 + T.minute // 5
            pw = dict(pri_link=np.tile(pri["rate_link"], (HOR, 1)),
                      pri_tod=pri["rate_tod"][[(tod0 + s) % 288 for s in range(1, HOR + 1)]],
                      pri_on=pri["P_on"], pri_ong=pri["P_ong"],
                      pri_cond=pri["P_on"] if cond == 0 else pri["P_ong"])
            for n, b in bundles.items():
                R = (qramp.history_for(panel, SPLIT, lids, T, HIST)
                     if b["ramp"] else None)
                H = history_summary(sp, fl, oc, pc, D["stat"]["vf"], D["stat"]["cap"],
                                    shock=b["shock"], ramp=R,
                                    lanes=D["stat"]["lanes"])
                X, _ = build_rows(H, D["stat"], pw, tod0, int(T.dayofweek), cond, pid,
                                  shock=b["shock"], noprior=b["noprior"],
                                  ramp=b["ramp"])
                probs[n].append(qkind.to_score(b["kind"], b["models"][cond].predict(X)))
        print(f"  {panel} done", flush=True)
    return probs, truths, keys


def score(pv, truths, keys, pstar):
    """Leaderboard aggregation of IoU at one threshold."""
    per = {}
    for p, t, (panel, cond) in zip(pv, truths, keys):
        pred = p >= pstar
        u = np.logical_or(pred, t).sum()
        iou = 1.0 if u == 0 else float(np.logical_and(pred, t).sum() / u)
        per.setdefault((panel, cond), []).append(iou)
    pan = {}
    for (panel, cond), v in per.items():
        pan.setdefault(panel, []).append(float(np.mean(v)))
    pv2 = {p: float(np.mean(v)) for p, v in pan.items()}
    fams = sorted({FAMILY[p] for p in pv2})
    return float(np.mean([np.mean([v for p, v in pv2.items() if FAMILY[p] == f])
                          for f in fams]))


def main():
    bundles = discover()
    assert bundles, "no queue models found"
    print(f"models: {list(bundles)}\n")
    probs, truths, keys = collect_probs(bundles)

    cands = {n: probs[n] for n in bundles}
    names = sorted(bundles)
    for k in range(2, min(MAX_COMBO, len(names)) + 1):
        for combo in itertools.combinations(names, k):
            cands["avg(" + "+".join(combo) + ")"] = [
                np.mean([probs[n][i] for n in combo], axis=0)
                for i in range(len(truths))]

    rows = []
    for name, pv in cands.items():
        best = max(SWEEP, key=lambda ps: score(pv, truths, keys, ps))
        rows.append(dict(name=name, pstar=best, s_queue=score(pv, truths, keys, best)))
    rows.sort(key=lambda r: -r["s_queue"])

    base = next((r["s_queue"] for r in rows if r["name"] == "base"), None)
    print(f"\n{'candidate':34s} {'p*':>5s} {'S_queue':>9s} {'vs base':>9s} {'board est':>10s}")
    for r in rows:
        d = f"{r['s_queue']-base:+.4f}" if base is not None else "-"
        print(f"{r['name']:34s} {r['pstar']:5.2f} {r['s_queue']:9.4f} {d:>9s}"
              f" {r['s_queue']/1.20:10.4f}")
    pd.DataFrame(rows).to_csv(OUT / "qcompare.csv", index=False)
    print(f"\nthe harness reads ~1.20x high; 'board est' divides by that.")
    print(f"current shipped model on the board: S_queue 0.5025, composite 0.77393")


if __name__ == "__main__":
    main()
