"""What S_queue the genuine forecaster actually scores, aggregated the way the board does.

`queue_eval.py` is obsolete: it starts every grid from `mavail & (msp <= thr)` -
reading the published horizon - which is exactly the route the organizers closed
on 2026-09-10. It also reports one panel at a time and never aggregates.

This forecasts from history alone, over held-out train windows, on all eight
queue panels, and folds the result up the way the leaderboard does:

    IoU per window  ->  mean per (panel, condition)
                    ->  0.5*(onset + ongoing) per panel
                    ->  mean per family
                    ->  mean over families

The last two steps matter: eight panels in four families, two directions each, so
a panel in a small family carries more weight than one window count would
suggest. Reporting a flat mean over windows would flatter or punish us by a few
points for no reason.

    python qeval2.py                 # the shipped narrow model
    QSHOCK=1 python qeval2.py        # the space-time model, if it has been trained
    python qeval2.py --both          # both, side by side, same windows

The windows are the last EVAL_DAYS of train, sampled with a fixed seed, so two
runs compare like for like. Threshold: whatever the model bundle carries, plus a
sweep, because the tuned value came from the same script that trained it and is
worth re-checking against a metric this sensitive to set size.
"""
import os, pickle, sys, time
import numpy as np

from common import OUT, QUEUE_PANELS, FAMILY
from qdata import load_panel, make_priors, origins, window_xy
from qfeat import HOR

EVAL_DAYS = 60
N_WIN = 200                      # per condition per panel
SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60)


def panel_iou(panel, pid, bundles, rng_seed=3):
    """IoU per (bundle, condition, threshold) for one panel's held-out windows."""
    D = load_panel(panel)
    pri = make_priors(D)
    nT = D["q"].shape[0]
    nd = nT // 288
    rng = np.random.default_rng(rng_seed)
    on, ong = origins(D)
    on = on[on // 288 >= nd - EVAL_DAYS]
    ong = ong[ong // 288 >= nd - EVAL_DAYS]
    on = rng.choice(on, min(N_WIN, len(on)), replace=False)
    ong = rng.choice(ong, min(N_WIN, len(ong)), replace=False)

    acc = {name: {c: {p: [] for p in SWEEP} for c in ("onset", "ongoing")}
           for name in bundles}
    for cond, origs in ((0, on), (1, ong)):
        cname = "onset" if cond == 0 else "ongoing"
        for T in origs:
            sl = slice(T + 1, T + 1 + HOR)
            e = D["elig"][sl]
            truth = (D["q"][sl] & e).reshape(-1)
            emask = e.reshape(-1)
            for name, b in bundles.items():
                X, _, _, _ = window_xy(D, pri, T, cond, pid, with_y=False,
                                       shock=b["shock"])
                p = b["models"][cond].predict(X)
                for pstar in SWEEP:
                    pred = (p >= pstar) & emask
                    u = np.logical_or(pred, truth).sum()
                    acc[name][cname][pstar].append(
                        1.0 if u == 0 else
                        float(np.logical_and(pred, truth).sum() / u))
    return {n: {c: {p: float(np.mean(v)) for p, v in d.items()}
                for c, d in cd.items()} for n, cd in acc.items()}, len(on), len(ong)


def aggregate(per_panel, name, pstar):
    """panel -> family -> mean, mirroring the leaderboard."""
    fams = sorted({FAMILY[p] for p in per_panel})
    pv = {p: 0.5 * (per_panel[p][name]["onset"][pstar]
                    + per_panel[p][name]["ongoing"][pstar])
          for p in per_panel}
    return float(np.mean([np.mean([v for p, v in pv.items() if FAMILY[p] == f])
                          for f in fams]))


def load_bundles(which):
    out = {}
    for name, fn, shock in (("narrow", "queue_model.pkl", False),
                            ("shock", "queue_model_shock.pkl", True)):
        if name not in which:
            continue
        path = OUT / fn
        if not path.exists():
            print(f"  ({name}: {fn} not present, skipping)")
            continue
        with open(path, "rb") as fh:
            b = pickle.load(fh)
        out[name] = {"models": b["models"], "thr": b.get("thr"), "shock": shock}
    assert out, "no model bundles available"
    return out


def main():
    args = sys.argv[1:]
    if "--both" in args:
        which = ("narrow", "shock")
    elif os.environ.get("QSHOCK") == "1":
        which = ("shock",)
    else:
        which = ("narrow",)
    bundles = load_bundles(which)
    print(f"evaluating {list(bundles)} on {len(QUEUE_PANELS)} queue panels, "
          f"last {EVAL_DAYS} train days\n")

    t0 = time.time()
    per_panel = {}
    for pid, panel in enumerate(QUEUE_PANELS):
        per_panel[panel], n_on, n_ong = panel_iou(panel, pid, bundles)
        best = {n: max(SWEEP, key=lambda p: 0.5 * (per_panel[panel][n]["onset"][p]
                                                   + per_panel[panel][n]["ongoing"][p]))
                for n in bundles}
        msg = "  ".join(
            f"{n}: {0.5*(per_panel[panel][n]['onset'][best[n]]+per_panel[panel][n]['ongoing'][best[n]]):.4f}"
            f"@{best[n]:.2f}" for n in bundles)
        print(f"{panel:11s} {n_on}+{n_ong} windows   {msg}   [{time.time()-t0:.0f}s]",
              flush=True)

    print(f"\n{'threshold':>10s} " + "".join(f"{n:>12s}" for n in bundles))
    rows = {}
    for pstar in SWEEP:
        rows[pstar] = {n: aggregate(per_panel, n, pstar) for n in bundles}
        print(f"{pstar:10.2f} " + "".join(f"{rows[pstar][n]:12.4f}" for n in bundles))

    print()
    for n in bundles:
        b = max(SWEEP, key=lambda p: rows[p][n])
        s = rows[b][n]
        print(f"{n:8s} best S_queue {s:.4f} at p*={b:.2f}   "
              f"-> composite contribution {0.30*s:.4f}")
    if len(bundles) == 2:
        bn = max(SWEEP, key=lambda p: rows[p]["narrow"])
        bs = max(SWEEP, key=lambda p: rows[p]["shock"])
        d = rows[bs]["shock"] - rows[bn]["narrow"]
        print(f"\nspace-time features: {d:+.4f} S_queue  ->  {0.30*d:+.4f} composite")


if __name__ == "__main__":
    main()
