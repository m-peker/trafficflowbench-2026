"""Two more Task 1 ensemble members, chosen for bias diversity rather than strength.

Averaging the per-panel and pooled fits was the only change in the five-run
batch that gained anything (+0.00132), so the question is what else to average.
An ensemble pays in proportion to how *differently* its members are wrong, not
how good each one is, so both additions are deliberately unlike the two we have.

  family - pooled over the two directions of one corridor family, five models.
    The existing pair sits at the two extremes of pooling: one corridor each, or
    all ten at once. A family shares geometry, demand pattern and sensor layout
    between its two directions but nothing with the other four, so its errors
    should correlate with neither extreme.

  alt - per-panel again, but a different learner: shallow trees (63 leaves not
    255), a slower rate, half the features per split, extra_trees randomization,
    and a different sampling seed. Same data, different inductive bias. Shallow
    heavily-randomized forests are wrong in smoother ways than deep ones, which
    is what makes their average with a deep fit worth more than either.

  phys - the same learner as `own`, trained to minimise the error in *density*
    rather than in speed. S_LWR reads k = q/v, and dk ~ dq/v - (k/v)*dv, so a
    speed error is amplified by k/v. Measured, that makes speed responsible for
    82 percent of the density error and concentrates it savagely: cells whose
    true speed is under 50 km/h are 3.8 percent of the data and 60 percent of
    the density error, at 11-39x the per-cell average. Weighting each row by
    (k/v)^2 for the speed model and 1/v^2 for the flow model makes the loss the
    one the metric reads. Density needs no weight; it is already the target.

Neither family nor alt is expected to beat `own` alone; they are not meant to.
phys should be clearly worse on speed RMSE and clearly better on density -
whether that trade is positive is for the ensemble search to say, since S_state
carries weight 0.35 against S_physics's 0.15.

    python t1_members.py family
    python t1_members.py alt
    python t1_members.py phys
    python t1_members.py both

A caveat that belongs in the record: every member early-stops on the last 45
train days, which is also the window ens_search scores on. Iteration count is
the only thing fitted there and it is fitted identically for all members, so
comparisons between weightings stay fair, but the absolute level is mildly
optimistic. This argues for preferring round, near-equal weights over a sharply
optimized vector - which is the safer choice for the private split anyway.
"""
import nice  # noqa: F401  - before numpy and lightgbm
import gc, pickle, sys, time
import numpy as np
import lightgbm as lgb

from common import OUT, PANELS, FAMILY
from t1build import build
from task1_run import PARAMS, NUM_THREADS

FAMILY_SAMPLE = 1_200_000        # per panel, so ~2.4M rows per family model
ALT_SAMPLE = 1_500_000           # diversity matters more than sample size here
TARGETS = (("speed", 25.0), ("flow", 600.0), ("density", None))

ALT_PARAMS = dict(objective="l2", learning_rate=0.035, num_leaves=63,
                  min_data_in_leaf=40, feature_fraction=0.55,
                  bagging_fraction=0.7, bagging_freq=1, extra_trees=True,
                  lambda_l2=1.0, verbosity=-1, num_threads=NUM_THREADS)


# how hard to chase the density error. 1.0 is the exact sensitivity weight; the
# top percentile is clipped so a handful of near-stopped cells cannot own the fit
PHYS_POWER = 1.0
PHYS_CLIP = 99.0


def phys_weights(y_speed, y_flowpl, target):
    """Row weights that turn a speed or flow loss into a density loss."""
    v = np.maximum(y_speed, 5.0)
    if target == "speed":
        w = (y_flowpl / v ** 2) ** 2            # (dk/dv)^2, k/v = q/v^2
    elif target == "flow":
        w = (1.0 / v) ** 2                      # (dk/dq)^2
    else:
        return None                             # density is already the target
    w = np.where(np.isfinite(w), w, 0.0) ** PHYS_POWER
    hi = np.percentile(w[w > 0], PHYS_CLIP) if (w > 0).any() else 1.0
    w = np.clip(w, 0.0, hi)
    m = w.mean()
    return w / m if m > 0 else None


def collect(panels, sample, seed):
    """Design matrix over the given panels, with the day index kept for the split."""
    Xs, ys, ds, cols = [], {n: [] for n, _ in TARGETS}, [], None
    for panel in panels:
        r = build(panel, "train", truth="train", sample=sample, seed=seed,
                  verbose=False)
        cols = r["cols"]
        ok = np.isfinite(r["y_speed"]) & np.isfinite(r["y_flowpl"])
        Xs.append(r["X"][ok].astype(np.float32))
        ys["speed"].append(r["y_speed"][ok])
        ys["flow"].append(r["y_flowpl"][ok])
        ys["density"].append(r["y_flowpl"][ok] / np.maximum(r["y_speed"][ok], 1.0))
        ds.append((r["ti"][ok] // 288).astype(np.int16))
        del r
        gc.collect()
    X = np.vstack(Xs) if len(Xs) > 1 else Xs[0]
    del Xs
    gc.collect()
    return (X, {n: np.concatenate(v) for n, v in ys.items()},
            np.concatenate(ds), cols)


def fit(X, y, day, cols, params, rounds, tag, t0, weighted=False):
    nd = day.max() + 1
    tr, va = day < nd - 45, day >= nd - 45
    models = {}
    for name, scale in TARGETS:
        m = np.isfinite(y[name])
        sw = phys_weights(y["speed"], y["flow"], name) if weighted else None
        ds = lgb.Dataset(X[tr & m], label=y[name][tr & m], feature_name=cols,
                         weight=None if sw is None else sw[tr & m])
        dv = lgb.Dataset(X[va & m], label=y[name][va & m], feature_name=cols,
                         weight=None if sw is None else sw[va & m], reference=ds)
        b = lgb.train(params, ds, rounds, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        p = b.predict(X[va & m], num_iteration=b.best_iteration)
        e = float(np.sqrt(np.mean((p - y[name][va & m]) ** 2)))
        models[name] = lgb.Booster(model_str=b.model_to_string(
            num_iteration=b.best_iteration))
        models[name].best_iteration = b.best_iteration
        note = f"S={max(0, 1 - e / scale):.4f}" if scale else ""
        print(f"  [{tag}] {name}: it={b.best_iteration} RMSE={e:.4f} {note} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del ds, dv, b, p
        gc.collect()
    return models


def train_family(fams=None):
    t0 = time.time()
    groups = {}
    for p in PANELS:
        groups.setdefault(FAMILY[p], []).append(p)
    for fam in (fams or sorted(groups)):
        path = OUT / f"t1_family_{fam}.pkl"
        if path.exists():
            print(f"[{fam}] already trained, skipping", flush=True)
            continue
        panels = groups[fam]
        X, y, day, cols = collect(panels, FAMILY_SAMPLE, seed=1)
        print(f"[{fam}] {panels} -> {X.shape} [{time.time()-t0:.0f}s]", flush=True)
        models = fit(X, y, day, cols, PARAMS, 3000, fam, t0)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"models": models, "cols": cols, "panels": panels}, fh)
        tmp.replace(path)                     # atomic: a kill leaves no stub
        print(f"[{fam}] saved {path.name}", flush=True)
        del X, y, day, models
        gc.collect()


def train_alt(panels=None):
    t0 = time.time()
    for panel in (panels or PANELS):
        path = OUT / f"t1_alt_{panel}.pkl"
        if path.exists():
            print(f"[{panel}] already trained, skipping", flush=True)
            continue
        X, y, day, cols = collect([panel], ALT_SAMPLE, seed=11)
        print(f"[{panel}] alt {X.shape} [{time.time()-t0:.0f}s]", flush=True)
        models = fit(X, y, day, cols, ALT_PARAMS, 4000, panel, t0)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"models": models, "cols": cols}, fh)
        tmp.replace(path)                     # atomic: a kill leaves no stub
        print(f"[{panel}] saved {path.name}", flush=True)
        del X, y, day, models
        gc.collect()


def train_phys(panels=None):
    t0 = time.time()
    for panel in (panels or PANELS):
        path = OUT / f"t1_phys_{panel}.pkl"
        if path.exists():
            print(f"[{panel}] already trained, skipping", flush=True)
            continue
        X, y, day, cols = collect([panel], ALT_SAMPLE, seed=1)
        print(f"[{panel}] phys {X.shape} [{time.time()-t0:.0f}s]", flush=True)
        models = fit(X, y, day, cols, PARAMS, 3000, panel, t0, weighted=True)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"models": models, "cols": cols}, fh)
        tmp.replace(path)                     # atomic: a kill leaves no stub
        print(f"[{panel}] saved {path.name}", flush=True)
        del X, y, day, models
        gc.collect()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "both"
    if cmd == "phys":
        train_phys()
    if cmd in ("family", "both"):
        train_family()
    if cmd in ("alt", "both"):
        train_alt()
