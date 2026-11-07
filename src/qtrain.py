"""Train the Task 2 queue model and tune the decision threshold by IoU."""
import sys, time, pickle
import numpy as np
import pandas as pd
import nice
import lightgbm as lgb
from common import WORK, OUT, QUEUE_PANELS, FAMILY
from qfeat import HIST, HOR
from qdata import load_panel, origins, make_priors, window_xy

import os
# space-time front features (qshock). QSHOCK=1 to train the wider model;
# the shipped out/queue_model.pkl is the narrow one and must stay loadable.
SHOCK = os.environ.get("QSHOCK", "0") == "1"
NOPRIOR = os.environ.get("QNOPRIOR", "0") == "1"
TARGET = os.environ.get("QTARGET", "flag")
RAMP = os.environ.get("QRAMP", "0") == "1"
# Averaging paid on Task 1 and lost on Task 2, because every Task 2 member so far
# differed only in how much data it saw. QPARAMS=alt swaps the deep 255-leaf fit
# for a shallow, heavily randomised one - the same substitution that made `alt` a
# useful Task 1 member - so an average finally has two different biases to
# cancel.
ALT_PARAMS = dict(objective="binary", learning_rate=0.035, num_leaves=63,
                  min_data_in_leaf=40, feature_fraction=0.55,
                  bagging_fraction=0.7, bagging_freq=1, extra_trees=True,
                  lambda_l2=1.0, verbosity=-1)
MODEL_NAME = os.environ.get(
    "QMODEL", "queue_model_shock.pkl" if SHOCK else "queue_model.pkl")

RNG = np.random.default_rng(0)
# Both were caps chosen before anything was measured, and both throw data away.
# The onset pool runs 600-1400 windows per panel so 700 halves it; the ongoing
# pool is far larger - 22,970 candidates in ten days alone - so 700 uses a few
# percent of it. NEG_KEEP drops 70 percent of the negative cells and reweights
# the survivors, which is standard but was never checked against keeping them.
# Env-configurable so the calibrated harness can rank the choices.
NEG_KEEP = float(os.environ.get("QNEG", "0.30"))
MAX_ONSET = int(os.environ.get("QWIN_ON", "700"))
MAX_ONGOING = int(os.environ.get("QWIN_ONG", "700"))
# The last N train days are kept out of the tree fit and used to tune the
# threshold. They are also the days closest to the scored months, and the model
# decays about 0.04 over nine months, so holding out 60 of them may cost more
# than the threshold tuning is worth. Env-configurable to find out.
VAL_DAYS = int(os.environ.get("QVAL_DAYS", "60"))
ADD_SPLIT = os.environ.get("QADD_SPLIT", "")
ADD_ONSET = int(os.environ.get("QADD_ONSET", "1000"))
ADD_ONGOING = int(os.environ.get("QADD_ONGOING", "1500"))
HOLD_FIRST = int(os.environ.get("QHOLD_FIRST", "0"))


def sample(arr, n, rng):
    return np.sort(rng.choice(arr, n, replace=False)) if len(arr) > n else np.sort(arr)


def collect(panels, verbose=True):
    """Return per-panel dicts holding data, priors, and fit/val window lists."""
    store = {}
    for pi, panel in enumerate(panels):
        D = load_panel(panel)
        nd = len(D["days"])
        day_fit = np.zeros(nd, bool); day_fit[: nd - VAL_DAYS] = True
        # QHOLD_FIRST=k also drops the first k days from the fit, so the
        # organizers' own train windows (all in 2030-06-01..20) can be scored
        # out of sample - the only selector-drawn windows whose truth we have.
        if HOLD_FIRST:
            day_fit[:HOLD_FIRST] = False
        day_val = ~day_fit
        pri = make_priors(D, day_fit)
        on_f, ong_f = origins(D, day_fit)
        on_v, ong_v = origins(D, day_val)
        rng = np.random.default_rng(100 + pi)
        store[panel] = dict(
            D=D, pri=pri, pid=pi,
            fit=[(T, 0) for T in sample(on_f, MAX_ONSET, rng)] +
                [(T, 1) for T in sample(ong_f, MAX_ONGOING, rng)],
            val=[(T, 0) for T in sample(on_v, 250, rng)] +
                [(T, 1) for T in sample(ong_v, 250, rng)])
        if ADD_SPLIT:
            # scored-month windows from the masked layer (for PRIVATE predictions only:
            # the validation month precedes it). Windows touching a blanked row are skipped.
            from qdata import load_masked_split
            Dm = load_masked_split(panel, ADD_SPLIT)
            blank = ~np.isfinite(Dm["speed"]).any(1)
            cb = np.concatenate([[0], np.cumsum(blank)])
            ok = lambda T: T[(cb[T + HOR + 1] - cb[T - HIST]) == 0]
            on_m, ong_m = origins(Dm)
            on_m, ong_m = ok(on_m), ok(ong_m)
            store[panel + "@" + ADD_SPLIT] = dict(
                D=Dm, pri=None, pid=pi,
                fit=[(T, 0) for T in sample(on_m, ADD_ONSET, rng)] +
                    [(T, 1) for T in sample(ong_m, ADD_ONGOING, rng)], val=[])
            print(f"[{panel}@{ADD_SPLIT}] onset {len(on_m)} ongoing {len(ong_m)}", flush=True)
        if verbose:
            print(f"[{panel}] nL={D['nL']} fit={len(store[panel]['fit'])} "
                  f"val={len(store[panel]['val'])} onset_pool={len(on_f)}", flush=True)
    return store


def build_fit_matrix(store):
    Xs, ys, ws, cs = [], [], [], []
    cols = None
    for panel, S in store.items():
        D, pri, pid = S["D"], S["pri"], S["pid"]
        rng = np.random.default_rng(7 + pid)
        for T, cond in S["fit"]:
            X, cols, y, m = window_xy(D, pri, T, cond, pid, shock=SHOCK, noprior=NOPRIOR,
                                      ramp=RAMP)
            if TARGET == "speed":
                keep = m          # every observed cell carries signal here
            else:
                keep = m & ((y == 1) | (rng.random(len(y)) < NEG_KEEP))
            Xs.append(X[keep]); ys.append(y[keep]); cs.append(np.full(keep.sum(), cond, np.int8))
            w = (np.ones(int(keep.sum()), np.float32) if TARGET == "speed"
                 else np.where(y[keep] == 1, 1.0, 1.0 / NEG_KEEP).astype(np.float32))
            ws.append(w)
        print(f"  built {panel}: rows so far {sum(len(a) for a in ys):,}", flush=True)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws),
            np.concatenate(cs), cols)


def window_iou_from_prob(D, T, prob, thr):
    sl = slice(T + 1, T + 1 + HOR)
    e = D["elig"][sl].reshape(-1)
    truth = (D["q"][sl] & D["elig"][sl]).reshape(-1)
    p = (prob >= thr) & e
    inter = np.logical_and(p, truth).sum()
    union = np.logical_or(p, truth).sum()
    return 1.0 if union == 0 else inter / union


def evaluate(store, models, thr_on, thr_ong, cache=None):
    """Leaderboard-style aggregation: panel-condition -> panel -> family -> overall."""
    rows = []
    for panel, S in store.items():
        D, pri, pid = S["D"], S["pri"], S["pid"]
        for T, cond in S["val"]:
            key = (panel, T, cond)
            prob = cache[key] if cache is not None and key in cache else None
            if prob is None:
                X, _, _, _ = window_xy(D, pri, T, cond, pid, with_y=False, shock=SHOCK,
                                          noprior=NOPRIOR, ramp=RAMP)
                prob = models[cond].predict(X, num_iteration=models[cond].best_iteration)
                if cache is not None:
                    cache[key] = prob
            thr = thr_on if cond == 0 else thr_ong
            rows.append((panel, cond, window_iou_from_prob(D, T, prob, thr)))
    df = pd.DataFrame(rows, columns=["panel", "cond", "iou"])
    pc = df.groupby(["panel", "cond"]).iou.mean().reset_index()
    pm = pc.groupby("panel").iou.mean().reset_index()
    pm["family"] = pm.panel.map(FAMILY)
    fam = pm.groupby("family").iou.mean()
    return float(fam.mean()), pc, pm


def main():
    t0 = time.time()
    store = collect(QUEUE_PANELS)
    X, y, w, c, cols = build_fit_matrix(store)
    print(f"fit matrix {X.shape} pos={y.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

    models = {}
    for cond in (0, 1):
        sel = c == cond
        ds = lgb.Dataset(X[sel], label=y[sel], weight=w[sel], feature_name=cols,
                         free_raw_data=False)
        if TARGET == "speed":
            params = dict(objective="l2", learning_rate=0.05, num_leaves=127,
                          min_data_in_leaf=200, feature_fraction=0.8,
                          bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                          verbosity=-1, num_threads=nice.THREADS)
            rounds = 600
        elif os.environ.get("QPARAMS") == "alt":
            params = dict(ALT_PARAMS, num_threads=nice.THREADS)
            rounds = 1200            # shallower trees need more of them
        else:
            params = dict(objective="binary", learning_rate=0.05, num_leaves=127,
                          min_data_in_leaf=200, feature_fraction=0.8,
                          bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                          verbosity=-1, num_threads=nice.THREADS)
            rounds = 500
        if os.environ.get("QSEED"):
            params["seed"] = int(os.environ["QSEED"])      # bagging / feature-fraction seed
        models[cond] = lgb.train(params, ds, num_boost_round=rounds)
        models[cond].best_iteration = rounds
        print(f"model cond={cond} trained on {sel.sum():,} rows "
              f"pos={y[sel].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

    cache = {}
    best = {}
    for cond, name in ((0, "onset"), (1, "ongoing")):
        grid = np.arange(0.05, 0.95, 0.025)
        scores = []
        for th in grid:
            s, _, _ = evaluate(store, models,
                               th if cond == 0 else best.get("onset", 0.3),
                               th if cond == 1 else best.get("ongoing", 0.5),
                               cache=cache)
            scores.append(s)
        # score the condition in isolation
        rows = []
        for panel, S in store.items():
            D = S["D"]
            for T, cd in S["val"]:
                if cd != cond:
                    continue
                prob = cache[(panel, T, cd)]
                rows.append([panel] + [window_iou_from_prob(D, T, prob, th) for th in grid])
        df = pd.DataFrame(rows, columns=["panel"] + [f"t{i}" for i in range(len(grid))])
        pm = df.groupby("panel").mean()
        pm["family"] = [FAMILY[p] for p in pm.index]
        fam = pm.groupby("family").mean().mean(0).to_numpy()
        i = int(np.argmax(fam))
        best[name] = float(grid[i])
        print(f"{name}: best thr={grid[i]:.3f} IoU={fam[i]:.4f}  "
              f"(curve {np.round(fam, 3)})", flush=True)

    total, pc, pm = evaluate(store, models, best["onset"], best["ongoing"], cache=cache)
    print("\npanel-condition IoU:\n", pc.to_string())
    print("\npanel IoU:\n", pm.to_string())
    print(f"\nS_queue (held-out train windows) = {total:.4f}")
    with open(OUT / MODEL_NAME, "wb") as fh:
        pickle.dump({"models": models, "thr": best, "cols": cols, "kind": ("reg" if TARGET == "speed" else "clf")}, fh)
    print("saved", OUT / MODEL_NAME, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
