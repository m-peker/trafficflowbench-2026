"""Predict Task 2 queue flags for the released validation/private windows.

Takes the model - or models - and the threshold from the command line, because
what ships is whatever `qcompare.py` ranks first, and that may be an average:

    python qpredict.py                                  # the shipped default
    python qpredict.py --models win4x --thr 0.50        # one variant
    python qpredict.py --models win4x,both --thr 0.55   # their average

Averaging is over probabilities, before the threshold, which is the same thing
qcompare scores. A model whose name contains `shock` or `noprior` gets the
matching feature set automatically, so mixed averages are safe.
"""
import sys, pickle
import numpy as np
import pandas as pd
from common import ROOT, OUT, QUEUE_PANELS
import qkind
import qramp
from qfeat import HIST, HOR, history_summary, build_rows
from qdata import load_panel, make_priors, window_prior

SPLITS = ["validation", "private"]


def history_arrays(hist, link_ids, times):
    """window_history rows -> (HIST, nL) arrays in corridor order."""
    nL = len(link_ids)
    li = {l: i for i, l in enumerate(link_ids)}
    ti = {t: i for i, t in enumerate(times)}
    sp = np.full((HIST, nL), np.nan, np.float32)
    fl = np.full((HIST, nL), np.nan, np.float32)
    oc = np.full((HIST, nL), np.nan, np.float32)
    pc = np.zeros((HIST, nL), np.int8)
    r = hist.timestamp.map(ti).to_numpy()
    c = hist.link_id.map(li).to_numpy()
    ok = pd.notna(r) & pd.notna(c)
    r = r[ok].astype(int); c = c[ok].astype(int)
    sp[r, c] = hist.speed_kmh.to_numpy(np.float32)[ok]
    fl[r, c] = hist.flow_vph.to_numpy(np.float32)[ok]
    oc[r, c] = hist.occupancy.to_numpy(np.float32)[ok]
    pc[r, c] = hist.pct_observed.to_numpy(np.int8)[ok]
    return sp, fl, oc, pc


def parse_args():
    argv = sys.argv[1:]
    names, thr = ["base"], None
    if "--models" in argv:
        names = argv[argv.index("--models") + 1].split(",")
    if "--thr" in argv:                      # one value for both conditions
        v = float(argv[argv.index("--thr") + 1])
        thr = {"onset": v, "ongoing": v}
    # onset and ongoing are different problems - about 1.5 percent of cells
    # positive against 14 - so one cutoff cannot suit both. s12 shipped 0.575
    # for each because --thr was the only option.
    if "--thr-onset" in argv or "--thr-ongoing" in argv:
        thr = dict(thr or {"onset": 0.5, "ongoing": 0.5})
        if "--thr-onset" in argv:
            thr["onset"] = float(argv[argv.index("--thr-onset") + 1])
        if "--thr-ongoing" in argv:
            thr["ongoing"] = float(argv[argv.index("--thr-ongoing") + 1])
    return names, thr


# The organizers build every queue_onset window with its origin exactly thirty
# minutes before the queue begins. On train, all 39 onset windows that contain a
# queue first show it at step 6 and not before, on every one of the eight
# panels - so for onset the truth at steps 1-5 is empty by construction, and
# any positive predicted there is a guaranteed false positive. With about three
# queued cells per onset window, a handful of those collapse the window's IoU.
ONSET_LAST_ONLY = "--onset-last-only" in sys.argv[1:]


def load(names):
    out = []
    for n in names:
        fn = "queue_model.pkl" if n == "base" else f"queue_model_{n}.pkl"
        with open(OUT / fn, "rb") as fh:
            b = pickle.load(fh)
        names = set(b["models"][0].feature_name())
        out.append(dict(name=n, models=b["models"], thr=b.get("thr"),
                        kind=qkind.kind_of(b),
                        shock="d_dn_q" in names, noprior="pri_on" not in names,
                        ramp="merge_load" in names))
    return out


def main():
    names, thr_override = parse_args()
    bundles = load(names)
    if thr_override is not None:
        thr = thr_override
    else:
        assert len(bundles) == 1, "averaging needs an explicit --thr"
        thr = bundles[0]["thr"]
    print(f"models: {[b['name'] for b in bundles]}   thresholds: {thr}")
    out = []
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel)
        pri = make_priors(D)                       # priors from the whole train split
        link_ids = D["link_ids"]
        stat = D["stat"]
        for split in SPLITS:
            base = ROOT / "task2" / panel / split
            if not (base / "window_index.csv").exists():
                continue
            wi = pd.read_csv(base / "window_index.csv")
            hist = pd.read_parquet(base / "window_history.parquet")
            hist["link_id"] = hist.link_id.astype(str)
            hist["timestamp"] = pd.to_datetime(hist.timestamp, utc=True)
            tmpl = pd.read_csv(base / "sample_submission_queue.csv")
            tmpl["link_id"] = tmpl.link_id.astype(str)
            tmpl["timestamp"] = pd.to_datetime(tmpl.timestamp, utc=True)
            for w in wi.itertuples(index=False):
                T = pd.Timestamp(w.forecast_origin)
                times = [T - pd.Timedelta(minutes=5 * (HIST - i)) for i in range(HIST)]
                h = hist[hist.window_id == w.window_id]
                sp, fl, oc, pc = history_arrays(h, link_ids, times)
                H = history_summary(sp, fl, oc, pc, stat["vf"], stat["cap"])
                cond = 0 if "onset" in str(w.condition) else 1
                tod0 = T.hour * 12 + T.minute // 5
                # step index of T inside the train grid is irrelevant; priors need tods
                pw = dict(pri_link=np.tile(pri["rate_link"], (HOR, 1)),
                          pri_tod=pri["rate_tod"][[(tod0 + s) % 288 for s in range(1, HOR + 1)]],
                          pri_on=pri["P_on"], pri_ong=pri["P_ong"],
                          pri_cond=pri["P_on"] if cond == 0 else pri["P_ong"])
                ps = []
                for b in bundles:
                    if b["shock"] or b["ramp"]:
                        R = (qramp.history_for(panel, split, link_ids, T, HIST)
                             if b["ramp"] else None)
                        Hb = history_summary(sp, fl, oc, pc, stat["vf"],
                                             stat["cap"], shock=b["shock"],
                                             ramp=R, lanes=stat["lanes"])
                    else:
                        Hb = H
                    X, _ = build_rows(Hb, stat, pw, tod0, int(T.dayofweek), cond,
                                      pid, shock=b["shock"], noprior=b["noprior"],
                                      ramp=b["ramp"])
                    ps.append(qkind.to_score(b["kind"],
                                             b["models"][cond].predict(X)))
                p = np.mean(ps, axis=0)
                flag = (p >= (thr["onset"] if cond == 0 else thr["ongoing"])).astype(int)
                if cond == 0 and ONSET_LAST_ONLY:
                    flag = flag.reshape(HOR, len(link_ids))
                    flag[:HOR - 1] = 0
                    flag = flag.reshape(-1)
                grid = pd.DataFrame({
                    "window_id": w.window_id,
                    "timestamp": np.repeat(
                        [T + pd.Timedelta(minutes=5 * s) for s in range(1, HOR + 1)],
                        len(link_ids)),
                    "link_id": np.tile(link_ids, HOR),
                    "queue_pred": flag})
                out.append(grid)
            print(f"[{panel}/{split}] {len(wi)} windows", flush=True)
    sub = pd.concat(out, ignore_index=True)
    sub["timestamp"] = pd.to_datetime(sub.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    sub = sub.drop_duplicates(["window_id", "timestamp", "link_id"])
    sub.to_csv(OUT / "queue_submission.csv", index=False)
    print("wrote", OUT / "queue_submission.csv", len(sub), "rows,",
          f"positive rate {sub.queue_pred.mean():.4f}")


if __name__ == "__main__":
    main()
