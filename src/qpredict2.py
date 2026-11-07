"""Task 2 predictions with a separate model mix and threshold per condition.

qpredict.py takes one model list for both conditions. The out-of-sample
harnesses (oos_onset.py, oos_ongoing.py, nn_compare.py) found the best mixes
differ by condition - trees for onset, trees plus the conv net for ongoing - so
this takes one spec per condition. Averaging is over probabilities, before the
threshold, exactly as the harnesses score it. Onset steps 1-5 are always empty
(the s15 construction rule).

    python qpredict2.py OUT.csv --onset both,both6,negall@0.60 \
        --ongoing both,nn:c1,nn:c2@0.50

`nn:NAME` loads out/nn_queue_NAME_c{0,1}.pt (per-condition checkpoints; a
single nn_queue_NAME.pt is used for both if those are absent). Tree models get
priors from the whole train split, as in qpredict.py; the nets get the fit-day
priors they were trained with.
"""
import nice  # noqa: F401
import sys, pickle
import os
import numpy as np
import pandas as pd
import torch

from common import ROOT, OUT, QUEUE_PANELS
import qkind
from qfeat import HIST, HOR, history_summary, build_rows
from qdata import load_panel, make_priors
from qpredict import history_arrays
import nn_queue as NQ

SPLITS = os.environ.get("QP_SPLITS", "validation,private").split(",")
QT_T = os.environ.get("QT_T", "0") == "1"   # include the released row at the origin T
QROLL = int(os.environ.get("QROLL", "0"))    # priors from the window's own split, previous N days
QLONG = os.environ.get("QLONG", "0") == "1"  # models trained with the 3-h masked-layer context
_D0 = {}


def _roll(panel, split, T):
    """Rolling same-split priors for the window's day (QROLL)."""
    from qdata import rolling_priors
    from t1feat import load_grid as _lg
    if (panel, split) not in _D0:
        _D0[(panel, split)] = pd.Timestamp(_lg(panel, split, masked=True)["days"][0], tz="UTC")
    return rolling_priors(panel, split, int((T - _D0[(panel, split)]).days))

# --elig-mask: predict 0 on horizon cells the release marks is_score_eligible=0.
# mainline_states_masked blanks the horizon's values but keeps its quality flags,
# and on train those flags equal the true eligibility on every horizon cell
# checked (57k/57k); on validation/private pct_observed>=75 <=> eligible on
# every horizon cell. The spec scores only eligible cells, and on the OOS
# harness 92% of our ongoing false positives sit on ineligible horizon cells.
ELIG_MASK = "--elig-mask" in sys.argv[1:]
# --dump PATH: also pickle each window's averaged probabilities, so threshold
# variants can be written without re-running the models (qvariant.py)
DUMP = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
DUMPED = {}
_DAY = {}


def horizon_elig(panel, split, T, link_ids):
    """(HOR, nL) bool: is_score_eligible of the horizon cells, from the release."""
    import glob
    day = T.strftime("%Y_%m_%d")
    key = (panel, split, day)
    if key not in _DAY:
        base = ROOT / "corridors" / panel / split / "mainline_states_masked"
        f = glob.glob(str(base / "*" / f"*{day}.parquet"))
        assert len(f) == 1, f"{panel}/{split}/{day}: {len(f)} files"
        d = pd.read_parquet(f[0], columns=["timestamp", "link_id", "is_score_eligible"])
        d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
        d["link_id"] = d.link_id.astype(str)
        _DAY.clear()
        _DAY[key] = d
    d = _DAY[key]
    hz = d[(d.timestamp > T) & (d.timestamp <= T + pd.Timedelta(minutes=5 * HOR))]
    li = {l: i for i, l in enumerate(link_ids)}
    out = np.zeros((HOR, len(link_ids)), bool)
    r = ((hz.timestamp - T).dt.total_seconds() // 300).astype(int).to_numpy() - 1
    c = hz.link_id.map(li)
    ok = c.notna().to_numpy() & (r >= 0) & (r < HOR)
    out[r[ok], c[ok].astype(int).to_numpy()] = hz.is_score_eligible.to_numpy()[ok] == 1
    assert ok.sum() == HOR * len(link_ids), f"{panel}/{split} {T}: {ok.sum()} horizon cells"
    return out


def parse(spec):
    models, thr = spec.rsplit("@", 1)
    return models.split(","), float(thr)


def load_tree(n):
    fn = "queue_model.pkl" if n == "base" else f"queue_model_{n}.pkl"
    b = pickle.load(open(OUT / fn, "rb"))
    names = set(b["models"][0].feature_name())
    assert "d_dn_q" not in names and "merge_load" not in names, f"{n}: extra features"
    return dict(kind="tree", models=b["models"], score=qkind.kind_of(b),
                noprior="pri_on" not in names, long="lq_frac" in names)


def load_nn(n):
    nets = []
    for c in (0, 1):
        f = OUT / f"nn_queue_{n}_c{c}.pt"
        if not f.exists():
            f = OUT / f"nn_queue_{n}.pt"
        net = NQ.make_net(n).to(NQ.DEV)
        net.load_state_dict(torch.load(f, map_location=NQ.DEV))
        nets.append(net.eval())
    return dict(kind="nn", nets=nets)


def load_nn2(n):
    """A 2-D (time x link) queue net from nn_queue2.py, one checkpoint per condition."""
    os.environ.setdefault("NN2_PAD", "batch")
    import nn_queue2 as N2
    return dict(kind="nn2", nets=[N2.load_net(n, c) for c in (0, 1)], mod=N2)


def main():
    out_csv = sys.argv[1]
    a = sys.argv[2:]
    spec = {0: parse(a[a.index("--onset") + 1]), 1: parse(a[a.index("--ongoing") + 1])}
    cache = {}
    get = lambda n: cache.setdefault(n, load_nn2(n[4:]) if n.startswith("nn2:") else
                                     load_nn(n[3:]) if n.startswith("nn:") else load_tree(n))
    for c in (0, 1):
        for n in spec[c][0]:
            get(n)
    print("onset", spec[0], " ongoing", spec[1], " elig-mask", ELIG_MASK, flush=True)

    rows = []
    for pid, panel in enumerate(QUEUE_PANELS):
        D = load_panel(panel); lids = D["link_ids"]; stat = D["stat"]; nL = len(lids)
        nd = len(D["days"]); fit = np.zeros(nd, bool); fit[:nd - NQ.VAL] = True
        pri_all = make_priors(D)
        pri_fit = make_priors(D, fit)
        occ_scale = float(np.nanpercentile(D["occ"][::7], 99)) or 1.0
        for split in SPLITS:
            base = ROOT / "task2" / panel / split
            wi = pd.read_csv(base / "window_index.csv")
            hist = pd.read_parquet(base / "window_history.parquet")
            hist["link_id"] = hist.link_id.astype(str)
            hist["timestamp"] = pd.to_datetime(hist.timestamp, utc=True)
            if QT_T or QLONG:
                from t1feat import load_grid
                Gm = load_grid(panel, split, masked=True)
                gli = {str(l): i for i, l in enumerate(Gm["link_ids"])}; gcol = [gli[str(l)] for l in lids]
                gday0 = pd.Timestamp(Gm["days"][0], tz="UTC")
            for w in wi.itertuples(index=False):
                T = pd.Timestamp(w.forecast_origin)
                times = [T - pd.Timedelta(minutes=5 * (HIST - i)) for i in range(HIST)]
                if QT_T:
                    # history T-55..T: the window history's last 11 steps + the released
                    # row AT T from the Task 1 masked layer (allowed by the organizers)
                    times = times[1:] + [T]
                sp, fl, oc, pc = history_arrays(hist[hist.window_id == w.window_id], lids, times)
                if QT_T:
                    ti = int((T - gday0).total_seconds() // 300)
                    sp[-1] = Gm["speed"][ti, gcol]; fl[-1] = Gm["flow"][ti, gcol]
                    oc[-1] = Gm["occ"][ti, gcol]; pc[-1] = np.nan_to_num(Gm["pct"][ti, gcol]).astype(np.int8)
                cond = 0 if "onset" in str(w.condition) else 1
                tod0 = T.hour * 12 + T.minute // 5
                names, thr = spec[cond]
                ps = []
                for n in names:
                    m = get(n)
                    if m["kind"] == "tree":
                        H = history_summary(sp, fl, oc, pc, stat["vf"], stat["cap"])
                        pa = pri_all
                        if QROLL:
                            from qdata import rolling_priors
                            from t1feat import load_grid as _lg
                            if (panel, split) not in _D0:
                                _D0[(panel, split)] = pd.Timestamp(_lg(panel, split, masked=True)["days"][0], tz="UTC")
                            pa = rolling_priors(panel, split, int((T - _D0[(panel, split)]).days))
                        pw = dict(pri_link=np.tile(pa["rate_link"], (HOR, 1)),
                                  pri_tod=pa["rate_tod"][[(tod0 + s) % 288 for s in range(1, HOR + 1)]],
                                  pri_on=pa["P_on"], pri_ong=pa["P_ong"],
                                  pri_cond=pa["P_on"] if cond == 0 else pa["P_ong"])
                        LF = None
                        if m.get("long"):
                            from qfeat import long_feats, LONG
                            ti = int((T - gday0).total_seconds() // 300)
                            LF = long_feats(Gm["speed"][ti - LONG:ti + 1][:, gcol], stat["vf"])
                        X, _ = build_rows(H, stat, pw, tod0, int(T.dayofweek), cond, pid,
                                          noprior=m["noprior"], long=LF)
                        ps.append(qkind.to_score(m["score"], m["models"][cond].predict(X)))
                    elif m["kind"] == "nn2":
                        N2 = m["mod"]
                        x = N2.make_input2(sp, fl, oc, pc, stat, _roll(panel, split, T) if QROLL else pri_fit, tod0,
                                           int(T.dayofweek), cond, occ_scale)
                        X, _, _, Cd = N2.batch([(x, None, None, cond)])
                        with torch.no_grad():
                            pr = torch.sigmoid(m["nets"][cond](X, Cd)).cpu().numpy()[0, :, :nL]
                        ps.append(pr.reshape(-1))
                    else:
                        x = NQ.make_input(sp, fl, oc, pc, stat, _roll(panel, split, T) if QROLL else pri_fit, tod0,
                                          int(T.dayofweek), cond, occ_scale)
                        X, _, _, Cd = NQ.batch([(x, None, None, cond)])
                        with torch.no_grad():
                            pr = torch.sigmoid(m["nets"][cond](X, Cd)).cpu().numpy()[0, :, :nL]
                        ps.append(pr.reshape(-1))
                p = np.mean(ps, axis=0)
                if DUMP:
                    DUMPED[w.window_id] = dict(p=p.astype(np.float32).reshape(HOR, nL), cond=cond,
                                               T=str(T), links=np.array(lids))
                flag = (p >= thr).astype(np.int8).reshape(HOR, nL)
                if cond == 0:
                    flag[:HOR - 1] = 0
                if ELIG_MASK:
                    flag &= horizon_elig(panel, split, T, lids)
                rows.append(pd.DataFrame({
                    "window_id": w.window_id,
                    "timestamp": np.repeat([T + pd.Timedelta(minutes=5 * s) for s in range(1, HOR + 1)], nL),
                    "link_id": np.tile(lids, HOR), "queue_pred": flag.reshape(-1)}))
            print(f"[{panel}/{split}] {len(wi)} windows", flush=True)
    sub = pd.concat(rows, ignore_index=True)
    sub["timestamp"] = pd.to_datetime(sub.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    sub = sub.drop_duplicates(["window_id", "timestamp", "link_id"])
    sub.to_csv(out_csv, index=False)
    if DUMP:
        pickle.dump(DUMPED, open(DUMP, "wb"))
    print("wrote", out_csv, len(sub), "rows, positive rate", round(sub.queue_pred.mean(), 4))


if __name__ == "__main__":
    main()
