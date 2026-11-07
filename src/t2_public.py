"""Task 2 of finals s101 / s103, from archived weights (no training).

Two public Kaggle pipelines, retrained locally from their published code
(repro/src_t2, train/t2_public.sh), are combined:

  onset   (anubisyy, catboost-lgbm-nnls-pipeline-top-50-0-86517):
          p = 0.5 LightGBM + 0.5 CatBoost on the final step's links, queued at
          T+30 where p >= 0.55, else the single most likely link
  ongoing p = 0.5 x the 7-member LightGBM ensemble of yeashusemwal's
          trafficflowbench-trees-u-net-cnn-lb-0-86 (0.8 x [0.5 mean(v11..v14) +
          0.5 mean(v15, v16)] + 0.2 x v20, averaged over its two heads)
            + 0.5 x anubisyy's ongoing LightGBM;  queued where p >= 0.40
          s105: equal thirds of the ensemble, anubisyy's LightGBM and our own
            no-prior LightGBM (T2_W3, T2_OURS_PKL), queued where p >= 0.43

Both use only the 60-minute window history (data at or before the forecast
origin), as the organizers' online rule requires.

    python t2_public.py <out_queue_csv>
"""
import glob, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path(os.environ["TFB_DATA"])
MODELS = Path(os.environ.get("T2_MODELS", ROOT / "models" / "t2"))
os.environ.setdefault("TFB_DATA_ROOT", str(DATA))
os.environ.setdefault("TFB_OUTPUT_ROOT", str(Path(os.environ.get("TFB_WORK", ".")) / "t2_scratch"))
os.environ.setdefault("TFB_THREADS", "8")
sys.path[:0] = [str(ROOT / "src_t2" / "pub"), str(ROOT / "src_t2" / "yeash" / "nb"), str(ROOT / "src_t2" / "yeash" / "v2")]

import lightgbm as lgb
from catboost import CatBoostClassifier
from queue_features import Panel, panel_names, HISTORY, HORIZON
from queue_ongoing import Panel as OngoingPanel
import t2_gbm_infer as g2

ONSET_THR, V20 = float(os.environ.get("T2_ON_THR", "0.55")), 0.20
ENS_ONSET, ENS_ON_W = {}, float(os.environ.get("T2_ENS_ON_W", "0"))
ONSET, ONSET_W = None, float(os.environ.get("T2_ONSET_W", "1.0"))
# s105: a third ongoing member - our own no-prior LightGBM (queue_model_bothnp),
# probabilities dumped by qpredict2.py --dump; T2_W3 = weights of ensemble, public, ours
W3 = [float(x) for x in os.environ.get("T2_W3", "0.5,0.5,0").split(",")]
ONG_THR = float(os.environ.get("T2_ONG_THR", "0.40"))
OURS = None
if W3[2] > 0:
    import pickle
    # T2_OURS_PKL may list several dumps (s108: bothnp, q1np) - averaged
    # T2_OURS_PKL lists the member dumps; "a+b" averages seeds of one member first
    def _load(spec):
        ds = [pickle.load(open(f, "rb")) for f in spec.split("+")]
        return {w: dict(v, p=np.mean([d[w]["p"] for d in ds], 0)) for w, v in ds[0].items()}
    _d = [_load(f) for f in os.environ["T2_OURS_PKL"].split(",")]
    OURS = {w: dict(v, p=np.mean([d[w]["p"] for d in _d], 0)) for w, v in _d[0].items()}
if os.environ.get("T2_ONSET_PKL"):
    _o = [_load(f) for f in os.environ["T2_ONSET_PKL"].split(",")]
    ONSET = {w: dict(v, p=np.mean([d[w]["p"] for d in _o], 0)) for w, v in _o[0].items()}


def ensemble_ongoing():
    """{(window_id, h, link_id): p} for the 7-member ensemble."""
    P = {}
    for family, names in (("v11", ["v11", "v12", "v13", "v14"]), ("v15", ["v15", "v16"]), ("v18", ["v20"])):
        dirs = [str(MODELS / f"t2_{n}") for n in names]
        out = g2.dump(family, dirs, threads=int(os.environ["TFB_THREADS"]))
        P.update({n: out[d] for n, d in zip(names, dirs)})
        print(f"  ensemble family {family} done", flush=True)
    key = ["window_id", "h", "link_id"]
    base = P["v11"][key + ["condition"]].copy()
    for n, d in P.items():
        assert (d[key].values == base[key].values).all(), n
    mean = lambda names, c: np.mean([P[n][c].to_numpy() for n in names], 0)
    group = lambda c: 0.5 * mean(["v11", "v12", "v13", "v14"], c) + 0.5 * mean(["v15", "v16"], c)
    base["p"] = np.mean([(1 - V20) * group(c) + V20 * P["v20"][c].to_numpy() for c in ("p_general", "p_general_w")], 0)
    global ENS_ONSET
    on = base[(base.condition == "queue_onset") & (base.h == 6)].copy()
    on["p6"] = 0.5 * mean(["v11", "v12", "v13", "v14"], "p_onsetT30")[on.index] + 0.5 * mean(["v15", "v16"], "p_onsetT30")[on.index]
    on["p6"] = (1 - V20) * on.p6 + V20 * P["v20"]["p_onsetT30"].to_numpy()[on.index]
    ENS_ONSET = {(w, str(l)): p for w, l, p in zip(on.window_id, on.link_id, on.p6)}
    base = base[base.condition == "queue_ongoing"]
    return {(w, int(h), str(l)): p for w, h, l, p in zip(base.window_id, base.h, base.link_id, base.p)}


def main():
    out_csv = sys.argv[1]
    ens = ensemble_ongoing()
    ong = lgb.Booster(model_file=str(MODELS / "pub_ongoing_lgb.txt"))
    on_lgb = lgb.Booster(model_file=str(MODELS / "pub_onset_lgb.txt"))
    on_cb = CatBoostClassifier(); on_cb.load_model(str(MODELS / "pub_onset.cbm"))
    parts = []
    for split in ("validation", "private"):
        for no, name in enumerate(panel_names()):
            panel, opanel = Panel(name, no), OngoingPanel(name, no)
            folder = DATA / "task2" / name / split
            wi = pd.read_csv(folder / "window_index.csv"); hist = pd.read_parquet(folder / "window_history.parquet")
            tpl = pd.read_csv(folder / "sample_submission_queue.csv", dtype={"link_id": str})
            pieces = []
            for w in wi.itertuples(index=False):
                origin = pd.Timestamp(w.forecast_origin)
                times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min",
                                      tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                h = hist[hist.window_id == w.window_id].copy(); h["link_id"] = h.link_id.astype(str)
                assert pd.to_datetime(h.timestamp, utc=True).max() < origin        # strictly before T
                mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(
                    index=times, columns=panel.links).to_numpy(dtype=np.float32)
                sp, fl, el = mat("speed_kmh"), mat("flow_vph"), mat("is_score_eligible")
                pred = np.zeros((HORIZON, len(panel.links)), dtype=np.uint8)
                if w.condition == "queue_onset":
                    X = panel.features(sp, fl, el, origin, w.condition)[-len(panel.links):]
                    p = 0.5 * on_lgb.predict(X) + 0.5 * on_cb.predict_proba(X)[:, 1]
                    if ENS_ON_W > 0:
                        # T2_ENS_ON_W: mix in the yeash members' own onset heads (p_onsetT30)
                        p = (1 - ENS_ON_W) * p + ENS_ON_W * np.array([ENS_ONSET[(w.window_id, str(l))] for l in panel.links])
                    if ONSET is not None:
                        # T2_ONSET_PKL: our own onset heads (step 6), mixed in with weight T2_ONSET_W
                        o = ONSET[w.window_id]; li = {str(x): k for k, x in enumerate(o["links"])}
                        p = (1 - ONSET_W) * p + ONSET_W * o["p"][-1, [li[str(l)] for l in panel.links]]
                    sel = p >= ONSET_THR
                    if not sel.any():
                        sel[np.argmax(p)] = True
                    pred[-1] = sel
                else:
                    X = opanel.features(sp, fl, np.nan_to_num(el, nan=0).astype(bool), origin, w.condition)
                    p_pub = ong.predict(X).reshape(HORIZON, len(panel.links))
                    p_ens = np.array([[ens[(w.window_id, s + 1, str(l))] for l in panel.links] for s in range(HORIZON)])
                    p = W3[0] * p_ens + W3[1] * p_pub
                    if OURS is not None:
                        o = OURS[w.window_id]; li = {str(x): k for k, x in enumerate(o["links"])}
                        p = p + W3[2] * o["p"][:, [li[str(l)] for l in panel.links]]
                    pred = (p >= ONG_THR).astype(np.uint8)
                stamps = [(origin + pd.Timedelta(minutes=5 * s)).strftime("%Y-%m-%dT%H:%M:%SZ") for s in range(1, HORIZON + 1)]
                pieces.append(pd.DataFrame({"window_id": w.window_id, "timestamp": np.repeat(stamps, len(panel.links)),
                                            "link_id": np.tile([str(l) for l in panel.links], HORIZON),
                                            "queue_pred": pred.ravel()}))
            keys = ["window_id", "timestamp", "link_id"]
            res = tpl[keys].merge(pd.concat(pieces, ignore_index=True), on=keys, how="left", validate="one_to_one")
            assert res.queue_pred.notna().all() and len(res) == len(tpl)
            parts.append(res)
            print(f"  [{name}/{split}] {len(res):,} rows", flush=True)
    q = pd.concat(parts, ignore_index=True)
    q["queue_pred"] = q.queue_pred.astype(int)
    q.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(q):,} rows, {int(q.queue_pred.sum()):,} positives")


if __name__ == "__main__":
    main()
