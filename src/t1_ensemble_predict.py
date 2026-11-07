"""Write a Task 1 submission from a weighted ensemble of any members.

s05 hard-coded one blend, 0.7 own + 0.3 pooled. Once there are four members the
weighting is whatever ens_search returns, so this takes it as an argument:

    python t1_ensemble_predict.py <out_dir> "own=0.5,pooled=0.2,family=0.2,alt=0.1"
    python t1_ensemble_predict.py <out_dir> "speed:own=.51,pooled=.12,family=.21,alt=.16;flow:own=.47,pooled=.25,family=.17,alt=.12;density:own=.44,pooled=.16,family=.17,alt=.23"

Members resolve through ens_search.MEMBERS, so anything the search can score,
this can ship - the two files cannot disagree about what a weighting means.
Weights are normalised per target, and a member missing for some panel simply
drops out of that panel's mix rather than failing the run.

Everything downstream of the mix - the free-flow speed clip, the half-and-half
flow/density combination, the capacity clip - is copied from task1_run's
reconstruct() so the only difference from the frozen v3 pipeline is which
numbers go into the average.

Restart-safe: each panel-split lands in out/parts/ as it finishes, atomically,
and a rerun skips what is already there. A full pass is an hour and this machine
gets shut down, so losing it to a kill at minute fifty is not acceptable.
"""
import nice  # noqa: F401  - thread caps and priority, before numpy
import gc, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from common import ROOT, OUT, PANELS
from t1build import build
from t1feat import load_grid
from task1_run import FLOW_BLEND, predict_chunked
from ens_search import MEMBERS, TARGETS

SPLITS = ["validation", "private"]


import json as _json

# congested / transition / free by own-model speed over free-flow speed - must be
# the exact rule ens_search.Panel uses, or the shipped mix is not the one measured
REGIME_EDGES = (0.6, 0.8)


def load_regime_weights(path):
    """{regime: {target: {member: weight}}} from JSON, keys coerced to int."""
    raw = _json.loads(Path(path).read_text())
    return {int(r): v for r, v in raw.items()}


def parse_weights(spec):
    """One vector for every target, or one per target.

        "own=0.7,pooled=0.3"
        "speed:own=.51,pooled=.12,family=.21,alt=.16;flow:...;density:..."

    The per-target form matters because the least-squares optima genuinely
    differ between them - density leans on the pooled and alt members more than
    speed does.
    """
    def one(t):
        w = {}
        for part in t.split(","):
            k, v = part.split("=")
            w[k.strip()] = float(v)
        unknown = set(w) - set(MEMBERS)
        assert not unknown, f"unknown members: {sorted(unknown)}"
        assert w, "no weights given"
        return w

    if ":" in spec:
        out = {}
        for chunk in spec.split(";"):
            target, rest = chunk.split(":", 1)
            target = target.strip()
            assert target in TARGETS, f"unknown target {target}"
            out[target] = one(rest)
        missing = set(TARGETS) - set(out)
        assert not missing, f"no weights for {sorted(missing)}"
        return out
    w = one(spec)
    return {t: dict(w) for t in TARGETS}


def load_members(panel, w):
    """The boosters for every weighted member that exists for this panel."""
    if "by_regime" in w:
        wanted = {m for reg in w["by_regime"].values() for wt in reg.values()
                  for m, v in wt.items() if v > 0}
        wanted.add("own")                    # the regime is read off own's speed
    else:
        wanted = {m for wt in w.values() for m, v in wt.items() if v > 0}
    got = {}
    for name in sorted(wanted):
        try:
            got[name] = MEMBERS[name](panel)
        except (FileNotFoundError, OSError):
            print(f"  [{panel}] member {name} missing, dropping it", flush=True)
    assert got, f"{panel}: no members available"
    return got


def target_mask(panel, split, days, link_ids):
    """Boolean (nT, nL) marking exactly the rows the Task 1 template asks for.

    Not the blanked cells. Since the 2026-09-10 re-release the Task 2 horizon and
    a 60-minute buffer are blanked too and are not scored for Task 1, so "blank"
    over-selects - the organizers flagged it explicitly. It over-selected before
    the re-release as well, by about 215k cells per panel-split of genuinely
    missing observations, so this is a correctness fix and a 40 percent saving on
    the feature build at the same time.
    """
    key = pd.read_csv(ROOT / "task1" / panel / split / "sample_submission_state.csv",
                      usecols=["timestamp", "link_id"], dtype={"link_id": str})
    di = {d: i for i, d in enumerate(days)}
    li = {l: i for i, l in enumerate(link_ids)}
    ts = pd.to_datetime(key.timestamp)
    row = ts.dt.strftime("%Y-%m-%d").map(di).to_numpy()
    row = row * 288 + (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
    col = key.link_id.map(li).to_numpy()
    m = np.zeros((len(days) * 288, len(link_ids)), bool)
    m[row.astype(np.int64), col.astype(np.int64)] = True
    return m


def reconstruct(panel, split, members, w):
    G = load_grid(panel, split, masked=True)
    stat = G["stat"]
    link_ids = np.array([str(x) for x in G["link_ids"]])
    days = np.array(G["days"])
    del G
    gc.collect()

    want = target_mask(panel, split, days, link_ids)
    r = build(panel, split, targets="blanked", target_mask=want, verbose=False)
    del want
    ti, tj = r["ti"], r["tj"]
    mixed = {}
    if isinstance(w, dict) and "by_regime" in w:
        # Route by regime. Predict every member once, then take each cell's mix
        # from the weights of the regime its own-model speed puts it in. The
        # congestion specialist `phys` carries 0.47 of the speed weight among
        # congested cells and 0.02 in free flow, which a single global vector
        # cannot express.
        raw = {(n, t): predict_chunked(M[t], r["X"])
               for n, M in members.items() for t in TARGETS}
        ratio = raw[("own", "speed")] / np.maximum(stat["vf"][tj], 1.0)
        regime = np.digitize(ratio, REGIME_EDGES)
        for target in TARGETS:
            acc = np.zeros(len(tj))
            for reg, wreg in w["by_regime"].items():
                sel = regime == reg
                if not sel.any():
                    continue
                wt = wreg[target]
                tot = sum(wt.get(n, 0.0) for n in members)
                assert tot > 0, f"{panel}: no member for {target} in regime {reg}"
                for name in members:
                    if wt.get(name, 0.0):
                        acc[sel] += raw[(name, target)][sel] * (wt[name] / tot)
            mixed[target] = acc
        del raw
    else:
        for target in TARGETS:
            wt = w[target]
            tot = sum(wt.get(n, 0.0) for n in members)
            assert tot > 0, f"{panel}: no member available for {target}"
            acc = None
            for name, M in members.items():
                if not wt.get(name, 0.0):
                    continue
                a = predict_chunked(M[target], r["X"]) * (wt[name] / tot)
                acc = a if acc is None else acc + a
            mixed[target] = acc
    del r["X"], r
    gc.collect()

    sp = np.clip(mixed["speed"], 5.0, stat["vf"][tj] * 1.05)
    fl_pl = (1 - FLOW_BLEND) * mixed["flow"] + FLOW_BLEND * mixed["density"] * sp
    flow = np.clip(fl_pl * stat["lanes"][tj], 5.0, stat["cap"][tj] * 1.15)
    del mixed, fl_pl

    ts = pd.to_datetime(days[ti // 288]) + pd.to_timedelta((ti % 288) * 5, unit="m")
    d = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "link_id": link_ids[tj], "speed_kmh": sp, "flow_vph": flow})
    del sp, flow, ti, tj
    gc.collect()
    return d


def main():
    out_dir = Path(sys.argv[1])
    if sys.argv[2].endswith(".json"):
        w = {"by_regime": load_regime_weights(sys.argv[2])}
    else:
        w = parse_weights(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"weights {w} -> {out_dir}", flush=True)

    # one file per panel-split, so a kill costs one panel and not the whole run
    parts = out_dir / "parts"
    parts.mkdir(exist_ok=True)
    for panel in PANELS:
        todo = [sp for sp in SPLITS if not (parts / f"{panel}_{sp}.csv").exists()]
        if not todo:
            print(f"  [{panel}] both splits already written, skipping", flush=True)
            continue
        members = load_members(panel, w)
        for split in todo:
            d = reconstruct(panel, split, members, w)
            key = pd.read_csv(ROOT / "task1" / panel / split /
                              "sample_submission_state.csv",
                              dtype={"link_id": str, "station_id": str})
            merged = key[["panel", "timestamp", "station_id", "link_id",
                          "mask_regime"]].merge(d, on=["timestamp", "link_id"],
                                                how="left")
            miss = int(merged.speed_kmh.isna().sum())
            assert miss == 0, f"{panel}/{split}: {miss} unmatched target rows"
            tmp = parts / f"{panel}_{split}.csv.tmp"
            merged.to_csv(tmp, index=False)
            tmp.replace(parts / f"{panel}_{split}.csv")     # atomic
            print(f"  [{panel}/{split}] {len(merged):,} rows "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            del d, key, merged
            gc.collect()
        del members
        gc.collect()

    sub = pd.concat([pd.read_csv(parts / f"{p}_{sp}.csv",
                                 dtype={"link_id": str, "station_id": str})
                     for p in PANELS for sp in SPLITS], ignore_index=True)
    assert sub.speed_kmh.notna().all() and sub.flow_vph.notna().all()
    sub.to_csv(out_dir / "state_submission.csv", index=False)
    if "by_regime" in w:
        (out_dir / "weights.txt").write_text(_json.dumps(
            {str(k): v for k, v in w["by_regime"].items()}, indent=1))
    else:
        (out_dir / "weights.txt").write_text("".join(
            f"{t}: " + ", ".join(f"{k}={v}" for k, v in sorted(w[t].items()))
            + chr(10) for t in TARGETS))
    print(f"\nwrote {len(sub):,} rows to {out_dir/'state_submission.csv'}")


if __name__ == "__main__":
    main()
