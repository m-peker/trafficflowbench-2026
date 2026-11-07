"""Replace Task 1 rows inside the Task 2 gaps with an edge + profile fill.

Validation/private blank the whole corridor for each Task 2 horizon plus the 60
minutes after it; ~1.9% of the scored Task 1 rows sit there. The nets never saw
a full-corridor blackout and fill those cells with near free-flow values at
queue events (t1_gap_probe.py: speed RMSE 12.6, flow 300 vph/lane in-gap vs
~1.3 / ~36 outside). This rewrites exactly those rows:

    edges    mean of the observed values in the 6 steps before the gap (the raw
             Task 2 history) and the 6 steps after it
    profile  per-link time-of-day mean over all train days (weekday/weekend),
             scaled to the level seen at the two edges
    rule     speed and flow = a(d) * interp + (1 - a(d)) * profile,
             a(d) = exp(-d / TAU), d = steps to the nearer edge   (GAP_RULE=wexp)
             or speed = mean(interp, profile), flow = profile     (GAP_RULE=mix)

    python t1_gapfill.py IN_STATE.csv OUT_STATE.csv
"""
import os, sys
import numpy as np
import pandas as pd
from common import ROOT
from t1feat import load_grid
from t1_gap_analog import analog_fill

GAP, EDGE = 18, 6
RULE = os.environ.get("GAP_RULE", "wexp")   # wexp | mix | analog (mix as fallback)
TAU = float(os.environ.get("GAP_TAU", "3.0"))
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N",
          "D7_I405_S", "D12_I5_N", "D12_I5_S"]


def train_profile(panel, link_ids):
    Gt = load_grid(panel, "train", masked=False)
    o = [Gt["link_ids"].index(l) for l in link_ids]
    wk = (Gt["dow"] >= 5).astype(int); tod = Gt["tod"]
    out = {}
    for c in ("speed", "flow"):
        x = np.where(Gt["pct"][:, o] >= 75, Gt[c][:, o], np.nan)
        prof = np.full((2, 288, len(o)), np.nan)
        for w in (0, 1):
            m = wk == w
            s = np.zeros((288, len(o))); n = np.zeros((288, len(o)))
            ok = np.isfinite(x[m])
            np.add.at(s, tod[m], np.where(ok, x[m], 0.0)); np.add.at(n, tod[m], ok)
            p = s / np.maximum(n, 1); p[n == 0] = np.nan
            prof[w] = p
        out[c] = prof
    return out


def fill_panel(panel, split):
    G = load_grid(panel, split, masked=True)
    nT, nL = G["nT"], G["nL"]
    prof = train_profile(panel, G["link_ids"])
    wk = (G["dow"] >= 5).astype(int); tod = G["tod"]
    day0 = pd.Timestamp(G["days"][0], tz="UTC")
    wi = pd.read_csv(ROOT / "task2" / panel / split / "window_index.csv")
    vals = {c: np.full((nT, nL), np.nan) for c in ("speed", "flow")}
    for r in wi.itertuples():
        a = int((pd.Timestamp(r.forecast_start) - day0).total_seconds() // 300)
        b = a + GAP - 1
        if a - EDGE < 0 or b + 1 + EDGE > nT:
            continue
        for c in ("speed", "flow"):
            x = np.where(G["pct"] >= 75, G[c], np.nan)
            with np.errstate(all="ignore"):
                left = np.nanmean(x[a - EDGE:a], 0); right = np.nanmean(x[b + 1:b + 1 + EDGE], 0)
                pl = np.nanmean(prof[c][wk[a - 1], tod[a - EDGE:a]], 0)
                pr = np.nanmean(prof[c][wk[b + 1], tod[b + 1:b + 1 + EDGE]], 0)
                rl, rr = left / pl, right / pr
            rl = np.where(np.isfinite(rl), rl, rr); rr = np.where(np.isfinite(rr), rr, rl)
            rl = np.where(np.isfinite(rl), rl, 1.0); rr = np.where(np.isfinite(rr), rr, 1.0)
            L = np.where(np.isfinite(left), left, right); R = np.where(np.isfinite(right), right, left)
            for i, t in enumerate(range(a, b + 1)):
                w = (i + 1) / (GAP + 1)
                li = (1 - w) * L + w * R
                p = prof[c][wk[t], tod[t]] * ((1 - w) * rl + w * rr)
                p = np.where(np.isfinite(p), p, li); li = np.where(np.isfinite(li), li, p)
                d = min(i, GAP - 1 - i)
                if RULE == "wexp":
                    al = np.exp(-d / TAU)
                    vals[c][t] = al * li + (1 - al) * p
                else:                                    # mix, and analog's fallback
                    vals[c][t] = 0.5 * (li + p) if c == "speed" else p
    if os.environ.get("GAP_NET"):
        # fallback = mean(edge+profile fill, a gap-aware net); analogs still override below
        from common import OUT
        Z = [np.load(OUT / f"gapnet_{n}_{panel}_{split}.npz") for n in os.environ["GAP_NET"].split(",")]
        w0 = float(os.environ.get("GAP_NETW", "0.5"))         # weight of the nets' mean
        for c, k in (("speed", "v"), ("flow", "q")):
            w = float(os.environ.get("GAP_NETW_Q", w0)) if c == "flow" else w0
            net = np.mean([z[k] for z in Z], 0)
            vals[c] = np.where(np.isfinite(vals[c]), (1 - w) * vals[c] + w * net, np.nan)
    if RULE == "analog":
        Gt = load_grid(panel, "train", masked=False)
        o = [Gt["link_ids"].index(l) for l in G["link_ids"]]
        Lv = np.where(Gt["pct"][:, o] >= 75, Gt["speed"][:, o], np.nan)
        Lq = np.where(Gt["pct"][:, o] >= 75, Gt["flow"][:, o], np.nan)
        Ev = np.where(G["pct"] >= 75, G["speed"], np.nan)
        Eq = np.where(G["pct"] >= 75, G["flow"], np.nan)
        wins = [int((pd.Timestamp(r.forecast_start) - day0).total_seconds() // 300) - 1
                for r in wi.itertuples()]
        wins = [t for t in wins if t + 1 - EDGE >= 0 and t + 1 + GAP + EDGE <= nT]
        av, aq = analog_fill(Ev, Eq, wins, Lv, Lq, np.ones(Lv.shape[0], bool),
                             G["stat"]["vf"][None, :], G["stat"]["cap"][None, :], adjust=False,
                             max_dist=float(os.environ["GAP_MAXD"]) if os.environ.get("GAP_MAXD") else None)
        sm = os.environ.get("GAP_SM", "")
        if sm:
            # same-month library: this split's own masked grid, gaps blank, starts away
            # from every gap span and more than one day from the window being filled
            from t1_samemonth_dist import scores
            vf_, cap_ = G["stat"]["vf"][None, :], G["stat"]["cap"][None, :]
            Bv, Bq = Ev.copy(), Eq.copy()
            lib_sm = np.ones(nT, bool)
            for t in wins:
                Bv[t + 1:t + 1 + GAP] = np.nan; Bq[t + 1:t + 1 + GAP] = np.nan
                lib_sm[max(0, t + 1 - GAP - 2 * EDGE):t + 1 + GAP + EDGE] = False
            dA = scores(Ev, Eq, wins, Lv, Lq, np.ones(Lv.shape[0], bool), vf_, cap_)[:, 0]
            dB = scores(Ev, Eq, wins, Bv, Bq, lib_sm, vf_, cap_, excl=True)[:, 0]
            md = float(os.environ.get("GAP_MAXD", "0.002"))
            n_sw = 0
            for i, t in enumerate(wins):
                rr = slice(t + 1, t + 1 + GAP)
                useB = (dB[i] <= md) and ((dA[i] > md) if sm == "rescue" else (dB[i] < dA[i]))
                if not useB:
                    continue
                lib_i = lib_sm & (np.abs(np.arange(nT) - (t + 1)) > 288)
                bv, bq = analog_fill(Ev, Eq, [t], Bv, Bq, lib_i, vf_, cap_, adjust=False)
                av[rr] = bv[rr]; aq[rr] = bq[rr]; n_sw += 1
            print(f"    same-month ({sm}): {n_sw} of {len(wins)} windows from the month's own days; "
                  f"A good {int((dA <= md).sum())}, B good {int((dB <= md).sum())}", flush=True)
        n_an = int(np.isfinite(av).sum())
        vals["speed"] = np.where(np.isfinite(av), av, vals["speed"])
        vals["flow"] = np.where(np.isfinite(aq), aq, vals["flow"])
        print(f"    analog filled {n_an} gap cells", flush=True)
    return G, day0, vals


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sub = pd.read_csv(src, dtype={"link_id": str})
    ts = pd.to_datetime(sub.timestamp, utc=True)
    changed = 0
    for panel in PANELS:
        for split in ("validation", "private"):
            G, day0, vals = fill_panel(panel, split)
            lix = {l: j for j, l in enumerate(G["link_ids"])}
            m = (sub.panel == panel).to_numpy()
            ti = ((ts[m] - day0).dt.total_seconds() // 300).astype(int).to_numpy()
            inr = (ti >= 0) & (ti < G["nT"])
            idx = np.flatnonzero(m)[inr]; ti = ti[inr]
            tj = sub.link_id.to_numpy()[idx]
            tj = np.array([lix.get(l, -1) for l in tj])
            ok = tj >= 0
            idx, ti, tj = idx[ok], ti[ok], tj[ok]
            fs, fq = vals["speed"][ti, tj], vals["flow"][ti, tj]
            good = np.isfinite(fs) & np.isfinite(fq)
            vf = G["stat"]["vf"][tj]; cap = G["stat"]["cap"][tj]
            sub.loc[idx[good], "speed_kmh"] = np.clip(fs[good], 5.0, vf[good] * 1.05)
            sub.loc[idx[good], "flow_vph"] = np.clip(fq[good], 52.0, cap[good] * 1.15)
            changed += int(good.sum())
            print(f"  {panel}/{split}: {int(good.sum())} rows rewritten", flush=True)
    sub.to_csv(dst, index=False)
    print(f"wrote {dst}: {changed} rows ({changed / len(sub):.3%}) rewritten")


if __name__ == "__main__":
    main()
