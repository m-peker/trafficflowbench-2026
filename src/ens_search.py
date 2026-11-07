"""Score a Task 1 ensemble locally on the part of the composite it can move.

The five-run batch settled the question of what to optimize. The pooled/per-panel
blend was chosen on S_state alone, where it was worth +0.0003 - about +0.0001 on
the composite - and it delivered +0.00132 on the leaderboard. The holdout was
not wrong, it was incomplete: Task 1 feeds S_state (weight 0.35) *and*, through
the density q/v it implies, S_physics (weight 0.15), and only the first was
being measured.

So the harness scores

    partial = 0.35 * S_state + 0.15 * S_physics

aggregated per panel -> per family -> mean, the way the leaderboard aggregates.
It is not the composite - S_queue and S_ODME are untouched by Task 1 and drop
out of any comparison - but differences in `partial` are differences in the
composite, which is what a weight search needs.

S_physics needs a proxy, because the organizer residual cannot be reproduced
from the release. Their balance is

    resid = |dN - rhs|,  rhs = dt * (inflow + on_ramp - outflow - off_ramp)

and `lwr_mainline_topology.csv` shows why rhs is out of reach: every link has at
least one unsensored CONN-* boundary, and on D7_I10_E only 14 of 146 links have
both sides fully observable. Building rhs from what *is* released - either
positional neighbours (physics_local.py) or the topology's real incoming and
outgoing sets (lwr_topology.py) - gives mean|rhs| ~ 9.8 vehicles against
mean|dN| ~ 0.5, an 18x mismatch that scores the *ground truth itself* at
e_lwr ~ 0.99. That is not a near miss; it is the wrong quantity.

What is reachable comes from inverting the organizers' own number. They score a
perfect answer at S_physics 0.9544, so with S_FD ~ 0.98 their S_LWR on the truth
is ~0.94, which means their rhs and the truth's own dN agree to about six
percent. Then

    e_lwr  ~  sum|dN_ours - dN_true| / sum|dN_true|

with no boundary flows required. Measured on two panels it lands at S_LWR ~ 0.64
where the leaderboard implies ~0.79, so it is pessimistic in level - their
projection makes the target smoother than raw truth - but it responds the right
way: 0.3 pooled beats both 0.0 and 1.0 on both panels, which is what the
leaderboard said. Treat it as an ordering instrument, not a score. Its
sensitivity still understates the leaderboard by about 3x, so a weighting that
wins here by a hair is likely to win there by more.

Two phases, because building the feature matrix is the expensive part and the
weight search has to see it many times:

    python ens_search.py cache            # build X once per panel, predict every
                                          # member, save predictions and the
                                          # arrays the physics score needs
    python ens_search.py search           # sweep weights over the cached members

Adding an ensemble member means training it, registering it in MEMBERS, and
re-running `cache` for that member only; the search then picks it up.
"""
import gc, itertools, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from common import OUT, PANELS, FAMILY, net_dir
from t1feat import load_grid
from t1build import build, ramp_matrices
from task1_run import FLOW_BLEND, predict_chunked
from physics_local import lwr_score, fd_score, corridor_arrays, EVAL_DAYS, DT_H

import os as _os
# overridable so a second cache can be built alongside the first - used to
# measure what the regenerated ramp data did to models fitted on the old one
CACHE = OUT / _os.environ.get("ENS_CACHE", "ens_cache")
TARGETS = ("speed", "flow", "density")
# congested / transitional / free, by predicted speed over free-flow speed.
# Congestion is 0.6-7.6 percent of cells but carries 30-61 percent of the
# density error, and the members disagree most there (error correlation 0.72-0.89
# against 0.87-0.92 in free flow), so it is where a weighting has anything to
# decide.
REGIME_EDGES = (0.6, 0.8)
REGIME_NAMES = {0: "congested", 1: "transition", 2: "free"}

# name -> how to load its three boosters for a panel
MEMBERS = {
    "own":    lambda p: pickle.load(open(OUT / f"t1_model_{p}.pkl", "rb"))["models"],
    "pooled": lambda p: pickle.load(open(OUT / "t1_pooled.pkl", "rb"))["models"],
    "family": lambda p: pickle.load(
        open(OUT / f"t1_family_{FAMILY[p]}.pkl", "rb"))["models"],
    "alt":    lambda p: pickle.load(open(OUT / f"t1_alt_{p}.pkl", "rb"))["models"],
    "phys":   lambda p: pickle.load(open(OUT / f"t1_phys_{p}.pkl", "rb"))["models"],
}


def available(panel):
    """Members whose model file exists for this panel."""
    out = []
    for name in MEMBERS:
        try:
            MEMBERS[name](panel)
            out.append(name)
        except (FileNotFoundError, OSError):
            pass
    return out


# ---------------------------------------------------------------- cache phase

def cache_panel(panel, members=None):
    t0 = time.time()
    path = CACHE / f"{panel}.npz"
    if members and path.exists():
        try:
            have = {k[5:].split(".")[0] for k in np.load(path, allow_pickle=True).files
                    if k.startswith("pred_")}
            if set(members) <= have:
                print(f"[{panel}] {sorted(members)} already cached, skipping",
                      flush=True)
                return
        except Exception as e:                # a half-written file from a kill
            print(f"[{panel}] cache unreadable ({e}); rebuilding", flush=True)
            path.unlink(missing_ok=True)
    Gm, truth_sp, truth_fl, R = corridor_arrays(panel)
    stat = Gm["stat"]
    nT, nL = Gm["speed"].shape
    nd = nT // 288
    lo = (nd - EVAL_DAYS) * 288                    # a contiguous block of rows
    rows = np.zeros(nT, bool)
    rows[lo:] = True

    target_mask = np.zeros((nT, nL), bool)
    target_mask[lo:] = True
    r = build(panel, "train", targets="blanked", target_mask=target_mask,
              verbose=False)
    ti, tj = r["ti"], r["tj"]
    X = r["X"]
    del r

    names = members or available(panel)
    preds = {}
    for name in names:
        M = MEMBERS[name](panel)
        for t in TARGETS:
            preds[f"{name}.{t}"] = predict_chunked(M[t], X).astype(np.float32)
        del M
        gc.collect()
        print(f"  [{panel}] predicted {name} [{time.time()-t0:.0f}s]", flush=True)
    del X
    gc.collect()

    # the FD score needs per-link parameters; read them now so search needs no I/O
    fd = pd.read_csv(net_dir(panel) / "fd_parameters.csv",
                     dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    kcrit = np.array([fd.critical_density.get(l, 55.0) for l in Gm["link_ids"]])
    kjam = np.array([fd.k_jam.get(l, 300.0) for l in Gm["link_ids"]])

    # everything the scorer needs, sliced to the eval block to keep it small
    old = dict(np.load(path, allow_pickle=True)) if path.exists() else {}
    old.update({f"pred_{k}": v for k, v in preds.items()})
    old.update(dict(
        ti=(ti - lo).astype(np.int32), tj=tj.astype(np.int32),
        obs_sp=Gm["speed"][lo:].astype(np.float32),
        obs_fl=Gm["flow"][lo:].astype(np.float32),
        truth_sp=truth_sp[lo:].astype(np.float32),
        truth_fl=truth_fl[lo:].astype(np.float32),
        ramp_on=R["on"][lo:].astype(np.float32),
        ramp_off=R["off"][lo:].astype(np.float32),
        ramp_ok=(R["miss"][lo:] == 0),
        vf=stat["vf"], lanes=stat["lanes"], cap=stat["cap"],
        length=stat["length"], kcrit=kcrit, kjam=kjam,
        members=np.array(sorted({k[5:].split(".")[0] for k in old
                                 if k.startswith("pred_")} | set(names)),
                         dtype=object)))
    CACHE.mkdir(exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")   # savez appends .npz
    np.savez_compressed(tmp, **old)
    tmp.replace(path)                         # atomic; a kill leaves the old file
    print(f"[{panel}] cached {len(ti):,} cells, members "
          f"{list(old['members'])} [{time.time()-t0:.0f}s]", flush=True)
    del Gm, truth_sp, truth_fl, R, preds, old
    gc.collect()


# --------------------------------------------------------------- search phase

class Panel:
    """One panel's cached arrays, with a score() that takes ensemble weights."""

    def __init__(self, panel):
        z = np.load(CACHE / f"{panel}.npz", allow_pickle=True)
        self.panel = panel
        self.p = {k[5:]: z[k] for k in z.files if k.startswith("pred_")}
        self.members = sorted({k.split(".")[0] for k in self.p})
        self.ti, self.tj = z["ti"], z["tj"]
        self.obs_sp, self.obs_fl = z["obs_sp"], z["obs_fl"]
        self.truth_sp, self.truth_fl = z["truth_sp"], z["truth_fl"]
        self.on, self.off, self.ramp_ok = z["ramp_on"], z["ramp_off"], z["ramp_ok"]
        self.vf, self.lanes, self.cap = z["vf"], z["lanes"], z["cap"]
        self.length, self.kcrit, self.kjam = z["length"], z["kcrit"], z["kjam"]
        self.stat = dict(lanes=self.lanes, cap=self.cap, vf=self.vf,
                         length=self.length, panel=panel)
        # truth at the filled cells, for S_state
        self.v_t = self.truth_sp[self.ti, self.tj].astype(np.float64)
        self.q_t = (self.truth_fl[self.ti, self.tj]
                    / np.maximum(self.lanes[self.tj], 1.0))
        self.ok = np.isfinite(self.v_t) & np.isfinite(self.q_t)
        self.rows = np.ones(self.obs_sp.shape[0], bool)
        # the physics reference: the true vehicle count's own time difference
        Nt = (self.truth_fl.astype(np.float64)
              / np.maximum(self.truth_sp.astype(np.float64), 1.0)) * self.length[None, :]
        self.dNt = Nt[1:] - Nt[:-1]
        self.dNt_ok = np.isfinite(self.dNt)
        # regime index, from one member's speed rather than the blend's, so the
        # bin a cell falls in does not move as the weights being fitted change
        ref = self.p.get("own.speed", next(iter(self.p.values())))
        ratio = ref.astype(np.float64) / np.maximum(self.vf[self.tj], 1.0)
        self.regime = np.digitize(ratio, REGIME_EDGES)

    def mix(self, w, target):
        """w is {member: weight}, or {target: {member: weight}} to weight each
        target separately - the per-target optima differ enough to be worth it."""
        if "by_regime" in w:
            acc = np.zeros(len(self.tj))
            for reg, ws in w["by_regime"].items():
                m = self.regime == reg
                if m.any():
                    acc[m] = self._mix_flat(ws, target, m)
            return acc
        wt = w.get(target, w) if target in w else w
        tot = sum(wt.get(m, 0.0) for m in self.members if f"{m}.{target}" in self.p)
        assert tot > 0, f"{self.panel}: no weighted member available for {target}"
        acc = None
        for m in self.members:
            k = f"{m}.{target}"
            if k not in self.p or not wt.get(m, 0.0):
                continue
            a = self.p[k].astype(np.float64) * (wt[m] / tot)
            acc = a if acc is None else acc + a
        return acc

    def _mix_flat(self, w, target, sel):
        """One regime's slice of the mix; w is {target: {member: weight}}."""
        wt = w.get(target, w) if target in w else w
        tot = sum(wt.get(m, 0.0) for m in self.members if f"{m}.{target}" in self.p)
        assert tot > 0, f"{self.panel}: no weighted member for {target}"
        acc = None
        for m in self.members:
            k = f"{m}.{target}"
            if k not in self.p or not wt.get(m, 0.0):
                continue
            a = self.p[k].astype(np.float64)[sel] * (wt[m] / tot)
            acc = a if acc is None else acc + a
        return acc

    def score(self, w):
        tj = self.tj
        v = np.clip(self.mix(w, "speed"), 5.0, self.vf[tj] * 1.05)
        k = self.mix(w, "density")
        q_pl = (1 - FLOW_BLEND) * self.mix(w, "flow") + FLOW_BLEND * k * v
        q = np.clip(q_pl * self.lanes[tj], 5.0, self.cap[tj] * 1.15)

        o = self.ok
        S_v = 1 - float(np.sqrt(np.mean((v[o] - self.v_t[o]) ** 2))) / 25.0
        S_q = 1 - float(np.sqrt(np.mean(
            (q[o] / self.lanes[tj][o] - self.q_t[o]) ** 2))) / 600.0
        S_state = 0.54 * S_v + 0.46 * S_q

        sp = self.obs_sp.astype(np.float64).copy()
        fl = self.obs_fl.astype(np.float64).copy()
        sp[self.ti, tj] = v
        fl[self.ti, tj] = q
        N = (fl / np.maximum(sp, 1.0)) * self.length[None, :]
        dN = N[1:] - N[:-1]
        m = np.isfinite(dN) & self.dNt_ok
        e = (np.abs(dN[m] - self.dNt[m]).sum()
             / (np.abs(self.dNt[m]).sum() + 1e-9))
        S_lwr = max(0.0, 1.0 - min(1.0, float(e)))
        S_fd = self._fd(sp, fl)
        S_phys = (S_fd + 2 * S_lwr) / 3.0
        dens = float(np.sqrt(np.mean((q[o] / v[o] - self.q_t[o] /
                                      np.maximum(self.v_t[o], 1.0)) ** 2)))
        return dict(S_state=S_state, S_v=S_v, S_q=S_q, S_lwr=S_lwr, S_fd=S_fd,
                    S_physics=S_phys, partial=0.35 * S_state + 0.15 * S_phys,
                    e_lwr=e, dens_rmse=dens)

    def _fd(self, speed, flow):
        lanes = np.maximum(self.lanes, 1.0)[None, :]
        q_lane = flow / lanes
        k_lane = flow / np.maximum(speed, 1.0) / lanes
        kcrit, kjam = self.kcrit[None, :] / lanes, self.kjam[None, :] / lanes
        cap, vf = self.cap[None, :] / lanes, self.vf[None, :]
        q_fd = np.where(k_lane <= kcrit, vf * k_lane,
                        (cap / np.maximum(kjam - kcrit, 1e-9))
                        * np.maximum(kjam - k_lane, 0))
        m = np.isfinite(q_lane) & np.isfinite(q_fd)
        return float(max(0.0, 1.0 - np.abs(q_lane[m] - q_fd[m]).sum()
                         / (np.abs(q_lane[m]).sum() + 1e-9)))


def aggregate(per_panel):
    """panel -> family mean -> mean over families, as the leaderboard does."""
    fams = sorted({FAMILY[p] for p in per_panel})
    out = {}
    for key in next(iter(per_panel.values())):
        out[key] = float(np.mean([
            np.mean([v[key] for p, v in per_panel.items() if FAMILY[p] == f])
            for f in fams]))
    return out


def stack_weights(P, target, ridge=1e-8, sel=None):
    """Non-negative weights summing to 1 that best predict `target` on holdout.

    This is the statistically right way to weight an ensemble and it costs
    nothing: the members' predictions and the truth are already in memory, so it
    is an n x n least-squares problem per panel per target rather than a grid
    search over the simplex. Non-negativity and the sum-to-one constraint are
    what keep it from behaving like an unconstrained regression on correlated
    columns, where the weights blow up in opposite directions and do not
    generalise.

    Solved exactly by active-set enumeration rather than by descent. The members
    agree with each other far more than they disagree - error correlations run
    0.83 to 0.97 - so the quadratic is a long thin valley whose informative
    direction is its flattest one, and a projected gradient crawls along it: on
    D7_I10_W two thousand steps still read 0.50 where the true optimum is 0.60.
    With at most four members there are only 2^n - 1 subsets to try, so the
    exact answer is cheaper than an approximate one.
    """
    names = [m for m in P.members if f"{m}.{target}" in P.p]
    o = P.ok if sel is None else (P.ok & sel)
    if o.sum() < 500:                         # too thin to fit; fall back to all
        o = P.ok
    y = {"speed": P.v_t, "flow": P.q_t,
         "density": P.q_t / np.maximum(P.v_t, 1.0)}[target][o]
    A = np.column_stack([P.p[f"{m}.{target}"].astype(np.float64)[o] for m in names])
    n = len(names)
    G = A.T @ A / len(y) + ridge * np.eye(n)
    c = A.T @ y / len(y)

    best, best_obj = None, np.inf
    for k in range(1, n + 1):
        for S in itertools.combinations(range(n), k):
            S = list(S)
            # minimise w'Gw - 2c'w over the active set, subject to sum w = 1
            m = len(S)
            K = np.zeros((m + 1, m + 1))
            K[:m, :m] = G[np.ix_(S, S)]
            K[:m, m] = K[m, :m] = 1.0
            rhs = np.append(c[S], 1.0)
            try:
                sol = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError:
                continue
            w_S = sol[:m]
            if (w_S < -1e-9).any():
                continue                      # infeasible on this active set
            w = np.zeros(n)
            w[S] = np.maximum(w_S, 0.0)
            obj = float(w @ G @ w - 2 * c @ w)
            if obj < best_obj:
                best, best_obj = w, obj
    if best is None:                          # degenerate; fall back to equal
        best = np.full(n, 1.0 / n)
    return dict(zip(names, best))


def mean_weights(per_panel_w, members):
    """Average a per-panel weighting into one global vector."""
    out = {}
    for t in TARGETS:
        v = np.array([[w[t].get(m, 0.0) for m in members]
                      for w in per_panel_w.values()]).mean(0)
        out[t] = dict(zip(members, v / v.sum()))
    return out


def fmt(w):
    if set(w) & set(TARGETS):
        return " | ".join(f"{t[:4]}:" + "/".join(f"{v:.2f}" for v in w[t].values())
                          for t in TARGETS)
    return "/".join(f"{k} {v:.2f}" for k, v in w.items())


def search():
    panels = [p for p in PANELS if (CACHE / f"{p}.npz").exists()]
    assert panels, "run `python ens_search.py cache` first"
    P = {p: Panel(p) for p in panels}
    have = sorted(set.intersection(*[set(x.members) for x in P.values()]))
    print(f"{len(panels)} panels cached, members present in all: {have}" + chr(10))

    # per panel, per target: the least-squares optimal simplex weights
    per_panel = {p: {t: stack_weights(P[p], t) for t in TARGETS} for p in panels}
    print("least-squares weights per panel (" + ", ".join(have) + ")")
    for p in panels:
        print(f"  {p:11s} " + " | ".join(
            f"{t[:4]}:" + "/".join(f"{per_panel[p][t].get(m,0):.2f}" for m in have)
            for t in TARGETS))
    glob = mean_weights(per_panel, have)
    print(chr(10) + "global average: " + fmt(glob) + chr(10))

    candidates = [("v3 shipped: own only", {"own": 1.0}),
                  ("flat 70/30 own/pooled", {"own": 0.7, "pooled": 0.3}),
                  ("equal over all members", {m: 1.0 for m in have}),
                  ("least-squares, global per-target", glob)]
    if len(have) > 2:
        for m in have:
            if m not in ("own",):
                candidates.append((f"own 0.7 / {m} 0.3", {"own": 0.7, m: 0.3}))

    print(f"{'candidate':34s} {'S_state':>8s} {'S_phys':>7s} {'partial':>8s} "
          f"{'delta':>9s}")
    base, rows = None, []
    for label, w in candidates:
        agg = aggregate({p: P[p].score(w) for p in panels})
        if base is None:
            base = agg["partial"]
        rows.append(dict(label=label, weights=fmt(w), **agg))
        print(f"{label:34s} {agg['S_state']:8.4f} {agg['S_physics']:7.4f} "
              f"{agg['partial']:8.5f} {agg['partial']-base:+9.5f}", flush=True)

    # weightings that vary per panel or per cell cannot be written as one vector,
    # so score them directly
    extra = [("least-squares, per panel per target",
              {p: per_panel[p] for p in panels})]

    # per regime: the members disagree most in congestion, and the optimum there
    # is far from the free-flow optimum, so let the weight follow the regime
    reg_w = {p: {"by_regime": {r: {t: stack_weights(P[p], t, sel=P[p].regime == r)
                                   for t in TARGETS}
                               for r in sorted(REGIME_NAMES)}}
             for p in panels}
    extra.append(("least-squares, per regime (per panel)", reg_w))

    # and the same thing averaged into one global per-regime vector, which is the
    # version worth shipping if it holds up: three regimes x three targets, fitted
    # on millions of cells, rather than one vector per corridor
    glob_reg = {"by_regime": {}}
    for r in sorted(REGIME_NAMES):
        acc = {t: np.array([[reg_w[p]["by_regime"][r][t].get(m, 0.0) for m in have]
                            for p in panels]).mean(0) for t in TARGETS}
        glob_reg["by_regime"][r] = {t: dict(zip(have, v / v.sum()))
                                    for t, v in acc.items()}
    extra.append(("least-squares, per regime (global)",
                  {p: glob_reg for p in panels}))

    for label, wmap in extra:
        agg = aggregate({p: P[p].score(wmap[p]) for p in panels})
        rows.append(dict(label=label, weights="(varies)", **agg))
        print(f"{label:34s} {agg['S_state']:8.4f} {agg['S_physics']:7.4f} "
              f"{agg['partial']:8.5f} {agg['partial']-base:+9.5f}", flush=True)

    print()
    for r in sorted(REGIME_NAMES):
        share = float(np.mean([(P[p].regime == r).mean() for p in panels]))
        print(f"  {REGIME_NAMES[r]:11s} {share:6.2%} of cells   " + " | ".join(
            f"{t[:4]}:" + "/".join(f"{glob_reg['by_regime'][r][t][m]:.2f}"
                                   for m in have) for t in TARGETS))

    pd.DataFrame(rows).to_csv(OUT / "ens_search.csv", index=False)
    best = max(rows, key=lambda r: r["partial"])
    print(chr(10) + f"best: {best['label']}   partial {best['partial']:.5f} "
          f"({best['partial']-base:+.5f} vs v3)")
    print(f"weights: {best['weights']}")
    print(chr(10) + "Note: the physics proxy understates the leaderboard by "
          "roughly 3x, so treat these deltas as a lower bound.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    cmd = argv[0] if argv else "search"
    argv = argv[1:]
    only = None
    if argv and argv[0] == "--members":
        only = argv[1].split(",")
        argv = argv[2:]
    if cmd == "cache":
        CACHE.mkdir(exist_ok=True)
        for p in (argv or PANELS):
            cache_panel(p, members=only)
    else:
        search()
