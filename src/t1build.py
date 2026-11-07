"""Assemble the Task 1 design matrix for a panel/split.

Everything is evaluated at the target cells directly.  The only full
(time x link) arrays alive at once are the four channel grids, the availability
mask, the four index grids and one profile-ratio scratch array.
"""
import gc
import numpy as np
from common import WORK
from t1feat import (load_grid, time_indices, link_indices, gather_t, gather_l,
                    lin_interp, profile, smooth_tod, rolling_sums, rolling_at)


def ramp_matrices(panel, split, link_ids):
    z = np.load(WORK / "cache" / f"{panel}_{split}_ramp.npz", allow_pickle=True)
    src = [str(x) for x in z["link_ids"]]
    o = [src.index(l) for l in link_ids]
    return {k: z[k][:, o].astype(np.float32) for k in ("on", "off", "miss")}


def build(panel, split, truth=None, sample=None, seed=0, verbose=True,
          targets="eligible", target_mask=None):
    """targets='eligible' -> only scoreable blanked cells (training/Task 1),
    targets='blanked'  -> every blanked cell, so a full grid can be filled in."""
    G = load_grid(panel, split, masked=True)
    nT, nL = G["nT"], G["nL"]
    stat = G["stat"]
    lanes_v = stat["lanes"]
    sp, oc, pct = G["speed"], G["occ"], G["pct"]
    tot = G["flow"]
    fpl = (tot / lanes_v[None, :]).astype(np.float32)
    avail = np.isfinite(sp) & np.isfinite(tot)

    tgt = ((pct >= 75) & ~avail) if targets == "eligible" else ~avail
    if target_mask is not None:
        tgt = tgt & target_mask
    ti, tj = np.nonzero(tgt)
    ti = ti.astype(np.int32)
    tj = tj.astype(np.int32)
    n_all = len(ti)
    if sample is not None and n_all > sample:
        k = np.sort(np.random.default_rng(seed).choice(n_all, sample, replace=False))
        ti, tj = ti[k], tj[k]
    del tgt
    if verbose:
        print(f"[{panel}/{split}] {nT}x{nL} avail={avail.mean():.3f} "
              f"targets={n_all:,} used={len(ti):,}", flush=True)

    cols, blocks = [], []

    def add(name, vec):
        cols.append(name)
        blocks.append(np.asarray(vec, np.float32))

    def get(name):
        return blocks[cols.index(name)]

    # ---- index grids, built once and read at the targets ----
    prev_i, next_i = time_indices(avail)
    pi = prev_i[ti, tj].astype(np.int64)
    ni = next_i[ti, tj].astype(np.int64)
    pi2 = np.where(pi >= 1, prev_i[np.clip(pi - 1, 0, nT - 1), tj], -1).astype(np.int64)
    ni2 = np.where(ni + 1 < nT, next_i[np.clip(ni + 1, 0, nT - 1), tj], nT).astype(np.int64)
    del prev_i, next_i

    up_i, dn_i = link_indices(avail)
    ui = up_i[ti, tj].astype(np.int64)
    di = dn_i[ti, tj].astype(np.int64)
    del up_i, dn_i
    gc.collect()

    pg = np.where(pi >= 0, ti - pi, 999).astype(np.float32)
    ng = np.where(ni < nT, ni - ti, 999).astype(np.float32)
    pg2 = np.where(pi2 >= 0, ti - pi2, 999).astype(np.float32)
    ng2 = np.where(ni2 < nT, ni2 - ti, 999).astype(np.float32)
    ud = np.where(ui >= 0, tj - ui, 999).astype(np.float32)
    dd = np.where(di < nL, di - tj, 999).astype(np.float32)

    # ---- per-channel temporal / spatial context ----
    for nm, mat in (("sp", sp), ("fl", fpl), ("oc", oc)):
        pv = gather_t(mat, pi, tj, nT)
        nv = gather_t(mat, ni, tj, nT)
        uv = gather_l(mat, ti, ui, nL)
        dv = gather_l(mat, ti, di, nL)
        li = lin_interp(pv, pg, nv, ng)
        si = lin_interp(uv, ud, dv, dd)
        add(f"{nm}_prev", pv); add(f"{nm}_prevg", pg)
        add(f"{nm}_next", nv); add(f"{nm}_nextg", ng)
        add(f"{nm}_tlin", li)
        add(f"{nm}_up", uv); add(f"{nm}_upd", ud)
        add(f"{nm}_dn", dv); add(f"{nm}_dnd", dd)
        add(f"{nm}_slin", si)
        add(f"{nm}_blend", 0.85 * li + 0.15 * np.where(np.isfinite(si), si, li))
        add(f"{nm}_prev2", gather_t(mat, pi2, tj, nT)); add(f"{nm}_prev2g", pg2)
        add(f"{nm}_next2", gather_t(mat, ni2, tj, nT)); add(f"{nm}_next2g", ng2)
        cs_v, cs_a = rolling_sums(mat, avail)
        add(f"{nm}_roll3", rolling_at(cs_v, cs_a, ti, tj, 3, nT))
        add(f"{nm}_roll9", rolling_at(cs_v, cs_a, ti, tj, 9, nT))
        del cs_v, cs_a
        for k in (1, 2, -1, -2):
            jj = tj - k
            ok = (jj >= 0) & (jj < nL)
            v = np.full(len(ti), np.nan, np.float32)
            idx = np.nonzero(ok)[0]
            if len(idx):
                a, b = ti[idx], jj[idx]
                v[idx] = np.where(avail[a, b], mat[a, b], np.nan)
            add(f"{nm}_{'up' if k > 0 else 'dn'}{abs(k)}", v)
        gc.collect()

    # ---- historical profiles from the observed cells of this split ----
    tod_t = G["tod"].astype(np.int64)
    dt_t = (G["dow"].astype(np.int64) * 288 + tod_t)
    for nm, mat in (("sp", sp), ("fl", fpl)):
        p_tod = smooth_tod(profile(mat, avail, tod_t, 288), 3)
        p_dt = profile(mat, avail, dt_t, 7 * 288)
        base = p_tod[tod_t[ti], tj]
        add(f"{nm}_prof_tod", base)
        add(f"{nm}_prof_dt", p_dt[dt_t[ti], tj])
        del p_dt
        ratio = np.where(avail, mat / np.maximum(p_tod[tod_t], 1e-3), np.nan
                         ).astype(np.float32)
        with np.errstate(invalid="ignore"):
            corr = np.nanmedian(ratio, 1)
        corr = np.where(np.isfinite(corr), corr, 1.0).astype(np.float32)
        add(f"{nm}_corr_ratio", corr[ti])
        add(f"{nm}_prof_scaled", base * corr[ti])
        loc = np.empty(len(ti), np.float32)
        order = np.argsort(tj, kind="stable")
        with np.errstate(invalid="ignore", all="ignore"):
            start = 0
            while start < len(order):
                j = tj[order[start]]
                end = start
                while end < len(order) and tj[order[end]] == j:
                    end += 1
                sel = order[start:end]
                a, b = max(0, j - 3), min(nL, j + 4)
                loc[sel] = np.nanmedian(ratio[ti[sel], a:b], 1)
                start = end
        loc = np.where(np.isfinite(loc), loc, 1.0).astype(np.float32)
        add(f"{nm}_loc_ratio", loc)
        add(f"{nm}_prof_loc", base * loc)
        del ratio, p_tod, corr, loc
        gc.collect()

    # ---- conservation of vehicles along the corridor ----
    q_up = gather_l(tot, ti, ui, nL)
    q_dn = gather_l(tot, ti, di, nL)
    own_lanes = lanes_v[tj]
    add("q_up_tot", q_up); add("q_dn_tot", q_dn)
    add("q_up_pl", q_up / own_lanes); add("q_dn_pl", q_dn / own_lanes)
    lu = np.where(ui >= 0, lanes_v[np.clip(ui, 0, nL - 1)], np.nan)
    ld = np.where(di < nL, lanes_v[np.clip(di, 0, nL - 1)], np.nan)
    add("q_up_pl_own", q_up / np.maximum(lu, 1))
    add("q_dn_pl_own", q_dn / np.maximum(ld, 1))

    R = ramp_matrices(panel, split, G["link_ids"])
    C = np.cumsum(R["on"] - R["off"], 1).astype(np.float32)
    Cm = np.cumsum(R["miss"], 1).astype(np.float32)
    C_here = C[ti, tj]
    cons_up = q_up + (C_here - gather_l(C, ti, ui, nL))
    cons_dn = q_dn + (C_here - gather_l(C, ti, di, nL))
    add("cons_up_pl", cons_up / own_lanes)
    add("cons_dn_pl", cons_dn / own_lanes)
    with np.errstate(invalid="ignore"):
        add("cons_mean_pl", np.nanmean(np.stack([cons_up, cons_dn]), 0) / own_lanes)
    add("cons_up_miss", np.abs(Cm[ti, tj] - gather_l(Cm, ti, ui, nL)))
    add("ramp_on", R["on"][ti, tj]); add("ramp_off", R["off"][ti, tj])
    add("ramp_miss", R["miss"][ti, tj])
    del R, C, Cm
    gc.collect()

    # ---- static and calendar ----
    for nm in ("vf", "lanes", "cap", "length"):
        add(nm, stat[nm][tj])
    add("cap_pl", (stat["cap"] / np.maximum(stat["lanes"], 1))[tj])
    add("rank_norm", stat["rank"][tj] / max(nL - 1, 1))
    add("tod", tod_t[ti])
    add("dow", G["dow"][ti])
    add("weekend", (G["dow"][ti] >= 5))
    add("pct_observed", pct[ti, tj])
    if "regime" in G:
        add("regime", np.repeat(G["regime"], 288)[ti])

    # ---- cross-channel identities ----
    with np.errstate(invalid="ignore", divide="ignore"):
        extra = [
            ("k_from_tlin", get("fl_tlin") / np.maximum(get("sp_tlin"), 1.0)),
            ("v_from_qk", get("fl_tlin") / np.maximum(get("oc_tlin"), 1e-4) * 1e-3),
            ("sp_over_vf", get("sp_tlin") / np.maximum(get("vf"), 1.0)),
            ("fl_over_cap", get("fl_tlin") / np.maximum(get("cap_pl"), 1.0)),
            ("cons_minus_tlin", get("cons_mean_pl") - get("fl_tlin")),
            ("sp_tlin_minus_prof", get("sp_tlin") - get("sp_prof_loc")),
            ("fl_tlin_minus_prof", get("fl_tlin") - get("fl_prof_loc")),
        ]
    for nm, v in extra:
        add(nm, v)

    # fill the design matrix column by column so the block list can be released
    X = np.empty((len(ti), len(blocks)), np.float32)
    for i in range(len(blocks)):
        X[:, i] = blocks[i]
        blocks[i] = None
    del blocks
    gc.collect()

    res = dict(X=X, cols=cols, ti=ti, tj=tj, link_ids=G["link_ids"],
               days=G["days"], lanes=lanes_v)
    if truth is not None:
        Tz = load_grid(panel, truth, masked=False)
        o = [Tz["link_ids"].index(l) for l in G["link_ids"]]
        res["y_speed"] = Tz["speed"][:, o][ti, tj]
        res["y_flowpl"] = (Tz["flow"][:, o] / lanes_v[None, :])[ti, tj]
        del Tz
    del sp, oc, fpl, tot, avail, G
    gc.collect()
    return res
